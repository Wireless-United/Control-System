
import numpy as np
import matplotlib.pyplot as plt
import logging

from pypower.api import ppoption
from pypower.case39 import case39
from pypower.runpf import runpf

# Reuse your IEEE-style dynamic building blocks
from dynamic_models import IEEET1, IEEET1Params, TGOV1, TGOV1Params, GenParams, AGC, AGCParams , BatteryStorage, BatteryParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _coerce_index_dtypes(c):
    """Make sure all index columns are integer dtype for PYPOWER indexing."""
    # bus: BUS_I is col 0
    c['bus'][:, 0] = np.asarray(np.round(c['bus'][:, 0]), dtype=np.int64)
    # gen: GEN_BUS is col 0
    c['gen'][:, 0] = np.asarray(np.round(c['gen'][:, 0]), dtype=np.int64)
    # branch: F_BUS, T_BUS are cols 0,1
    c['branch'][:, 0:2] = np.asarray(np.round(c['branch'][:, 0:2]), dtype=np.int64)
    # areas (if present): col 0 is AREA number, col 1 is REF_BUS
    if 'areas' in c and c['areas'] is not None and len(c['areas']) > 0:
        c['areas'][:, 0:2] = np.asarray(np.round(c['areas'][:, 0:2]), dtype=np.int64)
    return c


def _fix_case_indices(case):
    case['bus'][:, 0] = np.round(case['bus'][:, 0]).astype(np.int64)
    case['gen'][:, 0] = np.round(case['gen'][:, 0]).astype(np.int64)
    case['branch'][:, 0:2] = np.round(case['branch'][:, 0:2]).astype(np.int64)
    if 'areas' in case:
        case['areas'][:, 0:2] = np.round(case['areas'][:, 0:2]).astype(np.int64)
    return case

class IEEE39QSTS:
    '''
    Quasi–steady-state time-domain simulation on IEEE 39-bus system that
    respects IEEE-style control blocks:
      - AVR (IEEET1) -> updates generator bus voltage setpoints (VG in PYPOWER)
      - Governor (TGOV1) -> updates generator active power setpoints (PG)
      - AGC secondary control -> adjusts governor Pref via participation factors
    Network equations are solved each time step with AC power flow (ENFORCE_Q_LIMS=1).
    '''
    def __init__(self, dt=0.02, t_end=30.0, beta=20.0):
        self.dt = float(dt)
        self.t_end = float(t_end)

        # Load case and sanitize indices
        self.case = _fix_case_indices(case39())
        self.case = _coerce_index_dtypes(self.case)
        self.baseMVA = float(self.case['baseMVA'])

        # Initial power flow
        ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0, ENFORCE_Q_LIMS=1)
        self.pf, ok = runpf(self.case, ppopt)
        if not ok:
            raise RuntimeError("Initial power flow did not converge.")

        # Build generator objects (one per online generator row)
        self.gens = []
        for gi in range(self.pf['gen'].shape[0]):
            bus_num = int(self.pf['gen'][gi, 0])
            bus_idx = bus_num - 1
            Vt0 = float(self.pf['bus'][bus_idx, 7])  # VM in p.u.
            Pg0 = float(self.pf['gen'][gi, 1]) / self.baseMVA

            exc = IEEET1(IEEET1Params(
                KA=50.0, TA=0.1, TR=0.05, KE=1.0, TE=0.5,
                VRMAX=3.0, VRMIN=-3.0, EMIN=0.0, EMAX=3.0,
                Vref=Vt0, S10=0.1, S12=0.5
            ))
            # Initialize exciter around steady state
            exc.Vt_f = Vt0

            gov = TGOV1(TGOV1Params(
                R=0.05, TG=0.5, TT=1.0, RATE=0.05,
                Pmax=float(self.pf['gen'][gi, 8]) / self.baseMVA,
                Pmin=float(self.pf['gen'][gi, 9]) / self.baseMVA,
                Pref=Pg0, DB=0.001
            ))
            gov.xg = Pg0; gov.xt = Pg0; gov.xg_prev = Pg0

            gen_par = GenParams(H=5.0, D=2.0, MVA_base=self.baseMVA)

            self.gens.append({
                'bus_idx': bus_idx,
                'exc': exc,
                'gov': gov,
                'par': gen_par,
                'Pm_pu': Pg0,
                'VG_sp': Vt0,   # generator bus voltage setpoint used in PF (VG column)
            })

        # Secondary control (AGC)
        self.agc = AGC(AGCParams(beta=float(beta), Kp=50.0, Ki=0.05, ACE_max=100.0))
        self.agc.set_participation(list(range(len(self.gens))), [1.0]*len(self.gens))

        # Frequency state (center-of-inertia approximation)
        self.f0 = 60.0
        self.freq_Hz = 60.0
        self.freq_min, self.freq_max = 59.0, 61.0
        self.H_sys = sum(g['par'].H * g['par'].MVA_base for g in self.gens) / self.baseMVA
        self.D_sys = sum(g['par'].D * g['par'].MVA_base for g in self.gens) / self.baseMVA

        # Load scaling
        self.original_PD = self.case['bus'][:, 2].copy()
        self.original_QD = self.case['bus'][:, 3].copy()
        self.load_factor = 1.0

        # Logging
        self.logs = {k: [] for k in [
            't', 'freq_Hz', 'sumGen_MW', 'sumLoad_MW', 'Vmax', 'Vmin', 'ACE', 'AGC_signal', 'load_factor'
        ]}

    # ---------- Load model (simple daily + step disturbance) ----------
    def _update_loads(self, t):
        # Base daily ripple (small and slow) + optional step at t=5s
        diurnal = 1.0 + 0.02*np.sin(2*np.pi * t / 86400.0)
        step = 1.10 if (t >= 5.0) else 1.00
        target = diurnal * step
        # Rate-limit the change to emulate thermostatic diversity
        rate = 0.05  # per second
        max_delta = rate * self.dt
        delta = np.clip(target - self.load_factor, -max_delta, max_delta)
        self.load_factor += delta

        # Apply to PD only; keep QD constant power for simplicity
        self.case['bus'][:, 2] = self.original_PD * self.load_factor
        return self.load_factor

    # ---------- Frequency integration (COI), RK4 ----------
    def _coi_freq_rk4(self, sum_Pm_MW, sum_Pe_MW):
        def deriv(f):
            dP = sum_Pm_MW - sum_Pe_MW
            freq_error_pu = (f - self.f0) / self.f0
            damping_MW = self.D_sys * freq_error_pu * self.baseMVA
            dfdt_pu = (dP - damping_MW) / (2.0 * self.H_sys * self.baseMVA)
            return self.f0 * dfdt_pu
        k1 = deriv(self.freq_Hz)
        k2 = deriv(self.freq_Hz + 0.5*self.dt*k1)
        k3 = deriv(self.freq_Hz + 0.5*self.dt*k2)
        k4 = deriv(self.freq_Hz + self.dt*k3)
        dfdt = (k1 + 2*k2 + 2*k3 + k4)/6.0
        dfdt = np.clip(dfdt, -2.0, 2.0)  # Hz/s clamp
        self.freq_Hz = np.clip(self.freq_Hz + self.dt*dfdt, self.freq_min, self.freq_max)

    # ---------- One simulation step ----------
    def _step_with_control_convergence(self, t):
        """Enhanced step method with AGC/AVR convergence iterations"""
        
        # Apply load changes
        lf = self._update_loads(t)
        
        # Initialize convergence tracking
        freq_prev = self.freq_Hz
        iterations = 0
        converged = False
        
        logging.debug(f"t={t:.3f}s: Starting control convergence, initial freq={freq_prev:.4f} Hz")
        
        while not converged and iterations < self.max_control_iterations:
            iterations += 1
            
            # AGC step
            agc_signal_prev = self.agc.get_control_signals().get('AGC_output', 0.0)
            agc_signal = self.agc.step(self.freq_Hz, self.dt, target_freq=self.f0)
            
            # Governor updates
            for gi, g in enumerate(self.gens):
                if hasattr(self.agc, 'participation') and gi in self.agc.participation:
                    share = self.agc.participation[gi]
                    delta_pref = (agc_signal * share) / g['par'].MVA_base
                    g['gov'].p.Pref = np.clip(g['gov'].p.Pref + delta_pref, g['gov'].p.Pmin, g['gov'].p.Pmax)
                
                w_pu = self.freq_Hz / self.f0
                Pm = g['gov'].step(w_pu, dt=self.dt)
                g['Pm_pu'] += np.clip(Pm - g['Pm_pu'], -0.01, 0.01)
            
            # AVR updates
            Vt = self.pf['bus'][:, 7] if hasattr(self, 'pf') else np.ones(len(self.gens))
            for g in self.gens:
                Vmeas = float(Vt[g['bus_idx']])
                Efd = g['exc'].step(Vmeas, dt=self.dt)
                Verr = g['exc'].p.Vref - Vmeas
                Kv = 5.0
                dVG = Kv * Verr * self.dt
                g['VG_sp'] = float(np.clip(g.get('VG_sp', Vmeas) + dVG, 0.90, 1.10))
            
            # Update power flow
            for gi, g in enumerate(self.gens):
                PG_MW = np.clip(g['Pm_pu'] * g['par'].MVA_base,
                                self.case['gen'][gi, 9], self.case['gen'][gi, 8])
                self.case['gen'][gi, 1] = PG_MW
                self.case['gen'][gi, 5] = g['VG_sp']
            
            # Solve power flow
            self.case = _coerce_index_dtypes(self.case)
            ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0, ENFORCE_Q_LIMS=1)
            self.pf, ok = runpf(self.case, ppopt)
            
            if not ok:
                logging.warning(f"t={t:.3f}s: Power flow convergence failed at iteration {iterations}")
                break
            
            # Update frequency
            sum_Pm_MW = sum(g['Pm_pu'] * g['par'].MVA_base for g in self.gens)
            sum_Pe_MW = float(np.sum(self.pf['gen'][:, 1]))
            self._coi_freq_rk4(sum_Pm_MW, sum_Pe_MW)
            
            # Check convergence
            freq_change = abs(self.freq_Hz - freq_prev)
            if freq_change < self.freq_tolerance:
                converged = True
                logging.debug(f"t={t:.3f}s: Converged in {iterations} iterations, freq={self.freq_Hz:.4f} Hz")
            else:
                freq_prev = self.freq_Hz
                logging.debug(f"t={t:.3f}s: Iteration {iterations}, freq={self.freq_Hz:.4f} Hz, change={freq_change:.6f}")
        
        if not converged:
            logging.warning(f"t={t:.3f}s: Control did not converge in {iterations} iterations")
        
        # Log results
        self._log_step_results(t, lf, agc_signal, iterations, converged)

    def _log_step_results(self, t, load_factor, agc_signal, iterations, converged):
        """Log step results with detailed control information"""
        sum_Pe_MW = float(np.sum(self.pf['gen'][:, 1]))
        sum_Pd_MW = float(np.sum(self.pf['bus'][:, 2]))
        
        self.logs['t'].append(t)
        self.logs['freq_Hz'].append(self.freq_Hz)
        self.logs['sumGen_MW'].append(sum_Pe_MW)
        self.logs['sumLoad_MW'].append(sum_Pd_MW)
        self.logs['Vmax'].append(float(np.max(self.pf['bus'][:, 7])))
        self.logs['Vmin'].append(float(np.min(self.pf['bus'][:, 7])))
        self.logs['ACE'].append(self.agc.p.beta * (self.freq_Hz - self.f0))
        self.logs['AGC_signal'].append(agc_signal)
        self.logs['load_factor'].append(load_factor)
        
        # Add convergence tracking
        if 'control_iterations' not in self.logs:
            self.logs['control_iterations'] = []
        self.logs['control_iterations'].append(iterations)
        
        # Periodic detailed logging
        if int(t / self.dt) % max(1, int(1.0 / self.dt)) == 0:  # Every second
            logging.info(f"t={t:6.2f}s | f={self.freq_Hz:7.4f}Hz | Gen={sum_Pe_MW:7.1f}MW | Load={sum_Pd_MW:7.1f}MW | "
                        f"ACE={self.logs['ACE'][-1]:6.2f} | Iter={iterations} | Conv={'✓' if converged else '✗'}")
        # ---------- Public API ----------
        def run(self):
            steps = int(self.t_end / self.dt) + 1
            logger.info(f"Running IEEE39 QSTS: dt={self.dt}s, t_end={self.t_end}s, steps={steps}")
            for k in range(steps):
                t = k * self.dt
                self._step(t)
                if k % max(1, steps//10) == 0:
                    logger.info(f"t={t:.2f}s  f={self.freq_Hz:.3f} Hz  Gen={self.logs['sumGen_MW'][-1]:.1f} MW  Load={self.logs['sumLoad_MW'][-1]:.1f} MW")
            logger.info("Simulation finished.")
            return self.logs

        def plot(self):
            if not self.logs['t']:
                logger.warning("No data to plot.")
                return
            t = np.array(self.logs['t'])
            fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
            fig.suptitle("IEEE 39‑Bus QSTS (AVR + TGOV1 + AGC, AC PF each step)", fontsize=14)

            ax[0].plot(t, self.logs['freq_Hz'], linewidth=2)
            ax[0].axhline(60.0, ls='--')
            ax[0].set_ylabel("Freq (Hz)"); ax[0].grid(True)

            ax[1].plot(t, self.logs['sumGen_MW'], linewidth=2, label="Gen")
            ax[1].plot(t, self.logs['sumLoad_MW'], linewidth=2, label="Load")
            ax[1].set_ylabel("Power (MW)"); ax[1].legend(); ax[1].grid(True)

            ax[2].plot(t, self.logs['Vmax'], linewidth=2, label="Vmax")
            ax[2].plot(t, self.logs['Vmin'], linewidth=2, label="Vmin")
            ax[2].set_ylabel("Bus V (pu)"); ax[2].legend(); ax[2].grid(True)

            ax[3].plot(t, self.logs['ACE'], linewidth=2, label="ACE")
            ax[3].plot(t, self.logs['AGC_signal'], linewidth=2, label="AGC signal")
            ax[3].set_ylabel("AGC"); ax[3].set_xlabel("Time (s)"); ax[3].legend(); ax[3].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
