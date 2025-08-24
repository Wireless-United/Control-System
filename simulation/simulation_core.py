# simulation_core.py
# Contains the main simulation class with enhanced IEEE-style dynamics
# Author: Power System Engineer

import numpy as np
import matplotlib.pyplot as plt
from pypower.api import runpf, ppoption
from pypower.case39 import case39
import logging

# Import the dynamic model classes
from dynamic_models import Generator, IEEET1, IEEET1Params, TGOV1, TGOV1Params, GenParams, ZIPLoad, AGC, AGCParams, GeneratorEMFParams

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load and power convergence controllers
# -----------------------------------------------------------------------------
class LoadController:
    def __init__(self, case, target_load_factor=1.0):
        self.case = case
        self.original_loads = case['bus'][:, 2].copy()
        self.target_factor = target_load_factor
        self.current_factor = 1.0

    def gradual_load_change(self, target_factor, rate_per_second=0.1, dt=0.005):
        max_change = rate_per_second * dt
        change = np.clip(target_factor - self.current_factor, -max_change, max_change)
        self.current_factor += change
        for i, original_load in enumerate(self.original_loads):
            if original_load > 0:
                self.case['bus'][i, 2] = original_load * self.current_factor
        return self.current_factor

    def emergency_load_shed(self, shed_percentage=0.1, priority_buses=None):
        if priority_buses is None:
            load_indices = np.argsort(-self.case['bus'][:, 2])
            priority_buses = load_indices[:10]
        total_shed = 0.0
        for bus_idx in priority_buses:
            if self.case['bus'][bus_idx, 2] > 0:
                shed_amount = self.case['bus'][bus_idx, 2] * shed_percentage
                self.case['bus'][bus_idx, 2] -= shed_amount
                total_shed += shed_amount
        return total_shed

class PowerConvergenceController:
    def __init__(self, generators):
        self.generators = generators
        self.total_capacity = sum(g.par.MVA_base for g in generators)

    def balance_power(self, load_demand_MW, dt=0.005):
        current_gen = sum(g.Pm_pu * g.par.MVA_base for g in self.generators)
        power_error = load_demand_MW - current_gen
        if abs(power_error) < 1.0:
            return True
        for g in self.generators:
            capacity_fraction = g.par.MVA_base / self.total_capacity
            power_adjustment = power_error * capacity_fraction
            max_rate = 0.05 * g.par.MVA_base * dt
            power_adjustment = np.clip(power_adjustment, -max_rate, max_rate)
            delta_pref = power_adjustment / g.par.MVA_base
            new_pref = g.gov.p.Pref + delta_pref
            g.gov.p.Pref = np.clip(new_pref, g.gov.p.Pmin, g.gov.p.Pmax)
        return abs(power_error) < 10.0

# -----------------------------------------------------------------------------
# Simulation core
# -----------------------------------------------------------------------------
class IEEE39DynamicSim:
    def __init__(self, dt=0.005, t_end=20.0, use_emf_model=False):
        self.dt = dt
        self.t_end = t_end
        self.use_emf_model = use_emf_model
        self.case = case39()
        self.baseMVA = self.case['baseMVA']
        self.original_case = case39()

        # Initial PF
        ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0, ENFORCE_Q_LIMS=1)
        self.pf_result, pf_ok = runpf(self.case, ppopt)
        if not pf_ok:
            raise RuntimeError("Initial power flow failed!")

        logger.info(f"Initial: Gen={np.sum(self.pf_result['gen'][:, 1]):.1f}MW, Load={np.sum(self.pf_result['bus'][:, 2]):.1f}MW")

        nG = self.case['gen'].shape[0]
        self.gens = []
        for gi in range(nG):
            gen_bus_idx = int(self.pf_result['gen'][gi, 0]) - 1
            steady_state_v = self.pf_result['bus'][gen_bus_idx, 7]
            steady_state_pg = self.pf_result['gen'][gi, 1] / self.baseMVA
            steady_state_qg = self.pf_result['gen'][gi, 2] / self.baseMVA

            exc_params = IEEET1Params(
                KA=50.0, TA=0.1, TR=0.05, KE=1.0, TE=0.5,
                VRMAX=3.0, VRMIN=-3.0, EMIN=0.0, EMAX=3.0,
                Vref=steady_state_v, S10=0.1, S12=0.5
            )
            gov_params = TGOV1Params(
                R=0.05, TG=0.5, TT=1.0, RATE=0.05,
                Pmax=min(1.5 * steady_state_pg, self.case['gen'][gi, 8]/self.baseMVA),
                Pmin=max(0.1 * steady_state_pg, self.case['gen'][gi, 9]/self.baseMVA),
                Pref=steady_state_pg, DB=0.001
            )
            gen_params = GenParams(H=5.0, D=2.0, MVA_base=self.baseMVA)
            emf_params = GeneratorEMFParams() if use_emf_model else None
            gen = Generator(gi, gen_params, IEEET1(exc_params), TGOV1(gov_params), emf_params)
            gen.Pm_pu = steady_state_pg
            gen.Pe_pu = steady_state_pg
            gen.Qgen_pu = steady_state_qg
            gen.w = 1.0
            gen.delta = 0.0

            # AVR steady state init
            gen.exc.Vt_f = steady_state_v
            Verr_ss = exc_params.Vref - steady_state_v
            Vr_ideal = exc_params.KA * Verr_ss
            gen.exc.Vr = np.clip(Vr_ideal, exc_params.VRMIN, exc_params.VRMAX)
            gen.exc.Vr_limited = gen.exc.Vr
            gen.exc.Efd = steady_state_v
            for _ in range(10):
                Se_val = gen.exc.Se(gen.exc.Efd)
                residual = gen.exc.Efd - (gen.exc.Vr - Se_val) / exc_params.KE
                if abs(residual) < 1e-6:
                    break
                eps = 1e-6
                Se_plus = gen.exc.Se(gen.exc.Efd + eps)
                dSe = (Se_plus - Se_val) / eps
                jacobian = 1.0 + dSe / exc_params.KE
                if abs(jacobian) > 1e-8:
                    gen.exc.Efd -= residual / jacobian
            gen.exc.Efd_limited = gen.exc.Efd

            # Governor init
            gen.gov.xg = steady_state_pg
            gen.gov.xt = steady_state_pg
            gen.gov.xg_prev = steady_state_pg
            self.gens.append(gen)

        self.load_controller = LoadController(self.case)
        self.power_controller = PowerConvergenceController(self.gens)
        self.zip_loads = {i: ZIPLoad(P0=bus_data[2], Q0=bus_data[3], a=0.1, b=0.3, c=0.6, Kf=1.2)
                          for i, bus_data in enumerate(self.case['bus']) if bus_data[2] > 0}
        self.f0 = 60.0
        self.freq_Hz = self.f0
        self.freq_min, self.freq_max = 59.0, 61.0
        self.H_sys = sum(g.par.H * g.par.MVA_base for g in self.gens) / self.baseMVA
        self.D_sys = sum(g.par.D * g.par.MVA_base for g in self.gens) / self.baseMVA
        agc_params = AGCParams(beta=20.0, Kp=50.0, Ki=0.05, ACE_max=100.0)
        self.agc = AGC(agc_params)
        gen_indices = list(range(nG))
        equal_factors = [1.0] * nG
        self.agc.set_participation(gen_indices, equal_factors)
        self.logs = {'t': [], 'freq_Hz': [], 'Pg_MW': [], 'Load_MW': [], 'Vmax': [], 'Vmin': [],
                     'gen_speeds': [], 'ACE': [], 'AGC_signal': [], 'Q_violations': [], 'load_factor': [], 'power_imbalance': []}

    # -------------------------------------------------------------------------
    def coi_frequency_rk4(self, sum_Pm_MW, sum_Pe_MW):
        def freq_derivative(freq):
            dP = sum_Pm_MW - sum_Pe_MW
            freq_error_pu = (freq - self.f0) / self.f0
            damping_MW = self.D_sys * freq_error_pu * self.baseMVA
            dfdt_pu = (dP - damping_MW) / (2.0 * self.H_sys * self.baseMVA)
            return self.f0 * dfdt_pu
        k1 = freq_derivative(self.freq_Hz)
        k2 = freq_derivative(self.freq_Hz + 0.5*self.dt*k1)
        k3 = freq_derivative(self.freq_Hz + 0.5*self.dt*k2)
        k4 = freq_derivative(self.freq_Hz + self.dt*k3)
        dfdt = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        dfdt = np.clip(dfdt, -2.0, 2.0)
        self.freq_Hz += self.dt * dfdt
        self.freq_Hz = np.clip(self.freq_Hz, self.freq_min, self.freq_max)

    # -------------------------------------------------------------------------
    def apply_load_profile(self, t):
        base_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t / 86400)
        noise_factor = 1.0 + 0.02 * np.random.normal()
        total_factor = base_factor * noise_factor
        self.load_controller.gradual_load_change(total_factor, rate_per_second=0.05, dt=self.dt)
        return total_factor

    def emergency_event(self, t, event_time=5.0, event_type="load_jump"):
        if abs(t - event_time) < self.dt/2:
            if event_type == "load_jump":
                logger.info(f"EMERGENCY: 15% load jump at t={t:.2f}s")
                self.load_controller.gradual_load_change(1.15, rate_per_second=2.0, dt=self.dt)
            elif event_type == "gen_trip":
                max_gen = max(self.gens, key=lambda g: g.par.MVA_base)
                max_gen.gov.p.Pmax = 0.0
                max_gen.gov.p.Pmin = 0.0
                logger.info(f"EMERGENCY: Generator {max_gen.idx} tripped at t={t:.2f}s")
            elif event_type == "line_outage":
                critical_buses = [15, 20, 25]
                for bus_idx in critical_buses:
                    if bus_idx < len(self.case['bus']):
                        self.case['bus'][bus_idx, 2] *= 1.1
                logger.info(f"EMERGENCY: Line outage simulation at t={t:.2f}s")

    def check_and_enforce_q_limits(self):
        q_violations = 0
        for gi, g in enumerate(self.gens):
            if gi < len(self.case['gen']):
                current_qg = self.case['gen'][gi, 2]
                qmax = self.case['gen'][gi, 3]
                qmin = self.case['gen'][gi, 4]
                if current_qg > qmax and not g.Q_limited:
                    self.case['gen'][gi, 2] = qmax
                    g.Q_limited = True
                    g.exc.p.Vref += 0.001
                    q_violations += 1
                elif current_qg < qmin and not g.Q_limited:
                    self.case['gen'][gi, 2] = qmin
                    g.Q_limited = True
                    g.exc.p.Vref -= 0.001
                    q_violations += 1
                elif g.Q_limited and qmin < current_qg < qmax:
                    g.Q_limited = False
        return q_violations

    # -------------------------------------------------------------------------
    def run(self):
        t = 0.0
        steps = int(self.t_end / self.dt) + 1
        logger.info(f"Starting simulation. dt={self.dt}s, t_end={self.t_end}s, steps={steps}")

        Vt0 = self.pf_result['bus'][self.case['gen'][:, 0].astype(int)-1, 7]
        Pg0 = self.pf_result['gen'][:, 1]
        Load0 = np.sum(self.pf_result['bus'][:, 2])

        for k in range(steps):
            t = k * self.dt
            load_factor = self.apply_load_profile(t)
            self.emergency_event(t, event_time=5.0, event_type="load_jump")
            total_load = np.sum(self.case['bus'][:, 2])
            self.power_controller.balance_power(total_load, self.dt)

            # Governors
            for g in self.gens:
                w_input = self.freq_Hz / self.f0
                new_Pm = g.gov.step(w_input, dt=self.dt)
                g.Pm_pu += np.clip(new_Pm - g.Pm_pu, -0.005, 0.005)

            # AGC
            agc_signal = self.agc.step(self.freq_Hz, self.dt)
            for gi, g in enumerate(self.gens):
                if gi in self.agc.participation and not g.Q_limited:
                    agc_contribution = agc_signal * self.agc.participation[gi]
                    delta_pref = agc_contribution / g.par.MVA_base
                    g.gov.p.Pref = np.clip(g.gov.p.Pref + delta_pref, g.gov.p.Pmin, g.gov.p.Pmax)

            q_violations = self.check_and_enforce_q_limits()

            Pg_MW = self.case['gen'][:, 1]
            sum_Pm_MW = sum(g.Pm_pu * g.par.MVA_base for g in self.gens)
            sum_Pe_MW = np.sum(Pg_MW)
            for gi, g in enumerate(self.gens):
                if gi < len(Pg_MW):
                    g.Pe_pu = Pg_MW[gi] / g.par.MVA_base
                g.w = self.freq_Hz / self.f0
                g.swing_step_rk4(self.dt)
            self.coi_frequency_rk4(sum_Pm_MW, sum_Pe_MW)

            self.logs['t'].append(t)
            self.logs['freq_Hz'].append(self.freq_Hz)
            self.logs['Pg_MW'].append(sum_Pe_MW)
            self.logs['Load_MW'].append(total_load)
            self.logs['Vmax'].append(np.max(self.case['bus'][:, 7]))
            self.logs['Vmin'].append(np.min(self.case['bus'][:, 7]))
            self.logs['gen_speeds'].append([g.w for g in self.gens])
            ace_val = self.agc.p.beta * (self.freq_Hz - 60.0)
            power_imbalance = sum_Pm_MW - sum_Pe_MW
            self.logs['ACE'].append(ace_val)
            self.logs['AGC_signal'].append(agc_signal)
            self.logs['Q_violations'].append(q_violations)
            self.logs['load_factor'].append(load_factor)
            self.logs['power_imbalance'].append(power_imbalance)

            if k % 200 == 0:
                logger.info(f"t={t:.2f}s f={self.freq_Hz:.3f}Hz Gen={sum_Pe_MW:.1f}MW Load={total_load:.1f}MW")

        logger.info("Simulation complete")
        return self.logs

    # -------------------------------------------------------------------------
    def plot(self):
        if not self.logs['t']:
            logger.error("No data to plot")
            return
        t = np.array(self.logs['t'])
        fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle("Power System Simulation Results (IEEE 39-Bus)", fontsize=16)
        ax[0].plot(t, self.logs['freq_Hz'], 'b-', linewidth=2)
        ax[0].axhline(60.0, ls='--', color='r', alpha=0.7)
        ax[0].set_ylabel('Frequency (Hz)')
        ax[0].grid(True)
        ax[1].plot(t, self.logs['Pg_MW'], 'g-', linewidth=2, label='Gen')
        ax[1].plot(t, self.logs['Load_MW'], 'r-', linewidth=2, label='Load')
        ax[1].set_ylabel('Power (MW)')
        ax[1].legend()