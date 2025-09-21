import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pandapower as pp
import pandapower.networks as pn
import pandapower.converter

from controllers import (
    IEEET1, IEEET1Params,
    TGOV1, TGOV1Params,
    AGC, AGCParams,
    BatteryStorage, BatteryParams
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add GenParams class if not in controllers
class GenParams:
    def __init__(self, H=5.0, D=2.0, MVA_base=100.0):
        self.H = H
        self.D = D
        self.MVA_base = MVA_base

class IEEE39QSTS:
    '''
    Quasi-steady-state time-domain simulation on IEEE 39-bus system using pandapower
    that respects IEEE-style control blocks:
      - AVR (IEEET1) -> updates generator bus voltage setpoints
      - Governor (TGOV1) -> updates generator active power setpoints
      - AGC secondary control -> adjusts governor Pref via participation factors
    Network equations are solved each time step with pandapower AC power flow.
    '''
    def __init__(self, dt=0.02, t_end=30.0, beta=20.0):
        self.dt = float(dt)
        self.t_end = float(t_end)
        
        # Convergence parameters (will be set by user input)
        self.max_control_iterations = 10
        self.freq_tolerance = 0.001
        self.step_disturbance = {'enabled': False}

        # Create IEEE39 network using pandapower
        self._create_ieee39_network()
        
        # Run initial power flow
        try:
            pp.runpp(self.net, algorithm='nr', init='flat', calculate_voltage_angles=True)
            if not self.net.converged:
                raise RuntimeError("Initial power flow did not converge.")
        except Exception as e:
            raise RuntimeError(f"Initial power flow failed: {str(e)}")

        # Build generator objects
        self._initialize_generators()
        
        # Initialize AGC
        self._initialize_agc(beta)
        
        # Add battery storage
        self.battery = BatteryStorage(BatteryParams(
            capacity_MWh=100.0,
            soc_init=0.5,
            max_power_MW=50.0
        ))
        
        # System frequency parameters
        self._initialize_frequency_dynamics()
        
        # Load scaling
        self.original_PD = self.net.load.p_mw.copy()
        self.original_QD = self.net.load.q_mvar.copy()
        self.load_factor = 1.0

        # Logging
        self.logs = {k: [] for k in [
            't', 'freq_Hz', 'sumGen_MW', 'sumLoad_MW', 'Vmax', 'Vmin', 
            'ACE', 'AGC_signal', 'load_factor', 'control_iterations',
            'battery_soc', 'battery_power'
        ]}

    def _create_ieee39_network(self):
        """Create IEEE39 network using pandapower"""
        try:
            # Try to use built-in IEEE39 case if available
            self.net = pn.case_ieee39()
            logger.info("Created IEEE39 network using pandapower built-in case")
        except:
            # Fallback: create from pypower case and convert
            try:
                from pypower.case39 import case39
                ppc = case39()
                self.net = pandapower.converter.from_ppc(ppc, f_hz=60)
                logger.info("Created IEEE39 network by converting from pypower case")
            except Exception as e:
                logger.error(f"Failed to create IEEE39 network: {e}")
                # Create a simplified network as last resort
                self.net = self._create_simplified_network()
                logger.warning("Created simplified test network instead of IEEE39")
        
        # Ensure we have controllable generators
        if len(self.net.gen) == 0:
            raise RuntimeError("No generators found in network")
        
        self.baseMVA = 100.0  # Standard base MVA
        logger.info(f"Network created: {len(self.net.bus)} buses, {len(self.net.gen)} generators, {len(self.net.load)} loads")

    def _create_simplified_network(self):
        """Create a simplified test network if IEEE39 is not available"""
        net = pp.create_empty_network(f_hz=60.0, sn_mva=100.0)
        
        # Create buses
        bus1 = pp.create_bus(net, vn_kv=345, name="Gen Bus 1")
        bus2 = pp.create_bus(net, vn_kv=345, name="Gen Bus 2")
        bus3 = pp.create_bus(net, vn_kv=345, name="Load Bus 1")
        bus4 = pp.create_bus(net, vn_kv=345, name="Load Bus 2")
        
        # Create generators
        pp.create_gen(net, bus=bus1, p_mw=300, vm_pu=1.0, controllable=True, name="Gen 1")
        pp.create_gen(net, bus=bus2, p_mw=200, vm_pu=1.0, controllable=True, name="Gen 2")
        
        # Create loads
        pp.create_load(net, bus=bus3, p_mw=150, q_mvar=50, name="Load 1")
        pp.create_load(net, bus=bus4, p_mw=100, q_mvar=30, name="Load 2")
        
        # Create lines
        pp.create_line(net, from_bus=bus1, to_bus=bus3, length_km=100, std_type="490-AL1/64-ST1A 380.0")
        pp.create_line(net, from_bus=bus2, to_bus=bus4, length_km=80, std_type="490-AL1/64-ST1A 380.0")
        pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=50, std_type="490-AL1/64-ST1A 380.0")
        
        # Create external grid (slack bus)
        pp.create_ext_grid(net, bus=bus1, vm_pu=1.0, va_degree=0.0)
        
        return net

    def _initialize_generators(self):
        """Initialize generator control objects"""
        self.gens = []
        
        for gi, gen_row in self.net.gen.iterrows():
            bus_idx = gen_row['bus']
            
            # Get initial conditions from power flow results
            if hasattr(self.net, 'res_gen') and not self.net.res_gen.empty:
                Vt0 = float(self.net.res_bus.loc[bus_idx, 'vm_pu'])
                Pg0 = float(self.net.res_gen.loc[gi, 'p_mw']) / self.baseMVA
            else:
                Vt0 = 1.0
                Pg0 = float(gen_row['p_mw']) / self.baseMVA

            # Create exciter (AVR)
            exc = IEEET1(IEEET1Params(
                KA=50.0, TA=0.1, TR=0.05, KE=1.0, TE=0.5,
                VRMAX=3.0, VRMIN=-3.0, EMIN=0.0, EMAX=3.0,
                Vref=Vt0, S10=0.1, S12=0.5
            ))
            exc.Vt_f = Vt0

            # Create governor
            Pmax = float(gen_row.get('max_p_mw', gen_row['p_mw'] * 1.5)) / self.baseMVA
            Pmin = float(gen_row.get('min_p_mw', gen_row['p_mw'] * 0.1)) / self.baseMVA
            
            gov = TGOV1(TGOV1Params(
                R=0.05, TG=0.5, TT=1.0, RATE=0.05,
                Pmax=Pmax, Pmin=Pmin, Pref=Pg0, DB=0.001
            ))
            gov.xg = Pg0
            gov.xt = Pg0

            # Generator parameters
            gen_par = GenParams(H=5.0, D=2.0, MVA_base=self.baseMVA)

            self.gens.append({
                'index': gi,
                'bus_idx': bus_idx,
                'exc': exc,
                'gov': gov,
                'par': gen_par,
                'Pm_pu': Pg0,
                'VM_sp': Vt0,  # voltage setpoint
            })
        
        logger.info(f"Initialized {len(self.gens)} generator controllers")

    def _initialize_agc(self, beta):
        """Initialize AGC controller"""
        self.agc = AGC(AGCParams(beta=float(beta), Kp=50.0, Ki=0.05, ACE_max=100.0))
        
        # Set participation factors (equal participation for now)
        gen_indices = list(range(len(self.gens)))
        participation_factors = [1.0] * len(self.gens)
        self.agc.set_participation(gen_indices, participation_factors)
        
        logger.info(f"Initialized AGC with beta={beta}")

    def _initialize_frequency_dynamics(self):
        """Initialize system frequency dynamics parameters"""
        self.f0 = 60.0
        self.freq_Hz = 60.0
        self.freq_min, self.freq_max = 59.0, 61.0
        
        # Calculate system inertia and damping
        total_H_MVA = sum(g['par'].H * g['par'].MVA_base for g in self.gens)
        total_D_MVA = sum(g['par'].D * g['par'].MVA_base for g in self.gens)
        
        self.H_sys = total_H_MVA / self.baseMVA
        self.D_sys = total_D_MVA / self.baseMVA
        
        logger.info(f"System inertia H={self.H_sys:.2f}, damping D={self.D_sys:.2f}")

    def _update_loads(self, t):
        """Update loads based on user inputs and step disturbances"""
        # Base load factor from user input
        base_factor = self.load_factor
        
        # Add step disturbance if configured
        if self.step_disturbance['enabled'] and t >= self.step_disturbance.get('time', 5.0):
            step_factor = self.step_disturbance.get('magnitude', 1.1)
        else:
            step_factor = 1.0
        
        # Small diurnal variation (optional)
        diurnal = 1.0 + 0.005 * np.sin(2 * np.pi * t / 86400.0)
        
        # Combined target
        target = base_factor * step_factor * diurnal
        
        # Rate-limit the change
        rate = 0.1  # per second
        max_delta = rate * self.dt
        delta = np.clip(target - self.load_factor, -max_delta, max_delta)
        current_factor = self.load_factor + delta
        
        # Apply to loads
        self.net.load.p_mw = self.original_PD * current_factor
        self.net.load.q_mvar = self.original_QD * current_factor
        
        return current_factor

    def _coi_freq_rk4(self, sum_Pm_MW, sum_Pe_MW):
        """Frequency dynamics using center-of-inertia and RK4 integration"""
        def deriv(f):
            dP = sum_Pm_MW - sum_Pe_MW
            freq_error_pu = (f - self.f0) / self.f0
            damping_MW = self.D_sys * freq_error_pu * self.baseMVA
            dfdt_pu = (dP - damping_MW) / (2.0 * self.H_sys * self.baseMVA)
            return self.f0 * dfdt_pu
        
        k1 = deriv(self.freq_Hz)
        k2 = deriv(self.freq_Hz + 0.5 * self.dt * k1)
        k3 = deriv(self.freq_Hz + 0.5 * self.dt * k2)
        k4 = deriv(self.freq_Hz + self.dt * k3)
        dfdt = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        dfdt = np.clip(dfdt, -2.0, 2.0)  # Hz/s clamp
        self.freq_Hz = np.clip(self.freq_Hz + self.dt * dfdt, self.freq_min, self.freq_max)

    def _step_with_control_convergence(self, t):
        """Enhanced step method with AGC/AVR convergence iterations using pandapower"""
        
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
            agc_signal = self.agc.step(self.freq_Hz, self.dt, target_freq=self.f0)
            
            # Governor updates with AGC participation
            for gi, g in enumerate(self.gens):
                if hasattr(self.agc, 'participation') and gi in self.agc.participation:
                    share = self.agc.participation[gi]
                    delta_pref = (agc_signal * share) / g['par'].MVA_base
                    g['gov'].p.Pref = np.clip(g['gov'].p.Pref + delta_pref, g['gov'].p.Pmin, g['gov'].p.Pmax)
                
                w_pu = self.freq_Hz / self.f0
                Pm = g['gov'].step(w_pu, dt=self.dt)
                g['Pm_pu'] += np.clip(Pm - g['Pm_pu'], -0.01, 0.01)
            
            # AVR updates - voltage regulation
            if hasattr(self.net, 'res_bus') and not self.net.res_bus.empty:
                for g in self.gens:
                    Vmeas = float(self.net.res_bus.loc[g['bus_idx'], 'vm_pu'])
                    Efd = g['exc'].step(Vmeas, dt=self.dt)
                    Verr = g['exc'].p.Vref - Vmeas
                    Kv = 3.0  # Voltage regulation gain
                    dVG = Kv * Verr * self.dt
                    g['VM_sp'] = float(np.clip(g['VM_sp'] + dVG, 0.90, 1.10))
            
            # Update generator setpoints in pandapower network
            for g in self.gens:
                gen_idx = g['index']
                
                # Update active power setpoint
                P_MW = g['Pm_pu'] * g['par'].MVA_base
                
                # Apply generator limits
                if 'max_p_mw' in self.net.gen.columns:
                    P_max = self.net.gen.loc[gen_idx, 'max_p_mw']
                    P_min = self.net.gen.loc[gen_idx, 'min_p_mw'] if 'min_p_mw' in self.net.gen.columns else 0
                else:
                    P_max = P_MW * 2.0  # Default limit
                    P_min = P_MW * 0.1
                
                P_MW = np.clip(P_MW, P_min, P_max)
                self.net.gen.loc[gen_idx, 'p_mw'] = P_MW
                
                # Update voltage setpoint
                self.net.gen.loc[gen_idx, 'vm_pu'] = g['VM_sp']
            
            # Solve power flow with pandapower
            try:
                pp.runpp(self.net, algorithm='nr', init='results', calculate_voltage_angles=True, 
                        max_iteration=20, tolerance_mva=1e-6)
                pf_converged = self.net.converged
            except Exception as e:
                logging.warning(f"t={t:.3f}s: Power flow exception at iteration {iterations}: {e}")
                pf_converged = False
            
            if not pf_converged:
                logging.warning(f"t={t:.3f}s: Power flow convergence failed at iteration {iterations}")
                # Emergency load shedding
                load_reduction = 0.02  # 2% reduction
                self.net.load.p_mw *= (1.0 - load_reduction)
                self.net.load.q_mvar *= (1.0 - load_reduction)
                
                try:
                    pp.runpp(self.net, algorithm='nr', init='flat', calculate_voltage_angles=True)
                    if not self.net.converged:
                        logging.error(f"t={t:.3f}s: Power flow failed completely")
                        break
                except:
                    logging.error(f"t={t:.3f}s: Power flow failed completely")
                    break
            
            # Update frequency with new power balance
            if hasattr(self.net, 'res_gen') and not self.net.res_gen.empty:
                sum_Pe_MW = float(self.net.res_gen.p_mw.sum())
                sum_Pd_MW = float(self.net.res_load.p_mw.sum()) if hasattr(self.net, 'res_load') else float(self.net.load.p_mw.sum())
            else:
                sum_Pe_MW = float(self.net.gen.p_mw.sum())
                sum_Pd_MW = float(self.net.load.p_mw.sum())
            
            sum_Pm_MW = sum(g['Pm_pu'] * g['par'].MVA_base for g in self.gens)
            
            # Battery operation for frequency support
            power_imbalance = sum_Pm_MW - sum_Pe_MW
            battery_power = 0.0
            
            if abs(power_imbalance) > 5.0:  # MW threshold
                if power_imbalance > 0 and self.battery.soc < self.battery.p.capacity_MWh * 0.9:
                    # Excess generation - charge battery
                    battery_power = min(power_imbalance * 0.5, self.battery.p.max_power_MW)
                    self.battery.charge(battery_power)
                elif power_imbalance < 0 and self.battery.soc > self.battery.p.capacity_MWh * 0.1:
                    # Generation deficit - discharge battery
                    battery_power = max(power_imbalance * 0.5, -self.battery.p.max_power_MW)
                    self.battery.discharge(-battery_power)
            
            # Update frequency dynamics
            effective_gen = sum_Pe_MW + battery_power
            self._coi_freq_rk4(effective_gen, sum_Pd_MW)
            
            # Check convergence
            freq_change = abs(self.freq_Hz - freq_prev)
            if freq_change < self.freq_tolerance:
                converged = True
                logging.debug(f"t={t:.3f}s: Converged in {iterations} iterations, freq={self.freq_Hz:.4f} Hz")
            else:
                freq_prev = self.freq_Hz
                logging.debug(f"t={t:.3f}s: Iteration {iterations}, freq={self.freq_Hz:.4f} Hz, change={freq_change:.6f}")
        
        if not converged:
            logging.warning(f"t={t:.3f}s: Control did not converge in {iterations} iterations, final freq={self.freq_Hz:.4f} Hz")
        
        # Log results
        self._log_step_results(t, lf, agc_signal, iterations, converged, battery_power)

    def _log_step_results(self, t, load_factor, agc_signal, iterations, converged, battery_power):
        """Log step results with detailed control information"""
        if hasattr(self.net, 'res_gen') and not self.net.res_gen.empty:
            sum_Pe_MW = float(self.net.res_gen.p_mw.sum())
        else:
            sum_Pe_MW = float(self.net.gen.p_mw.sum())
        
        if hasattr(self.net, 'res_load') and not self.net.res_load.empty:
            sum_Pd_MW = float(self.net.res_load.p_mw.sum())
        else:
            sum_Pd_MW = float(self.net.load.p_mw.sum())
        
        self.logs['t'].append(t)
        self.logs['freq_Hz'].append(self.freq_Hz)
        self.logs['sumGen_MW'].append(sum_Pe_MW)
        self.logs['sumLoad_MW'].append(sum_Pd_MW)
        
        if hasattr(self.net, 'res_bus') and not self.net.res_bus.empty:
            self.logs['Vmax'].append(float(self.net.res_bus.vm_pu.max()))
            self.logs['Vmin'].append(float(self.net.res_bus.vm_pu.min()))
        else:
            self.logs['Vmax'].append(1.0)
            self.logs['Vmin'].append(1.0)
        
        self.logs['ACE'].append(self.agc.p.beta * (self.freq_Hz - self.f0))
        self.logs['AGC_signal'].append(agc_signal)
        self.logs['load_factor'].append(load_factor)
        self.logs['control_iterations'].append(iterations)
        self.logs['battery_soc'].append(self.battery.soc)
        self.logs['battery_power'].append(battery_power)
        
        # Periodic detailed logging
        if int(t / self.dt) % max(1, int(1.0 / self.dt)) == 0:  # Every second
            logging.info(f"t={t:6.2f}s | f={self.freq_Hz:7.4f}Hz | Gen={sum_Pe_MW:7.1f}MW | Load={sum_Pd_MW:7.1f}MW | "
                        f"ACE={self.logs['ACE'][-1]:6.2f} | Bat={battery_power:5.1f}MW | Iter={iterations} | Conv={'✓' if converged else '✗'}")

    def run(self):
        """Run the simulation"""
        steps = int(self.t_end / self.dt) + 1
        logger.info(f"Running IEEE39 QSTS with pandapower: dt={self.dt}s, t_end={self.t_end}s, steps={steps}")
        logger.info(f"Max control iterations: {self.max_control_iterations}, Frequency tolerance: {self.freq_tolerance} Hz")
        
        for k in range(steps):
            t = k * self.dt
            try:
                self._step_with_control_convergence(t)
            except Exception as e:
                logging.error(f"Simulation failed at t={t:.2f}s: {e}")
                break
            
            if k % max(1, steps // 20) == 0:  # Progress updates
                logger.info(f"Progress: {k}/{steps} steps ({100*k/steps:.1f}%) - t={t:.2f}s")
        
        logger.info("Simulation finished.")
        
        # Final statistics
        if self.logs['control_iterations']:
            avg_iterations = np.mean(self.logs['control_iterations'])
            max_iterations = np.max(self.logs['control_iterations'])
            logger.info(f"Control convergence stats: avg={avg_iterations:.1f}, max={max_iterations} iterations")
        
        return self.logs

    def plot(self):
        """Plot simulation results"""
        if not self.logs['t']:
            logger.warning("No data to plot.")
            return
            
        t = np.array(self.logs['t'])
        fig, ax = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
        fig.suptitle("IEEE 39-Bus QSTS with Pandapower (AVR + TGOV1 + AGC + Battery)", fontsize=14)

        # Frequency
        ax[0].plot(t, self.logs['freq_Hz'], linewidth=2, color='blue')
        ax[0].axhline(60.0, ls='--', color='red', alpha=0.7)
        ax[0].axhline(59.8, ls=':', color='orange', alpha=0.7, label='±0.2 Hz')
        ax[0].axhline(60.2, ls=':', color='orange', alpha=0.7)
        ax[0].set_ylabel("Frequency (Hz)")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()
        ax[0].set_ylim([59.5, 60.5])

        # Power
        ax[1].plot(t, self.logs['sumGen_MW'], linewidth=2, label="Generation", color='green')
        ax[1].plot(t, self.logs['sumLoad_MW'], linewidth=2, label="Load", color='red')
        ax[1].plot(t, np.array(self.logs['sumGen_MW']) + np.array(self.logs['battery_power']), 
                  linewidth=2, label="Gen + Battery", color='purple', linestyle='--')
        ax[1].set_ylabel("Power (MW)")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        # Voltage
        ax[2].plot(t, self.logs['Vmax'], linewidth=2, label="V_max", color='red')
        ax[2].plot(t, self.logs['Vmin'], linewidth=2, label="V_min", color='blue')
        ax[2].axhline(1.05, ls='--', color='red', alpha=0.5)
        ax[2].axhline(0.95, ls='--', color='blue', alpha=0.5)
        ax[2].set_ylabel("Bus Voltage (pu)")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)

        # AGC and Control
        ax[3].plot(t, self.logs['ACE'], linewidth=2, label="ACE", color='red')
        ax[3].plot(t, self.logs['AGC_signal'], linewidth=2, label="AGC Signal", color='blue')
        ax[3].plot(t, self.logs['control_iterations'], linewidth=1, label="Control Iterations", color='green', alpha=0.7)
        ax[3].set_ylabel("Control Signals")
        ax[3].legend()
        ax[3].grid(True, alpha=0.3)

        # Battery
        ax[4].plot(t, self.logs['battery_soc'], linewidth=2, label="SOC (MWh)", color='orange')
        ax[4].plot(t, self.logs['battery_power'], linewidth=2, label="Power (MW)", color='purple')
        ax[4].set_ylabel("Battery")
        ax[4].set_xlabel("Time (s)")
        ax[4].legend()
        ax[4].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

        # Additional analysis plot
        self._plot_control_analysis()

    def _plot_control_analysis(self):
        """Additional analysis plots for control system performance"""
        if not self.logs['t']:
            return
            
        t = np.array(self.logs['t'])
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Control System Performance Analysis", fontsize=14)

        # Frequency deviation histogram
        freq_dev = np.array(self.logs['freq_Hz']) - 60.0
        ax[0,0].hist(freq_dev, bins=30, alpha=0.7, color='blue')
        ax[0,0].set_xlabel('Frequency Deviation (Hz)')
        ax[0,0].set_ylabel('Count')
        ax[0,0].set_title('Frequency Deviation Distribution')
        ax[0,0].grid(True, alpha=0.3)

        # Control iterations over time
        ax[0,1].plot(t, self.logs['control_iterations'], linewidth=1, marker='o', markersize=2)
        ax[0,1].set_xlabel('Time (s)')
        ax[0,1].set_ylabel('Iterations')
        ax[0,1].set_title('Control Convergence Iterations')
        ax[0,1].grid(True, alpha=0.3)

        # Power balance
        power_balance = np.array(self.logs['sumGen_MW']) - np.array(self.logs['sumLoad_MW'])
        ax[1,0].plot(t, power_balance, linewidth=2, color='red')
        ax[1,0].axhline(0, ls='--', color='black', alpha=0.5)
        ax[1,0].set_xlabel('Time (s)')
        ax[1,0].set_ylabel('Power Imbalance (MW)')
        ax[1,0].set_title('Generation - Load Balance')
        ax[1,0].grid(True, alpha=0.3)

        # Load factor
        ax[1,1].plot(t, self.logs['load_factor'], linewidth=2, color='green')
        ax[1,1].set_xlabel('Time (s)')
        ax[1,1].set_ylabel('Load Factor')
        ax[1,1].set_title('Load Scaling Factor')
        ax[1,1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()