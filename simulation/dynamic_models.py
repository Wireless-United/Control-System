# dynamic_models.py
#
# Contains the parameter data classes and the core dynamic models
# for the power system simulation with corrected swing equation.
#
# Author: Power System Engineer

from dataclasses import dataclass
import numpy as np

# ---------------------------- Parameter Classes ----------------------------- #

@dataclass
class IEEET1Params:
    """Parameters for the IEEE Type 1 exciter model (simplified)."""
    KA: float = 200.0   # Regulator gain
    TA: float = 0.02    # Regulator time constant (s)
    TR: float = 0.02    # Voltage transducer time constant (s)
    KE: float = 1.0     # Exciter feedback gain
    TE: float = 0.5     # Exciter time constant (s)
    VRMAX: float = 5.0  # Regulator output max (pu)
    VRMIN: float = -5.0 # Regulator output min (pu)
    EMIN: float = 0.0   # Field lower limit (pu)
    EMAX: float = 5.0   # Field upper limit (pu)
    # Standard IEEE saturation parameters
    S10: float = 0.1    # Saturation at 1.0 pu
    S12: float = 0.5    # Saturation at 1.2 pu
    Vref: float = 1.0   # Reference voltage (pu)

@dataclass
class TGOV1Params:
    """Parameters for the IEEE TGOV1 governor-turbine model."""
    R: float = 0.05     # Droop (pu speed change for 1 pu power)
    TG: float = 0.2     # Governor time constant (s)
    TT: float = 0.5     # Turbine time constant (s)
    Pmax: float = 1.2   # Mechanical power max (pu)
    Pmin: float = 0.0   # Mechanical power min (pu)
    Pref: float = 1.0   # Reference power (pu)
    RATE: float = 0.1   # Ramp rate (pu/s)
    DB: float = 0.001   # Speed dead-band

@dataclass
class GenParams:
    """Parameters for the generator swing model."""
    H: float = 3.5      # Inertia constant (s)
    D: float = 2.0      # Damping coefficient (pu MW/Hz on machine base)
    MVA_base: float = 100.0 # Machine MVA base

# ----------------------------- Dynamic Model Classes ----------------------------- #

class IEEET1:
    """
    IEEE Type 1 DC commutator exciter (simplified continuous form).
    This class models the AVR's dynamic behavior.
    """
    def __init__(self, p: IEEET1Params):
        self.p = p
        self.Vt_f = 1.0  # filtered terminal voltage state
        self.Vr = 0.0    # regulator output state
        self.Efd = 1.0   # field voltage state
        # Anti-windup states
        self.Vr_limited = 0.0
        self.Efd_limited = 1.0

    def Se(self, Efd):
        """Exponential saturation function for the exciter field."""
        if Efd <= 0.75:
            return 0.0
        if self.p.S12 > self.p.S10 > 0:
            B = np.log(self.p.S12 / self.p.S10) / 0.2
            A = self.p.S10 / np.exp(B * 1.0)
            return A * np.exp(B * Efd)
        else:
            return 0.0

    def step(self, Vt, dt=0.01):
        """Step forward the exciter dynamics."""
        p = self.p
        # Measurement lag
        self.Vt_f += dt * (Vt - self.Vt_f) / max(p.TR, 1e-6)
        
        # Regulator with anti-windup
        Verr = p.Vref - self.Vt_f
        dVr_ideal = (p.KA * Verr - self.Vr) / max(p.TA, 1e-6)
        Vr_unlimited = self.Vr + dt * dVr_ideal
        self.Vr_limited = np.clip(Vr_unlimited, p.VRMIN, p.VRMAX)
        self.Vr = self.Vr_limited

        # Exciter with anti-windup
        dEfd_ideal = (self.Vr_limited - p.KE * self.Efd - self.Se(self.Efd)) / max(p.TE, 1e-6)
        Efd_unlimited = self.Efd + dt * dEfd_ideal
        self.Efd_limited = np.clip(Efd_unlimited, p.EMIN, p.EMAX)
        self.Efd = self.Efd_limited

        return self.Efd

@dataclass
class GeneratorEMFParams:
    """Parameters for internal EMF model."""
    Xd: float = 1.8
    Xq: float = 1.7
    Xdp: float = 0.3
    Xqp: float = 0.55
    Tdp0: float = 8.0
    Tqp0: float = 0.4
    Ra: float = 0.003

class GeneratorEMF:
    """Internal EMF generator model for DAE formulation."""
    def __init__(self, params: GeneratorEMFParams):
        self.p = params
        self.Eqp = 1.0
        self.Edp = 0.0
        self.Id = 0.0
        self.Iq = 0.0
        self.Vt = 1.0
        self.delta = 0.0

    def compute_currents(self, Vd, Vq):
        det = self.p.Ra**2 + self.p.Xdp * self.p.Xqp
        if abs(det) < 1e-8:
            return self.Id, self.Iq
        self.Id = (self.p.Ra * (Vd - self.Edp) + self.p.Xqp * (Vq - self.Eqp)) / det
        self.Iq = (self.p.Ra * (Vq - self.Eqp) - self.p.Xdp * (Vd - self.Edp)) / det
        return self.Id, self.Iq

    def step_emf(self, Efd, dt):
        dEqp_dt = (Efd - self.Eqp - (self.p.Xd - self.p.Xdp) * self.Id) / max(self.p.Tdp0, 1e-6)
        self.Eqp += dt * dEqp_dt
        dEdp_dt = (-self.Edp + (self.p.Xq - self.p.Xqp) * self.Iq) / max(self.p.Tqp0, 1e-6)
        self.Edp += dt * dEdp_dt
        return self.Eqp, self.Edp

class TGOV1:
    """IEEE TGOV1 turbine-governor model."""
    def __init__(self, p: TGOV1Params):
        self.p = p
        self.xg = p.Pref
        self.xt = p.Pref
        self.xg_prev = p.Pref

    def step(self, w_pu, dt=0.01):
        p = self.p
        Dw = w_pu - 1.0
        if abs(Dw) < p.DB:
            Dw = 0.0
        xg_ideal = (p.Pref - (1.0 / max(p.R, 1e-6)) * Dw - self.xg) / max(p.TG, 1e-6)
        xg_unlimited = self.xg + dt * xg_ideal
        max_change = p.RATE * dt
        xg_rate_limited = self.xg_prev + np.clip(xg_unlimited - self.xg_prev, -max_change, max_change)
        self.xg_prev = self.xg
        self.xg = xg_rate_limited
        xt_dot = (self.xg - self.xt) / max(p.TT, 1e-6)
        self.xt += dt * xt_dot
        self.xt = np.clip(self.xt, p.Pmin, p.Pmax)
        return self.xt

class Generator:
    """Enhanced generator with EMF model and proper swing equation."""
    def __init__(self, gen_idx: int, gen_par: GenParams, exc: IEEET1, gov: TGOV1, emf_par: GeneratorEMFParams = None):
        self.idx = gen_idx
        self.par = gen_par
        self.exc = exc
        self.gov = gov
        self.delta = 0.0
        self.w = 1.0
        self.Pm_pu = gov.p.Pref
        self.Pe_pu = gov.p.Pref
        self.emf = GeneratorEMF(emf_par) if emf_par else None
        self.Q_limited = False
        self.Qgen_pu = 0.0

    def swing_step_rk4(self, dt):
        def swing_derivatives(w, delta):
            H, D = self.par.H, self.par.D
            dw_dt = (self.Pm_pu - self.Pe_pu - D * (w - 1.0)) / (2.0 * H)
            ddelta_dt = 2 * np.pi * 60.0 * (w - 1.0)
            return np.clip(dw_dt, -10.0, 10.0), ddelta_dt

        k1_w, k1_d = swing_derivatives(self.w, self.delta)
        k2_w, k2_d = swing_derivatives(self.w + 0.5*dt*k1_w, self.delta + 0.5*dt*k1_d)
        k3_w, k3_d = swing_derivatives(self.w + 0.5*dt*k2_w, self.delta + 0.5*dt*k2_d)
        k4_w, k4_d = swing_derivatives(self.w + dt*k3_w, self.delta + dt*k3_d)
        self.w += dt/6 * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        self.delta += dt/6 * (k1_d + 2*k2_d + 2*k3_d + k4_d)

class ZIPLoad:
    """ZIP load model."""
    def __init__(self, P0, Q0, a=0.0, b=0.0, c=1.0, Kf=1.0):
        self.P0 = P0
        self.Q0 = Q0
        self.a, self.b, self.c = a, b, c
        self.Kf = Kf

    def compute(self, V, freq=60.0):
        Df = (freq - 60.0) / 60.0
        voltage_factor = self.a * V**2 + self.b * V + self.c
        freq_factor = 1.0 + self.Kf * Df
        P = self.P0 * voltage_factor * freq_factor
        Q = self.Q0 * voltage_factor * freq_factor
        return P, Q

@dataclass
class AGCParams:
    beta: float = 1.0
    Kp: float = 100.0
    Ki: float = 0.1
    ACE_max: float = 100.0

class AGC:
    def __init__(self, params: AGCParams):
        self.p = params
        self.ACE_integral = 0.0
        self.participation = {}

    def set_participation(self, gen_indices, factors):
        total = sum(factors)
        self.participation = {gen_indices[i]: factors[i]/total for i in range(len(gen_indices))}

    def step(self, freq_Hz, dt, target_freq=60.0):
        freq_error = freq_Hz - target_freq
        ACE = self.p.beta * freq_error
        self.ACE_integral += ACE * dt
        self.ACE_integral = np.clip(self.ACE_integral, -10.0, 10.0)
        agc_signal = -(self.p.Kp * ACE + self.p.Ki * self.ACE_integral)
        agc_signal = np.clip(agc_signal, -self.p.ACE_max, self.p.ACE_max)
        return agc_signal
