import numpy as np
from dataclasses import dataclass

# ---------------- Exciter ----------------
@dataclass
class IEEET1Params:
    KA: float = 50.0
    TA: float = 0.1
    TR: float = 0.05
    KE: float = 1.0
    TE: float = 0.5
    VRMAX: float = 3.0
    VRMIN: float = -3.0
    EMIN: float = 0.0
    EMAX: float = 3.0
    Vref: float = 1.0
    S10: float = 0.1
    S12: float = 0.5

class IEEET1:
    def __init__(self, p: IEEET1Params):
        self.p = p
        self.Vt_f = p.Vref
        self.Vr = 0.0
        self.Efd = 1.0

    def _sat(self, Efd):
        a = self.p.S10
        return a * (max(Efd, 0.0) - 1.0) ** 2

    def step(self, Vt_meas: float, dt: float) -> float:
        self.Vt_f += (Vt_meas - self.Vt_f) * dt / max(self.p.TR, 1e-6)
        Verr = self.p.Vref - self.Vt_f
        dVr = (self.p.KA * Verr - self.Vr) / max(self.p.TA, 1e-6)
        self.Vr += dVr * dt
        self.Vr = np.clip(self.Vr, self.p.VRMIN, self.p.VRMAX)
        sat = self._sat(self.Efd)
        dE = (self.Vr - self.p.KE * self.Efd - sat) / max(self.p.TE, 1e-6)
        self.Efd += dE * dt
        self.Efd = np.clip(self.Efd, self.p.EMIN, self.p.EMAX)
        return self.Efd

# ---------------- Governor ----------------
@dataclass
class TGOV1Params:
    R: float = 0.05
    TG: float = 0.5
    TT: float = 1.0
    RATE: float = 0.10
    Pmax: float = 1e9
    Pmin: float = 0.0
    Pref: float = 0.0
    DB: float = 0.0

class TGOV1:
    def __init__(self, p: TGOV1Params):
        self.p = p
        self.xg = p.Pref
        self.xt = p.Pref

    def step(self, w_pu: float, dt: float) -> float:
        de = (1.0 - w_pu)
        if abs(de) < self.p.DB:
            de = 0.0
        Pref_eff = self.p.Pref + de / max(self.p.R, 1e-6)
        dxg = (Pref_eff - self.xg) / max(self.p.TG, 1e-6)
        self.xg += dxg * dt
        self.xg = np.clip(self.xg, self.p.Pmin, self.p.Pmax)
        dxt = (self.xg - self.xt) / max(self.p.TT, 1e-6)
        self.xt += dxt * dt
        self.xt = np.clip(self.xt, self.p.Pmin, self.p.Pmax)
        return self.xt

# ---------------- AGC ----------------
@dataclass
class AGCParams:
    beta: float = 20.0
    Kp: float = 50.0
    Ki: float = 0.05
    ACE_max: float = 400.0

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

    def get_control_signals(self):
        return {"AGC_output": self.ACE_integral}

# ---------------- Battery ----------------
@dataclass
class BatteryParams:
    capacity_MWh: float = 100.0
    soc_init: float = 0.5
    max_power_MW: float = 50.0

class BatteryStorage:
    def __init__(self, p: BatteryParams):
        self.p = p
        self.soc = p.soc_init * p.capacity_MWh

    def update(self, dt):
        pass  # placeholder for dynamic update

    def charge(self, power_MW):
        self.soc = min(self.soc + power_MW * 1/60, self.p.capacity_MWh)

    def discharge(self, power_MW):
        self.soc = max(self.soc - power_MW * 1/60, 0.0)

