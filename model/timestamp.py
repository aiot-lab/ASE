import numpy as np


class Time():
    def __init__(self, start, end, step=None, num=None) -> None:
        self.start = start
        self.end = end
        if step is not None:
            assert isinstance(step, int)
            self.step = step
        elif num is not None:
            assert isinstance(num, int)
            self.num = num

    def _get_time(self) -> np.ndarray:
        if hasattr(self, "step"):
            return np.arange(self.start, self.end, self.step)
        elif hasattr(self, "num"):
            return np.linspace(self.start, self.end, self.num)
        else:
            raise ValueError("step or num must be checked")

    def __call__(self):
        return self._get_time()


def getTime(end, num):
    return Time(0, end, num=num)()


def get_time(window_step, total_duration, rho_time, CFR_time):
    end = (window_step * total_duration * rho_time - 1) / CFR_time
    num = rho_time
    return getTime(end, num)


def get_tau(window_time, CFR_rate=None, tau_time=None):
    end = window_time - 1 / CFR_rate
    num = tau_time
    return getTime(end, num)
