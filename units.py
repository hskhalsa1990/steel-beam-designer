"""
Display units for the dashboard. All calculations run in SI internally
(N, m, N·m, Pa). Conversions happen only at the input and output boundaries.

Deflections are always reported in mm and stresses in MPa, which are the
conventional units for those quantities in steel design.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Units:
    force: str        # "kN" or "N"
    length: str       # "m" or "mm"

    # factors to SI
    @property
    def f(self) -> float:
        """Newtons per force unit."""
        return {"kN": 1e3, "N": 1.0}[self.force]

    @property
    def l(self) -> float:
        """Metres per length unit."""
        return {"m": 1.0, "mm": 1e-3}[self.length]

    # labels
    @property
    def F(self) -> str:
        return self.force

    @property
    def L(self) -> str:
        return self.length

    @property
    def intensity(self) -> str:
        return f"{self.force}/{self.length}"

    @property
    def moment(self) -> str:
        return f"{self.force}·{self.length}"

    # conversions
    def force_to_N(self, x: float) -> float:
        return x * self.f

    def length_to_m(self, x: float) -> float:
        return x * self.l

    def intensity_to_Npm(self, w: float) -> float:
        """Force per length (display unit) to N/m."""
        return w * self.f / self.l

    def moment_to_Nm(self, M: float) -> float:
        return M * self.f * self.l

    def N_to_force(self, F: float) -> float:
        return F / self.f

    def m_to_length(self, x: float) -> float:
        return x / self.l

    def Nm_to_moment(self, M: float) -> float:
        return M / (self.f * self.l)

    def Npm_to_intensity(self, w: float) -> float:
        return w * self.l / self.f


SI_DEFAULT = Units("kN", "m")
