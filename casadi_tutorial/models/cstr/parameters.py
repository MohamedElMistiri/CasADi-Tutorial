"""Physical parameters and nominal values for the CSTR teaching model."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CSTRParameters:
    """Collect CSTR constants and the documented nominal operating point.

    Time is measured in hours, concentration in kmol/m^3, and temperature in
    kelvin. Keeping units beside each value makes dimensional checks visible to
    learners who modify the model.
    """

    flow_rate: float = 1.0  # Volumetric feed and outlet flow [m^3/h].
    volume: float = 1.0  # Well-mixed reactor volume [m^3].
    pre_exponential: float = 9703.0 * 3600.0  # Arrhenius factor [1/h].
    activation_energy: float = 11843.0  # Activation energy [kcal/kmol].
    heat_of_reaction: float = -5960.0  # Exothermic reaction heat [kcal/kmol].
    volumetric_heat_capacity: float = 500.0  # Mixture heat capacity [kcal/(m^3 K)].
    heat_transfer: float = 150.0  # Overall jacket heat transfer [kcal/(h K)].
    gas_constant: float = 1.98589  # Gas constant [kcal/(kmol K)].

    feed_concentration: float = 10.0  # Feed concentration [kmol/m^3].
    feed_temperature: float = 298.15  # Feed temperature [K].
    nominal_jacket_temperature: float = 298.15  # Jacket temperature [K].
    nominal_concentration: float = 8.5695  # Reactor concentration [kmol/m^3].
    nominal_temperature: float = 311.267  # Reactor temperature [K].

    @property
    def nominal_state(self) -> np.ndarray:
        """Return ``[concentration, temperature]`` at the nominal point."""
        return np.array([self.nominal_concentration, self.nominal_temperature])

    @property
    def nominal_input(self) -> np.ndarray:
        """Return the nominal jacket-temperature input as a one-vector."""
        return np.array([self.nominal_jacket_temperature])

    @property
    def nominal_disturbance(self) -> np.ndarray:
        """Return ``[feed concentration, feed temperature]``."""
        return np.array([self.feed_concentration, self.feed_temperature])
