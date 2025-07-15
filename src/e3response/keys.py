from typing import Final

# pylint: disable=unused-wildcard-import
from tensorial.gcnn.atomic.keys import *  # We want all atomic keys pylint: disable=wildcard-import
from tensorial.gcnn.keys import *  # We want all graph keys pylint: disable=wildcard-import

# Electric
EXTERNAL_ELECTRIC_FIELD: Final[str] = "external_electric_field"
POLARIZATION: Final[str] = "polarization"
POLARIZABILITY: Final[str] = "polarizability"
DIELECTRIC_TENSOR: Final[str] = "dielectric_tensor"
<<<<<<< Updated upstream
BORN_CHARGES: Final[str] = "born_charges"
RAMAN_TENSORS: Final[str] = "raman_tensors"

# Magnetic
EXTERNAL_MAGNETIC_FIELD: Final[str] = "external_magnetic_field"
INDUCED_MAGNETIC_FIELD: Final[str] = "induced_magnetic_field"
=======

EXTERNAL_MAGNETIC_FIELD: Final[str] = "external_magnetic_field"
LOCAL_MAGNETIC_FIELD: Final[str] = "local_magnetic_field"
MAGNETIC_SHIELDING_TENSOR: Final[str] = "magnetic_shielding_tensor"

>>>>>>> Stashed changes
