"""
The MRI parameters are defined here.

The paper references is listed below, which can be used to get more information
about some measures:

[1] Leonie Petitclerc, et al. "Ultra-long-TE arterial spin labeling reveals
rapid and brain-wide blood-to-CSF water transport in humans", Neuroimage
(2021). DOI: 10.1016/j.neuroimage.2021.118755
"""

# Relaxation constants (ms)
# from typing import Any


class MRIParameters:
    def __init__(self) -> None:
        self.T1bl = 1650.0   # T1 relaxation for the blood [1]
        self.T1csf = 1400.0   # T1 relaxation for the CSF [1] Paper Ultralong TE: T1csf = 4300!!!

        self.T2bl = 165.0  # T2 relaxation for the blood [1] PAPER CITE 150!!!!
        self.T2gm = 75.0   # T2 relaxation for the GM [1] PAPER CITE 60
        self.T2csf = 1500   # T2 relaxation for the CSF [1]

        # MRI constants
        self.Alpha = 0.85   # RF labeling efficiency
        self.Lambda = 0.98   # Blood-brain partition coefficient [1]

    def set_constant(self, value: float, param: str):
        try:
            match param:
                case 'T1bl':
                    self.T1bl = value
                case 'T1csf':
                    self.T1csf = value
                case 'T2bl':
                    self.T2bl = value
                case 'T2gm':
                    self.T2gm = value
                case 'T2csf':
                    self.T2csf = value
                case 'Alpha':
                    self.Alpha = value
                case 'Lambda':
                    self.Lambda = value
        except AttributeError:
            print(
                f'A parameters must be indicated. Choose one option indicated in the class constructor documentation.'
            )

    def get_constant(self, param: str) -> float:
        try:
            match param:
                case 'T1bl':
                    return self.T1bl
                case 'T1csf':
                    return self.T1csf
                case 'T2bl':
                    return self.T2bl
                case 'T2gm':
                    return self.T2gm
                case 'T2csf':
                    return self.T2csf
                case 'Alpha':
                    return self.Alpha
                case 'Lambda':
                    return self.Lambda
        except AttributeError:
            print(
                f'A parameters must be indicated. Choose one option indicated in the class constructor documentation.'
            )
