"""
The MRI parameters are defined here.

The paper references is listed below, which can be used to get more information
about some measures:

[1] Leonie Petitclerc, et al. "Ultra-long-TE arterial spin labeling reveals
rapid and brain-wide blood-to-CSF water transport in humans", Neuroimage
(2021). DOI: 10.1016/j.neuroimage.2021.118755

[2] R B Buxton, et al. "A general kinetic model for quantitative perfusion
imaging with arterial spin labeling". Magn Reson Med (1998). PMID: 9727941
DOI: 10.1002/mrm.1910400308
"""


class MRIParameters:
    def __init__(self) -> None:
        """Creates the basic MRIParameters object to define the main MRI
        constants and values for ASL processing

        To see all the parameters listed in the class, one can call print the
        values using the default `print()` function
        """
        self.T1bl = 1650.0   # T1 relaxation for the blood [1]
        self.T1csf = 1400.0   # T1 relaxation for the CSF [1] Paper Ultralong TE: T1csf = 4300!!!

        self.T2bl = 165.0  # T2 relaxation for the blood [1] PAPER CITE 150!!!!
        self.T2gm = 75.0   # T2 relaxation for the GM [1] PAPER CITE 60
        self.T2csf = 1500   # T2 relaxation for the CSF [1]

        # MRI constants
        self.Alpha = 0.85   # RF labeling efficiency
        self.Lambda = 0.98   # Blood-brain partition coefficient [1]

    def set_constant(self, value: float, param: str):
        """Set a different value for a parameter defined in the MRIParameter
        class.

        Args:
            value (float): The value to be assumed in the parameter
            param (str): The parameter that will receive the new value

        Raises:
            AttributeError: The parameter type must be already defined in the
            MRIParameters class.
        """
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
            case _:
                raise AttributeError(
                    f'Constant type {param} is not valid. Choose in the list available in the MRIParameter class.'
                )

    def get_constant(self, param: str) -> float:
        """Collect a parameter value from a defined type

        Args:
            param (str): The specific parameter that should return the storage
            value.

        Raises:
            AttributeError: The parameter type must be already defined in the
            MRIParameters class.

        Returns:
            float: The parameter value storage in the object instance
        """
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
            case _:
                raise AttributeError(
                    f'Constant type {param} is not valid. Choose in the list available in the MRIParameter class.'
                )
