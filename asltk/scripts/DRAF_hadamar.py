import os
from glob import glob

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.linalg import hadamard

hadline_cnt = 8  # number of different time-encoding lines applied
boli_cnt = hadline_cnt - 1  # number of sub-bolis
nTe = 8  # number of echo times
ndin = 2  # number of dynamic volumes

temp = hadamard(hadline_cnt)
subtr_mtrx = -temp[:, 1:hadline_cnt]

datdir = '/home/antonio/Experiments/LOAM-sample/ASL_DICOM_subtract/ASL_DICOM/MTE-ASL'
# datdir = "/Users/ampaschoal/Documents/Data/BBB/Patients/EpilepsiaDCF/20241125/ASL"
fn = glob(os.path.join(datdir, '*.nii'))
head = sitk.ReadImage(fn[0])

for cnt in range(nTe):
    dat = sitk.ReadImage(fn[cnt])
    dat_array = sitk.GetArrayFromImage(dat)
    dims = dat_array.shape

    if cnt == 0:
        dat_final = np.zeros(
            (nTe, boli_cnt, dims[1], dims[2], dims[3]), dtype=np.float64
        )

    dat_aux = np.zeros(
        (ndin, dims[0] // ndin, dims[1], dims[2], dims[3]), dtype=np.float64
    )

    dat_aux[0, ...] = dat_array[::2, ...]
    dat_aux[1, ...] = dat_array[1::2, ...]

    # voxel_size = dat.GetSpacing()

    # dyn_cnt, z_dim, y_dim, x_dim,  = dat_aux.shape[1:]

    unsubtr_data_mean = np.mean(dat_aux, axis=0)

    Subtr_phases = np.zeros(unsubtr_data_mean[1:, ...].shape, dtype=np.float64)

    for bolus in range(boli_cnt):
        vector = subtr_mtrx[:, bolus]
        for line in range(hadline_cnt):
            Subtr_phases[boli_cnt - bolus - 1, ...] += (
                vector[line] * unsubtr_data_mean[line, ...]
            )

    dat_final[cnt, ...] = Subtr_phases * float(
        dat.GetMetaData('scl_slope')
    ) + float(dat.GetMetaData('scl_inter'))

# Create the final NIfTI image
dat_final_sitk = sitk.GetImageFromArray(dat_final)
# dat_final_sitk.SetSpacing(head.GetSpacing())
# dat_final_sitk.SetDirection(head.GetDirection())
# dat_final_sitk.SetOrigin(head.GetOrigin())

sitk.WriteImage(dat_final_sitk, os.path.join(datdir, 'pcasl.nii.gz'))
