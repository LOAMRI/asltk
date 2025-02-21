using NIfTI
using Statistics
using Plots
using ColorSchemes
using ImageTransformations
using Hadamard
using Glob

hadline_cnt = 8;                   # number of different time-encoding lines applied   
boli_cnt = hadline_cnt-1;          # number of sub-bolis
nTe = 8;                           # number of echo times
ndin = 2;                          # number of dynamic volumes

temp = hadamard(hadline_cnt);
subtr_mtrx = -temp[:,2:hadline_cnt];

datdir = "/home/antonio/Experiments/LOAM-sample/ASL_DICOM_subtract/ASL_DICOM/MTE-ASL";
fn = glob("*.nii",datdir);
head = niread(fn[1]);

for cnt in 1:nTe
    println("Processing echo time: $cnt")
    dat = niread(fn[cnt]);
    dims = size(dat.raw);

    if cnt == 1 
        global dat_final = zeros(Float64, dims[1],dims[2],dims[3],boli_cnt,nTe); 
    end

    dat_aux = zeros(Float64, dims[1],dims[2],dims[3],map(Int,dims[4]/ndin),ndin);

    dat_aux[:,:,:,:,1] = dat.raw[:,:,:,1:2:end];
    dat_aux[:,:,:,:,2] = dat.raw[:,:,:,2:2:end];

    vox = dat.header.pixdim[2]; 
    voy = dat.header.pixdim[3];
    voz = dat.header.pixdim[4]; 
    voxel_size = [vox voy voz];

    x_dim, y_dim, z_dim, dyn_cnt = size(dat_aux);

    #dat_mirrored = zeros(Float64, dims[1],dims[2],dims[3],map(Int,dims[4]/ndin),ndin);

    #dat_mirrored = dat_aux[end:-1:1,:,:,:,:];

    origin = [round(x_dim/2) round(y_dim/2) round(z_dim/2)]; datatype = 16;
    unsubtr_data_mean=dropdims(mean(dat_aux,dims=5),dims=5);

    Subtr_phases = zeros(Float64, size(unsubtr_data_mean[:,:,:,2:end,:]));

    for bolus in 1:boli_cnt
        vector = subtr_mtrx[:,bolus];
        for line in 1:hadline_cnt
            Subtr_phases[:,:,:,boli_cnt-bolus+1] = Subtr_phases[:,:,:,boli_cnt-bolus+1] + vector[line] *  unsubtr_data_mean[:,:,:,line];
        end

    end

    dat_final[:,:,:,:,cnt] = Subtr_phases .* dat.header.scl_slope .+ dat.header.scl_inter;
    #dat_final = dat_final .* dat.header.scl_slope .+ dat.header.scl_inter;
    println("finished TE $cnt")

end

nii = NIVolume(dat_final, time_step=1, slice_duration=head.header.slice_duration, regular=head.header.regular,
voxel_size=(head.header.pixdim[2], head.header.pixdim[3], head.header.pixdim[4]), 
orientation=vcat(collect(head.header.srow_x)', collect(head.header.srow_y)', collect(head.header.srow_z)'),
);
niwrite(datdir*"/pcasl_julia.nii.gz",nii)
