# ASL Processing Workflow Examples

This document provides comprehensive examples of using the asltk library for various ASL processing workflows. Each example includes practical code snippets, parameter explanations, and result interpretation guidance.

## Table of Contents

1. [Basic CBF Mapping Workflow](#basic-cbf-mapping-workflow)
2. [Multi-Echo ASL Analysis](#multi-echo-asl-analysis)
3. [Diffusion-Weighted ASL Analysis](#diffusion-weighted-asl-analysis)
4. [Complete Analysis Pipeline](#complete-analysis-pipeline)
5. [Best Practices and Tips](#best-practices-and-tips)

## Basic CBF Mapping Workflow

This example shows the standard ASL processing workflow for generating CBF and ATT maps.

```python
from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping
from asltk.utils import load_image, save_image
import numpy as np

# Step 1: Load ASL data
asl_data = ASLData(
    pcasl='./data/pcasl.nii.gz',          # ASL time series
    m0='./data/m0.nii.gz',                # M0 reference image
    ld_values=[1.8, 1.8, 1.8],           # Labeling durations (s)
    pld_values=[0.8, 1.8, 2.8]           # Post-labeling delays (s)
)

# Step 2: Create CBF mapper
cbf_mapper = CBFMapping(asl_data)

# Step 3: Set brain mask (recommended for speed and quality)
brain_mask = load_image('./data/brain_mask.nii.gz')
cbf_mapper.set_brain_mask(brain_mask)

# Step 4: Generate CBF and ATT maps
# Note: All parameters (ub, lb, par0, cores) have default values and are optional
# You can simply call: results = cbf_mapper.create_map() for default behavior
results = cbf_mapper.create_map(
    ub=[1.0, 5000.0],      # Upper bounds: [CBF, ATT]
    lb=[0.0, 0.0],         # Lower bounds: [CBF, ATT] 
    par0=[1e-5, 1000],     # Initial guess: [CBF, ATT]
    cores=4                # Use 4 CPU cores
)

# Step 5: Access and save results
cbf_map = results['cbf_norm']    # CBF in mL/100g/min
att_map = results['att']         # ATT in milliseconds
cbf_raw = results['cbf']         # Raw CBF values

# Save results
# Important: Image saving formats are limited to asltk API settings
# Available formats: .nii, .nii.gz, .mha, .nrrd (see save_image documentation)
save_image(cbf_map, './results/cbf_map.nii.gz')
save_image(att_map, './results/att_map.nii.gz')

# Step 6: Quality assessment
print(f"CBF range: {cbf_map.min():.1f} - {cbf_map.max():.1f} mL/100g/min")
print(f"Mean CBF: {np.mean(cbf_map[brain_mask > 0]):.1f} mL/100g/min")
print(f"ATT range: {att_map.min():.0f} - {att_map.max():.0f} ms")
print(f"Mean ATT: {np.mean(att_map[brain_mask > 0]):.0f} ms")
```

### Expected Results
- **Healthy CBF values**: 50-80 mL/100g/min (grey matter), 20-30 mL/100g/min (white matter)
- **Healthy ATT values**: 800-2000 ms depending on vascular territory

### Example CBF and ATT Maps

![CBF Map](../assets/images/cbf_map.png)

A CBF map example, using multi PLD ASL imaging acquisition

![ATT Map](../assets/images/att_map.png)

An ATT map example, using multi PLD ASL imaging acquisition


## Multi-Echo ASL Analysis

This example demonstrates multi-TE ASL processing for T1 relaxometry and enhanced tissue characterization.

```python
from asltk.asldata import ASLData
from asltk.reconstruction import MultiTE_ASLMapping
from asltk.utils import save_image
import numpy as np

# Step 1: Load multi-TE ASL data
asl_data = ASLData(
    pcasl='./data/pcasl_multi_te.nii.gz',
    m0='./data/m0.nii.gz',
    te_values=[13.2, 25.7, 50.4],        # Echo times (ms)
    ld_values=[1.8, 1.8, 1.8],
    pld_values=[0.8, 1.8, 2.8]
)

# Step 2: Create multi-TE mapper
mte_mapper = MultiTE_ASLMapping(asl_data)

# Step 3: Adjust MRI parameters for specific field strength (3T example)
mte_mapper.set_constant(1600.0, 'T1csf')    # CSF T1 at 3T
mte_mapper.set_constant(1300.0, 'T1gm')     # GM T1 at 3T
mte_mapper.set_constant(800.0, 'T1wm')      # WM T1 at 3T

# Step 4: Set brain mask
brain_mask = load_image('./data/brain_mask.nii.gz')
mte_mapper.set_brain_mask(brain_mask)

# Step 5: Generate multi-TE maps
results = mte_mapper.create_map(
    ub=[800.0],           # Upper bound for T1blGM (ms)
    lb=[50.0],            # Lower bound for T1blGM (ms)
    par0=[400.0],         # Initial guess for T1blGM (ms)
    cores=4
)

# Step 6: Access results
cbf_norm = results['cbf_norm']      # Normalized CBF
att_map = results['att']            # Arterial transit time
t1blgm_map = results['t1blgm']      # T1 blood-grey matter exchange

# Step 7: Save results
save_image(cbf_norm, './results/mte_cbf.nii.gz')
save_image(att_map, './results/mte_att.nii.gz')
save_image(t1blgm_map, './results/t1blgm.nii.gz')

# Step 8: Analysis
print(f"T1blGM range: {t1blgm_map.min():.0f} - {t1blgm_map.max():.0f} ms")
print(f"Mean T1blGM: {np.mean(t1blgm_map[brain_mask > 0]):.0f} ms")

# Typical T1blGM values:
# Grey matter: 200-600 ms (faster exchange)
# White matter: 400-800 ms (slower exchange)
```

### Example Multi-TE Results

![T1blGM Map](../assets/images/t1_blood_gm_map.png)

A T1 blood-GM map example, using the Multi TE ASL imaging acquisition

## Diffusion-Weighted ASL Analysis

This example shows multi-DW ASL processing for compartment analysis and diffusion characterization.

```python
from asltk.asldata import ASLData
from asltk.reconstruction import MultiDW_ASLMapping
from asltk.utils import save_image
import numpy as np

# Step 1: Load multi-DW ASL data
asl_data = ASLData(
    pcasl='./data/pcasl_multi_dw.nii.gz',
    m0='./data/m0.nii.gz',
    dw_values=[0, 50, 100, 200],          # b-values (s/mm²)
    ld_values=[1.8, 1.8, 1.8, 1.8],
    pld_values=[0.8, 1.8, 2.8, 3.8]
)

# Step 2: Create multi-DW mapper
mdw_mapper = MultiDW_ASLMapping(asl_data)

# Step 3: Set conservative brain mask (important for processing time)
brain_mask = load_image('./data/brain_mask.nii.gz')
# Create more conservative mask for initial testing
conservative_mask = brain_mask.copy()
conservative_mask[0:2, :, :] = 0      # Remove edge slices
conservative_mask[-2:, :, :] = 0      # Remove edge slices
mdw_mapper.set_brain_mask(conservative_mask)

print(f"Processing {np.sum(conservative_mask)} voxels (reduced from {np.sum(brain_mask)})")

# Step 4: Generate multi-DW maps (this may take 30+ minutes)
print("Starting multi-DW processing... (this may take a while)")
results = mdw_mapper.create_map(
    lb=[0.1, 1e-6, 0.1, 1e-7],          # Lower bounds: [A1, D1, A2, D2]
    ub=[2.0, 1e-3, 2.0, 1e-4],          # Upper bounds: [A1, D1, A2, D2]
    par0=[0.8, 5e-5, 0.3, 1e-5]         # Initial guess: [A1, D1, A2, D2]
)

# Step 5: Access compartment results
cbf_norm = results['cbf_norm']          # Normalized CBF
att_map = results['att']                # Arterial transit time
A1_map = results['A1']                  # Fast compartment amplitude
D1_map = results['D1']                  # Fast compartment diffusion
A2_map = results['A2']                  # Slow compartment amplitude
D2_map = results['D2']                  # Slow compartment diffusion
kw_map = results['kw']                  # Water exchange parameter

# Step 6: Save all results
save_image(cbf_norm, './results/mdw_cbf.nii.gz')
save_image(att_map, './results/mdw_att.nii.gz')
save_image(A1_map, './results/A1_fast_compartment.nii.gz')
save_image(D1_map, './results/D1_fast_diffusion.nii.gz')
save_image(A2_map, './results/A2_slow_compartment.nii.gz')
save_image(D2_map, './results/D2_slow_diffusion.nii.gz')
save_image(kw_map, './results/water_exchange.nii.gz')

# Step 7: Analysis and interpretation
mask_indices = conservative_mask > 0
print(f"Fast diffusion (D1) range: {D1_map[mask_indices].min():.2e} - {D1_map[mask_indices].max():.2e} mm²/s")
print(f"Slow diffusion (D2) range: {D2_map[mask_indices].min():.2e} - {D2_map[mask_indices].max():.2e} mm²/s")
print(f"Amplitude ratio (A1/A2): {np.mean(A1_map[mask_indices]/A2_map[mask_indices]):.2f}")

# Typical interpretations:
# D1 > D2: Fast compartment likely represents intravascular component
# A1/A2 ratio: Relative contribution of each compartment
# Higher kw: Faster water exchange between compartments
```

## Complete Analysis Pipeline

This example demonstrates a complete ASL analysis pipeline comparing different reconstruction methods.

```python
from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping, MultiTE_ASLMapping
from asltk.utils import load_image, save_image, collect_data_volumes
import numpy as np

# Step 1: Load comprehensive ASL dataset
asl_data = ASLData(
    pcasl='./data/pcasl_complete.nii.gz',
    m0='./data/m0.nii.gz',
    ld_values=[1.8, 1.8, 1.8],
    pld_values=[0.8, 1.8, 2.8],
    te_values=[13.2, 25.7, 50.4]         # Multi-TE capability
)

# Step 2: Data exploration
print(f"ASL data shape: {asl_data('pcasl').shape}")
print(f"M0 data shape: {asl_data('m0').shape}")
print(f"LD values: {asl_data.get_ld()} s")
print(f"PLD values: {asl_data.get_pld()} s")
print(f"TE values: {asl_data.get_te()} ms")

# Extract individual volumes for quality assessment
volumes, original_shape = collect_data_volumes(asl_data('pcasl'))
print(f"Extracted {len(volumes)} individual volumes")

# Step 3: Basic CBF mapping
print("\n=== Basic CBF Mapping ===")
cbf_mapper = CBFMapping(asl_data)
brain_mask = load_image('./data/brain_mask.nii.gz')
cbf_mapper.set_brain_mask(brain_mask)

basic_results = cbf_mapper.create_map(cores=4)
basic_cbf = basic_results['cbf_norm']
basic_att = basic_results['att']

print(f"Basic CBF - Mean: {np.mean(basic_cbf[brain_mask > 0]):.1f} mL/100g/min")
print(f"Basic ATT - Mean: {np.mean(basic_att[brain_mask > 0]):.0f} ms")

# Step 4: Multi-TE ASL mapping
print("\n=== Multi-TE ASL Mapping ===")
mte_mapper = MultiTE_ASLMapping(asl_data)
mte_mapper.set_brain_mask(brain_mask)

# Use basic CBF/ATT as input for multi-TE analysis
mte_mapper.set_cbf_map(basic_results['cbf'])
mte_mapper.set_att_map(basic_results['att'])

mte_results = mte_mapper.create_map(
    par0=[400.0],     # Initial guess for T1blGM
    cores=4
)

mte_cbf = mte_results['cbf_norm']
mte_t1blgm = mte_results['t1blgm']

print(f"Multi-TE CBF - Mean: {np.mean(mte_cbf[brain_mask > 0]):.1f} mL/100g/min")
print(f"T1blGM - Mean: {np.mean(mte_t1blgm[brain_mask > 0]):.0f} ms")

# Step 5: Save all results with organized naming
save_image(basic_cbf, './results/basic_cbf.nii.gz')
save_image(basic_att, './results/basic_att.nii.gz')
save_image(mte_cbf, './results/multite_cbf.nii.gz')
save_image(mte_t1blgm, './results/t1blgm_map.nii.gz')

# Step 6: Comparison analysis
print("\n=== Method Comparison ===")
cbf_difference = mte_cbf - basic_cbf
print(f"CBF difference (Multi-TE - Basic): {np.mean(cbf_difference[brain_mask > 0]):.1f} ± {np.std(cbf_difference[brain_mask > 0]):.1f} mL/100g/min")

# Step 7: Regional analysis (example with simple ROIs)
# Create simple GM/WM masks based on CBF thresholds
gm_mask = (basic_cbf > 40) & (brain_mask > 0)  # High CBF regions (GM)
wm_mask = (basic_cbf <= 40) & (basic_cbf > 10) & (brain_mask > 0)  # Lower CBF regions (WM)

print(f"\nRegional Analysis:")
print(f"GM CBF: {np.mean(basic_cbf[gm_mask]):.1f} ± {np.std(basic_cbf[gm_mask]):.1f} mL/100g/min")
print(f"WM CBF: {np.mean(basic_cbf[wm_mask]):.1f} ± {np.std(basic_cbf[wm_mask]):.1f} mL/100g/min")
print(f"GM T1blGM: {np.mean(mte_t1blgm[gm_mask]):.0f} ± {np.std(mte_t1blgm[gm_mask]):.0f} ms")
print(f"WM T1blGM: {np.mean(mte_t1blgm[wm_mask]):.0f} ± {np.std(mte_t1blgm[wm_mask]):.0f} ms")

print(f"\nResults saved to ./results/ directory")
```

## Best Practices and Tips

### Performance Optimization

1. **Use Brain Masks**: Always use brain masks to reduce processing time and improve results quality.
   ```python
   # Conservative mask for initial testing
   conservative_mask = brain_mask.copy()
   conservative_mask[brain_mask < np.percentile(brain_mask[brain_mask > 0], 50)] = 0
   ```

2. **Parallel Processing**: Adjust CPU cores based on your system:
   ```python
   import multiprocessing
   max_cores = multiprocessing.cpu_count()
   use_cores = max(1, max_cores - 2)  # Leave 2 cores for system
   ```

3. **Memory Management**: For large datasets, process in chunks or use conservative masks first.

### Parameter Selection

1. **CBF Mapping Parameters**:
   - `ub=[1.0, 5000.0]`: Suitable for most datasets
   - `par0=[1e-5, 1000]`: Good starting point for most scanners
   - Adjust based on field strength and sequence parameters

2. **Multi-TE Parameters**:
   - `par0=[400]`: Good for healthy subjects at 3T
   - Lower values (200-300) for faster exchange
   - Higher values (500-600) for slower exchange

3. **Multi-DW Parameters**:
   - Start with small brain regions for parameter optimization
   - `A1, A2`: Amplitude parameters (0.1-2.0)
   - `D1, D2`: Diffusion parameters (1e-7 to 1e-3 mm²/s)

### Quality Assessment

1. **CBF Range Check**:
   ```python
   # Typical ranges for healthy adults
   if np.mean(cbf_map[brain_mask > 0]) < 20:
       print("Warning: CBF values seem low")
   elif np.mean(cbf_map[brain_mask > 0]) > 100:
       print("Warning: CBF values seem high")
   ```

2. **ATT Range Check**:
   ```python
   # Typical ranges
   if np.mean(att_map[brain_mask > 0]) < 500:
       print("Warning: ATT values seem low")
   elif np.mean(att_map[brain_mask > 0]) > 3000:
       print("Warning: ATT values seem high")
   ```

3. **Visual Inspection**: Always visually inspect results for anatomical plausibility and artifacts.

### Troubleshooting

1. **Low SNR**: Increase spatial smoothing or use larger brain masks
2. **Long Processing Times**: Use smaller brain masks or reduce CPU cores
3. **Unrealistic Values**: Check input parameters and data quality
4. **Convergence Issues**: Adjust initial guess parameters or bounds

This documentation provides a comprehensive guide to using asltk for various ASL processing workflows. Each example is designed to be practical and adaptable to different research needs and datasets.