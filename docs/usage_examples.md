# ASL Processing Examples

This document provides practical examples for common ASL processing tasks using asltk.

## Quick Start: Basic CBF Mapping

```python
from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping
from asltk.utils import load_image, save_image

# Load ASL data
asl_data = ASLData(
    pcasl='path/to/pcasl.nii.gz',
    m0='path/to/m0.nii.gz',
    ld_values=[1.8, 1.8, 1.8],
    pld_values=[0.8, 1.8, 2.8]
)

# Create CBF mapper and set brain mask
cbf_mapper = CBFMapping(asl_data)
brain_mask = load_image('path/to/brain_mask.nii.gz')
cbf_mapper.set_brain_mask(brain_mask)

# Generate CBF and ATT maps
results = cbf_mapper.create_map()
cbf_map = results['cbf_norm']  # CBF in mL/100g/min
att_map = results['att']       # ATT in milliseconds

# Save results
save_image(cbf_map, 'cbf_map.nii.gz')
save_image(att_map, 'att_map.nii.gz')
```

## Multi-Echo ASL Analysis

```python
from asltk.reconstruction import MultiTE_ASLMapping

# Load multi-TE ASL data
asl_data = ASLData(
    pcasl='path/to/multi_te_pcasl.nii.gz',
    m0='path/to/m0.nii.gz',
    te_values=[13.2, 25.7, 50.4],  # Echo times
    ld_values=[1.8, 1.8, 1.8],
    pld_values=[0.8, 1.8, 2.8]
)

# Create multi-TE mapper
mte_mapper = MultiTE_ASLMapping(asl_data)
mte_mapper.set_brain_mask(brain_mask)

# Generate maps including T1 blood-gray matter exchange
results = mte_mapper.create_map()
t1blgm_map = results['t1blgm']  # T1 exchange time
```

## Diffusion-Weighted ASL Analysis

```python
from asltk.reconstruction import MultiDW_ASLMapping

# Load multi-DW ASL data
asl_data = ASLData(
    pcasl='path/to/multi_dw_pcasl.nii.gz',
    m0='path/to/m0.nii.gz',
    dw_values=[0, 50, 100, 200],  # b-values
    ld_values=[1.8, 1.8, 1.8, 1.8],
    pld_values=[0.8, 1.8, 2.8, 3.8]
)

# Create multi-DW mapper (use conservative brain mask)
mdw_mapper = MultiDW_ASLMapping(asl_data)
mdw_mapper.set_brain_mask(conservative_mask)

# Generate compartment maps (computationally intensive)
results = mdw_mapper.create_map()
A1_map = results['A1']  # Fast compartment
D1_map = results['D1']  # Fast diffusion
```

## Working with BIDS Data

```python
from asltk.utils import load_image

# Load from BIDS structure
asl_image = load_image('/path/to/bids/dataset', 
                      subject='01', 
                      session='baseline',
                      suffix='asl')

# Save to BIDS structure
save_image(cbf_map, 
          bids_root='/path/to/output',
          subject='01', 
          session='baseline')
```

## Data Utilities

```python
from asltk.utils import collect_data_volumes, save_asl_data, load_asl_data

# Separate 4D/5D data into individual volumes
volumes, original_shape = collect_data_volumes(asl_data('pcasl'))
print(f"Extracted {len(volumes)} volumes from shape {original_shape}")

# Save/load complete ASL data objects
save_asl_data(asl_data, 'asl_dataset.pkl')
loaded_asl = load_asl_data('asl_dataset.pkl')
```

## Best Practices

1. **Always use brain masks** for better performance and quality
2. **Start with conservative parameters** and adjust based on results
3. **Check result ranges** against expected physiological values
4. **Use parallel processing** but leave cores for system operations

For complete workflow examples, see [docs/examples/workflow_examples.md](examples/workflow_examples.md).