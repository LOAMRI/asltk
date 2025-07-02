# ASL Processing Quick Reference

This document serves as a quick reference for common ASL processing tasks using asltk. For comprehensive workflow examples with detailed explanations, parameter guidance, and complete analysis pipelines, see [workflow_examples.md](examples/workflow_examples.md).

## Essential Usage Patterns

### Minimal CBF Mapping
The most basic ASL processing workflow:

```python
from asltk.asldata import ASLData
from asltk.reconstruction import CBFMapping

# Load data and create mapper
asl_data = ASLData(pcasl='data.nii.gz', m0='m0.nii.gz', 
                   ld_values=[1.8], pld_values=[1.8])
cbf_mapper = CBFMapping(asl_data)

# Generate maps with defaults
results = cbf_mapper.create_map()
cbf_map = results['cbf_norm']  # Results in mL/100g/min
```

### One-Line Data Loading
```python
from asltk.utils import load_image

# Direct file loading
image = load_image('path/to/image.nii.gz')

# BIDS structure loading
image = load_image('/bids/root', subject='01', suffix='asl')
```

### Rapid Results Saving
```python
from asltk.utils import save_image

# Simple file save
save_image(cbf_map, 'cbf_results.nii.gz')

# BIDS structure save
save_image(cbf_map, bids_root='/output', subject='01')
```

## Advanced Processing (One-Liners)

### Multi-Echo ASL
```python
from asltk.reconstruction import MultiTE_ASLMapping

mte_mapper = MultiTE_ASLMapping(asl_data_with_te)
results = mte_mapper.create_map()  # Includes T1blGM mapping
```

### Diffusion-Weighted ASL
```python
from asltk.reconstruction import MultiDW_ASLMapping

mdw_mapper = MultiDW_ASLMapping(asl_data_with_dw)
results = mdw_mapper.create_map()  # Includes compartment analysis
```

## Data Handling Shortcuts

### Volume Extraction
```python
from asltk.utils import collect_data_volumes

volumes, shape = collect_data_volumes(asl_4d_data)
# Automatically handles 4D/5D data separation
```

### Complete Dataset Management
```python
from asltk.utils import save_asl_data, load_asl_data

# Save entire ASL dataset
save_asl_data(asl_data, 'dataset.pkl')

# Reload for later analysis
asl_data = load_asl_data('dataset.pkl')
```

## Performance Tips

- **Use brain masks**: `mapper.set_brain_mask(mask)` for 3-10x speedup
- **Default parameters**: All `create_map()` parameters are optional with sensible defaults
- **Conservative processing**: Start with smaller brain regions for multi-DW analysis
- **Format support**: Save in .nii.gz, .nii, .mha, or .nrrd formats

## Expected Value Ranges

| Measurement | Healthy Range | Units |
|-------------|---------------|-------|
| CBF (GM) | 50-80 | mL/100g/min |
| CBF (WM) | 20-30 | mL/100g/min |
| ATT | 800-2000 | ms |
| T1blGM (GM) | 200-600 | ms |
| T1blGM (WM) | 400-800 | ms |

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Processing too slow | Use brain mask: `mapper.set_brain_mask(mask)` |
| Values seem wrong | Check input parameter units and field strength |
| Memory issues | Process smaller ROIs or reduce CPU cores |
| Format errors | Use supported formats: .nii.gz, .nii, .mha, .nrrd |

---

**For detailed workflows, parameter explanations, and complete analysis pipelines, see [workflow_examples.md](examples/workflow_examples.md)**