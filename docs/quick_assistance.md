# **Quick Assistance**

This section provides minimal examples of common tasks in `asltk`, to help new users get started quickly.

Each snippet includes links to full guides and deeper usage in the rest of the documentation.

---

## Load and Save an ASL Image

```python
from asltk.io import load_asl, save_asl

asl_data = load_asl("input_asl.nii.gz")
save_asl(asl_data, "output_asl.nii.gz")
```
* Related: [Examples](docs/usage_examples.md), [Getting started](docs/getting_started.md)
---
## Modify Parameters of an ASL Image
```python
from asltk.io import load_asl

asl_data = load_asl("image.nii.gz")
asl_data.pld_values = [1.5]  # Update post-label delay
asl_data.ld_values = [1.8]   # Update label duration
```
---
## Copy an ASL Dataset
```python
from copy import deepcopy
from asltk.io import load_asl

asl_data = load_asl("scan.nii.gz")
asl_copy = deepcopy(asl_data)
```
---
## Create T1-BLG Map from MultiTE-ASL
```python
from asltk.io import load_asl
from asltk.multite import generate_t1blgm_map

multi_te_asl = load_asl("multite_asl.nii.gz")
t1blgm_map = generate_t1blgm_map(multi_te_asl)
```
* see also: [usage examples](usage_examples.md)
---
## Modify Parameters Before Reconstruction
```python
from asltk.io import load_asl
from asltk.recon import CBFReconstruction

asl_data = load_asl("input_asl.nii.gz")
asl_data.pld_values = [2.0]
asl_data.ld_values = [1.5]

cbf = CBFReconstruction()
cbf_map = cbf.run(asl_data)
```
---
## Add Logging to a Script
```python
from asltk import setup_logging, get_logger

setup_logging(level="INFO", console_output=True)
logger = get_logger()
logger.info("Starting ASL processing...")
```
* See: [logging](logging.md)
* full logging guide: [logging](logging.md)
---
