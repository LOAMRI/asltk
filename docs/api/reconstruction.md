# Reconstruction Module

## ASL Mapping Reconstruction

This is the main module in `asltk` which aims to offer quantitative reconstruction algorithms for the `ASLData`. 

In general, the ASL processing requires data fitting and manipulation to result in quantitative mappings. A classical example are the CBF and ATT mapping.

One of the major objective in `asltk` is to provide the state-of-the-art reconstruction methods available in the ASL research community, allowing an easy to use application to many studies.

!!! note
    It is intended to the `reconstruction` module to be evolve by the contribution of scientific community. If you want to apply a new reconstruction method in the `asltk` project, please see more details at the `How to Contribute` section and into the reconstruction classes standards

### Reconstruction class standard

In order to comply a standarized way to build the reconstruction classes, we assumed the following structure:

1. All reconstruction classes must hierent from `MRIParameters` data class, which informs the base MRI parameters values for a generic data analysis and fitting
2. The base reconstruction class must have a constructor (`__init__` method) with an input data as the `ASLData` or other types (if needed).
3. Even though there is not obligation (regulated by abstract methods) for determined class methods, it is expected that the reconstruction class can implement a `create_map` method, which is the actual reconstruction calculation that exposes (as an method output) the reconstructed data from `ASLData`.

!!! info
    Depending on the specifity of the reconstruction method, it can vary the way that the inner reconstruction methods can be implemented. However, we try to maintain a basic API usage to get a more uniform implementation thoughout the community contribution.

## Reconstruction Module API

::: reconstruction