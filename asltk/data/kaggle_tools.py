import os

import kagglehub

# from cucaracha.ml_models import CUCARACHA_PRESETS, DEFAULT_MODEL_LAKE


def download_brain_atlas(dataset_url: str):
    """
    Downloads a Cucaracha model from the given URL.
    This function checks if the provided model URL is present in the CUCARACHA_PRESETS dictionary.
    If the URL is valid, it attempts to download the model using the kagglehub library.

    The downloaded files are located in the home/.cache folder.

    Note:
        We used the kagglehub library to make all the operations here. The
        CUCARACHA_PRESETS dictionary is expected to have a nested structure where
        the model variations are stored under a 'variation' key. If the URL is
        valid, it attempts to download the model using the kagglehub library.

    Args:
        model_url (str): The URL of the model to be downloaded. Must be a valid kagglehub input.
    Returns:
        str: The path where the model is downloaded.
    Raises:
        ValueError: If the model URL is not present in CUCARACHA_PRESETS or if there is an error during download.
    """

    found = False
    for url in CUCARACHA_PRESETS.values():
        for item in url.values():
            if dataset_url in item['variation']:
                found = True
                break
        if found:
            break

    if not found:
        raise ValueError(
            f'Model URL {dataset_url} is not present in CUCARACHA_PRESETS'
        )

    try:
        path = kagglehub.model_download(dataset_url)
    except Exception as e:
        raise ValueError(f'Error downloading the model: {e}')

    return path



# def collect_cucaracha_model(cucaracha_preset: str):
#     """
#     Collects the Cucaracha model from the given preset.

#     This function checks if the provided model preset is present in the CUCARACHA_PRESETS dictionary.
#     If the preset is valid, it attempts to download the model using the kagglehub library.

#     The downloaded files are located in the home/.cache folder.

#     Note:
#         We used the kagglehub library to make all the operations here. The
#         CUCARACHA_PRESETS dictionary is expected to have a nested structure where
#         the model variations are stored under a 'variation' key. If the URL is
#         valid, it attempts to download the model using the kagglehub library.

#     Args:
#         cucaracha_preset (str): The name of the model preset to be downloaded.
#     Returns:
#         str: The path where the model is downloaded.
#     Raises:
#         ValueError: If the model preset is not present in CUCARACHA_PRESETS or if there is an error during download.
#     """
#     found = False
#     modality = None
#     for mod in CUCARACHA_PRESETS.values():
#         if cucaracha_preset in mod:
#             found = True
#             modality = next(
#                 key
#                 for key, value in CUCARACHA_PRESETS.items()
#                 if cucaracha_preset in value
#             )
#             break

#     if not found:
#         raise ValueError(
#             f'Model preset {cucaracha_preset} is not present in CUCARACHA_PRESETS'
#         )

#     model_url = CUCARACHA_PRESETS[modality][cucaracha_preset]['variation']

#     output = {
#         'model_path': download_cucaracha_model(model_url),
#         'modality': modality,
#         'labels': CUCARACHA_PRESETS[modality][cucaracha_preset]['labels'],
#     }
#     return output