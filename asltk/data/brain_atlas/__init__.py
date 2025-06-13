# Brain atlas list for ASLtk
# All the data are storage in the Kaggle ASLtk project
# When a new data is called, then the brain atlas is allocated locally
from asltk.data.kaggle_tools import download_brain_atlas

BRAIN_ATLASES = {
    'MNI152ArterialTerritories': {
        'dataset_url': '',
        'official_url': 'https://www.nitrc.org/projects/arterialatlas',
        'description': 'atlas of brain arterial territories based on lesion distributions in 1,298 acute stroke patients.',
        'dataset_doi': '10.25790/bml0cm.109',
        'citation_doi': ['10.1038/s41597-022-01923-0'],
        'labels': {},
    },
    'HOCorticalSubcorticalParcellation': {
        'dataset_url': '',
        'official_url': 'https://neurovault.org/collections/262/',
        'description': 'Probabilistic atlases covering 48 cortical and 21 subcortical structural areas, derived from structural data and segmentations kindly provided by the Harvard Center for Morphometric Analysis.',
        'dataset_doi': '',
        'citation_doi': [
            '10.1016/j.schres.2005.11.020',
            '10.1176/appi.ajp.162.7.1256',
            '10.1016/j.neuroimage.2006.01.021',
            '10.1016/j.biopsych.2006.06.027',
        ],
        'labels': {},
    },
    'Automated Anatomical Labeling': {
        'dataset_url': '',
        'official_url': 'https://www.gin.cnrs.fr/en/tools/aal/',
        'description': 'The automated anatomical parcellation AAL3 of the spatially normalized single-subject high-resolution T1 volume provided by the Montreal Neurological Institute (MNI).',
        'dataset_doi': '',
        'citation_doi': [
            '10.1006/nimg.2001.0978',
            '10.1016/j.neuroimage.2015.07.075',
            '10.1006/nimg.2001.0978',
        ],
        'labels': {},
    },
    'Mindboggle 101': {
        'dataset_url': '',
        'official_url': 'https://mindboggle.info/data',
        'description': 'dataset consists of 101 labeled brain images that have been manually labeled largely following the Desikan protocol. It also consists of a group-level parcellation atlas which has been included into Lead-DBS for connectomic analyses.',
        'dataset_doi': '',
        'citation_doi': ['10.3389/fnins.2012.00171'],
        'labels': {},
    },
    'Cortical Area Parcellation from Resting-State Correlations': {},  # https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
    'Local-Global Parcellation of the Human Cerebral Cortex': {},  # https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
    'AICHA: An atlas of intrinsic connectivity of homotopic areas': {},  # https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
    'Hammersmith atlas': {},  # https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
    'JuBrain / Juelich histological atlas': {},  # https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
    'Desikan-Killiany Atlas': {},  # https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
    'Functional Connectivity Atlas 7 Networks': {},  # https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
    'MNI structural atlas': {  # TODO Check the FSL compatible atlas
        'dataset_url': '',
        'official_url': 'https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009',
        'description': 'A number of unbiased non-linear averages of the MNI152 database have been generated that combines the attractions of both high-spatial resolution and signal-to-noise while not being subject to the vagaries of any single brain.',
        'dataset_doi': '',
        'citation_doi': [],
        'labels': {},
    },
}

class BrainAtlas():

    def __init__(self):
        pass

    def get_atlas_url(self, atlas_name: str):
        pass

    def get_atlas_labels(self, atlas_name: str):
        pass

    def get_atlas_info(self, atlas_name: str):
        pass

    def list_atlas(self):
        pass

    def _check_atlas_name(self, atlas_name: str):
        # check if the atlas_name exist into the BRAIN_ATLASES database
        pass
