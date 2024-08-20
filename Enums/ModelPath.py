from enum import Enum

class ModelPath(Enum):
    BASELINE_PRETRAINED = "./pretrained_model.pth"
    BASELINE_CUSTOM = "./CustomModel.pth"
    PERMUTATION_FT_PRETRAINED = ''
    PERMUTATION_FT_CUSTOM = ''