from enum import Enum

class ModelPath(Enum):
    BASELINE_PRETRAINED = "./pretrained_model.pth"
    BASELINE_CUSTOM = "./CustomModel.pth"
    PERMUTATION_FT_PRETRAINED = './pretrained_model_premutation.pth'
    PERMUTATION_FT_CUSTOM = './CustomModel_premutation.pth'