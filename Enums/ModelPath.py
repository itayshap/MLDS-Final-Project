from enum import Enum

class ModelPath(Enum):
    BASELINE_PRETRAINED = "./Pretrained_Model.pth"
    BASELINE_CUSTOM = "./CustomModel.pth"
    BASELINE_IMPROVED_CUSTOM = "./Improved_CustomModel.pth"

    PERMUTATION_FT_PRETRAINED = './Pretrained_Model_Premutation.pth'
    PERMUTATION_FT_CUSTOM = './CustomModel_Premutation.pth'
    PERMUTATION_FT_IMPROVED_CUSTOM = './Improved_CustomModel_Premutation.pth'
    SEVERAL_PERMUTATION_FT_IMPROVED_CUSTOM = './Improved_CustomModel_SeveralPremutations.pth'
    PERMUTATION_FT_VIT = './Vit_premutation_classifier_layer.pth'

    