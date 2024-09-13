from enum import Enum

class ModelPath(Enum):
    BASELINE_PRETRAINED = "./Weights/Pretrained_Model.pth"
    BASELINE_CUSTOM = "./Weights/CustomModel.pth"
    BASELINE_IMPROVED_CUSTOM = "./Weights/Improved_CustomModel.pth"
    

    PERMUTATION_FT_PRETRAINED = './Weights/Pretrained_Model_Premutation.pth'
    PERMUTATION_FT_CUSTOM = './Weights/CustomModel_Premutation.pth'
    PERMUTATION_FT_IMPROVED_CUSTOM = './Weights/Improved_CustomModel_Premutation.pth'
    SEVERAL_PERMUTATION_FT_IMPROVED_CUSTOM = './Weights/Improved_CustomModel_SeveralPremutations.pth'
    PERMUTATION_FT_VIT = './Weights/Vit_premutation_classifier_layer.pth'

    