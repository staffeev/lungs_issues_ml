"""
Модуль с доступными функциями ошибок. 

Для расширямости и масштабируемости все функции ошибок будут 
объявлены или определены здесь вне зависимости от их происхождения.

"""

from torch.nn import *

__all__ = [
    'CrossEntropyLoss', 'BCELoss', 'L1Loss', 'MSELoss', 'CTCLoss', 'NLLLoss', 
    'PoissonNLLLoss', 'GaussianNLLLoss', 'KLDivLoss', 'BCELoss', 'BCEWithLogitsLoss', 
    'MarginRankingLoss', 'HingeEmbeddingLoss', 'MultiLabelMarginLoss', 'HuberLoss', 
    'SmoothL1Loss', 'SoftMarginLoss', 'MultiLabelSoftMarginLoss', 'CosineEmbeddingLoss', 
    'MultiMarginLoss', 'TripletMarginLoss', 'TripletMarginWithDistanceLoss'
]


