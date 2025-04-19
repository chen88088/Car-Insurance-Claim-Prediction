import lightgbm as lgb
print(lgb.__file__)

from lightgbm import LGBMClassifier
model = LGBMClassifier(device='gpu')
print(model.fit.__doc__)
