import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification

print("✅ LightGBM 路徑:", lgb.__file__)

# 建立小型資料集測試是否能跑 GPU
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

try:
    print("🚀 嘗試使用 GPU 啟動 LightGBMClassifier")
    model = lgb.LGBMClassifier(device="gpu")
    model.fit(X, y)
    print("✅ 成功使用 GPU 訓練！")
except Exception as e:
    print("❌ 無法啟用 GPU：", e)
