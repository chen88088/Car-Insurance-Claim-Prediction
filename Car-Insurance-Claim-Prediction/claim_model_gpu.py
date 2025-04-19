

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# [0] 建立輸出資料夾（依照時間命名）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# [1] 讀取資料
df = pd.read_csv("../DATA/train.csv")
X = df.drop(columns=["policy_id", "is_claim"])
y = df["is_claim"]

# [2] 特徵工程：Label Encoding + 數值標準化
print("[1] 前處理資料...")
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))

# [3] 切分資料
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# [4] Grid Search 優化 LGBMClassifier
print("[2] 調參中（GridSearch）...")
scale_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
params = {
    'num_leaves': [31, 50],
    'max_depth': [5, 10],
    'min_child_samples': [20, 50],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [scale_weight],
    'n_estimators': [100, 200]
}

lgb = LGBMClassifier(random_state=42, device='gpu')
gs = GridSearchCV(lgb, params, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
print("最佳參數：", gs.best_params_)

# [5] 預測與評估
print("[3] 預測與評估...")
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

report = classification_report(y_val, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_val, y_prob)
fpr, tpr, _ = roc_curve(y_val, y_prob)

# [6] 儲存結果
results = {
    "LightGBM_Optimized": {
        "report": report,
        "roc_auc": roc_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "feature_importances": best_model.feature_importances_.tolist(),
        "features": X.columns.tolist(),
        "best_params": gs.best_params_
    }
}

with open(os.path.join(output_dir, "optimized_lgbm_report.json"), "w") as f:
    json.dump(results["LightGBM_Optimized"], f, indent=2)

# [7] 圖表輸出
plt.figure(figsize=(10, 6))
sns.barplot(x=results["LightGBM_Optimized"]["feature_importances"],
            y=results["LightGBM_Optimized"]["features"])
plt.title("Optimized LightGBM Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "optimized_feature_importance.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(results["LightGBM_Optimized"]["fpr"], results["LightGBM_Optimized"]["tpr"],
         label=f"Optimized LightGBM ROC AUC: {results['LightGBM_Optimized']['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Optimized ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "optimized_roc_curve.png"))
plt.close()

print(f"\n✅ 優化模型訓練完成！ROC AUC 已達 {roc_auc:.3f}，所有結果已儲存於資料夾：{output_dir}")
