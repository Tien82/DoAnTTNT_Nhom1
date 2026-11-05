import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, roc_auc_score

# Import các hàm từ 2 module đã được chuẩn hóa trước đó
from phanchia_va_canbang import prepare_data
from evaluate import danh_gia_chi_tiet, ve_va_luu_confusion_matrix

def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN VÀ TỐI ƯU MÔ HÌNH ===")
    
    # 1. Đọc dữ liệu, chia tập Train/Test, cân bằng SMOTE và chuẩn hóa StandardScaler
    X_train_scaled, X_test_scaled, y_train_res, y_test, scaler, feature_names = prepare_data(
        data_path="data/processed/data_clean.csv", 
        target_col="Outcome"
    )
    
    # 2. Khai báo không gian siêu tham số thử nghiệm (Đã sửa 'max_depth': [3-5] thành [3, 4, 5])
    models = {
        'LogisticRegression': (
            LogisticRegression(random_state=42),
            {'C': [0.01, 0.1, 1, 10]}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100], 'max_depth': [3, 5, 7, None]}
        ),
        'XGBoost': (
            XGBClassifier(random_state=42, eval_metric='logloss'),
            {'n_estimators': [50, 100], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1]}
        )
    }
    
    best_model = None
    best_recall = 0
    best_name = ""
    
    # 3. Thử nghiệm huấn luyện và tối ưu siêu tham số ưu tiên chỉ số Recall
    print("\n--- KẾT QUẢ THỰC NGHIỆM MÔ HÌNH (GRIDSEARCHCV CV=5) ---")
    for name, (model, params) in models.items():
        # Tối ưu theo tiêu chuẩn Recall để bắt tối đa ca mắc bệnh
        grid = GridSearchCV(model, params, scoring='recall', cv=5, n_jobs=-1)
        grid.fit(X_train_scaled, y_train_res)
        
        y_pred = grid.predict(X_test_scaled)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, grid.predict_proba(X_test_scaled)[:, 1])
        
        print(f"\nMô hình: {name:20s} | Best Params: {grid.best_params_}")
        print(f"         Recall (Độ nhạy): {rec:.4f} | ROC-AUC: {auc:.4f}")
        
        # In chi tiết các chỉ số đo lường qua module evaluate.py
        danh_gia_chi_tiet(y_test, y_pred, ten_mo_hinh=name)
        
        # Chọn mô hình có Recall cao nhất
        if rec > best_recall:
            best_recall = rec
            best_model = grid.best_estimator_
            best_name = name

    print(f"\n🏆 MÔ HÌNH XUẤT SẮC NHẤT: {best_name} (Recall = {best_recall:.4f})")
    
    # 4. Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix) cho mô hình tốt nhất
    y_pred_best = best_model.predict(X_test_scaled)
    ve_va_luu_confusion_matrix(y_test, y_pred_best, save_path="reports/confusion_matrix.png")
    
    # 5. Đóng gói Payload và xuất thành file .pkl
    os.makedirs('mo_hinh', exist_ok=True)
    payload = {
        'mo_hinh': best_model,
        'bo_chuan_hoa': scaler,
        'ten_dac_trung': feature_names
    }
    
    output_path = 'mo_hinh/mo_hinh_tieu_duong_tot_nhat.pkl'
    joblib.dump(payload, output_path)
    print(f"-> Đã lưu mô hình thành công vào file: '{output_path}'")

if __name__ == "__main__":
    main()