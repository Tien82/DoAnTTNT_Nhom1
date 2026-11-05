import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def prepare_data(data_path="data/processed/data_clean.csv", target_col="Outcome", test_size=0.20):
    """
    Hàm đọc dữ liệu sạch, phân chia tập Train/Test, áp dụng SMOTE và chuẩn hóa dữ liệu.
    """
    # 1. Đọc dữ liệu từ file CSV đã làm sạch
    if not os.path.exists(data_path):
        data_path = "du_lieu/da_xu_ly/data_clean.csv"  # Đường dẫn dự phòng
        
    df = pd.read_csv(data_path)
    print(f"-> Đã nạp thành công dataset sạch: {df.shape} dòng, {df.shape[1]} cột.")
    
    # 2. Tách tập đặc trưng đầu vào (X) và nhãn mục tiêu (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. Phân chia tập Train (80%) và Test (20%)
    # Tham số stratify=y đảm bảo tỷ lệ phân bố nhãn đều ở cả 2 tập
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"-> Kích thước ban đầu - Train: {X_train.shape} mẫu | Test: {X_test.shape} mẫu")
    
    # 4. Cân bằng dữ liệu tập Train bằng kỹ thuật SMOTE (Chỉ áp dụng trên tập Train)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"-> Kích thước tập Train sau khi cân bằng SMOTE: {X_train_res.shape} mẫu")
    
    # 5. Chuẩn hóa thang đo bằng StandardScaler
    # fit_transform trên tập Train và chỉ transform trên tập Test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_res, y_test, scaler, list(X.columns)

# --- CHẠY THỬ NGHIỆM ĐỘC LẬP ---
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
        print("\n Hoàn tất chuẩn bị dữ liệu! Sẵn sàng cho bước huấn luyện mô hình.")
    except FileNotFoundError:
        print("❌ Chưa tìm thấy file 'data_clean.csv'. Vui lòng chạy file 'khao_sat_va_lam_sach.py' trước!")