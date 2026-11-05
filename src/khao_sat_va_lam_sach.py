import os
import numpy as np
import pandas as pd

def load_and_clean_data(file_path):
    # 1. Nạp tập dữ liệu từ file CSV
    df = pd.read_csv(file_path)
    
    # In kích thước dòng và cột chuẩn (sử dụng index 0 và 1)
    print(f"-> Kích thước tập dữ liệu ban đầu: {df.shape} dòng, {df.shape[1]} cột.")
    
    # 2. Xử lý giá trị missing ngụy trang (chuyển số 0 bất hợp lý thành NaN)
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    existing_cols = [col for col in zero_features if col in df.columns]
    
    for col in existing_cols:
        df[col] = df[col].replace(0, np.nan)
        # Thay thế NaN bằng giá trị Median được tính theo từng nhóm nhãn Outcome
        if 'Outcome' in df.columns:
            df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))
        else:
            df[col] = df[col].fillna(df[col].median())
            
    print("-> Đã xử lý xong các giá trị missing ngụy trang bằng Median.")
    return df

def roi_rac_hoa_du_lieu(df, num_cols):
    """
    (Tùy chọn) Rời rạc hóa các thuộc tính số liên tục thành các khoảng (Binning)
    phục vụ cho các thuật toán cây quyết định thuần (như ID3).
    """
    df_discrete = df.copy()
    for col in num_cols:
        if col in df_discrete.columns:
            df_discrete[col] = pd.qcut(
                df_discrete[col], 
                q=3, 
                labels=['Thap', 'TrungBinh', 'Cao']
            )
    print(f"-> Đã rời rạc hóa các cột số: {num_cols}")
    return df_discrete

# --- CHẠY THỰC THI ---
if __name__ == "__main__":
    # Tự động tìm đường dẫn file dữ liệu gốc
    raw_data_path = "data/raw/diabetes.csv"
    if not os.path.exists(raw_data_path):
        raw_data_path = "du_lieu/tho/diabetes.csv"
        
    try:
        # Làm sạch dữ liệu
        df_clean = load_and_clean_data(raw_data_path)
        
        # Đảm bảo thư mục lưu trữ đã tồn tại
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "data_clean.csv")
        
        # Xuất dữ liệu sạch ra file CSV mới
        df_clean.to_csv(output_path, index=False)
        print(f"-> Đã lưu dữ liệu sạch vào '{output_path}'.")
    except FileNotFoundError:
        print(f"❌ Chưa tìm thấy file dữ liệu gốc tại '{raw_data_path}'. Vui lòng kiểm tra lại đường dẫn!")