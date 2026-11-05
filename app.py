import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Cấu hình trang ứng dụng Streamlit
st.set_page_config(
    page_title="Cảnh Báo Nguy Cơ Bệnh Tiểu Đường",
    page_icon="🩺",
    layout="centered",
)


# 2. Tải mô hình đã đóng gói với cơ chế Cache
@st.cache_resource
def load_model_artifacts():
  possible_paths = [
      "mo_hinh/mo_hinh_tieu_duong_tot_nhat.pkl",
      "models/best_diabetes_model.pkl",
      "mo_hinh_tieu_duong_tot_nhat.pkl",
  ]
  for path in possible_paths:
    if os.path.exists(path):
      return joblib.load(path)
  return None


artifacts = load_model_artifacts()

if artifacts is None:
  st.error(
      "Chưa tìm thấy file mô hình `.pkl`. Vui lòng chạy file"
      " `huan_luyen_mo_hinh.py` (hoặc `02_model_training.ipynb`) trước!"
  )
  st.stop()

# Đọc các thành phần đóng gói trong payload
model = artifacts.get("mo_hinh") or artifacts.get("model")
scaler = artifacts.get("bo_chuan_hoa") or artifacts.get("scaler")
feature_names = artifacts.get("ten_dac_trung") or artifacts.get("feature_names")

# Tên cột dự phòng nếu file pkl không lưu feature_names
if not feature_names:
  feature_names = [
      "Pregnancies",
      "Glucose",
      "BloodPressure",
      "SkinThickness",
      "Insulin",
      "BMI",
      "DiabetesPedigreeFunction",
      "Age",
  ]

# 3. Giao diện chính của ứng dụng Web
st.title("Ứng Dụng Cảnh Báo Nguy Cơ Bệnh Tiểu Đường")
st.write(
    "Nhập các chỉ số sinh lý và sức khỏe cá nhân dưới đây để kiểm tra nguy cơ"
    " mắc bệnh tiểu đường."
)

st.markdown("---")

# Chia làm 2 cột nhập liệu cân đối
col1, col2 = st.columns(2)

with col1:
  pregnancies = st.number_input(
      "Số lần mang thai (Pregnancies)",
      min_value=0,
      max_value=20,
      value=1,
      step=1,
  )
  glucose = st.number_input(
      "Chỉ số Đường huyết (Glucose - mg/dL)",
      min_value=0,
      max_value=300,
      value=120,
      step=1,
  )
  blood_pressure = st.number_input(
      "Huyết áp tâm trương (BloodPressure - mmHg)",
      min_value=0,
      max_value=150,
      value=70,
      step=1,
  )
  skin_thickness = st.number_input(
      "Độ dày nếp gấp da (SkinThickness - mm)",
      min_value=0,
      max_value=100,
      value=20,
      step=1,
  )

with col2:
  insulin = st.number_input(
      "Chỉ số Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1
  )
  bmi = st.number_input(
      "Chỉ số khối cơ thể (BMI - kg/m²)",
      min_value=0.0,
      max_value=70.0,
      value=25.0,
      step=0.1,
  )
  dpf = st.number_input(
      "Hàm di truyền (DiabetesPedigreeFunction)",
      min_value=0.0,
      max_value=3.0,
      value=0.47,
      step=0.01,
  )
  age = st.number_input(
      "Tuổi (Age)", min_value=1, max_value=120, value=30, step=1
  )

st.markdown("---")

# 4. Thực hiện dự đoán khi người dùng bấm nút
if st.button("Dự Đoán Nguy Cơ", use_container_width=True):
  # Tạo DataFrame dữ liệu đầu vào
  input_df = pd.DataFrame(
      [
          [
              pregnancies,
              glucose,
              blood_pressure,
              skin_thickness,
              insulin,
              bmi,
              dpf,
              age,
          ]
      ],
      columns=feature_names,
  )

  # Chuẩn hóa dữ liệu bằng StandardScaler nếu có
  if scaler is not None:
    input_data = scaler.transform(input_df)
  else:
    input_data = input_df

  # SỬA LỖI CHỈ MỤC (INDEXING):
  # predict_proba trả về matrix shape (1, 2) -> dùng [0][1] để lấy xác suất của lớp 1 (mắc bệnh)
  probability = model.predict_proba(input_data)[0][1]

  # predict trả về array([0]) hoặc array([1]) -> dùng [0] để lấy nhãn số nguyên
  prediction = model.predict(input_data)[0]

  percentage = probability * 100

  st.subheader("Kết Quả Phân Tích:")
  st.progress(float(probability))
  st.write(f"Xác suất nguy cơ mắc bệnh: `{percentage:.2f}%`")

  if percentage >= 50:
    st.error(
        f"CẢNH BÁO: Bạn có nguy cơ CAO mắc bệnh tiểu đường (Nhãn dự đoán:"
        f" {prediction}). Bạn nên đến cơ sở y tế để được thăm khám và xét nghiệm"
        " chi tiết!"
    )
  else:
    st.success(
        f"AN TÂM: Bạn có nguy cơ THẤP mắc bệnh tiểu đường (Nhãn dự đoán:"
        f" {prediction}). Hãy tiếp tục duy trì chế độ ăn uống và lối sống lành"
        " mạnh!"
    )