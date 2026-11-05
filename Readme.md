# 🩺 Ứng Dụng Cảnh Báo Nguy Cơ Bệnh Tiểu Đường

Hệ thống hỗ trợ dự đoán và cảnh báo sớm nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số sinh lý cá nhân, ứng dụng các thuật toán Học Máy (Machine Learning) kết hợp với giao diện ứng dụng Web **Streamlit**.

---

## 📌 Giới Thiệu Đề Tài

Bệnh tiểu đường (Đái tháo đường) là một trong những bệnh mãn tính phổ biến và có tốc độ gia tăng nhanh chóng. Đồ án xây dựng một hệ thống Học Máy toàn trình giúp dự đoán nguy cơ mắc bệnh tiểu đường dựa trên tập dữ liệu chuẩn **Pima Indians Diabetes Dataset**:
- **Khảo sát & Tiền xử lý:** Kiểm tra giá trị thiếu ($0$), xử lý điền khuyết bằng giá trị trung vị (Median) theo từng nhóm nhãn.
- **Cân bằng & Chuẩn hóa dữ liệu:** Sử dụng kỹ thuật **SMOTE** và **StandardScaler** để nâng cao hiệu quả phân loại.
- **Huấn luyện & Đánh giá:** Áp dụng các thuật toán Học Máy (Logistic Regression, Random Forest, XGBoost,...) để chọn mô hình tối ưu nhất.
- **Triển khai ứng dụng Web:** Đóng gói mô hình và xây dựng giao diện Streamlit trực quan, thân thiện cho người dùng.

---

## 📂 Cấu Trúc Dự Án

```text
DoAnTTNT_Nhom1/
├── data/
│   ├── processed/
│   │   └── data_clean.csv            # Dữ liệu đã qua làm sạch & điền giá trị khuyết
│   └── raw/
│       └── diabetes.csv                # Dữ liệu thô ban đầu
├── mo_hinh/
│   ├── best_diabetes_model.pkl       # File đóng gói mô hình & scaler (Tên tiếng Anh)
│   └── mo_hinh_tieu_duong_tot_nhat.pkl # File đóng gói mô hình & scaler (Tên tiếng Việt)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb    # Notebook khảo sát & làm sạch dữ liệu
│   └── 02_model_training.ipynb       # Notebook huấn luyện & tối ưu mô hình
├── reports/
│   └── confusion_matrix.png          # Biểu đồ Ma trận nhầm lẫn (Confusion Matrix)
├── src/
│   ├── evaluate.py                   # Hàm đánh giá độ chính xác & vẽ Confusion Matrix
│   ├── huan_luyen_mo_hinh.py         # Script chính huấn luyện mô hình & lưu file .pkl
│   ├── khao_sat_va_lam_sach.py       # Script làm sạch dữ liệu thô
│   └── phanchia_va_canbang.py        # Script phân chia Train/Test, SMOTE & StandardScaler
├── app.py                            # Giao diện ứng dụng Web Streamlit
└── requirements.txt                  # Danh sách thư viện phụ thuộc của dự án
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Yêu cầu hệ thống
- **Python:** phiên bản `3.9` trở lên.

### 2. Cài đặt các thư viện phụ thuộc

Mở Terminal / PowerShell tại thư mục gốc dự án (`DoAnTTNT_Nhom1`) và chạy lệnh:

```bash
pip install -r requirements.txt
```

*(Nếu cài đặt thủ công các thư viện chính:)*
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib streamlit matplotlib seaborn
```

---

## 🚀 Quy Trình Thực Thi Chương Trình

Mở thư mục gốc dự án trong Terminal và chạy các lệnh theo đúng thứ tự sau:

### Bước 1: Làm sạch dữ liệu thô
```bash
python src/khao_sat_va_lam_sach.py
```
> **Đầu ra:** Tạo file dữ liệu sạch `data/processed/data_clean.csv`.

### Bước 2: Huấn luyện và đóng gói mô hình
```bash
python src/huan_luyen_mo_hinh.py
```
> **Đầu ra:** Thực hiện phân chia tập dữ liệu, áp dụng **SMOTE**, chuẩn hóa **StandardScaler**, tìm mô hình tối ưu và lưu vào thư mục `mo_hinh/`.

### Bước 3: Khởi chạy ứng dụng Web Streamlit
```bash
streamlit run app.py
```
> **Kết quả:** Trình duyệt web sẽ tự động kích hoạt tại địa chỉ: `http://localhost:8501`.

---

## 📊 Các Thuộc Tính Đầu Vào (Features)

1. **Pregnancies**: Số lần mang thai.
2. **Glucose**: Chỉ số đường huyết sau 2 giờ kiểm tra dung nạp ($mg/dL$).
3. **BloodPressure**: Huyết áp tâm trương ($mmHg$).
4. **SkinThickness**: Độ dày nếp gấp da cơ tam đầu ($mm$).
5. **Insulin**: Nồng độ insulin huyết thanh sau 2 giờ ($mu U/ml$).
6. **BMI**: Chỉ số khối cơ thể, tính bằng $\frac{\text{cân nặng}}{\text{chiều cao}^2}$ ($kg/m^2$).
7. **DiabetesPedigreeFunction**: Hàm di truyền bệnh tiểu đường.
8. **Age**: Tuổi (năm).

---

## 🧰 Thư Viện & Công Nghệ Sử Dụng

- **Ngôn ngữ:** Python
- **Xử lý dữ liệu:** `Pandas`, `NumPy`
- **Học máy & Tiền xử lý:** `Scikit-learn`, `Imbalanced-learn` (SMOTE), `XGBoost`
- **Đóng gói mô hình:** `Joblib`
- **Trực quan hóa:** `Matplotlib`, `Seaborn`
- **Giao diện Web:** `Streamlit`

---

## 👥 Thông Tin Thực Hiện

- **Môn học:** Trí Tuệ Nhân Tạo
- **Đơn vị:** Nhóm 1
- **Thành Viên:** 
1.    **Lê Thị Tường Vy - DTH235818**
2.    **Phan Thị Thanh Tiền - DTH235785**
3.    **Lâm Đức Trường - DTH235802**