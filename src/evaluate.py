import os
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend 'Agg' để không bị lỗi GUI khi lưu ảnh
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def danh_gia_chi_tiet(y_true, y_pred, ten_mo_hinh="Mô hình"):
    """
    Hàm tính toán các chỉ số hiệu suất phân loại (Accuracy, Precision, Recall, F1-Score)
    và in kết quả định dạng rõ ràng ra màn hình console.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n--- BẢNG KẾT QUẢ ĐÁNH GIÁ ({ten_mo_hinh}) ---")
    print(f" Độ chính xác (Accuracy)     : {acc:.4f} ({acc*100:.2f}%)")
    print(f" Độ chính xác lớp (Precision): {prec:.4f}")
    print(f" Độ nhạy (Recall)            : {rec:.4f}  [Metric ưu tiên y tế]")
    print(f" F1-Score                    : {f1:.4f}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def ve_va_luu_confusion_matrix(y_true, y_pred, save_path="reports/confusion_matrix.png"):
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap và xuất ra file ảnh .png.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Không bệnh (0)', 'Có bệnh (1)'],
                yticklabels=['Không bệnh (0)', 'Có bệnh (1)'])
    
    plt.title("Ma trận nhầm lẫn (Confusion Matrix)", fontsize=12, fontweight='bold')
    plt.xlabel("Nhãn dự đoán (Predicted)", fontsize=10)
    plt.ylabel("Nhãn thực tế (Actual)", fontsize=10)
    plt.tight_layout()
    
    # Tạo thư mục lưu trữ báo cáo nếu chưa tồn tại
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Đã xuất biểu đồ ma trận nhầm lẫn vào: '{save_path}'")


if __name__ == "__main__":
    # Khối chạy thử nghiệm độc lập module (Chỉ test tính toán chỉ số, không lưu ảnh)
    y_test_demo = [1, 0, 1, 1]
    y_pred_demo = [1, 0, 1, 0]
    
    print("Test chạy thử module đánh giá:")
    danh_gia_chi_tiet(y_test_demo, y_pred_demo, ten_mo_hinh="Thử nghiệm")