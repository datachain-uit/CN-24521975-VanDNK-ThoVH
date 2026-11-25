
import pandas as pd
from paths import DATA_DIR, OUT_DIR
# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
path = DATA_DIR / "DXSUM.csv"
df = pd.read_csv(path, low_memory=False)

# Các cột cần sử dụng
required_cols = ["RID", "ORIGPROT", "VISCODE2", "DIAGNOSIS"]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Không tìm thấy các cột: {missing_cols}")


# ==========================================
# 2. CHỌN BẢN GHI BASELINE
# ==========================================

# VISCODE2 = bl: lần khám baseline
baseline = df[
    df["VISCODE2"].astype("string").str.strip().str.lower() == "bl"
].copy()

print("Số dòng baseline:", len(baseline))
print("Số RID duy nhất:", baseline["RID"].nunique())


# ==========================================
# 3. KIỂM TRA RID TRÙNG
# ==========================================

# Mỗi người chỉ nên có một bản ghi baseline
duplicate_rid = baseline[
    baseline.duplicated(subset=["RID"], keep=False)
]

if not duplicate_rid.empty:
    print("Các RID có nhiều bản ghi baseline:")
    print(
        duplicate_rid[
            ["RID", "ORIGPROT", "VISCODE2", "EXAMDATE", "DIAGNOSIS"]
        ].sort_values("RID")
    )

    raise ValueError(
        "Phát hiện RID trùng tại baseline. "
        "Cần kiểm tra trước khi thống kê, không tự động xóa dòng."
    )


# ==========================================
# 4. KIỂM TRA GIÁ TRỊ CHẨN ĐOÁN
# ==========================================

print("\nCác giá trị DIAGNOSIS tại baseline:")
print(baseline["DIAGNOSIS"].value_counts(dropna=False))

print("\nCác giai đoạn tuyển ban đầu:")
print(baseline["ORIGPROT"].value_counts(dropna=False))

baseline["DIAGNOSIS"] = (
    df["DIAGNOSIS"]
    .astype("string")
    .str.strip()
)

baseline["DX_GROUP"] = (
    baseline["DIAGNOSIS"]
    .fillna("Thiếu/khác mã")
)

# ==========================================
# 5. CHUẨN HÓA VÀ XÁC ĐỊNH DÒNG THIẾU / KHÁC MÃ
# ==========================================

# 1. Làm sạch chuỗi
diag_clean = baseline["DIAGNOSIS"].astype("string").str.strip().str.upper()

# 2. Quy đổi các mã hợp lệ về tên chuẩn (hỗ trợ cả AD, DEMENTIA)
dx_map = {
    "CN": "CN",
    "MCI": "MCI",
    "DEMENTIA": "Dementia",
    "AD": "Dementia"
}

# Ánh xạ: cái nào không khớp với map sẽ tự động thành NaN
baseline["DX_GROUP"] = diag_clean.map(dx_map)

# 3. Lọc ra các dòng không thuộc nhóm chuẩn hoặc bị thiếu
mask_invalid = baseline["DX_GROUP"].isna()

# Gán nhãn "Thiếu/khác mã" cho những dòng này
baseline["DX_GROUP"] = baseline["DX_GROUP"].fillna("Thiếu/khác mã")

# Lọc danh sách để xem và đối chứng
invalid_rows = baseline[mask_invalid]

print(f"\n---> Số dòng Thiếu / Khác mã: {len(invalid_rows)}")

if not invalid_rows.empty:
    # Chọn các cột cần thiết để đối chiếu với dataset gốc
    cols_to_show = [col for col in ["RID", "ORIGPROT", "VISCODE2", "EXAMDATE", "DIAGNOSIS"] if col in baseline.columns]
    
    print("\nDanh sách các bản ghi Thiếu / Khác mã:")
    print(invalid_rows[cols_to_show].to_string(index=False))

# ==========================================
# 6. THỐNG KÊ THEO ORIGPROT VÀ CHẨN ĐOÁN
# ==========================================

# Không bỏ người bị thiếu ORIGPROT
baseline["ORIGPROT"] = baseline["ORIGPROT"].fillna(
    "Thiếu ORIGPROT"
)

result = pd.crosstab(
    baseline["ORIGPROT"],
    baseline["DX_GROUP"]
)

# Đảm bảo các cột luôn xuất hiện theo thứ tự
columns_order = ["CN", "MCI", "Dementia", "Thiếu/khác mã"]

result = result.reindex(
    columns=columns_order,
    fill_value=0
)

# Tính tổng từng giai đoạn
result["Tổng"] = result.sum(axis=1)

# Tính tổng toàn bộ
result.loc["Tổng"] = result.sum(axis=0)


# ==========================================
# 7. HIỂN THỊ
# ==========================================

print("\nPHÂN BỐ CHẨN ĐOÁN TẠI BASELINE:")
print(result)

