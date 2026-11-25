import pandas as pd
from paths import DATA_DIR, OUT_DIR
# 1. Đọc dữ liệu DXSUM
path = DATA_DIR / "DXSUM.csv"
df = pd.read_csv(path, low_memory=False)

# 2. Kiểm tra các cột cần thiết
required_columns = ["RID", "DIAGNOSIS", "COLPROT", "DXDDUE", "DXAD"]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Không tìm thấy các cột: {missing_columns}")

# 3. Chuẩn hóa cột DIAGNOSIS
df["DIAGNOSIS_CLEAN"] = (
    df["DIAGNOSIS"]
    .astype("string")
    .str.strip()
    .str.upper()
)

# 4. Chỉ lấy các bản ghi có chẩn đoán Dementia
dementia_df = df[
    df["DIAGNOSIS_CLEAN"].eq("DEMENTIA")
].copy()

# 5. Xác định giá trị trống ở DXDDUE và DXAD
# Bao gồm: NaN, chuỗi rỗng và chuỗi chỉ chứa khoảng trắng
def is_blank(series):
    return (
        series.isna()
        | series.astype("string").str.strip().eq("").fillna(False)
    )

dementia_df["DXDDUE_BLANK"] = is_blank(dementia_df["DXDDUE"])
dementia_df["DXAD_BLANK"] = is_blank(dementia_df["DXAD"])

# Cả hai trường đều trống
dementia_df["BOTH_BLANK"] = (
    dementia_df["DXDDUE_BLANK"]
    & dementia_df["DXAD_BLANK"]
)

# 6. Chuẩn hóa COLPROT
dementia_df["COLPROT_GROUP"] = (
    dementia_df["COLPROT"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .fillna("MISSING")
)

# 7. Thống kê theo giai đoạn thu thập
result = (
    dementia_df
    .groupby("COLPROT_GROUP", sort=False)
    .agg(
        Total_Dementia_Records=("RID", "size"),
        Both_Blank_Records=("BOTH_BLANK", "sum")
    )
    .reset_index()
)

# 8. Tính tỷ lệ
result["Both_Blank_Percentage"] = (
    result["Both_Blank_Records"]
    / result["Total_Dementia_Records"]
    * 100
).round(2)

# 9. Thống kê thêm số người tham gia có bản ghi bị thiếu cả hai trường
people_count = (
    dementia_df[dementia_df["BOTH_BLANK"]]
    .groupby("COLPROT_GROUP")["RID"]
    .nunique()
)

result["Unique_RID_Both_Blank"] = (
    result["COLPROT_GROUP"]
    .map(people_count)
    .fillna(0)
    .astype(int)
)

# 10. Hiển thị kết quả
print("\nTHỐNG KÊ BẢN GHI DEMENTIA THIẾU CẢ DXDDUE VÀ DXAD")
print(result.to_string(index=False))
# 11. Xuất kết quả CSV
result.to_csv(
    OUT_DIR / "dementia_missing_DXDDUE_DXAD_by_COLPROT.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã xuất dementia_missing_DXDDUE_DXAD_by_COLPROT.csv")
