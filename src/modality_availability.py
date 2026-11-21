# ============================================================
# MODALITY AVAILABILITY - MCI BASELINE COHORT (ADNI)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\24521718\Documents\KLTN\data\selected")
COH_DIR = Path(r"C:\Users\24521718\Documents\KLTN\data\selected\mci_cohort_results")

COHORT_PATH = COH_DIR / "mci_cohort_baseline.csv"
ADSL_PATH = BASE_DIR / "ADSL.csv"
MRI_PATH = BASE_DIR / "UCSFFSX7.csv"

OUTPUT_DIR = BASE_DIR / "modality_availability_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. GROUPS
# ============================================================

GROUPS = {
    "demo_gen": [
        "AGE", "SEX", "RACE", "ETHNIC", "EDUC", "MARISTAT", "BMI"
    ],

    "APOE": [
        "APOE"
    ],

    "cognitive": [
        "ADASTT11", "ADASTT13", "CDGLOBAL", "CDRSB",
        "ECOGPTTT", "ECOGSPTT", "FAQTOTAL", "FCISCORE",
        "GDTOTAL", "MMSCORE", "MOCA", "NPITOTAL",
        "NPIQTOTL", "DIGITSCR", "LDELTOTL", "LIMMTOTL",
        "RAVLTFG", "RAVLTFGP", "RAVLTIMM", "RAVLTLRN",
        "TRABSCOR", "MPACCDIGIT", "MPACCTRAILSB"
    ],

    "AMYSTAT": [
        "AMYSTAT"
    ],

    "MRI": [
        "MRI_AVAILABLE"
    ]
}

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clean_string(series):
    """Chuẩn hóa chuỗi, giữ giá trị thiếu."""
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def check_required_columns(df, required, file_name):
    """Kiểm tra các cột bắt buộc."""
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{file_name} thiếu các cột bắt buộc: {missing}"
        )


def make_rid_key(series):
    """Chuẩn hóa khóa RID dưới dạng chuỗi."""
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def group_complete_flags(df, groups):
    """
    Tạo flag cho từng nhóm:
    1 = tất cả biến trong nhóm có dữ liệu
    0 = thiếu ít nhất một biến trong nhóm
    """

    flags = pd.DataFrame(index=df.index)

    for group_name, columns in groups.items():

        available_columns = [
            c for c in columns if c in df.columns
        ]

        if not available_columns:
            raise ValueError(
                f"Nhóm {group_name} không có cột hợp lệ "
                f"trong bảng đầu vào."
            )

        flags[group_name] = (
            df[available_columns]
            .notna()
            .all(axis=1)
            .astype(int)
        )

    return flags


# ============================================================
# 4. LOAD MCI BASELINE COHORT
# ============================================================

print("=" * 60)
print("4. LOAD MCI BASELINE COHORT")
print("=" * 60)

cohort = pd.read_csv(COHORT_PATH, low_memory=False)

print("Shape cohort:", cohort.shape)
print("Columns:", cohort.columns.tolist())

# Xác định cột RID
if "RID_KEY" in cohort.columns:
    cohort["RID_KEY"] = make_rid_key(cohort["RID_KEY"])

elif "RID" in cohort.columns:
    cohort["RID_KEY"] = make_rid_key(cohort["RID"])

else:
    raise ValueError(
        "Cohort không có RID_KEY hoặc RID."
    )

check_required_columns(
    cohort,
    ["RID_KEY", "ORIGPROT"],
    "Cohort"
)

cohort["ORIGPROT"] = clean_string(cohort["ORIGPROT"])

cohort = cohort.dropna(subset=["RID_KEY"]).copy()

# Loại TEAM nếu còn trong cohort
cohort = cohort[
    cohort["ORIGPROT"].str.upper() != "TEAM"
].copy()

# Một dòng / RID
if cohort["RID_KEY"].duplicated().any():

    duplicated_rids = (
        cohort.loc[
            cohort["RID_KEY"].duplicated(keep=False),
            "RID_KEY"
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Cohort có RID trùng. Cần kiểm tra trước khi tiếp tục.\n"
        f"RID trùng: {duplicated_rids[:20]}"
    )

print("Số RID trong cohort:", cohort["RID_KEY"].nunique())

print("\nSố RID theo ORIGPROT:")
print(cohort["ORIGPROT"].value_counts(dropna=False))


# ============================================================
# 5. LOAD ADSL
# ============================================================

print("\n" + "=" * 60)
print("5. LOAD ADSL")
print("=" * 60)

adsl = pd.read_csv(ADSL_PATH, low_memory=False)

print("Shape ADSL:", adsl.shape)

check_required_columns(
    adsl,
    ["SUBJID", "ORIGPROT"],
    "ADSL"
)

# Loại TEAM
adsl = adsl[
    adsl["ORIGPROT"].astype("string").str.upper() != "TEAM"
].copy()

# Chuẩn hóa khóa nối
adsl["RID_KEY"] = make_rid_key(adsl["SUBJID"])

adsl["ORIGPROT"] = clean_string(adsl["ORIGPROT"])

adsl = adsl.dropna(subset=["RID_KEY"]).copy()

# Kiểm tra RID trùng trong ADSL
if adsl["RID_KEY"].duplicated().any():

    duplicated_rids = (
        adsl.loc[
            adsl["RID_KEY"].duplicated(keep=False),
            "RID_KEY"
        ]
        ["RID_KEY"]
        .unique()
        .tolist()
    )

    raise ValueError(
        "ADSL có SUBJID trùng. Không thể merge one-to-one.\n"
        f"SUBJID trùng: {duplicated_rids[:20]}"
    )

# Chỉ lấy những biến cần thiết
adsl_columns = [
    "RID_KEY",
    "ORIGPROT"
] + [
    c for group in GROUPS.values()
    for c in group
    if c != "MRI_AVAILABLE"
]

adsl_columns = list(dict.fromkeys(adsl_columns))

adsl_columns = [
    c for c in adsl_columns
    if c in adsl.columns
]

adsl = adsl[adsl_columns].copy()

# ============================================================
# 6. MERGE COHORT WITH ADSL
# ============================================================

print("\n" + "=" * 60)
print("6. MERGE COHORT WITH ADSL")
print("=" * 60)

# Các biến modality cần lấy từ ADSL
adsl_modality_columns = [
    c for group, columns in GROUPS.items()
    if group != "MRI"
    for c in columns
]

adsl_modality_columns = list(dict.fromkeys(
    adsl_modality_columns
))

# Chỉ lấy những cột modality thực sự có trong ADSL
adsl_modality_columns = [
    c for c in adsl_modality_columns
    if c in adsl.columns
]

# Loại những cột modality trùng tên trong cohort,
# để dữ liệu từ ADSL không bị đổi tên thành hậu tố _ADSL
overlap_columns = [
    c for c in adsl_modality_columns
    if c in cohort.columns
]

print("Các cột modality trùng trong cohort:")
print(overlap_columns)

cohort_merge = cohort.drop(
    columns=overlap_columns,
    errors="ignore"
).copy()

# Merge
bl = cohort_merge.merge(
    adsl,
    on="RID_KEY",
    how="left",
    suffixes=("", "_ADSL"),
    validate="one_to_one",
    indicator=True
)

print("Shape sau merge:", bl.shape)

print("\nKết quả merge:")
print(bl["_merge"].value_counts())

# Xuất RID không nối được
unmatched_adsl = bl.loc[
    bl["_merge"] == "left_only",
    ["RID_KEY", "ORIGPROT"]
].copy()

unmatched_adsl.to_csv(
    OUTPUT_DIR / "unmatched_adsl.csv",
    index=False
)

print(
    "Số RID không nối được ADSL:",
    len(unmatched_adsl)
)

bl.drop(columns="_merge", inplace=True)

# ============================================================
# 7. LOAD UCSFFSX7 - MRI
# ============================================================

print("\n" + "=" * 60)
print("7. LOAD UCSFFSX7")
print("=" * 60)

mri = pd.read_csv(MRI_PATH, low_memory=False)

print("Shape UCSFFSX7:", mri.shape)

check_required_columns(
    mri,
    ["RID", "ORIGPROT"],
    "UCSFFSX7"
)

# Chuẩn hóa RID
mri["RID_KEY"] = make_rid_key(mri["RID"])

# Chuẩn hóa protocol
mri["ORIGPROT"] = clean_string(mri["ORIGPROT"])

# Loại TEAM
mri = mri[
    mri["ORIGPROT"].str.upper() != "TEAM"
].copy()

# ============================================================
# 8. IDENTIFY BASELINE MRI
# ============================================================

print("\n" + "=" * 60)
print("8. IDENTIFY BASELINE MRI")
print("=" * 60)

# Kiểm tra các cột visit code
visit_columns = [
    c for c in ["VISCODE2", "VISCODE"]
    if c in mri.columns
]

if not visit_columns:
    raise ValueError(
        "UCSFFSX7 không có VISCODE2 hoặc VISCODE."
    )

# Chuẩn hóa mã lần khám
for c in visit_columns:
    mri[c] = clean_string(mri[c])

# Xác định baseline:
# ưu tiên VISCODE2 nếu có.
# nếu VISCODE2 thiếu ở một dòng thì dùng VISCODE.
if "VISCODE2" in mri.columns:

    baseline_mask = mri["VISCODE2"].str.upper().eq("BL")

    if "VISCODE" in mri.columns:

        baseline_mask = baseline_mask | (
            mri["VISCODE2"].isna()
            & mri["VISCODE"].str.upper().eq("BL")
        )

else:

    baseline_mask = mri["VISCODE"].str.upper().eq("BL")


mri_bl = mri.loc[baseline_mask].copy()

print("Số dòng MRI baseline:", len(mri_bl))

# ============================================================
# 9. IDENTIFY MRI RECORD AVAILABILITY
# ============================================================

# Chỉ xác định RID có bản ghi MRI baseline.
# Nếu IMAGEUID tồn tại, yêu cầu IMAGEUID không thiếu.
# Không áp dụng OVERALLQC ở thống kê chính này.

if "IMAGEUID" in mri_bl.columns:

    mri_bl = mri_bl.dropna(subset=["IMAGEUID"]).copy()

# RID có MRI baseline
mri_rids = (
    mri_bl[["RID_KEY"]]
    .dropna()
    .drop_duplicates()
)

mri_rids["MRI_AVAILABLE"] = 1

print(
    "Số RID có MRI baseline trong UCSFFSX7:",
    mri_rids["RID_KEY"].nunique()
)

# Xuất danh sách MRI baseline
mri_bl.to_csv(
    OUTPUT_DIR / "mri_baseline_records.csv",
    index=False
)

# ============================================================
# 10. MERGE MRI AVAILABILITY INTO COHORT
# ============================================================

print("\n" + "=" * 60)
print("10. MERGE MRI AVAILABILITY")
print("=" * 60)

bl = bl.merge(
    mri_rids,
    on="RID_KEY",
    how="left",
    validate="one_to_one"
)

bl["MRI_AVAILABLE"] = (
    bl["MRI_AVAILABLE"]
    .fillna(0)
    .astype(int)
)

print("Số RID có MRI baseline trong cohort:")

print(
    bl.groupby("ORIGPROT")["MRI_AVAILABLE"]
    .sum()
)

# ============================================================
# 11. DEFINE GROUPS AVAILABLE IN THE FINAL TABLE
# ============================================================

# Kiểm tra nhóm biến ADSL có tồn tại
groups_final = {}

for group_name, columns in GROUPS.items():

    if group_name == "MRI":

        groups_final[group_name] = [
            "MRI_AVAILABLE"
        ]

    else:

        available_columns = [
            c for c in columns if c in bl.columns
        ]

        if not available_columns:

            raise ValueError(
                f"Không tìm thấy biến nào của nhóm "
                f"{group_name}."
            )

        groups_final[group_name] = available_columns


print("\nCác nhóm biến sử dụng:")

for group_name, columns in groups_final.items():

    print(f"\n{group_name}:")
    print(columns)

# ============================================================
# 12. AVAILABILITY BY PHASE
# ============================================================

def availability_by_phase(bl):
    out = {}

    for g, cols in GROUPS.items():
        cols = [c for c in cols if c in bl.columns]

        if g == "MRI":
            out[g] = (
                bl["MRI_AVAILABLE"].eq(1)
                .groupby(bl["ORIGPROT"])
                .mean()
            )
        else:
            out[g] = (
                bl[cols].notna().all(axis=1)
                .groupby(bl["ORIGPROT"])
                .mean()
            )

    out["n"] = bl.groupby("ORIGPROT").size()

    return pd.DataFrame(out).round(3)


availability = availability_by_phase(bl)

print("\n" + "=" * 60)
print("12. AVAILABILITY BY PHASE")
print("=" * 60)

print(availability)

availability.to_csv(
    OUTPUT_DIR / "availability_by_phase.csv"
)

# ============================================================
# 13. MISSING PATTERNS
# ============================================================

def missing_patterns(bl: pd.DataFrame, top: int = 10) -> pd.Series:

    """
    Các missing patterns phổ biến.

    1 = đủ toàn bộ biến trong nhóm
    0 = thiếu ít nhất một biến trong nhóm
    """

    flags = group_complete_flags(
        bl,
        groups_final
    )

    # Chuỗi 0/1 cho từng RID
    patterns = (
        flags.astype(str)
        .agg("".join, axis=1)
    )

    return patterns.value_counts().head(top)


patterns = missing_patterns(bl, top=20)

print("\n" + "=" * 60)
print("13. MISSING PATTERNS")
print("=" * 60)

print(patterns)

patterns.to_csv(
    OUTPUT_DIR / "missing_patterns.csv",
    header=["count"]
)

# ============================================================
# 14. MISSING PATTERNS BY PHASE
# ============================================================

flags = group_complete_flags(
    bl,
    groups_final
)

# Thêm ORIGPROT để phân nhóm
flags["ORIGPROT"] = bl["ORIGPROT"].values

patterns_by_phase = (
    flags.groupby("ORIGPROT")
    .apply(
        lambda x: (
            x.astype(str)
            .agg("".join, axis=1)
            .value_counts()
        ),
        include_groups=False
    )
)

print("\n" + "=" * 60)
print("14. MISSING PATTERNS BY PHASE")
print("=" * 60)

print(patterns_by_phase)

patterns_by_phase.to_csv(
    OUTPUT_DIR / "missing_patterns_by_phase.csv"
)

# ============================================================
# 15. PER-VARIABLE AVAILABILITY
# ============================================================

print("\n" + "=" * 60)
print("15. PER-VARIABLE AVAILABILITY")
print("=" * 60)

variable_rows = []

for group_name, columns in groups_final.items():

    for column in columns:

        if column not in bl.columns:
            continue

        for phase, sub in bl.groupby("ORIGPROT"):

            n_total = len(sub)

            n_available = sub[column].notna().sum()

            n_missing = n_total - n_available

            percentage = (
                n_available / n_total * 100
                if n_total > 0
                else np.nan
            )

            variable_rows.append({

                "ORIGPROT": phase,

                "Group": group_name,

                "Variable": column,

                "N_cohort": n_total,

                "N_available": n_available,

                "N_missing": n_missing,

                "Availability_percent": round(
                    percentage, 2
                )

            })


variable_availability = pd.DataFrame(
    variable_rows
)

print(variable_availability)

variable_availability.to_csv(
    OUTPUT_DIR / "per_variable_availability.csv",
    index=False
)

# ============================================================
# 16. EXPORT FINAL BASELINE TABLE
# ============================================================

bl.to_csv(
    OUTPUT_DIR / "mci_baseline_with_modalities.csv",
    index=False
)

# ============================================================
# 17. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("17. SUMMARY")
print("=" * 60)

print("Số RID:", bl["RID_KEY"].nunique())

print("\nSố RID theo ORIGPROT:")

print(
    bl["ORIGPROT"]
    .value_counts(dropna=False)
)

print("\nAvailability by phase:")

print(availability)

print("\nMissing patterns:")

print(patterns)

print("\nĐã hoàn thành.")

print(
    "Các file kết quả được lưu tại:",
    OUTPUT_DIR
)