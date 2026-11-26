# ============================================================
# MODALITY AVAILABILITY - MCI BASELINE COHORT (ADNI)
# ============================================================

import numpy as np
import pandas as pd

from paths import DATA_DIR, OUT_DIR
from modality_utils import (
    ROI, CORE,
    clean_string, check_required_columns, make_rid_key,
    match_baseline_mri, mri_coverage,
    group_flags, pattern_table, flow_table, lost_vs_labeled,
    adsl_alignment,
)

HORIZON = 36
MRI_WINDOW = 90

# ============================================================
# 1. PATHS
# ============================================================
COH_DIR = OUT_DIR / "mci_cohort_results"
COHORT_PATH = COH_DIR / "mci_cohort_baseline.csv"
DETAIL_PATH = COH_DIR / "mci_horizon_classification_detail.csv"
ADSL_PATH = DATA_DIR / "ADSL.csv"
MRI_PATH = DATA_DIR / "UCSFFSX7.csv"

OUTPUT_DIR = OUT_DIR / "modality_availability_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD MCI BASELINE COHORT
# ============================================================
print("=" * 60)
print("2. LOAD MCI BASELINE COHORT")
print("=" * 60)

cohort = pd.read_csv(COHORT_PATH, low_memory=False,
                     dtype={"RID": str, "RID_KEY": str})
print("Shape cohort:", cohort.shape)

if "RID_KEY" in cohort.columns:
    cohort["RID_KEY"] = make_rid_key(cohort["RID_KEY"])
elif "RID" in cohort.columns:
    cohort["RID_KEY"] = make_rid_key(cohort["RID"])
else:
    raise ValueError("Cohort không có RID_KEY hoặc RID.")

check_required_columns(cohort, ["RID_KEY", "ORIGPROT", "BASELINE_DATE"], "Cohort")
cohort["ORIGPROT"] = clean_string(cohort["ORIGPROT"])
cohort["BASELINE_DATE"] = pd.to_datetime(cohort["BASELINE_DATE"], errors="coerce")
cohort = cohort.dropna(subset=["RID_KEY"]).copy()
cohort = cohort[cohort["ORIGPROT"].str.upper() != "TEAM"].copy()

if cohort["RID_KEY"].duplicated().any():
    dup = cohort.loc[cohort["RID_KEY"].duplicated(keep=False), "RID_KEY"].unique()[:20].tolist()
    raise ValueError(f"Cohort có RID trùng: {dup}")

n_no_date = int(cohort["BASELINE_DATE"].isna().sum())
if n_no_date:
    print(f"CẢNH BÁO: {n_no_date} người không có BASELINE_DATE (sẽ không khớp được MRI)")

print("Số RID trong cohort:", cohort["RID_KEY"].nunique())
print(cohort["ORIGPROT"].value_counts(dropna=False))

# ============================================================
# 3. LOAD ADSL + KIỂM TRA KHỚP VỚI DXSUM (O14)
# ============================================================
print("\n" + "=" * 60)
print("3. LOAD ADSL + ADSL ALIGNMENT")
print("=" * 60)

adsl = pd.read_csv(ADSL_PATH, low_memory=False, dtype={"SUBJID": str})
print("Shape ADSL:", adsl.shape)
check_required_columns(adsl, ["SUBJID", "ORIGPROT"], "ADSL")

adsl = adsl[adsl["ORIGPROT"].astype("string").str.upper() != "TEAM"].copy()
adsl["RID_KEY"] = make_rid_key(adsl["SUBJID"])
adsl["ORIGPROT"] = clean_string(adsl["ORIGPROT"])
adsl = adsl.dropna(subset=["RID_KEY"]).copy()

# Sửa nhánh báo trùng (Mục 7.4)
dup = adsl["RID_KEY"].duplicated(keep=False)
if dup.any():
    ids = adsl.loc[dup, "RID_KEY"].unique()[:20].tolist()
    raise ValueError(f"ADSL có SUBJID trùng: {ids}")

# RID của cohort không có trong ADSL (chỉ lưu trên máy)
unmatched_adsl = cohort.loc[~cohort["RID_KEY"].isin(adsl["RID_KEY"]),
                            ["RID_KEY", "ORIGPROT"]]
unmatched_adsl.to_csv(OUTPUT_DIR / "unmatched_adsl.csv", index=False, encoding="utf-8-sig")
print("Số RID không nối được ADSL:", len(unmatched_adsl))

# adsl_alignment chạy trên ADSL đầy đủ (trước khi giữ cột cần thiết).
# Chỉ truyền RID_KEY, DX, ENRLDT để DX của ADSL không bị đổi tên nếu cohort
# cũng có cột DX.
check_required_columns(adsl, ["DX", "ENRLDT"], "ADSL")
align = adsl_alignment(cohort, adsl[["RID_KEY", "DX", "ENRLDT"]])

# Lưu các bảng đếm cho data card (số liệu tổng hợp)
pd.crosstab(align["ORIGPROT"], align["DX"], dropna=False).to_csv(
    OUTPUT_DIR / "adsl_dx_by_phase.csv", encoding="utf-8-sig")
gap = (pd.to_datetime(align["ENRLDT"], errors="coerce")
       - align["BASELINE_DATE"]).dt.days
gap.groupby(align["ORIGPROT"]).describe(
    percentiles=[.05, .5, .95]).round(0).to_csv(OUTPUT_DIR / "adsl_enrldt_gap_days.csv", encoding="utf-8-sig")
n_off = int(((align["DX"].astype("string").str.upper() != "MCI") | (gap.abs() > 90)).sum())
print(f"Người có DX ADSL khác MCI hoặc lệch ENRLDT > 90 ngày: {n_off} "
      "(báo cáo riêng, chưa loại)")

# ============================================================
# 4. BIẾN BASELINE TỪ ADSL (QĐ25)
# ============================================================
print("\n" + "=" * 60)
print("4. ADSL FEATURES (CORE)")
print("=" * 60)

# APOE4: số alen ε4 từ kiểu gen ("ε3/ε4" -> 1). Kiểm tra định dạng APOE.
check_required_columns(adsl, ["APOE"], "ADSL")
print("Mẫu giá trị APOE:", adsl["APOE"].dropna().astype(str).unique()[:6].tolist())
adsl["APOE4"] = adsl["APOE"].astype("string").str.count("4")

adsl_vars = [c for g, cols in CORE.items() if g != "mri" for c in cols]
check_required_columns(adsl, [c for c in adsl_vars if c != "APOE4"], "ADSL (nhóm lõi)")
adsl_keep = ["RID_KEY", "ORIGPROT"] + list(dict.fromkeys(adsl_vars))
adsl = adsl[adsl_keep].copy()

# ============================================================
# 5. MRI BASELINE: KHỚP THEO CỬA SỔ NGÀY (R5, QĐ24)
# ============================================================
print("\n" + "=" * 60)
print("5. MRI BASELINE (UCSFFSX7)")
print("=" * 60)

mri = pd.read_csv(MRI_PATH, dtype=str)
print("Shape UCSFFSX7:", mri.shape)
check_required_columns(
    mri, ["RID", "IMAGEUID", "EXAMDATE", "OVERALLQC", "FIELD_STRENGTH",
          "STATUS", "VISCODE2"] + list(ROI), "UCSFFSX7")
mri["RID_KEY"] = make_rid_key(mri["RID"])

matched = match_baseline_mri(cohort, mri, window=MRI_WINDOW)
print("Số RID khớp được MRI baseline:", matched["RID_KEY"].nunique())

# Chẩn đoán: ảnh 'Hippocampus Only' có ICV không?
hip_only = matched["OVERALLQC"].astype("string").str.upper().eq("HIPPOCAMPUS ONLY")
print("Ảnh Hippocampus Only:", int(hip_only.sum()),
      "| trong đó thiếu MRI_ICV:", int((hip_only & matched["MRI_ICV"].isna()).sum()))
print("Ảnh có HIPPO đủ nhưng thiếu MRI_ICV (sẽ bị tính là 'thiếu' ở nhóm mri):",
      int((matched["MRI_HIPPO_L"].notna() & matched["MRI_HIPPO_R"].notna()
           & matched["MRI_ICV"].isna()).sum()))

matched.to_csv(OUTPUT_DIR / "mri_baseline_records.csv", index=False, encoding="utf-8-sig")  # chỉ trên máy

feats = adsl.merge(matched, on="RID_KEY", how="left", validate="one_to_one")

# ============================================================
# 6. BẢNG BASELINE ĐẦY ĐỦ (chỉ trên máy)
# ============================================================
overlap = [c for c in feats.columns
           if c in cohort.columns and c not in ("RID_KEY", "ORIGPROT")]
bl = cohort.drop(columns=overlap, errors="ignore").merge(
    feats.drop(columns="ORIGPROT"), on="RID_KEY", how="left", validate="one_to_one")
bl.to_csv(OUTPUT_DIR / "mci_baseline_with_modalities.csv", index=False, encoding="utf-8-sig")
print("\nShape bảng baseline:", bl.shape)

# ============================================================
# 7. COHORT PHÂN TÍCH Ở H = 36 (QĐ26, QĐ27)
# ============================================================
print("\n" + "=" * 60)
print(f"7. COHORT PHÂN TÍCH H = {HORIZON}")
print("=" * 60)

res = pd.read_csv(DETAIL_PATH, dtype={"RID": str})
res36 = res[res["Horizon_months"] == HORIZON].rename(columns={"RID": "RID_KEY"}).copy()
res36["RID_KEY"] = make_rid_key(res36["RID_KEY"])

flow = flow_table(res36, feats)
print(flow)
flow.to_csv(OUTPUT_DIR / "flow_table.csv", encoding="utf-8-sig")

lab = res36[res36["Outcome"].isin(["pMCI", "sMCI"])]
lab = lab.merge(feats.drop(columns="ORIGPROT"), on="RID_KEY", how="left")
lab["SPLIT"] = np.where(lab["ORIGPROT"].eq("ADNI1"), "train", "test")
print("\nSố người có nhãn:", len(lab))
print(lab["SPLIT"].value_counts())

# 7.1 Mức sẵn có theo nhóm lõi (theo giai đoạn và theo train/test)
def availability_table(df, by):
    f = group_flags(df)
    f[by] = df[by].values
    out = f.groupby(by).mean().round(3)
    out["n"] = f.groupby(by).size()
    return out

avail_phase = availability_table(lab, "ORIGPROT")
avail_split = availability_table(lab, "SPLIT")
print("\nMức sẵn có theo giai đoạn:\n", avail_phase)
print("\nMức sẵn có theo train/test:\n", avail_split)
avail_phase.to_csv(OUTPUT_DIR / "availability_by_phase.csv", encoding="utf-8-sig")
avail_split.to_csv(OUTPUT_DIR / "availability_by_split.csv", encoding="utf-8-sig")

# 7.2 Mức sẵn có theo từng biến (nhóm mri gồm đủ 7 ROI)
var_groups = dict(CORE)
var_groups["mri"] = ["MRI_" + v for v in ROI.values()]

def per_variable_availability(df, by, groups):
    rows = []
    for g, cols in groups.items():
        for c in cols:
            for key, sub in df.groupby(by):
                n, a = len(sub), int(sub[c].notna().sum())
                rows.append({by: key, "Group": g, "Variable": c, "N": n,
                             "N_available": a, "N_missing": n - a,
                             "Availability_percent": round(a / n * 100, 2)})
    return pd.DataFrame(rows)

per_variable_availability(lab, "ORIGPROT", var_groups).to_csv(
    OUTPUT_DIR / "per_variable_availability_by_phase.csv", index=False, encoding="utf-8-sig")
per_variable_availability(lab, "SPLIT", var_groups).to_csv(
    OUTPUT_DIR / "per_variable_availability_by_split.csv", index=False, encoding="utf-8-sig")

# 7.3 Mẫu khuyết dạng cột (không mất số 0 đầu)
pat_phase = pattern_table(lab)
pat_split = pattern_table(lab, by="SPLIT")
print("\nMẫu khuyết theo giai đoạn:\n", pat_phase)
pat_phase.to_csv(OUTPUT_DIR / "missing_patterns_by_phase.csv", index=False, encoding="utf-8-sig")
pat_split.to_csv(OUTPUT_DIR / "missing_patterns_by_split.csv", index=False, encoding="utf-8-sig")

# 7.4 Bảng MRI của QĐ24 (cohort đầy đủ và cohort phân tích)
cohort_lab = cohort[cohort["RID_KEY"].isin(lab["RID_KEY"])]
for tag, coh in [("all_mci", cohort), ("analysis_h36", cohort_lab)]:
    cov, fs, st, vc = mri_coverage(coh, matched)
    print(f"\nMRI coverage ({tag}):\n", cov)
    cov.to_csv(OUTPUT_DIR / f"mri_coverage_{tag}.csv", encoding="utf-8-sig")
    fs.to_csv(OUTPUT_DIR / f"mri_field_strength_{tag}.csv", encoding="utf-8-sig")
    st.to_csv(OUTPUT_DIR / f"mri_status_{tag}.csv", encoding="utf-8-sig")
    vc.to_csv(OUTPUT_DIR / f"mri_viscode2_{tag}.csv", encoding="utf-8-sig")

# 7.5 Nhóm mất theo dõi so với nhóm có nhãn (O16)
smd = lost_vs_labeled(res36, feats)
print("\nSMD (mất theo dõi vs có nhãn):\n", smd)
smd.to_csv(OUTPUT_DIR / "smd_lost_vs_labeled.csv", index=False, encoding="utf-8-sig")
print("Các dòng |SMD| >= 0,2:")
print(smd[smd["SMD"].abs() >= 0.2])

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)
print("Số RID MCI baseline:", bl["RID_KEY"].nunique())
print("Số người có nhãn H=36:", len(lab))
print("Đã hoàn thành. Kết quả tại:", OUTPUT_DIR)
