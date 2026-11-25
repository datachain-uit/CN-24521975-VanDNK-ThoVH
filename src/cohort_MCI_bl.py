# ============================================================
# COHORT MCI BASELINE
# ============================================================
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA_DIR, OUT_DIR

# ============================================================
# 1. CAU HINH VA DOC DU LIEU
# ============================================================

DATA_PATH = DATA_DIR / "DXSUM.csv"

HORIZONS = [24, 36, 48]
FOLLOWUP_WINDOW_MONTHS = 3

AD_DUE_VALUE = "DEMENTIA DUE TO ALZHEIMER'S DISEASE"

df = pd.read_csv(DATA_PATH, low_memory=False)

required_columns = [
    "RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS",
    "ORIGPROT", "DXDDUE", "DXAD"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Thieu cac cot bat buoc: {missing_columns}")


# ============================================================
# 2. CAC HAM CHUAN HOA
# ============================================================

def clean_text(series):
    """Chuan hoa chuoi: bo khoang trang, viet hoa, chuan hoa dau nhay."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace("\u2019", "'", regex=False)
    )


def clean_rid(value):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def visit_month(value):
    """
    Lay thang danh nghia tu VISCODE2:
    bl -> 0; m36 -> 36; cac ma khong theo dang nay -> NaN.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value == "bl":
        return 0

    match = re.fullmatch(r"m(\d+)", value)
    if match:
        return int(match.group(1))

    return np.nan


def eligible_for_horizon(baseline_date, data_cutoff, horizon,
                         window=FOLLOWUP_WINDOW_MONTHS):
    """
    Chi du dieu kien neu baseline + (horizon + window) thang
    khong vuot qua ngay chot du lieu.
    """
    if pd.isna(baseline_date) or pd.isna(data_cutoff):
        return False

    latest_eligible_baseline = data_cutoff - pd.DateOffset(
        months=horizon + window
    )
    return baseline_date <= latest_eligible_baseline


def label_at_horizon(fu, horizon):
    """
    fu: cac lan kham sau baseline cua mot RID, da co:
    M_NOM, DIAGNOSIS_CLEAN, DXDDUE_CLEAN, DXAD_CLEAN.

    Chi xet su kien trong thang danh nghia <= horizon.
    """
    win = fu[fu["M_NOM"].notna() & (fu["M_NOM"] <= horizon)].copy()

    dementia = win[
        win["DIAGNOSIS_CLEAN"].eq("DEMENTIA")
    ].copy()

    condition_dxddue = (
        dementia["DXDDUE_CLEAN"].eq(AD_DUE_VALUE).fillna(False)
    )
    condition_dxad = (
        dementia["DXAD_CLEAN"].eq("YES").fillna(False)
    )

    ad_dementia = dementia[condition_dxddue | condition_dxad]

    if not ad_dementia.empty:
        return "pMCI"

    if not dementia.empty:
        return "Bị loại - Dementia không thỏa tiêu chí AD"

    # Dung thang danh nghia, khong so sanh EXAMDATE voi moc tron.
    observed_months = fu.loc[fu["M_NOM"].notna(), "M_NOM"]
    if not observed_months.empty and observed_months.max() >= horizon:
        return "sMCI"

    return "Bị loại - Mất theo dõi"


def analyze_diagnosis_sequence(states):
    """Dem so lan doi nhan va phat hien mot nhan quay lai sau nhan khac."""
    compressed = []
    for state in states:
        if not compressed or state != compressed[-1]:
            compressed.append(state)

    number_of_changes = max(len(compressed) - 1, 0)
    is_fluctuating = (
        len(compressed) >= 3
        and len(set(compressed)) < len(compressed)
    )
    return number_of_changes, is_fluctuating, compressed


# ============================================================
# 3. CHUAN HOA DU LIEU VA NGAY CHOT
# ============================================================

df["RID_KEY"] = df["RID"].apply(clean_rid)
df["VISCODE2_CLEAN"] = clean_text(df["VISCODE2"])
df["DIAGNOSIS_CLEAN"] = clean_text(df["DIAGNOSIS"])
df["ORIGPROT_CLEAN"] = clean_text(df["ORIGPROT"])
df["DXDDUE_CLEAN"] = clean_text(df["DXDDUE"])
df["DXAD_CLEAN"] = clean_text(df["DXAD"])

df["ORIGPROT_CLEAN"] = df["ORIGPROT_CLEAN"].fillna("KHONG RO")
df["EXAMDATE_DT"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

# Theo QD21: thoi gian duoc tinh bang thang danh nghia tu VISCODE2.
df["M_NOM"] = df["VISCODE2"].map(visit_month)

# Ngay chot du lieu la EXAMDATE lon nhat trong DXSUM.
data_cutoff = df["EXAMDATE_DT"].max()
if pd.isna(data_cutoff):
    raise ValueError("Khong xac dinh duoc ngay chot du lieu tu EXAMDATE.")

print("Ngay chot du lieu (EXAMDATE lon nhat):", data_cutoff.date())

# Dem cac lan kham co VISCODE2 khong doc duoc thang danh nghia.
unreadable_visits = df[df["M_NOM"].isna()].copy()
print("So lan kham co VISCODE2 khong doc duoc thang:", len(unreadable_visits))


# ============================================================
# 4. XAC DINH COHORT MCI TAI BASELINE
# ============================================================

baseline_mci = df[
    df["VISCODE2_CLEAN"].eq("BL")
    & df["DIAGNOSIS_CLEAN"].eq("MCI")
    & df["RID_KEY"].notna()
    & df["ORIGPROT_CLEAN"].ne("TEAM")
].copy()

# Kiem tra RID trung tai baseline; khong tu dong xoa dong.
duplicated_baseline = baseline_mci[
    baseline_mci.duplicated("RID_KEY", keep=False)
]

if not duplicated_baseline.empty:
    duplicate_path = OUT_DIR / "mci_baseline_duplicate_RID.csv"
    duplicated_baseline.to_csv(
        duplicate_path, index=False, encoding="utf-8-sig"
    )
    raise ValueError(
        "Co RID bi trung tai baseline. "
        f"Hay kiem tra: {duplicate_path}"
    )

cohort = baseline_mci[
    ["RID_KEY", "EXAMDATE_DT", "ORIGPROT_CLEAN"]
].copy()

cohort = cohort.rename(columns={
    "EXAMDATE_DT": "BASELINE_DATE",
    "ORIGPROT_CLEAN": "ORIGPROT"
})
cohort["RID"] = cohort["RID_KEY"]

if cohort["BASELINE_DATE"].isna().any():
    print(
        "CANH BAO:",
        int(cohort["BASELINE_DATE"].isna().sum()),
        "nguoi co EXAMDATE baseline bi thieu; se khong du dieu kien horizon."
    )

print("So nguoi MCI baseline sau khi loai ORIGPROT=TEAM:",
      cohort["RID"].nunique())


# ============================================================
# 5. LAY CAC LAN KHAM FOLLOW-UP THEO THANG DANH NGHIA
# ============================================================

followup = df.merge(
    cohort[["RID_KEY", "BASELINE_DATE"]],
    on="RID_KEY",
    how="inner",
    validate="many_to_one"
)

# Theo huong dan: dung M_NOM > 0, khong so sanh EXAMDATE > BASELINE_DATE.
# Chi giu cac nhan chan doan hop le de xac dinh su kien/theo doi.
followup = followup[
    followup["M_NOM"].notna()
    & followup["M_NOM"].gt(0)
    & followup["DIAGNOSIS_CLEAN"].isin(["CN", "MCI", "DEMENTIA"])
].copy()

followup = followup.sort_values(["RID_KEY", "M_NOM", "EXAMDATE_DT"])

followup_by_rid = {
    rid: group.reset_index(drop=True)
    for rid, group in followup.groupby("RID_KEY")
}


# ============================================================
# 6. PHAN LOAI pMCI / sMCI THEO HORIZON
# ============================================================

result_rows = []

for person in cohort.itertuples(index=False):
    rid = person.RID_KEY
    baseline_date = person.BASELINE_DATE
    phase = person.ORIGPROT

    person_followup = followup_by_rid.get(
        rid, pd.DataFrame(columns=followup.columns)
    )

    if not person_followup.empty:
        last_followup_month = person_followup["M_NOM"].max()
        last_followup_date = person_followup["EXAMDATE_DT"].max()
    else:
        last_followup_month = np.nan
        last_followup_date = pd.NaT

    for horizon in HORIZONS:
        horizon_date = (
            baseline_date + pd.DateOffset(months=horizon)
            if pd.notna(baseline_date) else pd.NaT
        )

        # QD21: loc dieu kien du thoi gian truoc khi phan loai.
        is_eligible = eligible_for_horizon(
            baseline_date,
            data_cutoff,
            horizon,
            FOLLOWUP_WINDOW_MONTHS
        )

        # Cua so su kien va cac thong ke tai horizon dung thang danh nghia.
        within_horizon = person_followup[
            person_followup["M_NOM"].notna()
            & (person_followup["M_NOM"] <= horizon)
        ].copy()

        dementia_records = within_horizon[
            within_horizon["DIAGNOSIS_CLEAN"].eq("DEMENTIA")
        ].copy()

        condition_dxddue = (
            dementia_records["DXDDUE_CLEAN"]
            .eq(AD_DUE_VALUE).fillna(False)
        )
        condition_dxad = (
            dementia_records["DXAD_CLEAN"]
            .eq("YES").fillna(False)
        )

        ad_dementia = dementia_records[
            condition_dxddue | condition_dxad
        ].copy()

        non_ad_criteria_dementia = dementia_records[
            ~(condition_dxddue | condition_dxad)
        ].copy()

        has_dementia = not dementia_records.empty
        has_ad_dementia = not ad_dementia.empty

        first_dementia_date = (
            dementia_records["EXAMDATE_DT"].min()
            if has_dementia else pd.NaT
        )

        first_ad_dementia_date = (
            ad_dementia["EXAMDATE_DT"].min()
            if has_ad_dementia else pd.NaT
        )

        # Thu tu uu tien:
        # 1) Chua the du thoi gian theo doi -> loai truoc.
        # 2) Neu du dieu kien: ap dung nhan tai horizon.
        if not is_eligible:
            outcome = "Bị loại - Chưa thể đủ thời gian"
            ad_evidence = "Không xét do chưa đủ điều kiện horizon"

        else:
            outcome = label_at_horizon(person_followup, horizon)

            # Luu bang chung tu ban ghi Dementia do AD dau tien.
            if has_ad_dementia:
                first_ad_record = (
                    ad_dementia.sort_values(
                        ["M_NOM", "EXAMDATE_DT"]
                    ).iloc[0]
                )

                evidence = []
                if (
                    pd.notna(first_ad_record["DXDDUE_CLEAN"])
                    and first_ad_record["DXDDUE_CLEAN"] == AD_DUE_VALUE
                ):
                    evidence.append("DXDDUE")

                if (
                    pd.notna(first_ad_record["DXAD_CLEAN"])
                    and first_ad_record["DXAD_CLEAN"] == "YES"
                ):
                    evidence.append("DXAD")

                ad_evidence = " + ".join(evidence)
            else:
                ad_evidence = "Không có"

        # Ngay Dementia va bang chung chi co y nghia trong horizon.
        cn_records = within_horizon[
            within_horizon["DIAGNOSIS_CLEAN"].eq("CN")
        ]
        has_reverted_to_cn = not cn_records.empty
        first_cn_date = (
            cn_records["EXAMDATE_DT"].min()
            if has_reverted_to_cn else pd.NaT
        )

        # Chan doan dao dong: giu logic cu, nhung loc theo M_NOM <= horizon.
        daily_diagnosis = (
            within_horizon
            .groupby("EXAMDATE_DT")["DIAGNOSIS_CLEAN"]
            .agg(lambda x: sorted(set(x.dropna())))
            .reset_index(name="DIAGNOSES")
        )

        # Chi su dung moc thang co mot nhan chan doan duy nhat.
        daily_diagnosis = daily_diagnosis[
            daily_diagnosis["DIAGNOSES"].apply(len).eq(1)
        ].copy()

        states = ["MCI"] + [
            diagnoses[0]
            for diagnoses in daily_diagnosis["DIAGNOSES"]
        ]

        changes, is_fluctuating, compressed_states = (
            analyze_diagnosis_sequence(states)
        )

        diagnostic_sequence = " -> ".join(states)

        result_rows.append({
            "RID": rid,
            "ORIGPROT": phase,
            "Horizon_months": horizon,
            "Baseline_date": baseline_date,
            "Horizon_date": horizon_date,
            "Data_cutoff": data_cutoff,
            "Eligible_for_horizon": is_eligible,
            "Outcome": outcome,
            "First_Dementia_date": first_dementia_date,
            "First_AD_Dementia_date": first_ad_dementia_date,
            "AD_evidence": ad_evidence,
            "Has_AD_Dementia": has_ad_dementia,
            "Number_of_Dementia_records": len(dementia_records),
            "Number_of_Dementia_not_meeting_AD_criteria":
                len(non_ad_criteria_dementia),
            "Last_followup_month_nominal": last_followup_month,
            "Last_followup_date": last_followup_date,
            "Reverted_to_CN": has_reverted_to_cn,
            "First_CN_date": first_cn_date,
            "Number_of_diagnosis_changes": changes,
            "Diagnosis_fluctuated": is_fluctuating,
            "Diagnostic_sequence": diagnostic_sequence,
            "Compressed_sequence": " -> ".join(compressed_states),
            "Number_of_followup_assessments": len(within_horizon)
        })

result = pd.DataFrame(result_rows)


# ============================================================
# 7. TONG HOP THEO HORIZON VA GIAI DOAN TUYEN
# ============================================================

def summarize_group(group):
    total_baseline = group["RID"].nunique()
    eligible = group.loc[
        group["Eligible_for_horizon"], "RID"
    ].nunique()

    pmci = group["Outcome"].eq("pMCI").sum()
    smci = group["Outcome"].eq("sMCI").sum()
    dementia_not_ad = group["Outcome"].eq(
        "Bị loại - Dementia không thỏa tiêu chí AD"
    ).sum()
    too_early = group["Outcome"].eq(
        "Bị loại - Chưa thể đủ thời gian"
    ).sum()
    lost_followup = group["Outcome"].eq(
        "Bị loại - Mất theo dõi"
    ).sum()

    reverted = group["Reverted_to_CN"].sum()
    fluctuated = group["Diagnosis_fluctuated"].sum()
    changed = group["Number_of_diagnosis_changes"].gt(0).sum()

    classified = pmci + smci

    return {
        "So_MCI_baseline": total_baseline,
        "Du_dieu_kien_horizon": eligible,
        "Chua_the_du_thoi_gian": too_early,
        "pMCI": pmci,
        "sMCI": smci,
        "Dementia_khong_thoa_tieu_chi_AD": dementia_not_ad,
        "Mat_theo_doi": lost_followup,
        "Quay_ve_CN": reverted,
        "Co_thay_doi_chan_doan": changed,
        "Chan_doan_dao_dong": fluctuated,
        "Ty_le_pMCI_tren_nhom_du_dieu_kien_pct": (
            round(pmci / eligible * 100, 2) if eligible else 0
        ),
        "Ty_le_pMCI_trong_nhom_pMCI_sMCI_pct": (
            round(pmci / classified * 100, 2) if classified else 0
        )
    }


summary_rows = []

# Tong hop theo ORIGPROT (giai doan tuyen ban dau).
for (horizon, phase), group in result.groupby(
    ["Horizon_months", "ORIGPROT"], dropna=False
):
    row = summarize_group(group)
    row["Horizon_thang"] = horizon
    row["ORIGPROT"] = phase
    summary_rows.append(row)

# Tong hop toan bo cohort theo tung horizon.
for horizon, group in result.groupby("Horizon_months"):
    row = summarize_group(group)
    row["Horizon_thang"] = horizon
    row["ORIGPROT"] = "Tổng"
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary["_total_sort"] = summary["ORIGPROT"].eq("Tổng").astype(int)
summary = summary.sort_values(
    ["Horizon_thang", "_total_sort", "ORIGPROT"]
).drop(columns="_total_sort")

summary = summary[
    [
        "Horizon_thang", "ORIGPROT",
        "So_MCI_baseline", "Du_dieu_kien_horizon",
        "Chua_the_du_thoi_gian", "pMCI", "sMCI",
        "Dementia_khong_thoa_tieu_chi_AD", "Mat_theo_doi",
        "Quay_ve_CN", "Co_thay_doi_chan_doan", "Chan_doan_dao_dong",
        "Ty_le_pMCI_tren_nhom_du_dieu_kien_pct",
        "Ty_le_pMCI_trong_nhom_pMCI_sMCI_pct"
    ]
]


# ============================================================
# 8. KIEM TRA VISCODE2 KHONG DOC DUOC
# ============================================================

unreadable_summary = (
    df[df["M_NOM"].isna()]
    .groupby("ORIGPROT_CLEAN", dropna=False)
    .agg(
        So_lan_kham_khong_doc_duoc_thang=("RID_KEY", "size"),
        So_RID=("RID_KEY", "nunique")
    )
    .reset_index()
    .rename(columns={"ORIGPROT_CLEAN": "ORIGPROT"})
)


# ============================================================
# 9. XUAT KET QUA CSV
# ============================================================

output_dir = OUT_DIR / "mci_cohort_results"
output_dir.mkdir(parents=True, exist_ok=True)

cohort.to_csv(
    output_dir / "mci_cohort_baseline.csv",
    index=False,
    encoding="utf-8-sig"
)

result.to_csv(
    output_dir / "mci_horizon_classification_detail.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    output_dir / "mci_horizon_phase_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

unreadable_summary.to_csv(
    output_dir / "mci_unreadable_VISCODE2_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

unreadable_visits.to_csv(
    output_dir / "mci_unreadable_VISCODE2_records.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. HIEN THI KET QUA
# ============================================================

print("\n" + "=" * 80)
print("COHORT MCI BASELINE (DA LOAI ORIGPROT=TEAM)")
print("=" * 80)
print("So nguoi MCI baseline:", cohort["RID"].nunique())

print("\n" + "=" * 80)
print("PHAN LOAI pMCI / sMCI / CAC TRUONG HOP BI LOAI")
print("=" * 80)
print(summary.to_string(index=False))

print("\n" + "=" * 80)
print("SO LAN KHAM CO VISCODE2 KHONG DOC DUOC THANG")
print("=" * 80)
print("Tong so lan kham:", len(unreadable_visits))
print(unreadable_summary.to_string(index=False))

print("\nNgay chot du lieu:", data_cutoff.date())
print("Da luu ket qua tai:", output_dir)
