
import pandas as pd
from pathlib import Path


# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================

path = r"C:\Users\24521718\Documents\KLTN\data\selected\DXSUM.csv"

df = pd.read_csv(path, low_memory=False)

required_columns = [
    "RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS",
    "ORIGPROT", "DXDDUE", "DXAD"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Thiếu các cột: {missing_columns}")


# ============================================================
# 2. CHUẨN HÓA DỮ LIỆU
# ============================================================

def clean_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace("’", "'", regex=False)
    )


def clean_rid(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


df["RID_KEY"] = df["RID"].apply(clean_rid)

df["VISCODE2_CLEAN"] = clean_text(df["VISCODE2"])
df["DIAGNOSIS_CLEAN"] = clean_text(df["DIAGNOSIS"])
df["ORIGPROT_CLEAN"] = clean_text(df["ORIGPROT"])

df["ORIGPROT_CLEAN"] = (
    df["ORIGPROT_CLEAN"].fillna("KHÔNG RÕ")
)

df["DXDDUE_CLEAN"] = clean_text(df["DXDDUE"])
df["DXAD_CLEAN"] = clean_text(df["DXAD"])

df["EXAMDATE_DT"] = pd.to_datetime(
    df["EXAMDATE"],
    errors="coerce"
)


# Giá trị DXDDUE được xác định là Dementia do Alzheimer
AD_DUE_VALUE = "DEMENTIA DUE TO ALZHEIMER'S DISEASE"

print("Các giá trị DXDDUE:")
print(df["DXDDUE_CLEAN"].value_counts(dropna=False))

print("\nCác giá trị DXAD:")
print(df["DXAD_CLEAN"].value_counts(dropna=False))


# ============================================================
# 3. XÁC ĐỊNH COHORT MCI TẠI BASELINE
# ============================================================

baseline_mci = df[
    (df["VISCODE2_CLEAN"] == "BL") &
    (df["DIAGNOSIS_CLEAN"] == "MCI") &
    (df["RID_KEY"].notna())
].copy()

# Kiểm tra RID trùng tại baseline
duplicated_baseline = baseline_mci[
    baseline_mci.duplicated("RID_KEY", keep=False)
]

if not duplicated_baseline.empty:

    duplicated_baseline.to_csv(
        "mci_baseline_duplicate_RID.csv",
        index=False,
        encoding="utf-8-sig"
    )

    raise ValueError(
        "Có RID bị trùng tại baseline. "
        "Hãy kiểm tra mci_baseline_duplicate_RID.csv."
    )

# Mỗi người chỉ có một dòng trong cohort
cohort = baseline_mci[
    ["RID_KEY", "EXAMDATE_DT", "ORIGPROT_CLEAN"]
].copy()

cohort = cohort.rename(columns={
    "EXAMDATE_DT": "BASELINE_DATE",
    "ORIGPROT_CLEAN": "ORIGPROT"
})

cohort["RID"] = cohort["RID_KEY"]

print("\nSố người MCI baseline:", cohort["RID"].nunique())


# ============================================================
# 4. LẤY DỮ LIỆU FOLLOW-UP
# ============================================================

followup = df.merge(
    cohort[["RID_KEY", "BASELINE_DATE"]],
    on="RID_KEY",
    how="inner"
)

# Chỉ lấy các lần đánh giá sau baseline
# Chỉ xét các chẩn đoán CN, MCI và Dementia

followup = followup[
    followup["EXAMDATE_DT"].notna() &
    followup["BASELINE_DATE"].notna() &
    (followup["EXAMDATE_DT"] > followup["BASELINE_DATE"]) &
    followup["DIAGNOSIS_CLEAN"].isin(
        ["CN", "MCI", "DEMENTIA"]
    )
].copy()

# Sắp xếp theo RID và ngày đánh giá
followup = followup.sort_values(
    ["RID_KEY", "EXAMDATE_DT"]
)

# Lưu tất cả các bản ghi theo RID
# Không gộp các dòng Dementia vì cần kiểm tra DXDDUE và DXAD
followup_by_rid = {
    rid: group.reset_index(drop=True)
    for rid, group in followup.groupby("RID_KEY")
}


# ============================================================
# 5. HÀM KIỂM TRA CHẨN ĐOÁN DAO ĐỘNG
# ============================================================

def analyze_diagnosis_sequence(states):

    # Loại bỏ các nhãn liên tiếp giống nhau
    compressed = []

    for state in states:
        if not compressed or state != compressed[-1]:
            compressed.append(state)

    # Số lần chuyển nhóm chẩn đoán
    number_of_changes = max(len(compressed) - 1, 0)

    # Có dao động nếu một nhãn xuất hiện lại
    is_fluctuating = (
        len(compressed) >= 3 and
        len(set(compressed)) < len(compressed)
    )

    return number_of_changes, is_fluctuating, compressed


# ============================================================
# 6. PHÂN LOẠI pMCI / sMCI THEO HORIZON
# ============================================================

horizons = [24, 36, 48]

result_rows = []

for person in cohort.itertuples(index=False):

    rid = person.RID_KEY
    baseline_date = person.BASELINE_DATE
    phase = person.ORIGPROT

    person_followup = followup_by_rid.get(
        rid,
        pd.DataFrame(columns=followup.columns)
    )

    # Ngày đánh giá cuối cùng
    if not person_followup.empty:
        last_followup_date = (
            person_followup["EXAMDATE_DT"].max()
        )
    else:
        last_followup_date = pd.NaT

    for horizon in horizons:

        # ----------------------------------------------------
        # 6.1. Xác định ngày kết thúc horizon
        # ----------------------------------------------------

        if pd.notna(baseline_date):
            horizon_date = (
                baseline_date + pd.DateOffset(months=horizon)
            )
        else:
            horizon_date = pd.NaT

        # Các bản ghi trong khoảng theo dõi
        if pd.notna(horizon_date):

            within_horizon = person_followup[
                person_followup["EXAMDATE_DT"] <= horizon_date
            ].copy()

        else:
            within_horizon = person_followup.iloc[0:0].copy()

        # ----------------------------------------------------
        # 6.2. Lấy các bản ghi Dementia
        # ----------------------------------------------------

        dementia_records = within_horizon[
            within_horizon["DIAGNOSIS_CLEAN"] == "DEMENTIA"
        ].copy()

        # ----------------------------------------------------
        # 6.3. XÁC ĐỊNH DEMENTIA DO ALZHEIMER
        # ----------------------------------------------------

        # Điều kiện 1:
        # DXDDUE = Dementia due to Alzheimer's Disease

        condition_dxddue = (
            dementia_records["DXDDUE_CLEAN"] == AD_DUE_VALUE
        )

        # Điều kiện 2:
        # DXAD = Yes

        condition_dxad = (
            dementia_records["DXAD_CLEAN"] == "YES"
        )

        # Chỉ cần thỏa mãn một trong hai điều kiện
        ad_dementia = dementia_records[
            condition_dxddue | condition_dxad
        ].copy()

        # Các bản ghi Dementia không thỏa tiêu chí AD
        non_ad_criteria_dementia = dementia_records[
            ~(condition_dxddue | condition_dxad)
        ].copy()

        has_dementia = not dementia_records.empty

        has_ad_dementia = not ad_dementia.empty

        has_non_ad_criteria_dementia = (
            not non_ad_criteria_dementia.empty
        )

        # ----------------------------------------------------
        # 6.4. XÁC ĐỊNH pMCI / sMCI / BỊ LOẠI
        # ----------------------------------------------------

        first_dementia_date = (
            dementia_records["EXAMDATE_DT"].min()
            if has_dementia else pd.NaT
        )
        
        if has_ad_dementia:

            # Có ít nhất một lần chuyển sang Dementia
            # thỏa mãn điều kiện AD
            outcome = "pMCI"

            first_ad_dementia_date = (
                ad_dementia["EXAMDATE_DT"].min()
            )

            # Lấy bản ghi AD Dementia đầu tiên
            first_ad_record = (
                ad_dementia
                .sort_values("EXAMDATE_DT")
                .iloc[0]
            )

            # Lưu bằng chứng xác định AD
            evidence = []

            dxddue_value = first_ad_record["DXDDUE_CLEAN"]
            dxad_value = first_ad_record["DXAD_CLEAN"]

            # Kiểm tra DXDDUE, bỏ qua nếu giá trị là NA
            if (
                pd.notna(dxddue_value)
                and dxddue_value == AD_DUE_VALUE
            ):
                evidence.append("DXDDUE")

            # Kiểm tra DXAD, bỏ qua nếu giá trị là NA
            if (
                pd.notna(dxad_value)
                and dxad_value == "YES"
            ):
                evidence.append("DXAD")

            ad_evidence = " + ".join(evidence)

        else:

            first_ad_dementia_date = pd.NaT
            ad_evidence = "Không có"

            if has_dementia:

                # Đã chuyển sang Dementia nhưng không thỏa
                # điều kiện xác nhận Dementia do AD
                outcome = "Bị loại - Dementia không thỏa tiêu chí AD"

            else:

                # Kiểm tra có đủ dữ liệu theo dõi đến horizon
                observed_to_horizon = (
                    pd.notna(horizon_date) and
                    pd.notna(last_followup_date) and
                    last_followup_date >= horizon_date
                )

                if observed_to_horizon:
                    outcome = "sMCI"
                else:
                    outcome = "Bị loại - Thiếu theo dõi"


        # ----------------------------------------------------
        # 6.5. KIỂM TRA NGƯỜI QUAY VỀ CN
        # ----------------------------------------------------

        cn_records = within_horizon[
            within_horizon["DIAGNOSIS_CLEAN"] == "CN"
        ]

        has_reverted_to_cn = not cn_records.empty

        if has_reverted_to_cn:
            first_cn_date = cn_records["EXAMDATE_DT"].min()
        else:
            first_cn_date = pd.NaT

        # ----------------------------------------------------
        # 6.6. KIỂM TRA CHẨN ĐOÁN DAO ĐỘNG
        # ----------------------------------------------------

        # Gộp theo ngày để tránh một người có nhiều dòng
        # cùng ngày tạo ra dao động giả.

        daily_diagnosis = (
            within_horizon
            .groupby("EXAMDATE_DT")["DIAGNOSIS_CLEAN"]
            .agg(lambda x: sorted(set(x.dropna())))
            .reset_index(name="DIAGNOSES")
        )

        # Chỉ sử dụng ngày có một chẩn đoán thống nhất
        daily_diagnosis = daily_diagnosis[
            daily_diagnosis["DIAGNOSES"].apply(len) == 1
        ].copy()

        states = ["MCI"] + [
            diagnoses[0]
            for diagnoses in daily_diagnosis["DIAGNOSES"]
        ]

        changes, is_fluctuating, compressed_states = (
            analyze_diagnosis_sequence(states)
        )

        diagnostic_sequence = " -> ".join(states)

        # ----------------------------------------------------
        # 6.7. LƯU KẾT QUẢ
        # ----------------------------------------------------

        result_rows.append({

            "RID": rid,
            "ORIGPROT": phase,

            "Horizon_months": horizon,

            "Baseline_date": baseline_date,
            "Horizon_date": horizon_date,

            "Outcome": outcome,

            "First_Dementia_date": first_dementia_date,
            "First_AD_Dementia_date": first_ad_dementia_date,

            "AD_evidence": ad_evidence,

            "Has_AD_Dementia": has_ad_dementia,

            "Number_of_Dementia_records": len(dementia_records),

            "Number_of_Dementia_not_meeting_AD_criteria": (
                len(non_ad_criteria_dementia)
            ),

            "Last_followup_date": last_followup_date,

            "Reverted_to_CN": has_reverted_to_cn,
            "First_CN_date": first_cn_date,

            "Number_of_diagnosis_changes": changes,
            "Diagnosis_fluctuated": is_fluctuating,

            "Diagnostic_sequence": diagnostic_sequence,
            "Compressed_sequence": " -> ".join(compressed_states),

            "Number_of_followup_assessments": len(within_horizon)
        })


# Tạo bảng kết quả chi tiết
result = pd.DataFrame(result_rows)


# ============================================================
# 7. TỔNG HỢP KẾT QUẢ
# ============================================================

def summarize_group(group):

    total = group["RID"].nunique()

    pmci = (group["Outcome"] == "pMCI").sum()

    smci = (group["Outcome"] == "sMCI").sum()

    dementia_not_ad = (
        group["Outcome"] ==
        "Bị loại - Dementia không thỏa tiêu chí AD"
    ).sum()

    excluded_followup = (
        group["Outcome"] ==
        "Bị loại - Thiếu theo dõi"
    ).sum()

    reverted = group["Reverted_to_CN"].sum()

    fluctuated = group["Diagnosis_fluctuated"].sum()

    changed = (
        group["Number_of_diagnosis_changes"] > 0
    ).sum()

    classified = pmci + smci

    return {

        "Số MCI baseline": total,

        "pMCI (Dementia do AD)": pmci,

        "sMCI": smci,

        "Dementia không thỏa tiêu chí AD": dementia_not_ad,

        "Bị loại do thiếu theo dõi": excluded_followup,

        "Quay về CN": reverted,

        "Có thay đổi chẩn đoán": changed,

        "Chẩn đoán dao động": fluctuated,

        "Tỷ lệ pMCI trên cohort (%)": (
            round(pmci / total * 100, 2)
            if total else 0
        ),

        "Tỷ lệ pMCI trong nhóm phân loại được (%)": (
            round(pmci / classified * 100, 2)
            if classified else 0
        )
    }


summary_rows = []

# Tổng hợp theo từng giai đoạn tuyển
for (horizon, phase), group in result.groupby(
    ["Horizon_months", "ORIGPROT"],
    dropna=False
):

    row = summarize_group(group)

    row["Horizon (tháng)"] = horizon
    row["ORIGPROT"] = phase

    summary_rows.append(row)


# Tổng hợp toàn bộ cohort theo từng horizon
for horizon, group in result.groupby("Horizon_months"):

    row = summarize_group(group)

    row["Horizon (tháng)"] = horizon
    row["ORIGPROT"] = "Tổng"

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)

summary["_total_sort"] = (
    summary["ORIGPROT"] == "Tổng"
).astype(int)

summary = summary.sort_values(
    ["Horizon (tháng)", "_total_sort", "ORIGPROT"]
).drop(columns="_total_sort")

summary = summary[
    [
        "Horizon (tháng)",
        "ORIGPROT",
        "Số MCI baseline",
        "pMCI (Dementia do AD)",
        "sMCI",
        "Dementia không thỏa tiêu chí AD",
        "Bị loại do thiếu theo dõi",
        "Quay về CN",
        "Có thay đổi chẩn đoán",
        "Chẩn đoán dao động",
        "Tỷ lệ pMCI trên cohort (%)",
        "Tỷ lệ pMCI trong nhóm phân loại được (%)"
    ]
]


# ============================================================
# 8. XUẤT KẾT QUẢ
# ============================================================

output_dir = Path(path).parent / "mci_cohort_results"
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


# ============================================================
# 9. HIỂN THỊ KẾT QUẢ
# ============================================================

print("\n" + "=" * 70)
print("COHORT MCI BASELINE")
print("=" * 70)

print("Số người MCI baseline:", cohort["RID"].nunique())

print("\n" + "=" * 70)
print("PHÂN LOẠI pMCI / sMCI / BỊ LOẠI")
print("=" * 70)

print(summary.to_string(index=False))

print("\nĐã lưu kết quả tại:")
print(output_dir)
