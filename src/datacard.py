
# src/datacard.py

import pandas as pd


def dataset_overview(df):
    """
    Thống kê tổng quan dữ liệu.
    """

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_RID": df["RID"].nunique(),
        "duplicate_RID_VISCODE": int(
            df.duplicated(
                ["RID", "VISCODE"]
            ).sum()
        )
    }


def diagnosis_distribution(df):
    """
    Phân bố chẩn đoán trong toàn bộ dữ liệu.
    """

    return df["DX"].value_counts(
        dropna=False
    ).rename_axis("DX").reset_index(
        name="count"
    )


def baseline_diagnosis(df):
    """
    Phân bố chẩn đoán baseline theo giai đoạn.
    """

    baseline = df[
        df["VISCODE"] == "bl"
    ]

    return pd.crosstab(
        baseline["ORIGPROT"],
        baseline["DX_bl"],
        margins=True,
        dropna=False
    )


def missingness_summary(df):
    """
    Tính tỷ lệ giá trị thiếu theo từng biến.
    """

    result = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean()
    })

    return result.sort_values(
        "missing_rate",
        ascending=False
    )


def check_csf_values(df):
    """
    Kiểm tra giá trị không phải số
    trong các biến CSF.
    """

    columns = ["ABETA", "TAU", "PTAU"]

    results = {}

    for col in columns:

        if col not in df.columns:
            results[col] = {
                "status": "Column not found"
            }
            continue

        s = df[col]

        invalid = s[
            s.notna()
            & pd.to_numeric(
                s, errors="coerce"
            ).isna()
        ]

        results[col] = {
            "invalid_count": len(invalid),
            "examples": (
                invalid.unique()[:5].tolist()
            )
        }

    return results
