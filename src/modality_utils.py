import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Hàm phụ trợ 
# ---------------------------------------------------------------
def clean_string(series):
    """Chuẩn hóa chuỗi, giữ giá trị thiếu."""
    return series.astype("string").str.strip().replace("", pd.NA)


def check_required_columns(df, required, file_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} thiếu các cột bắt buộc: {missing}")


def make_rid_key(series):
    """Chuẩn hóa khóa RID dưới dạng chuỗi."""
    return series.astype("string").str.strip().replace("", pd.NA)


# ---- QĐ24: khớp MRI baseline theo cửa sổ ngày ----
ROI = {"ST29SV": "HIPPO_L", "ST88SV": "HIPPO_R",
       "ST24CV": "ERC_L", "ST83CV": "ERC_R",
       "ST37SV": "LATVENT_L", "ST96SV": "LATVENT_R",
       "ST10CV": "ICV"}  # theo tài liệu UCSFFSX7; đối chiếu lại DATADIC


def tesla(s):
    num = s.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(num, errors="coerce")


def match_baseline_mri(cohort, mri, window=90):
    """cohort: RID_KEY, ORIGPROT, BASELINE_DATE (ngày khám bl của DXSUM).
    Trả về tối đa 1 ảnh/RID: gần baseline nhất trong ±window ngày."""
    m = mri.dropna(subset=["IMAGEUID"]).copy()
    m["MRI_DATE"] = pd.to_datetime(m["EXAMDATE"], errors="coerce")
    qc = m["OVERALLQC"].astype("string").str.upper()
    m = m[qc.ne("FAIL").fillna(True)]  # bỏ ảnh QC Fail
    x = cohort[["RID_KEY", "ORIGPROT", "BASELINE_DATE"]].merge(
        m.drop(columns=["ORIGPROT"], errors="ignore"), on="RID_KEY")
    x["DELTA_DAYS"] = (x["MRI_DATE"] - x["BASELINE_DATE"]).dt.days
    x = x[x["DELTA_DAYS"].abs() <= window].copy()
    want = np.where(x["ORIGPROT"].eq("ADNI1"), 1.5, 3.0)
    x["FS_KHAC"] = (tesla(x["FIELD_STRENGTH"]) != want).astype(int)
    x["CHUA_QC"] = x["STATUS"].astype("string").str.lower().ne(
        "complete").fillna(True).astype(int)
    x["ABS_DELTA"] = x["DELTA_DAYS"].abs()
    x = x.sort_values(["RID_KEY", "ABS_DELTA", "FS_KHAC", "CHUA_QC"])
    x = x.drop_duplicates("RID_KEY")
    for c in ROI:  # UCSFFSX7 lưu dạng chuỗi
        x[c] = pd.to_numeric(x[c], errors="coerce")
    keep = ["RID_KEY", "IMAGEUID", "VISCODE2", "DELTA_DAYS",
            "FIELD_STRENGTH", "FSVER", "STATUS", "OVERALLQC"] + list(ROI)
    return x[keep].rename(columns={c: "MRI_" + v for c, v in ROI.items()})


def mri_coverage(cohort, matched):
    t = cohort[["RID_KEY", "ORIGPROT"]].merge(
        matched, on="RID_KEY", how="left")
    t["CO_MRI"] = t["IMAGEUID"].notna()
    out = t.groupby("ORIGPROT").agg(
        n=("RID_KEY", "size"), co_MRI=("CO_MRI", "sum"),
        lech_ngay_trung_vi=("DELTA_DAYS", lambda s: s.abs().median()))
    out["ty_le"] = (out["co_MRI"] / out["n"]).round(3)
    fs = pd.crosstab(t["ORIGPROT"], t["FIELD_STRENGTH"].fillna("không có"))
    st = pd.crosstab(t["ORIGPROT"], t["STATUS"].fillna("không có"))
    vc = pd.crosstab(t["ORIGPROT"], t["VISCODE2"].fillna("không có"))
    return out, fs, st, vc


# ---- QĐ25-QĐ26: nhóm biến lõi, mẫu khuyết, sơ đồ chọn mẫu ----
CORE = {
    "demo": ["AGE", "SEX", "EDUC"],
    "apoe4": ["APOE4"],
    "cog": ["MMSCORE", "ADASTT13", "CDRSB", "FAQTOTAL", "RAVLTIMM"],
    "amyloid": ["AMYSTAT"],
    "mri": ["MRI_HIPPO_L", "MRI_HIPPO_R", "MRI_ICV"],
}


def group_flags(df, groups=CORE):
    # 1 = đủ mọi biến của nhóm; nhóm MRI xét giá trị ROI, không xét cờ 0/1
    return pd.DataFrame({g: df[c].notna().all(axis=1).astype(int)
                         for g, c in groups.items()}, index=df.index)


def pattern_table(df, by="ORIGPROT", groups=CORE):
    f = group_flags(df, groups)
    f[by] = df[by].values
    t = f.groupby([by] + list(groups)).size().rename("n").reset_index()
    t["ty_le"] = (t["n"] / t.groupby(by)["n"].transform("sum")).round(3)
    t[list(groups)] = t[list(groups)].replace({1: "có", 0: "thiếu"})
    return t.sort_values([by, "n"], ascending=[True, False])


def flow_table(res36, feats, groups=CORE):
    """res36: kết quả H=36, mỗi RID một dòng (RID_KEY, ORIGPROT, Outcome).
    feats: biến baseline đã nối ADSL + MRI, mỗi RID một dòng."""
    d = res36.merge(feats.drop(columns="ORIGPROT", errors="ignore"),
                    on="RID_KEY", how="left", validate="one_to_one")
    f = group_flags(d, groups)
    lab = d["Outcome"].isin(["pMCI", "sMCI"])
    core = lab & f[["demo", "apoe4", "cog"]].all(axis=1)
    steps = {
        "1. MCI baseline": pd.Series(True, index=d.index),
        "2. Đủ điều kiện H=36": d["Outcome"].ne("Bị loại - Chưa thể đủ thời gian"),
        "3. Có nhãn pMCI/sMCI": lab,
        "4. + demo, APOE4, nhận thức lõi": core,
        "5. + MRI baseline": core & f["mri"].eq(1),
        "6. + AMYSTAT": core & f["mri"].eq(1) & f["amyloid"].eq(1),
    }
    pm = d["Outcome"].eq("pMCI")
    out = {}
    for k, v in steps.items():
        n = v.groupby(d["ORIGPROT"]).sum()
        p = (v & pm).groupby(d["ORIGPROT"]).sum()
        out[k] = n.astype(str) + " (" + p.astype(str) + ")"
    return pd.DataFrame(out).T  # ô = số người (số pMCI)


def lost_vs_labeled(res36, feats,
                    cols=("AGE", "EDUC", "MMSCORE", "ADASTT13", "CDRSB", "APOE4")):
    d = res36.merge(feats.drop(columns="ORIGPROT", errors="ignore"),
                    on="RID_KEY", how="left", validate="one_to_one")
    lost_lab = "Bị loại - Mất theo dõi"
    d = d[d["Outcome"].isin(["pMCI", "sMCI", lost_lab])]
    rows = []
    for ph, g in d.groupby("ORIGPROT"):
        lost = g["Outcome"].eq(lost_lab)
        for c in cols:
            a = pd.to_numeric(g.loc[lost, c], errors="coerce").astype(float)
            b = pd.to_numeric(g.loc[~lost, c], errors="coerce").astype(float)
            sd = np.sqrt((a.var() + b.var()) / 2)
            smd = (a.mean() - b.mean()) / sd if sd > 0 else np.nan
            rows.append({"ORIGPROT": ph, "bien": c,
                         "n_mat": a.notna().sum(), "n_co_nhan": b.notna().sum(),
                         "TB_mat": round(a.mean(), 2),
                         "TB_co_nhan": round(b.mean(), 2),
                         "SMD": round(smd, 2)})
    return pd.DataFrame(rows)


# ---- O14: kiểm tra khớp ADSL - DXSUM ----
def adsl_alignment(cohort, adsl):
    m = cohort.merge(adsl, on="RID_KEY", how="left", indicator=True,
                     suffixes=("", "_ADSL"), validate="one_to_one")
    print(m["_merge"].value_counts())  # left_only = 0
    print(pd.crosstab(m["ORIGPROT"], m["DX"], dropna=False))  # kỳ vọng MCI
    gap = (pd.to_datetime(m["ENRLDT"], errors="coerce")
           - m["BASELINE_DATE"]).dt.days
    print(gap.groupby(m["ORIGPROT"]).describe(
        percentiles=[.05, .5, .95]).round(0))  # lệch ngày
    return m.drop(columns="_merge")
