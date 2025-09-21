# app.py
"""
HR Analytics Dashboard (Drive-enabled) — corrected derive_years bug
- Attempts to download dataset from Google Drive using gdown (fallback: requests)
- Allows upload fallback or sample dataset
- Robust date handling and filters
"""
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import requests
import gdown

st.set_page_config(page_title="HR Analytics — Fixed", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def read_csv_flexible(path_or_file):
    """Read CSV with common fallbacks"""
    try:
        return pd.read_csv(path_or_file)
    except Exception:
        try:
            return pd.read_csv(path_or_file, encoding="latin1")
        except Exception as e:
            raise e

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def safe_to_datetime(series):
    """Convert series-like to pd.Series of datetimes (errors -> NaT)."""
    # If input is a pandas Series -> convert directly
    if isinstance(series, pd.Series):
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    # If it's None -> return None (caller should handle)
    if series is None:
        return None
    # If scalar or array-like -> attempt conversion and return a Series
    try:
        converted = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        if isinstance(converted, pd.Series):
            return converted
        # converted might be a Timestamp (scalar) -> wrap as Series
        return pd.Series(converted)
    except Exception:
        return pd.Series(pd.to_datetime(series, errors="coerce"))

def derive_age(dob_series, ref_date=None):
    """Return an Int64 series of ages (years)."""
    if dob_series is None:
        return pd.Series(dtype="Int64")
    dob = safe_to_datetime(dob_series)
    if isinstance(dob, pd.Series):
        ref = pd.to_datetime(ref_date) if ref_date is not None else pd.Timestamp.today()
        # avoid .dt on non-series
        return ((ref - dob).dt.days // 365).astype("Int64")
    else:
        # fallback: wrap and return single-element series
        ref = pd.to_datetime(ref_date) if ref_date is not None else pd.Timestamp.today()
        val = (ref - pd.to_datetime(dob, errors="coerce")).days // 365
        return pd.Series([val], dtype="Int64")

def derive_years(start_series, end_series=None, ref_date=None):
    """
    Safely derive years between start and end (or reference date).
    - Ensures 'end' is a Series aligned to 'start' index before calling fillna.
    """
    if start_series is None:
        return pd.Series(dtype="Int64")

    start = safe_to_datetime(start_series)
    # ensure start is a Series
    if not isinstance(start, pd.Series):
        start = pd.Series(start)

    # Prepare 'end' as a Series aligned with 'start'
    if end_series is None:
        end = pd.Series([pd.NaT] * len(start), index=start.index)
    else:
        end = safe_to_datetime(end_series)
        # If end is not a Series (scalar), convert to Series with same index
        if not isinstance(end, pd.Series):
            end = pd.Series([end] * len(start), index=start.index)
        else:
            # If the index differs, reindex to start's index to guarantee alignment
            if not end.index.equals(start.index):
                end = end.reindex(start.index)

    ref = pd.to_datetime(ref_date) if ref_date is not None else pd.Timestamp.today()
    # Now safe to call fillna
    end = end.fillna(ref)

    # Compute integer years and return nullable Int64
    years = ((end - start).dt.days // 365).astype("Int64")
    return years

def pct(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{round(float(x), decimals)}%"

def detect_column(df, candidates):
    """Return first matching column name from df for any candidate (case-insensitive / partial match)."""
    if df is None:
        return None
    cols = list(df.columns)
    # exact match (case-insensitive)
    for cand in candidates:
        for c in cols:
            if c.strip().lower() == cand.strip().lower():
                return c
    # partial containment
    for cand in candidates:
        for c in cols:
            if cand.strip().lower() in c.strip().lower():
                return c
    return None

def df_is_empty_or_none(df):
    return df is None or (hasattr(df, "shape") and df.shape[0] == 0)

# -----------------------
# Google Drive downloader
# -----------------------
DRIVE_LINK_DEFAULT = "https://drive.google.com/file/d/1YnkDIjGs0ShOdEK0iq60O0fGarPqJhU2/view?usp=drive_link"
LOCAL_DATA_PATH = "./data/hr_data.csv"

def extract_drive_id(drive_link: str):
    if not drive_link:
        return None
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
    if m:
        return m.group(1)
    m2 = re.search(r"id=([a-zA-Z0-9_-]+)", drive_link)
    if m2:
        return m2.group(1)
    return None

def download_from_gdrive(drive_link: str, dest_path: str):
    """Download file using gdown; fallback to requests if necessary."""
    file_id = extract_drive_id(drive_link)
    if not file_id:
        raise ValueError("Could not extract file id from the provided Google Drive link.")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    # Skip download if present and non-empty
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
        return dest_path
    try:
        gdown.download(url, dest_path, quiet=False)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return dest_path
        else:
            raise RuntimeError("gdown finished but file is missing or empty.")
    except Exception as e:
        # fallback using requests (may fail for large files requiring confirmation)
        try:
            download_url = "https://docs.google.com/uc?export=download"
            with requests.get(download_url, params={"id": file_id}, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                return dest_path
            else:
                raise RuntimeError("Fallback requests download produced empty file.")
        except Exception as e2:
            raise RuntimeError(f"Both gdown and requests failed: {e} | {e2}")

# -----------------------
# Sidebar: controls & data source
# -----------------------
st.sidebar.title("📥 Data source & controls")
st.sidebar.write("Try: download from Drive → repo file → upload → sample.")

drive_link = st.sidebar.text_input("Google Drive link to dataset", value=DRIVE_LINK_DEFAULT)
use_drive = st.sidebar.checkbox("Download dataset from Google Drive", value=True)
use_repo = st.sidebar.checkbox("Load dataset from ./data/hr_data.csv (repo)", value=False)
uploaded_file = st.sidebar.file_uploader("Or upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use demo/sample data", value=False)

data_path = None
df_raw = None

# Attempt drive download (if requested)
if use_drive and drive_link:
    try:
        downloaded = download_from_gdrive(drive_link, LOCAL_DATA_PATH)
        data_path = downloaded
        st.sidebar.success(f"Downloaded dataset to {downloaded}")
    except Exception as e:
        st.sidebar.error(f"Drive download failed: {e}")
        data_path = None

# Repo fallback
if data_path is None and use_repo:
    if os.path.exists(LOCAL_DATA_PATH):
        data_path = LOCAL_DATA_PATH
        st.sidebar.success(f"Found dataset at {LOCAL_DATA_PATH}")
    else:
        st.sidebar.warning(f"No file found at {LOCAL_DATA_PATH} in repo.")

# Upload takes precedence
if uploaded_file is not None:
    try:
        df_raw = read_csv_flexible(uploaded_file)
        st.sidebar.success("Loaded uploaded CSV.")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded CSV: {e}")

# Load from file system if available
if df_raw is None and data_path is not None and os.path.exists(data_path):
    try:
        df_raw = read_csv_flexible(data_path)
        st.sidebar.success(f"Loaded dataset from {data_path}")
    except Exception as e:
        st.sidebar.error(f"Error reading dataset from {data_path}: {e}")
        df_raw = None

# Sample fallback
if df_raw is None and use_sample:
    rng = np.random.default_rng(42)
    n = 200
    df_raw = pd.DataFrame({
        "Employee ID": [f"E{1000+i}" for i in range(n)],
        "DOB": pd.to_datetime("1990-01-01") + pd.to_timedelta(rng.integers(0,365*30,n), unit="D"),
        "StartDate": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0,365*7,n), unit="D"),
        "ExitDate": [pd.NaT if p else pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0,365,1)[0], unit="D") for p in rng.choice([True, False], n, p=[0.85,0.15])],
        "Department": rng.choice(["Engineering","Sales","HR","Finance","Ops"], n),
        "Gender": rng.choice(["Male","Female","Other"], n),
        "Location": rng.choice(["Delhi","Bengaluru","Mumbai","Chennai"], n),
        "Performance Score": np.clip(np.round(rng.normal(70,12,n),1), 30, 100),
        "Engagement Score": np.clip(np.round(rng.normal(70,10,n),1), 20, 100),
        "Salary": rng.integers(300000,2000000,n)
    })
    st.sidebar.success("Sample dataset loaded.")

if df_raw is None:
    st.info("No dataset loaded. Use the sidebar to download/upload or enable sample data.")
else:
    # normalize column names
    df_raw.columns = [c.strip() for c in df_raw.columns]

# -----------------------
# Dashboard builder (uses fixed derive_years)
# -----------------------
def build_dashboard(df):
    # detect columns
    col_empid = detect_column(df, ["Employee ID", "EmployeeID", "EmpID", "ID"])
    col_dob = detect_column(df, ["DOB", "DateOfBirth", "Date of Birth", "BirthDate"])
    col_start = detect_column(df, ["StartDate", "DateOfHire", "Date of Joining", "DOJ"])
    col_exit = detect_column(df, ["ExitDate", "Separation Date", "DateOfExit"])
    col_dept = detect_column(df, ["Department", "DepartmentType", "Division"])
    col_title = detect_column(df, ["Title", "JobTitle", "Job Title"])
    col_gender = detect_column(df, ["Gender", "GenderCode"])
    col_loc = detect_column(df, ["Location", "State", "LocationCode"])
    col_perf = detect_column(df, ["Performance Score", "Performance", "PerformanceRating", "Rating"])
    col_salary = detect_column(df, ["Salary", "AnnualSalary", "CTC"])
    col_eng = detect_column(df, ["Engagement Score", "EmployeeEngagementScore", "Engagement"])
    col_status = detect_column(df, ["EmploymentStatus", "Status", "TerminationType"])
    col_survey_date = detect_column(df, ["Survey Date", "SurveyDate", "Survey_Date"])

    # Derived: Age
    if col_dob and col_dob in df.columns:
        df["Age"] = derive_age(df[col_dob])

    # Derived: YearsAtCompany (pass exit only if column exists)
    if col_start and col_start in df.columns:
        end_series = df[col_exit] if (col_exit and col_exit in df.columns) else None
        df["YearsAtCompany"] = derive_years(df[col_start], end_series)

    # Attrition logic
    if col_exit and col_exit in df.columns:
        df["Attrition"] = df[col_exit].notna().astype(int)
    elif col_status and col_status in df.columns:
        df["Attrition"] = df[col_status].astype(str).str.lower().isin(["resigned","terminated","left","separated"]).astype(int)

    # numeric conversions
    for c in (col_perf, col_salary, col_eng):
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Global Filters")
    dept_options = sorted(df[col_dept].dropna().unique().tolist()) if (col_dept and col_dept in df.columns) else []
    department_filter = st.sidebar.multiselect("Department", options=dept_options, default=[])

    title_options = sorted(df[col_title].dropna().unique().tolist()) if (col_title and col_title in df.columns) else []
    title_filter = st.sidebar.multiselect("Job Title", options=title_options, default=[])

    gender_options = sorted(df[col_gender].dropna().unique().tolist()) if (col_gender and col_gender in df.columns) else []
    gender_filter = st.sidebar.multiselect("Gender", options=gender_options, default=[])

    loc_options = sorted(df[col_loc].dropna().unique().tolist()) if (col_loc and col_loc in df.columns) else []
    location_filter = st.sidebar.multiselect("Location / State", options=loc_options, default=[])

    age_range = None
    if "Age" in df.columns:
        amin, amax = int(df["Age"].dropna().min()), int(df["Age"].dropna().max())
        age_range = st.sidebar.slider("Age range", amin, amax, (amin, amax))

    tenure_range = None
    if "YearsAtCompany" in df.columns:
        tmin, tmax = int(df["YearsAtCompany"].dropna().min()), int(df["YearsAtCompany"].dropna().max())
        tenure_range = st.sidebar.slider("Years at Company", tmin, tmax, (tmin, tmax))

    show_only_attrited = st.sidebar.checkbox("Show only employees who left (attrition)", value=False)

    if st.sidebar.button("Reset filters"):
        st.experimental_rerun()

    # apply filters
    def apply_filters_local(d):
        dd = d.copy()
        if department_filter and col_dept in dd.columns:
            dd = dd[dd[col_dept].isin(department_filter)]
        if title_filter and col_title in dd.columns:
            dd = dd[dd[col_title].isin(title_filter)]
        if gender_filter and col_gender in dd.columns:
            dd = dd[dd[col_gender].isin(gender_filter)]
        if location_filter and col_loc in dd.columns:
            dd = dd[dd[col_loc].isin(location_filter)]
        if age_range and "Age" in dd.columns:
            dd = dd[dd["Age"].between(age_range[0], age_range[1])]
        if tenure_range and "YearsAtCompany" in dd.columns:
            dd = dd[dd["YearsAtCompany"].between(tenure_range[0], tenure_range[1])]
        if show_only_attrited and "Attrition" in dd.columns:
            dd = dd[dd["Attrition"] == 1]
        return dd

    df_filtered = apply_filters_local(df)
    if df_is_empty_or_none(df_filtered):
        st.warning("No data available after applying filters.")
        return

    # Overview KPIs
    st.header("🔎 Overview — HR Dashboard (fixed)")
    c1, c2, c3, c4, c5 = st.columns(5)
    headcount = int(df_filtered[col_empid].nunique()) if (col_empid and col_empid in df_filtered.columns) else df_filtered.shape[0]
    attr_rate = (df_filtered["Attrition"].mean() * 100) if "Attrition" in df_filtered.columns else np.nan
    avg_tenure = df_filtered.get("YearsAtCompany").dropna().mean() if "YearsAtCompany" in df_filtered.columns else np.nan
    avg_age = df_filtered.get("Age").dropna().mean() if "Age" in df_filtered.columns else np.nan
    avg_eng = df_filtered.get(col_eng).dropna().mean() if (col_eng and col_eng in df_filtered.columns) else np.nan
    c1.metric("Headcount", headcount)
    c2.metric("Attrition Rate", pct(attr_rate) if not pd.isna(attr_rate) else "N/A")
    c3.metric("Avg Tenure (yrs)", round(avg_tenure,2) if not pd.isna(avg_tenure) else "N/A")
    c4.metric("Avg Age", round(avg_age,2) if not pd.isna(avg_age) else "N/A")
    c5.metric("Avg Engagement", round(avg_eng,2) if not pd.isna(avg_eng) else "N/A")

    st.markdown("---")
    # department counts
    if col_dept and col_dept in df_filtered.columns:
        st.subheader("Employees by Department")
        dept_counts = df_filtered[col_dept].value_counts().reset_index()
        dept_counts.columns = ["Department","Count"]
        st.plotly_chart(px.bar(dept_counts, x="Department", y="Count", text="Count"), use_container_width=True)

    # engagement distribution
    if col_eng and col_eng in df_filtered.columns:
        st.subheader("Engagement distribution")
        st.plotly_chart(px.histogram(df_filtered, x=col_eng, nbins=12), use_container_width=True)

    # performance vs tenure
    if col_perf and col_perf in df_filtered.columns and "YearsAtCompany" in df_filtered.columns:
        st.subheader("Performance vs Tenure")
        st.plotly_chart(px.scatter(df_filtered, x="YearsAtCompany", y=col_perf,
                                   color=col_dept if col_dept in df_filtered.columns else None,
                                   hover_data=[col_empid, col_title] if (col_empid and col_empid in df_filtered.columns and col_title and col_title in df_filtered.columns) else None),
                       use_container_width=True)

    # numeric correlations
    nums = df_filtered.select_dtypes(include=[np.number]).drop(columns=[col_empid] if (col_empid and col_empid in df_filtered.columns) else [], errors="ignore")
    if nums.shape[1] >= 2:
        st.subheader("Numeric correlations")
        st.plotly_chart(px.imshow(nums.corr(), text_auto=True, aspect="auto"), use_container_width=True)

    st.markdown("---")
    st.download_button("📥 Download filtered CSV", data=to_csv_bytes(df_filtered), file_name="hr_filtered.csv", mime="text/csv")

# Build dashboard if data exists
if df_raw is not None:
    build_dashboard(df_raw)

st.markdown("---")
st.caption("Fixed derive_years bug — now returns years reliably even when ExitDate is missing or scalar NaT.")
