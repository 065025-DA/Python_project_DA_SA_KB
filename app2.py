# app.py
"""
HR Analytics Dashboard (updated)
- Downloads dataset from a Google Drive link (using gdown) into ./data/hr_data.csv if not already present.
- Allows upload fallback, sample data, or repo-local data.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from datetime import datetime
import requests
import gdown  # used to download from Google Drive

st.set_page_config(page_title="HR Analytics — Drive-enabled Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Helper functions
# -----------------------
@st.cache_data(show_spinner=False)
def read_csv_flexible(path_or_file):
    try:
        return pd.read_csv(path_or_file)
    except Exception:
        try:
            return pd.read_csv(path_or_file, encoding="latin1")
        except Exception as e:
            raise e

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def derive_age(dob_series, ref=None):
    ref = pd.to_datetime(ref) if ref is not None else pd.Timestamp.today()
    dob = safe_to_datetime(dob_series)
    return ((ref - dob).dt.days // 365).astype("Int64")

def derive_years(start_series, end_series=None, ref=None):
    start = safe_to_datetime(start_series)
    end = safe_to_datetime(end_series) if end_series is not None else pd.NaT
    ref = pd.to_datetime(ref) if ref is not None else pd.Timestamp.today()
    end = end.fillna(ref)
    return ((end - start).dt.days // 365).astype("Int64")

def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else np.nan

def pct(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{round(float(x), decimals)}%"

def detect_column(df, candidates):
    """Return the first candidate found in df columns or None."""
    if df is None:
        return None
    cols = [c for c in df.columns]
    for cand in candidates:
        for c in cols:
            if c.strip().lower() == cand.strip().lower():
                return c
    # try partial matching
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
    # supports common google drive link patterns
    if drive_link is None:
        return None
    # try 'd/<id>/'
    import re
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
    if m:
        return m.group(1)
    # try id=...
    m2 = re.search(r"id=([a-zA-Z0-9_-]+)", drive_link)
    if m2:
        return m2.group(1)
    return None

def download_from_gdrive(drive_link: str, dest_path: str):
    file_id = extract_drive_id(drive_link)
    if not file_id:
        raise ValueError("Could not extract file id from the provided Google Drive link.")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    # Use gdown to download; gdown will handle large files & confirm tokens
    try:
        # If file already exists, skip download
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100:
            return dest_path
        gdown.download(url, dest_path, quiet=False)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return dest_path
        else:
            raise RuntimeError("gdown reported success but file is missing or empty.")
    except Exception as e:
        # fallback: try requests with export=download (may fail for large files)
        try:
            params = {"id": file_id}
            download_url = "https://docs.google.com/uc?export=download"
            with requests.get(download_url, params=params, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                return dest_path
            else:
                raise RuntimeError("Fallback requests download failed or produced empty file.")
        except Exception as e2:
            raise RuntimeError(f"Both gdown and requests downloads failed: {e} | {e2}")

# -----------------------
# Sidebar: data source & controls
# -----------------------
st.sidebar.title("📥 Data source & controls")
st.sidebar.write("This app will try (in order):\n1) download the dataset from the Google Drive link you provided, 2) load dataset from ./data/hr_data.csv in repo, 3) let you upload a CSV, or 4) use a small sample dataset for demo.")

# Show the provided link prefilled (we will use this link automatically)
drive_link = st.sidebar.text_input("Google Drive link to dataset", value=DRIVE_LINK_DEFAULT)

# Options
use_drive = st.sidebar.checkbox("Download dataset from Google Drive (recommended)", value=True)
use_repo = st.sidebar.checkbox("Load dataset from ./data/hr_data.csv (repo)", value=False)
uploaded_file = st.sidebar.file_uploader("Or upload a CSV instead", type=["csv"])

use_sample = st.sidebar.checkbox("Use small demo/sample dataset", value=False)

# Attempt to ensure data exists locally if requested
data_loaded_path = None
df_raw = None

if use_drive and drive_link:
    st.sidebar.info("Attempting to download dataset from Google Drive (this runs during app startup).")
    try:
        # Download to LOCAL_DATA_PATH
        downloaded = download_from_gdrive(drive_link, LOCAL_DATA_PATH)
        data_loaded_path = downloaded
        st.sidebar.success(f"Downloaded dataset to {downloaded}")
    except Exception as e:
        st.sidebar.error(f"Drive download failed: {e}")
        data_loaded_path = None

# If not downloaded and user selected repo load
if data_loaded_path is None and use_repo:
    if os.path.exists(LOCAL_DATA_PATH):
        data_loaded_path = LOCAL_DATA_PATH
        st.sidebar.success(f"Found dataset at {LOCAL_DATA_PATH}")
    else:
        st.sidebar.warning(f"No file found at {LOCAL_DATA_PATH} in repo.")

# If user uploaded a file, prefer that
if uploaded_file is not None:
    try:
        df_raw = read_csv_flexible(uploaded_file)
        st.sidebar.success("Loaded uploaded CSV.")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded CSV: {e}")

# If no upload but we have a path from drive/repo, load it
if df_raw is None and data_loaded_path is not None and os.path.exists(data_loaded_path):
    try:
        df_raw = read_csv_flexible(data_loaded_path)
        st.sidebar.success(f"Loaded dataset from {data_loaded_path}")
    except Exception as e:
        st.sidebar.error(f"Error reading dataset from {data_loaded_path}: {e}")
        df_raw = None

# If requested sample
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

# If still no data
if df_raw is None:
    st.info("No dataset loaded yet. Use the sidebar to download from Google Drive, upload a CSV, point to repo file, or enable sample data.")
else:
    df_raw.columns = [c.strip() for c in df_raw.columns]

# -----------------------
# If dataset present, continue to build dashboard (same detection & visuals as before)
# -----------------------
def build_dashboard(df):
    # detect some likely columns
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

    # derived columns
    if col_dob:
        df["Age"] = derive_age(df[col_dob])
    if col_start:
        df["YearsAtCompany"] = derive_years(df[col_start], df.get(col_exit))

    if col_exit:
        df["Attrition"] = df[col_exit].notna().astype(int)
    elif col_status:
        df["Attrition"] = df[col_status].astype(str).str.lower().isin(["resigned","terminated","left","separated"]).astype(int)

    # make numeric conversions
    for c in [col_perf, col_salary, col_eng]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sidebar global filters (dynamic)
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Global Filters")
    dept_options = sorted(df[col_dept].dropna().unique().tolist()) if col_dept and col_dept in df.columns else []
    department_filter = st.sidebar.multiselect("Department", options=dept_options, default=[])
    gender_options = sorted(df[col_gender].dropna().unique().tolist()) if col_gender and col_gender in df.columns else []
    gender_filter = st.sidebar.multiselect("Gender", options=gender_options, default=[])
    loc_options = sorted(df[col_loc].dropna().unique().tolist()) if col_loc and col_loc in df.columns else []
    location_filter = st.sidebar.multiselect("Location / State", options=loc_options, default=[])

    age_range = None
    if "Age" in df.columns:
        amin, amax = int(df["Age"].dropna().min()), int(df["Age"].dropna().max())
        age_range = st.sidebar.slider("Age range", amin, amax, (amin, amax))

    tenure_range = None
    if "YearsAtCompany" in df.columns:
        tmin, tmax = int(df["YearsAtCompany"].dropna().min()), int(df["YearsAtCompany"].dropna().max())
        tenure_range = st.sidebar.slider("Years at Company", tmin, tmax, (tmin, tmax))

    # Apply filters function
    def apply_filters_local(d):
        dd = d.copy()
        if department_filter and col_dept in dd.columns:
            dd = dd[dd[col_dept].isin(department_filter)]
        if gender_filter and col_gender in dd.columns:
            dd = dd[dd[col_gender].isin(gender_filter)]
        if location_filter and col_loc in dd.columns:
            dd = dd[dd[col_loc].isin(location_filter)]
        if age_range and "Age" in dd.columns:
            dd = dd[dd["Age"].between(age_range[0], age_range[1])]
        if tenure_range and "YearsAtCompany" in dd.columns:
            dd = dd[dd["YearsAtCompany"].between(tenure_range[0], tenure_range[1])]
        return dd

    df_filtered = apply_filters_local(df)
    if df_is_empty_or_none(df_filtered):
        st.warning("No data after applying filters. Try clearing filters or checking column detection.")
        return

    # Overview KPIs
    st.header("🔎 Overview — HR Dashboard (Drive dataset)")
    c1, c2, c3, c4, c5 = st.columns(5)
    headcount = int(df_filtered[col_empid].nunique()) if col_empid and col_empid in df_filtered.columns else df_filtered.shape[0]
    attr_rate = (df_filtered["Attrition"].mean() * 100) if "Attrition" in df_filtered.columns else np.nan
    avg_tenure = safe_mean(df_filtered.get("YearsAtCompany", pd.Series(dtype=float)))
    avg_age = safe_mean(df_filtered.get("Age", pd.Series(dtype=float)))
    avg_eng = safe_mean(df_filtered.get(col_eng, pd.Series(dtype=float))) if col_eng else np.nan
    c1.metric("Headcount", headcount)
    c2.metric("Attrition Rate", pct(attr_rate) if not pd.isna(attr_rate) else "N/A")
    c3.metric("Avg Tenure (yrs)", round(avg_tenure,2) if not pd.isna(avg_tenure) else "N/A")
    c4.metric("Avg Age", round(avg_age,2) if not pd.isna(avg_age) else "N/A")
    c5.metric("Avg Engagement", round(avg_eng,2) if not pd.isna(avg_eng) else "N/A")

    st.markdown("---")
    # visual: department counts
    if col_dept in df_filtered.columns:
        st.subheader("Employees by Department")
        dept_counts = df_filtered[col_dept].value_counts().reset_index()
        dept_counts.columns = ["Department","Count"]
        fig = px.bar(dept_counts, x="Department", y="Count", text="Count")
        st.plotly_chart(fig, use_container_width=True)

    # engagement distribution
    if col_eng in df_filtered.columns:
        st.subheader("Engagement distribution")
        st.plotly_chart(px.histogram(df_filtered, x=col_eng, nbins=12), use_container_width=True)

    # performance vs tenure
    if col_perf in df_filtered.columns and "YearsAtCompany" in df_filtered.columns:
        st.subheader("Performance vs Tenure")
        st.plotly_chart(px.scatter(df_filtered, x="YearsAtCompany", y=col_perf,
                                   color=col_dept if col_dept in df_filtered.columns else None,
                                   hover_data=[col_empid, col_title] if col_empid in df_filtered.columns and col_title in df_filtered.columns else None),
                       use_container_width=True)

    # correlation heatmap
    nums = df_filtered.select_dtypes(include=[np.number]).drop(columns=[col_empid] if col_empid in df_filtered.columns else [], errors="ignore")
    if nums.shape[1] >= 2:
        st.subheader("Numeric correlations")
        st.plotly_chart(px.imshow(nums.corr(), text_auto=True, aspect="auto"), use_container_width=True)

    st.markdown("---")
    st.download_button("📥 Download current filtered dataset", data=to_csv_bytes(df_filtered), file_name="hr_filtered.csv", mime="text/csv")

# Build dashboard if data exists
if df_raw is not None:
    build_dashboard(df_raw)

# footer
st.markdown("---")
st.caption("Built with ❤️ — Dataset downloaded using gdown/requests. If download fails, upload the CSV manually via the sidebar.")
