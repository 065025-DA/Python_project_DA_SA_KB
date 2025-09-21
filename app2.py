# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="HR Analytics — Powerful Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Helper Functions
# -----------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.mean() if not s.empty else np.nan

def pct(value):
    return f"{round(value,2)}%" if not pd.isna(value) else "N/A"

# -----------------------
# Sidebar Upload
# -----------------------
st.sidebar.title("📂 Upload Employee Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

df = None
if uploaded_file:
    df = load_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]  # clean col names

# -----------------------
# Sidebar Filters
# -----------------------
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Global Filters")

    department_filter = st.sidebar.multiselect("Department", sorted(df["Department"].dropna().unique().tolist()))
    gender_filter = st.sidebar.multiselect("Gender", sorted(df["Gender"].dropna().unique().tolist()))
    job_filter = st.sidebar.multiselect("Job Title", sorted(df["JobTitle"].dropna().unique().tolist()))
    status_filter = st.sidebar.multiselect("Employment Status", sorted(df["EmploymentStatus"].dropna().unique().tolist()))

    # Apply filters
    if department_filter:
        df = df[df["Department"].isin(department_filter)]
    if gender_filter:
        df = df[df["Gender"].isin(gender_filter)]
    if job_filter:
        df = df[df["JobTitle"].isin(job_filter)]
    if status_filter:
        df = df[df["EmploymentStatus"].isin(status_filter)]

# -----------------------
# Tabs
# -----------------------
tabs = st.tabs(["Overview", "Employee Insights", "Performance", "Engagement", "Compensation & Risk"])

# -----------------------
# Overview Tab
# -----------------------
with tabs[0]:
    st.header("📊 Company Overview")
    if df is None:
        st.info("Please upload the dataset.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Headcount", df["EmployeeID"].nunique())
        c2.metric("Avg Tenure (yrs)", round(safe_mean(df["YearsAtCompany"]),1))
        c3.metric("Avg Age", round(safe_mean((pd.to_datetime("today") - pd.to_datetime(df["DateOfBirth"], errors="coerce")).dt.days/365),1))
        c4.metric("Avg Salary", round(safe_mean(df["Salary"]),2))
        c5.metric("Avg Engagement", round(safe_mean(df["EmployeeEngagementScore"]),2))

        st.markdown("---")
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Employees by Department")
            dept = df["Department"].value_counts().reset_index()
            dept.columns = ["Department", "Count"]
            fig = px.bar(dept, x="Department", y="Count", text="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Gender Split")
            g = df["Gender"].value_counts().reset_index()
            g.columns = ["Gender", "Count"]
            st.plotly_chart(px.pie(g, names="Gender", values="Count", hole=0.4), use_container_width=True)

# -----------------------
# Employee Insights
# -----------------------
with tabs[1]:
    st.header("🧾 Employee Insights")
    if df is None:
        st.info("Upload dataset to see insights.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Job Titles")
            top_jobs = df["JobTitle"].value_counts().head(15).reset_index()
            top_jobs.columns = ["Job Title", "Count"]
            st.plotly_chart(px.bar(top_jobs, x="Job Title", y="Count"), use_container_width=True)

        with col2:
            st.subheader("Employment Status Distribution")
            status = df["EmploymentStatus"].value_counts().reset_index()
            status.columns = ["Status", "Count"]
            st.plotly_chart(px.pie(status, names="Status", values="Count", hole=0.4), use_container_width=True)

# -----------------------
# Performance
# -----------------------
with tabs[2]:
    st.header("📈 Performance Insights")
    if df is None:
        st.info("Upload dataset to view performance insights.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Performance Rating Distribution")
            st.plotly_chart(px.histogram(df, x="PerformanceRating", nbins=10), use_container_width=True)

        with col2:
            st.subheader("Performance vs Tenure")
            st.plotly_chart(px.scatter(df, x="YearsAtCompany", y="PerformanceRating", color="Department",
                                       hover_data=["EmployeeID","JobTitle"]), use_container_width=True)

# -----------------------
# Engagement
# -----------------------
with tabs[3]:
    st.header("💬 Engagement Insights")
    if df is None:
        st.info("Upload dataset to view engagement insights.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Job Satisfaction", round(safe_mean(df["JobSatisfactionScore"]),2))
        col2.metric("Avg Stress Level", round(safe_mean(df["StressLevelScore"]),2))
        col3.metric("Avg WFH Days", round(safe_mean(df["WorkFromHomeDays"]),1))

        st.markdown("---")
        st.subheader("Engagement vs Job Satisfaction")
        st.plotly_chart(px.scatter(df, x="EmployeeEngagementScore", y="JobSatisfactionScore",
                                   color="Department", hover_data=["EmployeeID","JobTitle"]), use_container_width=True)

# -----------------------
# Compensation & Risk
# -----------------------
with tabs[4]:
    st.header("⚠️ Compensation & Retention Risk")
    if df is None:
        st.info("Upload dataset to view compensation and risk insights.")
    else:
        # --- Fix RetentionRisk handling ---
        if "RetentionRisk" in df.columns:
            # Try numeric, else map categories
            if df["RetentionRisk"].dtype == "object":
                mapping = {"Low": 1, "Medium": 2, "High": 3}
                df["RetentionRisk"] = df["RetentionRisk"].map(mapping).fillna(np.nan)
            else:
                df["RetentionRisk"] = pd.to_numeric(df["RetentionRisk"], errors="coerce")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Absences vs Retention Risk")
            if "RetentionRisk" in df.columns and "Absences" in df.columns:
                fig = px.scatter(
                    df, x="Absences", y="RetentionRisk", color="Department",
                    hover_data=["EmployeeID","JobTitle"], opacity=0.6
                )
                fig.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("RetentionRisk or Absences column missing.")

        with col2:
            st.subheader("Retention Risk by Department")
            if "RetentionRisk" in df.columns and "Department" in df.columns:
                risk = df.groupby("Department", as_index=False)["RetentionRisk"].mean()
                fig = px.bar(
                    risk, x="Department", y="RetentionRisk", text="RetentionRisk",
                    color="Department"
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig.update_layout(yaxis=dict(title="Avg Retention Risk"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("RetentionRisk or Department column missing.")

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Promotion Distribution")
            if "PromotionCount" in df.columns:
                promo = df["PromotionCount"].value_counts(normalize=True).reset_index()
                promo.columns = ["PromotionCount","Percent"]
                fig = px.bar(
                    promo, x="PromotionCount", y="Percent", text="Percent",
                    color="PromotionCount"
                )
                fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                fig.update_layout(yaxis=dict(title="Percentage"), bargap=0.3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("PromotionCount column missing.")

        with col4:
            st.subheader("Absences Distribution")
            if "Absences" in df.columns:
                fig = px.histogram(df, x="Absences", nbins=20, marginal="box")
                fig.update_traces(marker_color="blue", opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Absences column missing.")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Built with ❤️ — Expected Columns: EmployeeID, DateOfHire, DateOfBirth, Gender, Department, JobTitle, EmploymentStatus, PerformanceRating, Salary, TrainingHours, Absences, RetentionRisk, Engagement scores, etc.")
