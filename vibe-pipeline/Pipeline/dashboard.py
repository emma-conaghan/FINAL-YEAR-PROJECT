import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# PAGE SETUP
st.set_page_config(
    page_title="LLM Code Evaluation Dashboard",
    layout="wide"
)

# FILE PATHS
SCORESHEET = "Scoring/scoresheet.csv"
SONAR_ISSUES = "Scoring/sonar_issues.csv"
ANALYSIS_DIR = "Scoring/analysis"

SYNTAX_SUMMARY = os.path.join(ANALYSIS_DIR, "syntax_summary.csv")
ISSUE_COUNTS = os.path.join(ANALYSIS_DIR, "issue_counts.csv")
SEVERITY_BREAKDOWN = os.path.join(ANALYSIS_DIR, "severity_breakdown.csv")
COMBINED_METRICS = os.path.join(ANALYSIS_DIR, "combined_metrics.csv")
ISSUE_TYPE_TABLE = os.path.join(ANALYSIS_DIR, "issue_type_vs_model.csv")
 
# HELPERS
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def safe_load(path, friendly_name):
    if not os.path.exists(path):
        st.error(f"{friendly_name} not found at: {path}")
        st.stop()
    return load_csv(path)

# LOAD DATA
scores_df = safe_load(SCORESHEET, "scoresheet.csv")
sonar_df = safe_load(SONAR_ISSUES, "sonar_issues.csv")
syntax_summary_df = safe_load(SYNTAX_SUMMARY, "syntax_summary.csv")
issue_counts_df = safe_load(ISSUE_COUNTS, "issue_counts.csv")
severity_breakdown_df = safe_load(SEVERITY_BREAKDOWN, "severity_breakdown.csv")
combined_metrics_df = safe_load(COMBINED_METRICS, "combined_metrics.csv")
issue_type_table_df = safe_load(ISSUE_TYPE_TABLE, "issue_type_vs_model.csv")

# CLEAN COLUMNS
scores_df.columns = scores_df.columns.str.strip()
sonar_df.columns = sonar_df.columns.str.strip()
syntax_summary_df.columns = syntax_summary_df.columns.str.strip()
issue_counts_df.columns = issue_counts_df.columns.str.strip()
severity_breakdown_df.columns = severity_breakdown_df.columns.str.strip()
combined_metrics_df.columns = combined_metrics_df.columns.str.strip()
issue_type_table_df.columns = issue_type_table_df.columns.str.strip()

# Make sure "model" is a proper column after CSV export
if "model" not in syntax_summary_df.columns and syntax_summary_df.columns[0] != "model":
    syntax_summary_df = syntax_summary_df.reset_index()

if "model" not in issue_counts_df.columns and issue_counts_df.columns[0] != "model":
    issue_counts_df = issue_counts_df.reset_index()

if "model" not in severity_breakdown_df.columns and severity_breakdown_df.columns[0] != "model":
    severity_breakdown_df = severity_breakdown_df.reset_index()

if "model" not in combined_metrics_df.columns and combined_metrics_df.columns[0] != "model":
    combined_metrics_df = combined_metrics_df.reset_index()

# If pandas saved the index as the first unnamed column, rename it
for df in [syntax_summary_df, issue_counts_df, severity_breakdown_df, combined_metrics_df]:
    if df.columns[0].startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "model"}, inplace=True)

if issue_type_table_df.columns[0].startswith("Unnamed"):
    issue_type_table_df.rename(columns={issue_type_table_df.columns[0]: "issue_name"}, inplace=True)

# Convert booleans if needed
if "syntax_valid" in scores_df.columns:
    scores_df["syntax_valid"] = scores_df["syntax_valid"].astype(str).str.strip().str.lower().map({
        "true": True,
        "false": False
    })

# TITLE
st.title("LLM Code Evaluation Dashboard")
st.markdown("### Automated Comparative Analysis of Multi-Model Code Generation")
st.caption(
    "This dashboard presents results from an automated evaluation pipeline "
    "for LLM-generated Python code, including syntax validity, SonarQube "
    "issue counts, severity breakdowns, and rule-level issue distributions."
)
st.divider()

# TOP METRICS
total_runs = len(scores_df)
valid_runs = int(scores_df["syntax_valid"].sum()) if "syntax_valid" in scores_df.columns else 0
valid_rate = (valid_runs / total_runs) if total_runs > 0 else 0
total_issues = len(sonar_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Runs Logged", total_runs)
col2.metric("Valid Python Outputs", valid_runs)
col3.metric("Syntax Validity Rate", f"{valid_rate:.1%}")
col4.metric("Total Sonar Issues", total_issues)

st.divider()

# SIDEBAR FILTERS
st.sidebar.header("Filters")
st.sidebar.caption("Filter dashboard views by model.")

available_models = sorted(scores_df["model"].dropna().unique().tolist()) if "model" in scores_df.columns else []
selected_models = st.sidebar.multiselect(
    "Choose model(s)",
    options=available_models,
    default=available_models
)

# FILTER DATA
filtered_scores_df = scores_df.copy()
if selected_models and "model" in filtered_scores_df.columns:
    filtered_scores_df = filtered_scores_df[filtered_scores_df["model"].isin(selected_models)]

filtered_combined_df = combined_metrics_df.copy()
if selected_models and "model" in filtered_combined_df.columns:
    filtered_combined_df = filtered_combined_df[filtered_combined_df["model"].isin(selected_models)]

filtered_issue_counts_df = issue_counts_df.copy()
if selected_models and "model" in filtered_issue_counts_df.columns:
    filtered_issue_counts_df = filtered_issue_counts_df[filtered_issue_counts_df["model"].isin(selected_models)]

filtered_severity_df = severity_breakdown_df.copy()
if selected_models and "model" in filtered_severity_df.columns:
    filtered_severity_df = filtered_severity_df[filtered_severity_df["model"].isin(selected_models)]

filtered_syntax_df = syntax_summary_df.copy()
if selected_models and "model" in filtered_syntax_df.columns:
    filtered_syntax_df = filtered_syntax_df[filtered_syntax_df["model"].isin(selected_models)]

filtered_issue_type_df = issue_type_table_df.copy()
issue_type_model_cols = [c for c in filtered_issue_type_df.columns if c != "issue_name"]
keep_cols = ["issue_name"] + [c for c in issue_type_model_cols if c in selected_models]
filtered_issue_type_df = filtered_issue_type_df[keep_cols]

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Syntax & Reliability",
    "Security Findings",
    "Issue Types",
    "Raw Data"
])

# TAB 1: OVERVIEW
with tab1:
    st.subheader("Combined Model Metrics")
    st.dataframe(filtered_combined_df, use_container_width=True, height=300)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Syntax Validity Rate")
        if (
            not filtered_combined_df.empty
            and "model" in filtered_combined_df.columns
            and "valid_rate" in filtered_combined_df.columns
        ):
            fig, ax = plt.subplots()
            ax.bar(filtered_combined_df["model"], filtered_combined_df["valid_rate"])
            ax.set_ylabel("Valid Rate")
            ax.set_xlabel("Model")
            ax.set_title("Syntax Validity Rate by Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

    with col_right:
        st.subheader("Issues per Valid File")
        if (
            not filtered_combined_df.empty
            and "model" in filtered_combined_df.columns
            and "issues_per_valid_file" in filtered_combined_df.columns
        ):
            fig, ax = plt.subplots()
            ax.bar(filtered_combined_df["model"], filtered_combined_df["issues_per_valid_file"])
            ax.set_ylabel("Issues per Valid File")
            ax.set_xlabel("Model")
            ax.set_title("Issues per Valid File by Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# TAB 2: SYNTAX & RELIABILITY
with tab2:
    st.subheader("Syntax Summary")
    st.dataframe(filtered_syntax_df, use_container_width=True, height=250)

    st.divider()

    st.subheader("Scoresheet Entries")
    st.dataframe(filtered_scores_df, use_container_width=True, height=400)

    if (
        not filtered_scores_df.empty
        and "model" in filtered_scores_df.columns
        and "syntax_valid" in filtered_scores_df.columns
    ):
        syntax_counts = (
            filtered_scores_df.groupby(["model", "syntax_valid"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        st.divider()
        st.subheader("Valid vs Invalid Outputs")
        st.dataframe(syntax_counts, use_container_width=True, height=200)

# TAB 3: SECURITY FINDINGS
with tab3:
    st.subheader("Issue Counts by Model")
    st.dataframe(filtered_issue_counts_df, use_container_width=True, height=250)

    st.divider()

    st.subheader("Severity Breakdown")
    st.dataframe(filtered_severity_df, use_container_width=True, height=300)

    if not filtered_severity_df.empty and "model" in filtered_severity_df.columns:
        plot_df = filtered_severity_df.set_index("model")
        numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            fig, ax = plt.subplots()
            plot_df[numeric_cols].plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Issue Count")
            ax.set_xlabel("Model")
            ax.set_title("Severity Breakdown by Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# TAB 4: ISSUE TYPES
with tab4:
    st.subheader("Issue Type vs Model")
    st.dataframe(filtered_issue_type_df, use_container_width=True, height=500)

    st.divider()

    top_n = st.slider("Show top N issue types", min_value=5, max_value=50, value=20, step=5)

    chart_df = filtered_issue_type_df.copy()
    model_cols = [c for c in chart_df.columns if c != "issue_name"]

    if model_cols:
        chart_df["total"] = chart_df[model_cols].sum(axis=1)
        chart_df = chart_df.sort_values("total", ascending=False).head(top_n)

        st.subheader(f"Top {top_n} Issue Types")
        st.dataframe(chart_df.drop(columns=["total"]), use_container_width=True, height=400)

        plot_df = chart_df[["issue_name", "total"]].sort_values("total", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(plot_df["issue_name"], plot_df["total"])
        ax.set_xlabel("Total Issue Count")
        ax.set_ylabel("Issue Type")
        ax.set_title(f"Top {top_n} Issue Types Across Selected Models")
        plt.tight_layout()
        st.pyplot(fig)

# TAB 5: RAW DATA
with tab5:
    st.subheader("Raw scoresheet.csv")
    st.dataframe(scores_df, use_container_width=True, height=350)

    st.divider()

    st.subheader("Raw sonar_issues.csv")
    st.dataframe(sonar_df, use_container_width=True, height=350)

    st.divider()

    st.subheader("Download Analysis Files")
    for file_name in [
        "syntax_summary.csv",
        "issue_counts.csv",
        "severity_breakdown.csv",
        "combined_metrics.csv",
        "issue_type_vs_model.csv"
    ]:
        full_path = os.path.join(ANALYSIS_DIR, file_name)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                st.download_button(
                    label=f"Download {file_name}",
                    data=f,
                    file_name=file_name,
                    mime="text/csv"
                )

st.divider()
st.caption(
    "Dashboard generated from the automated LLM evaluation pipeline. "
    "Results shown here are based on exported CSV outputs from syntax validation, "
    "SonarQube issue extraction, and downstream analysis scripts."
)