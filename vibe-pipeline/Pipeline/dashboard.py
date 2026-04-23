import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="LLM Code Evaluation Dashboard",
    layout="wide"
)

# ---------------------------
# FILE PATHS
# ---------------------------
SCORESHEET = "Scoring/scoresheet.csv"
SONAR_ISSUES = "Scoring/sonar_issues.csv"
ANALYSIS_DIR = "Scoring/analysis"

SYNTAX_SUMMARY = os.path.join(ANALYSIS_DIR, "syntax_summary.csv")
ISSUE_COUNTS = os.path.join(ANALYSIS_DIR, "issue_counts.csv")
SEVERITY_BREAKDOWN = os.path.join(ANALYSIS_DIR, "severity_breakdown.csv")
COMBINED_METRICS = os.path.join(ANALYSIS_DIR, "combined_metrics.csv")
ISSUE_TYPE_TABLE = os.path.join(ANALYSIS_DIR, "issue_type_vs_model.csv")


# ---------------------------
# HELPERS
# ---------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def safe_load(path, name):
    if not os.path.exists(path):
        st.error(f"{name} not found")
        st.stop()
    return load_csv(path)


def clean_columns(df, first_col_name=None):
    df.columns = df.columns.str.strip()

    if len(df.columns) > 0 and df.columns[0].startswith("Unnamed"):
        if first_col_name:
            df.rename(columns={df.columns[0]: first_col_name}, inplace=True)
        else:
            df.rename(columns={df.columns[0]: "model"}, inplace=True)

    return df


def get_model_row(df, selected_model):
    if "model" not in df.columns:
        return None
    rows = df[df["model"] == selected_model]
    if rows.empty:
        return None
    return rows.iloc[0]


def classify_score(score):
    if score >= 80:
        return "#0CCE6B"
    if score >= 50:
        return "#FFA400"
    return "#FF4E42"


def make_score_circle(label, score, raw_text):
    score = max(0, min(100, int(round(score))))
    colour = classify_score(score)

    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:10px 0;">
        <div style="
            width:115px;
            height:115px;
            border-radius:50%;
            border:8px solid {colour};
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:32px;
            font-weight:700;
            color:{colour};
            background-color:white;
            margin-bottom:10px;
        ">
            {score}
        </div>
        <div style="
            font-size:16px;
            font-weight:600;
            text-align:center;
            color:#202124;
            margin-bottom:4px;
        ">
            {label}
        </div>
        <div style="
            font-size:13px;
            text-align:center;
            color:#5f6368;
        ">
            {raw_text}
        </div>
    </div>
    """


def build_takeaway_text(model_name, validity_rate, issues_per_valid_file, total_issues):
    return (
        f"**Key takeaway:** {model_name} achieved **{validity_rate:.1f}%** valid outputs, "
        f"with **{issues_per_valid_file:.2f} issues per valid file** and "
        f"**{total_issues} total issues**."
    )


# ---------------------------
# LOAD DATA
# ---------------------------
scores_df = clean_columns(safe_load(SCORESHEET, "scoresheet.csv"))
sonar_df = clean_columns(safe_load(SONAR_ISSUES, "sonar_issues.csv"))
syntax_summary_df = clean_columns(safe_load(SYNTAX_SUMMARY, "syntax_summary.csv"))
issue_counts_df = clean_columns(safe_load(ISSUE_COUNTS, "issue_counts.csv"))
severity_breakdown_df = clean_columns(safe_load(SEVERITY_BREAKDOWN, "severity_breakdown.csv"))
combined_metrics_df = clean_columns(safe_load(COMBINED_METRICS, "combined_metrics.csv"))
issue_type_table_df = clean_columns(safe_load(ISSUE_TYPE_TABLE, "issue_type_vs_model.csv"), first_col_name="issue_name")

# Fix syntax_valid booleans if present
if "syntax_valid" in scores_df.columns:
    scores_df["syntax_valid"] = (
        scores_df["syntax_valid"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

# Ensure model column exists
for df in [syntax_summary_df, issue_counts_df, severity_breakdown_df, combined_metrics_df]:
    if "model" not in df.columns and len(df.columns) > 0:
        df.rename(columns={df.columns[0]: "model"}, inplace=True)

if "issue_name" not in issue_type_table_df.columns and len(issue_type_table_df.columns) > 0:
    issue_type_table_df.rename(columns={issue_type_table_df.columns[0]: "issue_name"}, inplace=True)


# ---------------------------
# PAGE HEADER
# ---------------------------
st.title("LLM Code Evaluation Dashboard")
st.caption("Comparison of syntax validity and static analysis findings for LLM-generated Python code.")


# ---------------------------
# SIDEBAR
# ---------------------------
available_models = sorted(scores_df["model"].dropna().unique().tolist()) if "model" in scores_df.columns else []

if not available_models:
    st.error("No models found.")
    st.stop()

st.sidebar.header("Controls")

selected_model = st.sidebar.selectbox(
    "Focus model",
    options=available_models,
    index=0
)

comparison_models = st.sidebar.multiselect(
    "Models to compare",
    options=available_models,
    default=available_models
)

if not comparison_models:
    comparison_models = available_models


# ---------------------------
# FILTER DATA
# ---------------------------
filtered_combined_df = combined_metrics_df.copy()
if "model" in filtered_combined_df.columns:
    filtered_combined_df = filtered_combined_df[filtered_combined_df["model"].isin(comparison_models)]

filtered_sonar_df = sonar_df.copy()
if "model" in filtered_sonar_df.columns:
    filtered_sonar_df = filtered_sonar_df[filtered_sonar_df["model"] == selected_model]

filtered_severity_df = severity_breakdown_df.copy()
if "model" in filtered_severity_df.columns:
    filtered_severity_df = filtered_severity_df[filtered_severity_df["model"] == selected_model]

filtered_issue_counts_df = issue_counts_df.copy()
if "model" in filtered_issue_counts_df.columns:
    filtered_issue_counts_df = filtered_issue_counts_df[filtered_issue_counts_df["model"] == selected_model]

filtered_issue_type_df = issue_type_table_df.copy()
if "issue_name" in filtered_issue_type_df.columns:
    keep_cols = ["issue_name"] + [m for m in comparison_models if m in filtered_issue_type_df.columns]
    filtered_issue_type_df = filtered_issue_type_df[keep_cols]


# ---------------------------
# FOCUS MODEL METRICS
# ---------------------------
focus_combined_row = get_model_row(combined_metrics_df, selected_model)
focus_syntax_row = get_model_row(syntax_summary_df, selected_model)

validity_rate = 0.0
issues_per_valid_file = 0.0
total_issues_model = 0
invalid_outputs = 0

if focus_combined_row is not None:
    validity_rate = float(focus_combined_row.get("valid_rate", 0)) * 100
    issues_per_valid_file = float(focus_combined_row.get("issues_per_valid_file", 0))
    total_issues_model = int(focus_combined_row.get("issue_count", 0))

if focus_syntax_row is not None and "invalid_files" in focus_syntax_row.index:
    invalid_outputs = int(focus_syntax_row.get("invalid_files", 0))

# ---------------------------
# LIGHTHOUSE-STYLE SCORES
# Higher score = better for all circles
# ---------------------------
max_issue_density = combined_metrics_df["issues_per_valid_file"].max() if "issues_per_valid_file" in combined_metrics_df.columns else 1
max_total_issues = combined_metrics_df["issue_count"].max() if "issue_count" in combined_metrics_df.columns else 1

if pd.isna(max_issue_density) or max_issue_density == 0:
    max_issue_density = 1

if pd.isna(max_total_issues) or max_total_issues == 0:
    max_total_issues = 1

valid_outputs_score = validity_rate
issues_per_file_score = max(0, 100 * (1 - (issues_per_valid_file / max_issue_density)))
total_issues_score = max(0, 100 * (1 - (total_issues_model / max_total_issues)))

# ---------------------------
# HERO SECTION
# ---------------------------
st.subheader(f"Focus Model: {selected_model}")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        make_score_circle(
            "Valid Outputs",
            valid_outputs_score,
            f"{validity_rate:.1f}% valid outputs"
        ),
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        make_score_circle(
            "Issues per Valid File",
            issues_per_file_score,
            f"{issues_per_valid_file:.2f} issues per valid file"
        ),
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        make_score_circle(
            "Total Issues",
            total_issues_score,
            f"{total_issues_model} total issues"
        ),
        unsafe_allow_html=True
    )

st.markdown(
    "<div style='text-align:center; color:#5f6368; font-size:14px; margin-bottom:18px;'>Higher scores indicate better relative performance across the evaluated models.</div>",
    unsafe_allow_html=True
)

st.info(build_takeaway_text(selected_model, validity_rate, issues_per_valid_file, total_issues_model))

st.divider()


# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Severity Breakdown",
    "Issue Details"
])


# ---------------------------
# TAB 1: OVERVIEW
# ---------------------------
with tab1:
    st.subheader("Model Comparison")

    comparison_metric = st.radio(
        "Choose comparison metric",
        ["Valid Outputs (%)", "Issues per Valid File", "Total Issues"],
        horizontal=True
    )

    fig, ax = plt.subplots()

    if comparison_metric == "Valid Outputs (%)" and "valid_rate" in filtered_combined_df.columns:
        plot_df = filtered_combined_df[["model", "valid_rate"]].copy()
        plot_df["metric_value"] = plot_df["valid_rate"] * 100
        ylabel = "Percentage"
        title = "Valid Outputs by Model"

    elif comparison_metric == "Issues per Valid File" and "issues_per_valid_file" in filtered_combined_df.columns:
        plot_df = filtered_combined_df[["model", "issues_per_valid_file"]].copy()
        plot_df["metric_value"] = plot_df["issues_per_valid_file"]
        ylabel = "Issues per Valid File"
        title = "Issues per Valid File by Model"

    else:
        plot_df = filtered_combined_df[["model", "issue_count"]].copy()
        plot_df["metric_value"] = plot_df["issue_count"]
        ylabel = "Issue Count"
        title = "Total Issues by Model"

    bar_colours = ["#1f77b4" if model != selected_model else "#0CCE6B" for model in plot_df["model"]]
    ax.bar(plot_df["model"], plot_df["metric_value"], color=bar_colours)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Combined Metrics Table")
    st.dataframe(filtered_combined_df, use_container_width=True, height=250)


# ---------------------------
# TAB 2: SEVERITY BREAKDOWN
# ---------------------------
with tab2:
    st.subheader(f"Severity Breakdown for {selected_model}")

    if not filtered_severity_df.empty and "model" in filtered_severity_df.columns:
        severity_row = filtered_severity_df.iloc[0]
        severity_cols = [col for col in filtered_severity_df.columns if col != "model"]

        severity_names = []
        severity_values = []

        for col in severity_cols:
            if pd.api.types.is_numeric_dtype(filtered_severity_df[col]):
                severity_names.append(col)
                severity_values.append(severity_row[col])

        if severity_names:
            fig, ax = plt.subplots()
            ax.bar(severity_names, severity_values)
            ax.set_ylabel("Issue Count")
            ax.set_title("Issues by Severity")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Severity Table")
        st.dataframe(filtered_severity_df, use_container_width=True, height=220)

    with right:
        st.subheader("Issue Count Summary")
        st.dataframe(filtered_issue_counts_df, use_container_width=True, height=220)


# ---------------------------
# TAB 3: ISSUE DETAILS
# ---------------------------
with tab3:
    st.subheader(f"Detailed SonarQube Issues for {selected_model}")

    if filtered_sonar_df.empty:
        st.info("No issue records found for this model.")
    else:
        useful_cols = [col for col in ["severity", "message", "file", "line", "rule"] if col in filtered_sonar_df.columns]
        if useful_cols:
            st.dataframe(filtered_sonar_df[useful_cols], use_container_width=True, height=350)
        else:
            st.dataframe(filtered_sonar_df, use_container_width=True, height=350)

    if not filtered_issue_type_df.empty and "issue_name" in filtered_issue_type_df.columns:
        st.subheader("Top Issue Types Across Selected Models")

        plot_df = filtered_issue_type_df.copy()
        model_cols = [c for c in plot_df.columns if c != "issue_name"]

        if model_cols:
            plot_df["total"] = plot_df[model_cols].sum(axis=1)
            plot_df = plot_df.sort_values("total", ascending=False).head(10)

            chart_df = plot_df[["issue_name", "total"]].sort_values("total", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(chart_df["issue_name"], chart_df["total"])
            ax.set_xlabel("Count")
            ax.set_title("Top Issue Types")
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(plot_df.drop(columns=["total"]), use_container_width=True, height=250)