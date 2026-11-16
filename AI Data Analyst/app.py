import json
import pandas as pd
import plotly.express as px
import ollama
import streamlit as st

from ai.code_generator import generate_code

MODEL = "llama3.2"

st.set_page_config(
    page_title="InsightAI",
    page_icon="📊",
    layout="wide"
)

# Independent persistent state for both assistants
DEFAULTS = {
    "datasets": {},
    "selected_dataset": None,
    "analysis_result": None,
    "analysis_explanation": None,
    "analysis_plan": None,
    "analysis_chart": None,
    "analysis_error": None,
    "code_assistant_answer": None,
    "code_assistant_error": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("📊 InsightAI")
st.caption("Local AI Data Analyst | Powered by Ollama")

# --------------------------------------------------
# 1. CSV UPLOAD AND DATASET SELECTION
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key="csv_upload"
)

if uploaded_files:
    for file in uploaded_files:
        try:
            file.seek(0)
            st.session_state.datasets[file.name] = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read {file.name}: {e}")

datasets = st.session_state.datasets
df = None

if datasets:
    names = list(datasets.keys())
    previous = st.session_state.selected_dataset
    default_index = names.index(previous) if previous in names else 0

    selected = st.selectbox(
        "Select a dataset",
        names,
        index=default_index,
        key="dataset_selector"
    )
    st.session_state.selected_dataset = selected
    df = datasets[selected]

    st.subheader(f"Dataset: {selected}")

    # --------------------------------------------------
    # 2. DATA PROFILING
    # --------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    with st.expander("View Dataset Preview"):
        st.dataframe(df.head(100), use_container_width=True)

    with st.expander("View Dataset Statistics"):
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    with st.expander("View Column Information"):
        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique().values
        })
        st.dataframe(column_info, use_container_width=True)
else:
    st.session_state.selected_dataset = None
    st.info("Upload one or more CSV files to use dataset analysis. Code Assistant works without a CSV.")

# --------------------------------------------------
# 3. ASK INSIGHTAI (DATA ANALYSIS)
# --------------------------------------------------
st.divider()
st.header("🔎 Ask InsightAI")

analysis_question = st.text_area(
    "Ask a question about your dataset",
    placeholder="Example: Show total sales by category",
    key="analysis_question_input"
)

if st.button("Analyze Dataset", type="primary", key="analyze_dataset"):
    if df is None:
        st.session_state.analysis_error = "Upload and select a CSV dataset before asking data questions."
        st.session_state.analysis_result = None
        st.session_state.analysis_explanation = None
        st.session_state.analysis_plan = None
        st.session_state.analysis_chart = None
    elif not analysis_question.strip():
        st.session_state.analysis_error = "Please enter a question."
    else:
        st.session_state.analysis_error = None
        st.session_state.analysis_result = None
        st.session_state.analysis_explanation = None
        st.session_state.analysis_plan = None
        st.session_state.analysis_chart = None

        schema = {
            "columns": [
                {"name": str(col), "type": str(df[col].dtype)}
                for col in df.columns
            ]
        }

        system_prompt = """
You are InsightAI, a data analyst. Convert the user's question into a JSON analysis plan.
Allowed operations: overview, groupby, sort, filter, chart, correlation.
Return ONLY valid JSON using this schema:
{
  "operation": "overview",
  "group_column": null,
  "value_column": null,
  "aggregation": "sum",
  "sort_column": null,
  "sort_order": "descending",
  "filter_column": null,
  "filter_operator": "equals",
  "filter_value": null,
  "chart_type": "bar"
}
Rules:
- Use only columns present in the dataset.
- Never generate Python code or SQL for execution.
- Never invent column names. Use overview if unclear.
- Aggregations: sum, mean, count, min, max.
- Filter operators: equals, greater_than, less_than, contains.
- Charts: bar, line, scatter, histogram.
"""
        try:
            with st.spinner("InsightAI is analyzing your dataset..."):
                response = ollama.chat(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Dataset schema:\n" + json.dumps(schema) + "\nQuestion:\n" + analysis_question}
                    ],
                    format="json"
                )
                plan = json.loads(response.message.content)
                st.session_state.analysis_plan = plan

                operation = plan.get("operation")
                valid_operations = {"overview", "groupby", "sort", "filter", "chart", "correlation"}
                if operation not in valid_operations:
                    raise ValueError(f"Unsupported operation: {operation}")

                result = None
                fig = None

                if operation == "overview":
                    result = df.describe(include="all").transpose()

                elif operation == "groupby":
                    group_column = plan.get("group_column")
                    value_column = plan.get("value_column")
                    aggregation = plan.get("aggregation")
                    if group_column not in df.columns:
                        raise ValueError("Invalid grouping column returned by the model.")
                    if aggregation not in {"sum", "mean", "count", "min", "max"}:
                        raise ValueError("Invalid aggregation returned by the model.")
                    if aggregation == "count":
                        result = df.groupby(group_column).size().reset_index(name="count")
                    else:
                        if value_column not in df.columns:
                            raise ValueError("Invalid value column returned by the model.")
                        if not pd.api.types.is_numeric_dtype(df[value_column]):
                            raise ValueError(f"'{value_column}' must be numeric for {aggregation}.")
                        result = df.groupby(group_column)[value_column].agg(aggregation).reset_index()

                elif operation == "sort":
                    sort_column = plan.get("sort_column")
                    if sort_column not in df.columns:
                        raise ValueError("Invalid sorting column returned by the model.")
                    result = df.sort_values(
                        sort_column,
                        ascending=(plan.get("sort_order") == "ascending")
                    )

                elif operation == "filter":
                    filter_column = plan.get("filter_column")
                    operator = plan.get("filter_operator")
                    filter_value = plan.get("filter_value")
                    if filter_column not in df.columns:
                        raise ValueError("Invalid filtering column returned by the model.")
                    series = df[filter_column]
                    if operator == "equals":
                        result = df[series.astype(str) == str(filter_value)]
                    elif operator == "greater_than":
                        result = df[pd.to_numeric(series, errors="coerce") > float(filter_value)]
                    elif operator == "less_than":
                        result = df[pd.to_numeric(series, errors="coerce") < float(filter_value)]
                    elif operator == "contains":
                        result = df[series.astype(str).str.contains(str(filter_value), case=False, na=False, regex=False)]
                    else:
                        raise ValueError("Invalid filter operator returned by the model.")

                elif operation == "chart":
                    chart_type = plan.get("chart_type")
                    group_column = plan.get("group_column")
                    value_column = plan.get("value_column")
                    if group_column not in df.columns:
                        raise ValueError("Invalid chart/category column returned by the model.")
                    if chart_type == "histogram":
                        fig = px.histogram(df, x=group_column)
                    elif chart_type in {"bar", "line", "scatter"}:
                        if value_column not in df.columns:
                            raise ValueError("Invalid chart value column returned by the model.")
                        if chart_type == "bar":
                            fig = px.bar(df, x=group_column, y=value_column)
                        elif chart_type == "line":
                            fig = px.line(df, x=group_column, y=value_column)
                        else:
                            fig = px.scatter(df, x=group_column, y=value_column)
                    else:
                        raise ValueError("Invalid chart type returned by the model.")

                elif operation == "correlation":
                    numeric_df = df.select_dtypes(include="number")
                    if numeric_df.empty:
                        raise ValueError("This dataset has no numeric columns for correlation.")
                    result = numeric_df.corr()

                st.session_state.analysis_result = result
                st.session_state.analysis_chart = fig

                if result is not None:
                    with st.spinner("Generating a plain-English explanation..."):
                        explanation = ollama.chat(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "You are a data analyst. Explain results in simple English. Do not invent facts."},
                                {"role": "user", "content": "Question: " + analysis_question + "\nResults:\n" + result.head(20).to_string()}
                            ]
                        )
                        st.session_state.analysis_explanation = explanation.message.content
                elif fig is not None:
                    st.session_state.analysis_explanation = "The chart is shown below. Review its axes and values to interpret the pattern."
        except Exception as e:
            st.session_state.analysis_error = f"Analysis failed: {e}"

# Persist and display Ask InsightAI's output independently of button clicks
if st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)

if st.session_state.analysis_plan is not None:
    with st.expander("AI Analysis Plan", expanded=False):
        st.json(st.session_state.analysis_plan)

if st.session_state.analysis_chart is not None:
    st.subheader("Interactive Chart")
    st.plotly_chart(st.session_state.analysis_chart, use_container_width=True)

if st.session_state.analysis_result is not None:
    st.subheader("Analysis Results")
    st.dataframe(st.session_state.analysis_result, use_container_width=True)

if st.session_state.analysis_explanation:
    st.subheader("AI Explanation")
    st.markdown(st.session_state.analysis_explanation)

# --------------------------------------------------
# 4. INSIGHTAI CODE ASSISTANT
# --------------------------------------------------
st.divider()
st.header("💻 InsightAI Code Assistant")
st.caption("Generate SQL, DAX, Excel, Python, ML, Deep Learning, NLP, and Generative AI code locally.")

domain = st.selectbox(
    "Select Programming Domain",
    [
        "SQL", "Power BI DAX", "Power Query M", "Excel Formulas",
        "Excel VBA", "Excel Office Scripts", "Python", "Pandas",
        "Machine Learning", "Deep Learning", "NLP", "Generative AI"
    ],
    key="code_domain"
)

task = st.selectbox(
    "Select Task",
    ["Generate Code", "Explain Code", "Debug Code", "Optimize Code", "Convert Code", "Build Complete Workflow"],
    key="code_task"
)

code_question = st.text_area(
    "Describe what you want",
    placeholder="Example: Generate a SQL query to calculate monthly revenue by category.",
    height=150,
    key="code_question_input"
)

dataset_columns = list(df.columns) if df is not None else []

if st.button("Generate Code", type="primary", key="generate_code"):
    if not code_question.strip():
        st.session_state.code_assistant_error = "Enter your coding question."
    else:
        st.session_state.code_assistant_error = None
        try:
            with st.spinner("Local AI is generating your code..."):
                st.session_state.code_assistant_answer = generate_code(
                    domain=domain,
                    task=task,
                    question=code_question,
                    dataset_columns=dataset_columns
                )
        except Exception as e:
            st.session_state.code_assistant_error = f"Code generation failed: {e}"

# Persist and display Code Assistant output independently
if st.session_state.code_assistant_error:
    st.error(st.session_state.code_assistant_error)

if st.session_state.code_assistant_answer:
    st.subheader("Generated Solution")
    st.markdown(st.session_state.code_assistant_answer)
    st.download_button(
        label="Download Generated Code",
        data=st.session_state.code_assistant_answer,
        file_name="insightai_generated_code.md",
        mime="text/markdown",
        key="download_generated_code"
    )
