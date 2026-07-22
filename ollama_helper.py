from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import re
import json
import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

model = OllamaLLM(model="llama3.2", base_url=OLLAMA_BASE_URL, num_ctx=4096, num_thread=12, streaming=True)

BASE_TEMPLATE = """You are a data analyst helping non-technical users on the Aplo Data Analytics Platform.

Dataset Information:
{dataset_info}

Analysis Result:
{analysis_result}

User Question: {question}

Rules:
- Answer directly and concisely. For simple questions, keep it to 1-3 sentences.
- Do NOT generate charts unless explicitly asked. Do NOT include <chart> blocks unless the user clearly requests a visualization.
- Do NOT describe charts/visualizations unless asked. Just state numbers directly.
- Do not mention ML unless the user explicitly asks about prediction, classification, regression, or models.
- Do not invent columns or capabilities not in the dataset.
- Address the user as "you". Do not repeat the question.
- If you provide follow-up questions, format them as:
Follow-up questions:
• First question
• Second question
Keep language simple. No hallucinations.
"""

CHART_TEMPLATE = """You are a data analyst. You MUST output a chart configuration.

Dataset columns: {dataset_info}

User request: {question}

INSTRUCTIONS:
1. Pick the correct column names from the dataset above.
2. Output EXACTLY this format (the chart block must be on ONE line, at the VERY END of your response):

<chart>{"chart_type": "bar", "x": "region", "y": ["sales_revenue_usd"], "agg": "sum"}</chart>

The text BEFORE the chart block should be 1-2 sentences explaining what the chart shows.
The chart block <chart>...</chart> MUST be the very last thing in your response. Nothing after it.

Valid chart_type values: bar, line, scatter, pie, gauge
Valid agg values: "sum", "mean", "median", null
x = single column name for the X axis (not required for gauge)
y = JSON array of numeric column names for the Y axis

For GAUGE charts, add these optional fields:
- time_period: "last_week", "last_month", "last_quarter", "last_year", "7d", "30d", "3m", "6m", "1y", or "all"
- target_source: "previous_period" (auto-compute from previous period) or "manual"
- target: numeric value (only if target_source is "manual")
- factor: multiplier or additive value (default 1.0)
- factor_mode: "multiply" or "add" (default "multiply")
- target_period: "last_week", "last_month", "last_quarter", "last_year" (which previous period to compare against; defaults to time_period if not specified)
- target_agg: "sum", "mean", or "median" (how to aggregate target period values; defaults to "sum")

Example gauge with previous period target (compares to last year):
<chart>{"chart_type": "gauge", "y": ["sales_revenue_usd"], "agg": "sum", "time_period": "last_month", "target_source": "previous_period", "target_period": "last_year", "target_agg": "sum", "factor": 1.0, "factor_mode": "multiply"}</chart>

Example gauge with manual target:
<chart>{"chart_type": "gauge", "y": ["sales_revenue_usd"], "agg": "sum", "target_source": "manual", "target": 500000}</chart>

IMPORTANT: The <chart> block is REQUIRED. You MUST include it. Output it exactly as shown above with real column names from the dataset."""

ML_TEMPLATE = """You are a data analyst helping non-technical users on the Aplo Data Analytics Platform.

Dataset Information:
{dataset_info}

User Question: {question}

Task: Answer the user's machine learning question.

Available ML models:
- Regression: Linear Regression, Ridge Regression, Gradient Boosting Regressor, Decision Tree Regressor, K-Neighbors Regressor
- Classification: Logistic Regression, Decision Tree Classifier
- Time-Series: ARIMAX, XGBoost

Guidance:
- Identify target column (y) and input features (X).
- Classify as regression (numeric target) or classification (categorical target).
- Avoid using ID columns as features.
- Explain R², MAPE, MAE, RMSE in simple terms when relevant.
- Only recommend models listed above. Do NOT mention Random Forest, SVM, Neural Networks, etc.
- Address the user as "you". Keep language simple. No hallucinations.
"""

base_prompt = ChatPromptTemplate.from_template(BASE_TEMPLATE)
chart_prompt = ChatPromptTemplate.from_template(CHART_TEMPLATE)
ml_prompt = ChatPromptTemplate.from_template(ML_TEMPLATE)

base_chain = base_prompt | model
chart_chain = chart_prompt | model
ml_chain = ml_prompt | model


def detect_query_type(question: str) -> bool:
    analytical_keywords = [
        'average', 'mean', 'sum', 'total', 'count', 'how many',
        'maximum', 'minimum', 'median', 'percentage', 'trend',
        'correlation', 'distribution', 'group by', 'aggregate',
        'compare', 'comparison', 'difference', 'highest', 'lowest',
        'overview', 'summary', 'describe'
    ]
    q = question.lower()
    return any(k in q for k in analytical_keywords)


def detect_ml_intent(question: str) -> bool:
    ml_keywords = [
        "machine learning", "ml", "predict", "prediction", "target column",
        "feature columns", "classification", "regression",
        "model", "models", "train model", "trained model"
    ]
    q = question.lower()
    return any(k in q for k in ml_keywords)


def detect_chart_intent(question: str) -> bool:
    chart_keywords = [
        "chart", "graph", "plot", "bar chart", "line chart",
        "scatter plot", "visualize", "visualise",
        "draw a chart", "draw a graph", "draw a plot",
        "create a chart", "create a graph", "create a plot",
        "generate a chart", "generate a graph", "generate a plot",
        "make a chart", "make a graph", "make a plot",
        "show me a chart", "show me a graph", "show me a plot",
        "display a chart", "display a graph", "display a plot"
    ]
    q = question.lower()
    return any(k in q for k in chart_keywords)


def parse_chart_from_response(text: str, question: str = "", available_columns: list[str] = None) -> dict | None:
    """Extract chart parameters from LLM response. Handles multiple formats."""
    # Try 1: standard <chart>{JSON}</chart> with optional spaces in tags
    match = re.search(r'<\s*chart\s*>(.*?)<\s*/\s*chart\s*>', text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        content = re.sub(r'^```\w*\s*', '', content).strip()
        content = re.sub(r'\s*```$', '', content).strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed.get("y"), str):
                parsed["y"] = [parsed["y"]]
            return parsed
        except Exception:
            pass

    # Try 2: find a JSON object with chart_type key anywhere in the response
    json_pattern = r'\{[^{}]*"chart_type"\s*:\s*"[^"]*"[^{}]*\}'
    json_match = re.search(json_pattern, text, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed.get("y"), str):
                parsed["y"] = [parsed["y"]]
            return parsed
        except Exception:
            pass

    # Try 3: extract key-value pairs from malformed XML-like output
    content = text
    try:
        chart_type_m = re.search(r'chart\s*_?\s*type\s*[=>:]\s*(\w+)', content)
        x_m = re.search(r'<\s*x\s*>\s*(\S+)', content)
        y_m = re.search(r'<\s*y\s*>\s*\[?\s*(\S+?)\s*\]?\s*</', content)
        agg_m = re.search(r'<\s*agg\s*>\s*(\w+|null)', content)

        if chart_type_m and x_m and y_m:
            result = {
                "chart_type": chart_type_m.group(1).strip(),
                "x": x_m.group(1).strip().rstrip(","),
                "y": [y_m.group(1).strip().rstrip(",")],
                "agg": agg_m.group(1).strip() if agg_m else None,
            }
            if result["agg"] == "null":
                result["agg"] = None
            return result
    except Exception:
        pass

    # Try 4: infer chart params from user question and response text using heuristics
    if available_columns:
        return _infer_chart_from_text(text, question, available_columns)

    return None


def _infer_chart_from_text(text: str, question: str, columns: list[str]) -> dict | None:
    """Infer chart parameters from question/response text using keyword heuristics."""
    combined = (question + " " + text).lower()
    col_lower = {c.lower().replace("_", " "): c for c in columns}
    q_lower = question.lower()

    # Detect chart type
    chart_type = "bar"
    if "line" in combined:
        chart_type = "line"
    elif "scatter" in combined:
        chart_type = "scatter"
    elif "pie" in combined:
        chart_type = "pie"
    elif "gauge" in combined:
        chart_type = "gauge"

    # Detect X axis (look for "by <column>" or "over time" pattern)
    x_col = None

    if "over time" in combined or "over period" in combined or "by date" in combined or "by time" in combined:
        for col_key, col_name in col_lower.items():
            if any(kw in col_key for kw in ["date", "time", "month", "year", "day"]):
                x_col = col_name
                break
    else:
        for phrase in ["by ", "per ", "across ", "for each "]:
            idx = combined.find(phrase)
            if idx >= 0:
                after = combined[idx + len(phrase):]
                word_match = re.match(r'(\w+(?:\s+\w+)?)', after)
                if word_match:
                    candidate = word_match.group(1).strip()
                    for col_key, col_name in col_lower.items():
                        if candidate == col_key or candidate in col_key or col_key.startswith(candidate):
                            x_col = col_name
                            break
                    if not x_col:
                        for w in candidate.split():
                            for col_key, col_name in col_lower.items():
                                if w in col_key:
                                    x_col = col_name
                                    break
                            if x_col:
                                break
                break

    # Detect Y axis — prefer the most specific numeric column mentioned in question
    y_cols = []

    # Priority 1: exact column name match (full name or with underscores removed)
    for col in columns:
        if col == x_col:
            continue
        col_normalized = col.lower().replace("_", "")
        q_normalized = q_lower.replace(" ", "")
        if col_normalized in q_normalized or col.lower() in q_lower.replace(" ", ""):
            y_cols.append(col)

    # Priority 2: specific compound keywords (e.g., "ad spend online" → ad_spend_online_usd)
    if not y_cols:
        specific_map = [
            ("ad spend online", "ad_spend_online"),
            ("ad spend offline", "ad_spend_offline"),
            ("marketing budget", "marketing_budget"),
            ("conversion rate", "conversion_rate"),
            ("website traffic", "website_traffic"),
            ("social media", "social_media"),
            ("customer satisfaction", "customer_satisfaction"),
            ("competitor price", "competitor_price"),
            ("email open", "email_open"),
            ("customer age", "customer_age"),
        ]
        for phrase, col_prefix in specific_map:
            if phrase in q_lower:
                for col in columns:
                    if col == x_col:
                        continue
                    if col.lower().startswith(col_prefix):
                        y_cols.append(col)
                        break

    # Priority 3: general numeric keywords
    if not y_cols:
        general_keywords = ["revenue", "sales", "budget", "spend", "price", "cost",
                            "profit", "income", "traffic", "followers", "score",
                            "rate", "percentage", "index"]
        for kw in general_keywords:
            for col in columns:
                if col == x_col:
                    continue
                if kw in col.lower() and col not in y_cols:
                    y_cols.append(col)
                    break
            if y_cols:
                break

    # Detect aggregation
    agg = "sum"
    if any(w in combined for w in ["average", "mean"]):
        agg = "mean"
    elif "count" in combined:
        agg = "count"
    elif "max" in combined:
        agg = "max"
    elif "min" in combined:
        agg = "min"

    if not x_col or not y_cols:
        return None

    return {
        "chart_type": chart_type,
        "x": x_col,
        "y": y_cols[:1],
        "agg": agg if agg != "count" else None,
    }


def validate_chart_columns(chart_params: dict, available_columns: list[str]) -> dict:
    """Validate and correct column names in chart params against available columns."""
    col_names = [c["name"] if isinstance(c, dict) else c for c in available_columns]
    col_lower = {name.lower().replace(" ", "_"): name for name in col_names}

    def find_best_match(name: str) -> str | None:
        if not name:
            return None
        key = name.lower().replace(" ", "_")
        if key in col_lower:
            return col_lower[key]
        # Fuzzy: check if any column contains the name or vice versa
        for cn in col_names:
            cn_key = cn.lower().replace(" ", "_")
            if key in cn_key or cn_key in key:
                return cn
        return None

    result = dict(chart_params)

    matched_x = find_best_match(result.get("x", ""))
    if matched_x:
        result["x"] = matched_x
    else:
        result["x"] = col_names[0] if col_names else None

    y_list = result.get("y", [])
    matched_y = []
    for y_name in (y_list if isinstance(y_list, list) else [y_list]):
        m = find_best_match(y_name)
        if m:
            matched_y.append(m)
    result["y"] = matched_y if matched_y else col_names[1:2] if len(col_names) > 1 else [col_names[0]]

    return result


def _find_matching_column(name: str, available: list[str]) -> str | None:
    """Find the best matching column name from available columns."""
    if not name:
        return None
    key = name.lower().replace('_', ' ').strip()
    # Exact match (case-insensitive)
    for col in available:
        if col.lower() == key or col.lower().replace('_', ' ') == key:
            return col
    # Partial match
    for col in available:
        col_key = col.lower().replace('_', ' ')
        if key in col_key or col_key in key:
            return col
    return None


def _extract_groupby(question: str, dataframe: pd.DataFrame):
    """Detect 'by <column>' pattern and return (group_col, agg_type) or None."""
    q = question.lower()
    # Match patterns like "by sales channel", "by region", "by category"
    by_match = re.search(r'\bby\s+([a-z][a-z\s_]+?)(?:\s*$|\s*,|\s*\.|\s*\?)', q)
    if not by_match:
        return None
    group_name = by_match.group(1).strip()
    all_cols = dataframe.columns.tolist()
    group_col = _find_matching_column(group_name, all_cols)
    if not group_col:
        return None
    # Only group by non-numeric (categorical) columns
    if group_col in dataframe.select_dtypes(include=['number']).columns:
        return None
    # Determine aggregation type
    if 'average' in q or 'mean' in q:
        agg = 'mean'
    elif 'sum' in q or 'total' in q:
        agg = 'sum'
    elif 'count' in q or 'how many' in q:
        agg = 'count'
    elif 'maximum' in q or 'max' in q or 'highest' in q:
        agg = 'max'
    elif 'minimum' in q or 'min' in q or 'lowest' in q:
        agg = 'min'
    else:
        agg = 'mean'
    return group_col, agg


def _find_value_column(question: str, dataframe: pd.DataFrame):
    """Find the numeric column the user is asking about."""
    q = question.lower()
    numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
    # Check if user mentions a specific column
    mentioned = [col for col in numeric_cols if col.lower() in q or col.replace('_', ' ').lower() in q]
    if mentioned:
        return mentioned[0]
    # Try revenue/sales/price keywords
    key_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['revenue', 'sales', 'price', 'budget', 'spend', 'income', 'profit', 'cost'])]
    if key_cols:
        return key_cols[0]
    return numeric_cols[0] if numeric_cols else None


def _format_currency(val):
    """Format a value as currency if it looks like a monetary amount."""
    if isinstance(val, float):
        return f"${val:,.2f}"
    return f"{val:,.2f}"


def perform_calculation(question: str, dataframe: pd.DataFrame) -> str:
    q = question.lower()
    try:
        # Check for "by <column>" group-by pattern FIRST
        groupby_result = _extract_groupby(q, dataframe)
        if groupby_result:
            group_col, agg = groupby_result
            value_col = _find_value_column(q, dataframe)
            if value_col is None:
                return f"No numeric column found to aggregate."
            result = dataframe.groupby(group_col)[value_col].agg(agg)
            is_currency = any(kw in value_col.lower() for kw in ['revenue', 'price', 'budget', 'spend', 'income', 'profit', 'cost'])
            agg_label = {'mean': 'Average', 'sum': 'Total', 'count': 'Count', 'max': 'Highest', 'min': 'Lowest'}.get(agg, agg.title())
            if agg == 'count':
                formatted = "\n".join([f"- {cat}: {val:,}" for cat, val in result.items()])
            elif is_currency:
                formatted = "\n".join([f"- {cat}: ${val:,.2f}" for cat, val in result.items()])
            else:
                formatted = "\n".join([f"- {cat}: {val:,.2f}" for cat, val in result.items()])
            return f"{agg_label} {value_col} by {group_col}:\n{formatted}"

        if 'average' in q or 'mean' in q:
            numeric_cols = dataframe.select_dtypes(include=['number']).columns
            mentioned_cols = [col for col in numeric_cols if col.lower() in q or col.replace('_', ' ').lower() in q]
            if mentioned_cols:
                result = dataframe[mentioned_cols].mean()
                formatted = "\n".join([f"- Average {col}: ${val:,.2f}" if 'revenue' in col.lower() or 'price' in col.lower() or 'budget' in col.lower() or 'spend' in col.lower() else f"- Average {col}: {val:,.2f}" for col, val in result.items()])
                return f"Average values:\n{formatted}"
            else:
                key_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['revenue', 'sales', 'price', 'budget', 'traffic', 'conversion'])]
                if not key_cols:
                    key_cols = numeric_cols[:5]
                result = dataframe[key_cols].mean()
                formatted = "\n".join([f"- Average {col}: ${val:,.2f}" if 'revenue' in col.lower() or 'price' in col.lower() or 'budget' in col.lower() or 'spend' in col.lower() else f"- Average {col}: {val:,.2f}" for col, val in result.items()])
                return f"Average values:\n{formatted}"

        elif 'count' in q or 'how many' in q:
            return f"Total number of records: {len(dataframe):,}"

        elif 'sum' in q or 'total' in q:
            numeric_cols = dataframe.select_dtypes(include=['number']).columns
            mentioned_cols = [col for col in numeric_cols if col.lower() in q or col.replace('_', ' ').lower() in q]
            if mentioned_cols:
                result = dataframe[mentioned_cols].sum()
                formatted = "\n".join([f"- Total {col}: ${val:,.2f}" if 'revenue' in col.lower() or 'price' in col.lower() or 'budget' in col.lower() or 'spend' in col.lower() else f"- Total {col}: {val:,.2f}" for col, val in result.items()])
                return f"Total sums:\n{formatted}"
            else:
                key_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['revenue', 'sales', 'price', 'budget'])]
                if not key_cols:
                    key_cols = numeric_cols[:5]
                result = dataframe[key_cols].sum()
                formatted = "\n".join([f"- Total {col}: ${val:,.2f}" if 'revenue' in col.lower() or 'price' in col.lower() or 'budget' in col.lower() or 'spend' in col.lower() else f"- Total {col}: {val:,.2f}" for col, val in result.items()])
                return f"Total sums:\n{formatted}"

        elif 'maximum' in q or 'max' in q or 'highest' in q:
            numeric_cols = dataframe.select_dtypes(include=['number']).columns
            mentioned_cols = [col for col in numeric_cols if col.lower() in q or col.replace('_', ' ').lower() in q]
            if mentioned_cols:
                result = dataframe[mentioned_cols].max()
            else:
                key_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['revenue', 'sales', 'price', 'budget'])]
                if not key_cols:
                    key_cols = numeric_cols[:5]
                result = dataframe[key_cols].max()
            formatted = "\n".join([f"- Highest {col}: ${val:,.2f}" if 'revenue' in col.lower() or 'price' in col.lower() or 'budget' in col.lower() or 'spend' in col.lower() else f"- Highest {col}: {val:,.2f}" for col, val in result.items()])
            return f"Maximum values:\n{formatted}"

        elif 'minimum' in q or 'min' in q or 'lowest' in q:
            numeric_cols = dataframe.select_dtypes(include=['number']).columns
            mentioned_cols = [col for col in numeric_cols if col.lower() in q or col.replace('_', ' ').lower() in q]
            if mentioned_cols:
                result = dataframe[mentioned_cols].min()
            else:
                key_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['revenue', 'sales', 'price', 'budget'])]
                if not key_cols:
                    key_cols = numeric_cols[:5]
                result = dataframe[key_cols].min()
            formatted = "\n".join([f"- Lowest {col}: ${val:,.2f}" if 'revenue' in col.lower() or 'price' in col.lower() or 'budget' in col.lower() or 'spend' in col.lower() else f"- Lowest {col}: {val:,.2f}" for col, val in result.items()])
            return f"Minimum values:\n{formatted}"

        elif 'overview' in q or 'summary' in q or 'describe' in q:
            numeric_cols = dataframe.select_dtypes(include=['number']).columns
            text_cols = dataframe.select_dtypes(include=['object', 'string']).columns
            overview = f"""Dataset Overview:
- Total records: {len(dataframe):,}
- Numeric columns: {', '.join(numeric_cols.tolist())}
- Text columns: {', '.join(text_cols.tolist())}

Quick Statistics:
{dataframe[numeric_cols].describe().loc[['mean', 'min', 'max']].to_string()}
"""
            return overview

        # Default basic stats
        numeric_cols = dataframe.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result = dataframe[numeric_cols].describe().loc[['mean', 'min', 'max']]
            return f"Basic statistics:\n{result.to_string()}"
        else:
            return "No numeric data available for calculation."

    except Exception:
        return "Could not perform that calculation. Try asking in a different way."



def get_quick_dataset_info(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    return f"""
            Dataset loaded: {len(df):,} records

            Columns available:
            • Numeric: {', '.join(numeric_cols) if numeric_cols else 'None'}
            • Text: {', '.join(text_cols) if text_cols else 'None'}
            """


def stream_ollama_analysis(question: str, df: pd.DataFrame, dataset_info: str | None = None):
    """
    Generator that yields text chunks from the LLM.
    Yields tuples: ("text", chunk) or ("usage", {"input_tokens": ..., "output_tokens": ...})
    """
    if dataset_info is None:
        dataset_info = get_quick_dataset_info(df)

    is_analytical = detect_query_type(question)
    is_ml_question = detect_ml_intent(question)
    is_chart_request = detect_chart_intent(question)

    if is_chart_request:
        chain = chart_chain
        prompt_vars = {"dataset_info": dataset_info, "question": question}
    elif is_ml_question:
        chain = ml_chain
        prompt_vars = {"dataset_info": dataset_info, "question": question}
    else:
        chain = base_chain
        if is_analytical:
            calc_result = perform_calculation(question, df)
            analysis_result = (
                f"The pre-computed calculation results are below. You MUST present ALL items "
                f"from the results in a clear list format. Do NOT summarize, skip, or pick "
                f"only one item. Do NOT add chart/graph descriptions. Present every single "
                f"number from the results.\n\n"
                f"{calc_result}"
            )
        else:
            analysis_result = (
                "Provide a short and direct answer based on the dataset. "
                "Do not add ML guidance unless asked. Do NOT include charts unless requested."
            )
        prompt_vars = {"dataset_info": dataset_info, "analysis_result": analysis_result, "question": question}

    full_response = ""
    in_chart_block = False
    chart_buffer = ""
    input_tokens = 0
    output_tokens = 0

    # Build prompt text from template variables to estimate input tokens
    prompt_text = " ".join(str(v) for v in prompt_vars.values() if v)
    input_tokens = max(1, int(len(prompt_text.split()) * 1.3))

    for chunk in chain.stream(prompt_vars):
        text = getattr(chunk, "content", None) or str(chunk)
        if not text:
            continue

        # Try to extract real usage metadata if available
        if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
            meta = chunk.response_metadata
            if 'eval_count' in meta:
                output_tokens = meta['eval_count']
            if 'prompt_eval_count' in meta:
                input_tokens = meta['prompt_eval_count']
        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
            um = chunk.usage_metadata
            if 'input_tokens' in um:
                input_tokens = um['input_tokens']
            if 'output_tokens' in um:
                output_tokens = um['output_tokens']

        full_response += text

        if in_chart_block:
            chart_buffer += text
            if re.search(r'</\s*chart\s*>', chart_buffer):
                in_chart_block = False
                chart_buffer = ""
            continue

        if re.search(r'<\s*chart[\s>]', full_response) and not re.search(r'</\s*chart\s*>', full_response):
            in_chart_block = True
            chart_buffer = text
            continue

        if re.search(r'<\s*chart\s*>', text) and re.search(r'</\s*chart\s*>', text):
            continue

        yield ("text", text)

    # Estimate output tokens from generated text
    if output_tokens == 0 and full_response:
        output_tokens = max(1, int(len(full_response.split()) * 1.3))

    yield ("usage", {"input_tokens": input_tokens, "output_tokens": output_tokens})
    yield ("full_response", full_response)


# Non streaming version for Flask API
def run_ollama_analysis(question: str, df: pd.DataFrame, dataset_info: str | None = None) -> str:
    if dataset_info is None:
        dataset_info = get_quick_dataset_info(df)

    is_analytical = detect_query_type(question)
    is_ml_question = detect_ml_intent(question)
    is_chart_request = detect_chart_intent(question)

    if is_chart_request:
        chain = chart_chain
        prompt_vars = {"dataset_info": dataset_info, "question": question}
    elif is_ml_question:
        chain = ml_chain
        prompt_vars = {"dataset_info": dataset_info, "question": question}
    else:
        chain = base_chain
        if is_analytical:
            calc_result = perform_calculation(question, df)
            analysis_result = (
                f"The pre-computed calculation results are below. You MUST present ALL items "
                f"from the results in a clear list format. Do NOT summarize, skip, or pick "
                f"only one item. Do NOT add chart/graph descriptions. Present every single "
                f"number from the results.\n\n"
                f"{calc_result}"
            )
        else:
            analysis_result = (
                "Provide a short and direct answer based on the dataset. "
                "Do not add ML guidance unless asked. Do NOT include charts unless requested."
            )
        prompt_vars = {"dataset_info": dataset_info, "analysis_result": analysis_result, "question": question}

    response = chain.invoke(prompt_vars)
    result = getattr(response, "content", response)

    if not is_chart_request and result:
        result = re.sub(r'<\s*chart\s*>.*?<\s*/\s*chart\s*>', '', result, flags=re.DOTALL).strip()
        result = re.sub(
            r'\s*(This|The|A|An)\s+(bar|line|scatter|pie|gauge|chart|graph|plot|visualization|visualisation)\s*(chart|graph|plot)?\s*(that\s+)?(shows|displays|illustrates|represents|depicts|demonstrates|highlights|reveals|indicates|summarizes|summarises|compares|breaks down|illustrates).*',
            '', result, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        result = re.sub(
            r'\s*(The\s+)?(above|below|following)\s+(chart|graph|plot|visualization)\s+(that\s+)?(shows|displays|illustrates|represents|depicts).*',
            '', result, flags=re.DOTALL | re.IGNORECASE
        ).strip()

    return result