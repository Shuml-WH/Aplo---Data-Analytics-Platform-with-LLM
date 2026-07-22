from fileinput import filename
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import io
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from chart_service import build_chart_figure
from flask import Response, stream_with_context
from ollama_helper import run_ollama_analysis, get_quick_dataset_info, stream_ollama_analysis, parse_chart_from_response, validate_chart_columns, detect_chart_intent
from ml_service import infer_task_type, build_preprocessing_plan, train_models, predict_rows, compute_feature_sensitivity, decompose_datetimes
from timeseries_service import train_arimax_model, train_xgboost_model, generate_forecast, get_timeseries_info
import tempfile
import pickle
import json
import numpy as np
import math


class SafeJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN, Infinity, and numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            if math.isinf(obj):
                return 1e308 if obj > 0 else -1e308
        return super().default(obj)


app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

ACTIVE_DATASET_ID = "current" 
DATASETS = {}

# in‑memory audit log for the active dataset
AUDIT_LOG = []  # list of dicts: {id, timestamp, action, details}



## Helper functions

# Returns (df, filename) or (None, None) if no active dataset.
def get_active_dataset():
    
    ds = DATASETS.get(ACTIVE_DATASET_ID)
    if not ds:
        return None, None
    return ds.get("df"), ds.get("filename")


# Utility function to convert values to JSON-safe formats
def make_json_safe(value): 
    if pd.isna(value):  # check pandas's Nan, None, pd.Na, NaT (Not a Time) values
        return None

    if isinstance(value, pd.Timestamp):  # check if it's a pandas Timestamp, convert to ISO format string
        return value.isoformat()

    if hasattr(value, "item"):  # check if the value has an 'item' method (e.g., numpy scalar types) and try to convert it to a native Python type
        try:
            return value.item()
        except Exception:
            pass

    return value



 # takes a pandas DataFrame and converts it to a list of dictionaries, ensuring all values are JSON serializable by using the make_json_safe function
def records_json_safe(df_part): 
    return [
        {col: make_json_safe(val) for col, val in row.items()}
        for row in df_part.to_dict(orient="records")
    ]


# takes a nested dictionary (like the one produced by df.describe().to_dict(orient="index")) and applies make_json_safe to all values, ensuring the entire structure is JSON serializable
def nested_dict_json_safe(data):  
    safe = {}
    for outer_key, inner_dict in data.items():
        safe[outer_key] = {
            inner_key: make_json_safe(inner_val)
            for inner_key, inner_val in inner_dict.items()
        }
    return safe


# Tries multiple encodings to read a CSV file, returning the DataFrame if successful or raising the last encountered error if all attempts fail
def read_csv_flexible(filepath):  
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error = None

    for enc in encodings:  # Try reading the CSV file with each encoding in the list. If it succeeds, return the DataFrame. If it fails, catch the exception and store it as last_error before trying the next encoding.
        try:
            return pd.read_csv(filepath, encoding=enc)
        except Exception as e:
            last_error = e

    raise last_error



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



# Helper to build the datasetProfile JSON from a DataFrame
def build_dataset_profile(df, filename):
    preview_df = df.head(5)

    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    info_text = info_buffer.getvalue()

    columns = []
    for col in df.columns:
        col_info = {"name": str(col), "dtype": str(df[col].dtype)}
        # Detect datetime columns (dates stored as strings have dtype "object")
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                non_null = df[col].notna()
                if non_null.sum() > 0 and parsed[non_null].notna().mean() > 0.5:
                    col_info["is_datetime"] = True
                else:
                    col_info["is_datetime"] = False
            except Exception:
                col_info["is_datetime"] = False
        else:
            col_info["is_datetime"] = False
        columns.append(col_info)

    head_data = records_json_safe(preview_df)

    summary_df = df.describe(include="all").transpose()
    summary_stats = nested_dict_json_safe(summary_df.to_dict(orient="index"))

    null_counts = {
        str(col): int(count)
        for col, count in df.isnull().sum().to_dict().items()
    }


    combined_columns_profile = []
    for col in df.columns:
        col_name = str(col)
        series = df[col]
        dtype_str = str(series.dtype)
        non_null = int(series.notnull().sum())

        stats = summary_stats.get(col_name, {}) or {}

        # Simple numeric detection based on dtype string
        numeric_prefixes = ("int", "float", "double", "number")
        is_numeric = dtype_str.startswith(numeric_prefixes)
        dtype_label = "Numeric" if is_numeric else "Non-numeric"

        # Compute top directly from the column
        if non_null > 0:
            top_val = series.value_counts(dropna=True).idxmax()
        else:
            top_val = None

        combined_columns_profile.append({
            "feature": col_name,
            "non_null": non_null,
            "dtype_label": dtype_label,
            "dtype": dtype_str,
            "min": make_json_safe(stats.get("min")),
            "q25": make_json_safe(stats.get("25%")),
            "q50": make_json_safe(stats.get("50%")),
            "q75": make_json_safe(stats.get("75%")),
            "max": make_json_safe(stats.get("max")),
            "top": make_json_safe(top_val),  
        })
        
    # print("DEBUG top for price:", df["price"].value_counts(dropna=True).head())    
    # print("DEBUG columns_profile sample:", combined_columns_profile[0])
    return {
        "success": True,
        "filename": filename,
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "columns": columns,
        "info_text": info_text,
        "head": head_data,
        "summary_stats": summary_stats,
        "null_counts": null_counts,
        "columns_profile": combined_columns_profile
    }


def build_chat_ml_context(df, filename):
    profile = build_dataset_profile(df, filename)

    column_lines = []
    for col in profile["columns_profile"]:
        feature = col.get("feature")
        dtype_label = col.get("dtype_label")
        dtype_value = col.get("dtype")
        non_null = col.get("non_null")
        top = col.get("top")
        column_lines.append(
            f"- {feature}: {dtype_label} ({dtype_value}), non-null={non_null}, example/top={top}"
        )

    numeric_columns = [
        c["feature"] for c in profile["columns_profile"]
        if c.get("dtype_label") == "Numeric"
    ]
    non_numeric_columns = [
        c["feature"] for c in profile["columns_profile"]
        if c.get("dtype_label") != "Numeric"
    ]

    return f"""Dataset file: {filename}
Rows: {profile["shape"]["rows"]}
Columns: {profile["shape"]["columns"]}

Column profiles:
{chr(10).join(column_lines)}

Numeric columns: {", ".join(numeric_columns) if numeric_columns else "None"}
Non-numeric columns: {", ".join(non_numeric_columns) if non_numeric_columns else "None"}
"""


#  Append a new event to the audit log.
def add_audit_event(action, details="", strategy=None):
    event = {
        "id": len(AUDIT_LOG) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "details": details,
        "strategy": strategy,
    }
    AUDIT_LOG.append(event)


# Initialize ML state
def get_ml_state():
    ds = DATASETS.get(ACTIVE_DATASET_ID)
    if not ds:
        return None
    if "ml" not in ds:
        ds["ml"] = {
            "config": None,
            "plan": None,
            "results": None,
            "models": {},
            "best_model_key": None,
            "label_encoder": None,
            "label_classes": None,
        }
    return ds["ml"]



def build_export_model_bundle(ml_state, model_key):
    if not ml_state or model_key not in ml_state.get("models", {}):
        raise ValueError(f"Model '{model_key}' not found in ML state.")

    config = ml_state.get("config") or {}
    stored_model = ml_state["models"][model_key]

    return { 
        "model_key": model_key, 
        "task_type": config.get("task_type"),  # preserve regression/classification type
        "target_column": config.get("target_column"),  
        "feature_columns": config.get("feature_columns", []), 
        "numeric_strategy": config.get("numeric_strategy"),
        "categorical_strategy": config.get("categorical_strategy"), 
        "use_scaling": config.get("use_scaling"),
        "pipeline": stored_model.get("pipeline"),  # include the fitted sklearn pipeline object
        "metrics": stored_model.get("metrics"),  # 
        "insights": stored_model.get("insights"),  # include feature importance/coefficient insights
        "label_classes": ml_state.get("label_classes"),  # preserve classification label mapping info
        "datetime_columns": stored_model.get("datetime_columns", []),  # datetime columns that were decomposed during training
        "exported_at": datetime.now().isoformat(timespec="seconds"),  
    } 






## API endpoint

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})




@app.route("/api/upload-csv", methods=["POST", "OPTIONS"])
def upload_csv():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200


    print("[upload] request received")

    if "file" not in request.files:
        print("[upload] no file part in request")
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        print("[upload] empty filename")
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        print("[upload] invalid extension:", file.filename)
        return jsonify({"success": False, "error": "Only CSV files are allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"[upload] file saved to: {filepath}")



    try:
        df = read_csv_flexible(filepath)

        print(f"[upload] dataframe loaded successfully: shape={df.shape}")

        DATASETS[ACTIVE_DATASET_ID] = {
            "df": df.copy(),
            "filename": filename,
            "filepath": filepath,
            "undo_df": None,          # previous version of df
            "undo_label": None,       # e.g., "Before Auto-Clean 2026-05-08T00:54:00"
        }

        response = build_dataset_profile(df, filename)

        add_audit_event(
            action="Uploaded CSV",
            details=f"File '{filename}' loaded with {df.shape[0]} rows, {df.shape[1]} columns.", strategy=None
        )

        print(
            f"[upload] sending response with {len(response['columns'])} columns "
            f"and {len(response['head'])} preview rows"
        )
        return jsonify(response), 200

    except Exception as e:
        import traceback
        print("[upload] ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/dataset-profile", methods=["GET"])
def dataset_profile():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 404

    response = build_dataset_profile(df, filename)
    return jsonify(response), 200




# Endpoint to handle AI chat queries about the dataset. 
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"success": False, "error": "Query is required"}), 400

    # Get active dataset
    df, current_filename = get_active_dataset()
    if df is None:
        return jsonify({
            "success": True,
            "reply": "Please upload a CSV file first so I can answer questions about the dataset."
        }), 200

    try:
        dataset_info = build_chat_ml_context(df, current_filename)
        answer = run_ollama_analysis(query, df, dataset_info=dataset_info)

        # Check for chart intent in the LLM response
        # Only process chart if user explicitly asked for one
        chart_data = None
        if detect_chart_intent(query):
            chart_params = parse_chart_from_response(answer)
            if chart_params:
                try:
                    # Validate and correct column names against actual dataset columns
                    chart_params = validate_chart_columns(chart_params, df.columns.tolist())
                    chart_type = chart_params.get("chart_type", "bar")
                    x = chart_params.get("x")
                    y = chart_params.get("y", [])
                    agg = chart_params.get("agg")
                    group_by = chart_params.get("group_by")
                    time_period = chart_params.get("time_period")
                    target = chart_params.get("target")
                    factor = chart_params.get("factor")
                    factor_mode = chart_params.get("factor_mode")
                    target_source = chart_params.get("target_source")
                    target_period = chart_params.get("target_period")
                    target_agg = chart_params.get("target_agg")

                    if chart_type == "gauge" or (x and y and x in df.columns and all(col in df.columns for col in y)):
                        fig_json = build_chart_figure(
                            df, chart_type=chart_type, x=x, y=y, agg=agg,
                            group_by=group_by, time_period=time_period, target=target,
                            factor=factor, factor_mode=factor_mode, target_source=target_source,
                            target_period=target_period, target_agg=target_agg
                        )
                        chart_data = {
                            "figure_json": fig_json,
                            "chart_type": chart_type,
                            "x": x,
                            "y": y,
                            "agg": agg,
                            "group_by": group_by,
                            "time_period": time_period,
                            "target": target,
                            "factor": factor,
                            "factor_mode": factor_mode,
                            "target_source": target_source,
                            "target_period": target_period,
                            "target_agg": target_agg,
                        }
                except Exception as chart_err:
                    print(f"[chat] Chart build error: {chart_err}")

        # Optional: log to audit trail
        add_audit_event(
            action="Chat query",
            details=f"Query: {query[:200]}... Answer length: {len(answer)} chars",
            strategy="Ollama analyst"
        )

        return jsonify({
            "success": True,
            "reply": answer,
            "chart": chart_data,
        }), 200

    except Exception as e:
        # Log server-side
        import traceback
        print("chat ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    


# handle streaming AI chat responses using Server-Sent Events.
@app.route("/api/chat-stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"success": False, "error": "Query is required"}), 400

    df, current_filename = get_active_dataset()
    if df is None:
        def no_dataset():
            yield "data:" + "Please upload a CSV file first.\n\n"
        return Response(no_dataset(), mimetype="text/event-stream")

    dataset_info = build_chat_ml_context(df, current_filename)

    @stream_with_context
    def generate():
        try:
            full_answer = []
            usage_data = None
            full_response_text = ""
            for item in stream_ollama_analysis(query, df, dataset_info=dataset_info):
                if item[0] == "text":
                    full_answer.append(item[1])
                    safe = item[1].replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
                    yield f"data:{safe}\n\n"
                elif item[0] == "usage":
                    usage_data = item[1]
                elif item[0] == "full_response":
                    full_response_text = item[1]

            # Send token usage as a special event
            if usage_data:
                yield f"data:[USAGE]{json.dumps(usage_data)}[/USAGE]\n\n"

            # Use the full response (including chart blocks) for parsing
            answer_text = full_response_text if full_response_text else "".join(full_answer)

            if detect_chart_intent(query):
                chart_params = parse_chart_from_response(answer_text, question=query, available_columns=df.columns.tolist())
                if chart_params:
                    try:
                        chart_params = validate_chart_columns(chart_params, df.columns.tolist())
                        chart_type = chart_params.get("chart_type", "bar")
                        x = chart_params.get("x")
                        y = chart_params.get("y", [])
                        agg = chart_params.get("agg")
                        group_by = chart_params.get("group_by")
                        time_period = chart_params.get("time_period")
                        target = chart_params.get("target")
                        factor = chart_params.get("factor")
                        factor_mode = chart_params.get("factor_mode")
                        target_source = chart_params.get("target_source")
                        target_period = chart_params.get("target_period")
                        target_agg = chart_params.get("target_agg")

                        if chart_type == "gauge" or (x and y and x in df.columns and all(col in df.columns for col in y)):
                            fig_json = build_chart_figure(
                                df, chart_type=chart_type, x=x, y=y, agg=agg,
                                group_by=group_by, time_period=time_period, target=target,
                                factor=factor, factor_mode=factor_mode, target_source=target_source,
                                target_period=target_period, target_agg=target_agg
                            )
                            chart_data = json.dumps({
                                "figure_json": fig_json,
                                "chart_type": chart_type,
                                "x": x,
                                "y": y,
                                "agg": agg,
                                "group_by": group_by,
                                "time_period": time_period,
                                "target": target,
                                "factor": factor,
                                "factor_mode": factor_mode,
                                "target_source": target_source,
                                "target_period": target_period,
                                "target_agg": target_agg,
                            })
                            yield f"data:[CHART]{chart_data}[/CHART]\n\n"
                        else:
                            print(f"[chat-stream] Chart validation failed: x={x}, y={y}, df.columns={df.columns.tolist()}")
                    except Exception as chart_err:
                        print(f"[chat-stream] Chart build error: {chart_err}")
                else:
                    print(f"[chat-stream] Failed to parse chart params from response. Response preview: {answer_text[:500]}")

        except Exception as e:
            err_msg = f"[Error] {str(e)}"
            yield f"data:{err_msg}\n\n"

    return Response(generate(), mimetype="text/event-stream")
    




# Clear the active dataset from memory (and optionally on disk later).
@app.route("/api/reset-dataset", methods=["POST"])
def reset_dataset():

    # Clear the entry for the active dataset id if it exists
    if ACTIVE_DATASET_ID in DATASETS:
        del DATASETS[ACTIVE_DATASET_ID]

    AUDIT_LOG.clear()
    add_audit_event(action="Reset dataset", details="Active dataset and audit log cleared.", strategy=None)

    return jsonify({"success": True, "message": "Active dataset has been reset"}), 200



#  Return the current audit log for the active dataset.
@app.route("/api/audit-trail", methods=["GET"])
def get_audit_trail():
    return jsonify({
        "success": True,
        "events": AUDIT_LOG
    }), 200


# Perform auto-cleaning of the active dataset based on specified strategies for numeric and categorical columns. 
# The request body can include "numeric_strategy" (one of "mean", "median", "zero", "drop") and 
# "categorical_strategy" (one of "mode", "constant", "drop"). 
# The function applies the cleaning strategies, updates the active dataset, rebuilds the profile, and logs the action in the audit trail.
@app.route("/api/auto-clean", methods=["POST"])
def auto_clean():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    numeric_strategy = data.get("numeric_strategy", "median")
    categorical_strategy = data.get("categorical_strategy", "mode")

    # Work on a copy so we can assign back atomically
    df_clean = df.copy()

    # Snapshot of missing counts before cleaning
    missing_before = df_clean.isnull().sum().to_dict()  # {col: count}

    # Snapshot for undo
    ds_meta = DATASETS.get(ACTIVE_DATASET_ID, {})
    ds_meta["undo_df"] = df.copy()
    ds_meta["undo_label"] = f"Before Auto-Clean at {datetime.now().isoformat(timespec='seconds')}"
    DATASETS[ACTIVE_DATASET_ID] = ds_meta

    # Split columns by dtype
    numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        c for c in df_clean.columns if c not in numeric_cols
    ]

    column_summaries = []  # will be list of {column, strategy, filled_count, dropped_rows}


    # Apply numeric strategy
    if numeric_strategy in ("mean", "median", "zero", "drop"):
        if numeric_strategy == "drop":
            # drop rows that have NaN in ANY numeric column
            rows_before = df_clean.shape[0]
            df_clean = df_clean.dropna(subset=numeric_cols)
            rows_after = df_clean.shape[0]
            dropped_rows = rows_before - rows_after

            numeric_action = (
                f"Dropped {dropped_rows} rows with missing numeric values."
            )

            for col in numeric_cols:
                if missing_before.get(col, 0) > 0:
                    column_summaries.append({
                        "column": col,
                        "strategy": "drop_rows_numeric",
                        "filled_count": 0,
                        "dropped_rows": dropped_rows,
                    })
        else:
            if numeric_cols:
                if numeric_strategy == "mean":
                    fill_values = df_clean[numeric_cols].mean()
                elif numeric_strategy == "median":
                    fill_values = df_clean[numeric_cols].median()
                elif numeric_strategy == "zero":
                    fill_values = {col: 0 for col in numeric_cols}

                # compute how many NaNs will be filled per column
                for col in numeric_cols:
                    before = int(missing_before.get(col, 0))
                    if before > 0:
                        column_summaries.append({
                            "column": col,
                            "strategy": numeric_strategy,
                            "filled_count": before,
                            "dropped_rows": 0,
                        })

                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(fill_values)
                numeric_action = (
                    f"Filled missing numeric values using '{numeric_strategy}'."
                )
            else:
                numeric_action = "No numeric columns to clean."
    else:
        numeric_action = f"Unknown numeric strategy '{numeric_strategy}' (no changes)."

    # Apply categorical strategy
    # Categorical strategy
    if categorical_strategy in ("mode", "constant", "drop"):
        if categorical_strategy == "drop":
            rows_before = df_clean.shape[0]
            if categorical_cols:
                df_clean = df_clean.dropna(subset=categorical_cols)
                rows_after = df_clean.shape[0]
                dropped_rows = rows_before - rows_after
                cat_action = (
                    f"Dropped {dropped_rows} rows with missing categorical values."
                )
                for col in categorical_cols:
                    if missing_before.get(col, 0) > 0:
                        column_summaries.append({
                            "column": col,
                            "strategy": "drop_rows_categorical",
                            "filled_count": 0,
                            "dropped_rows": dropped_rows,
                        })
            else:
                cat_action = "No categorical columns to clean."
        else:
            if categorical_cols:
                if categorical_strategy == "mode":
                    fill_values = {}
                    for col in categorical_cols:
                        mode_series = df_clean[col].mode(dropna=True)
                        if not mode_series.empty:
                            fill_values[col] = mode_series.iloc[0]
                elif categorical_strategy == "constant":
                    fill_values = {col: "Missing" for col in categorical_cols}

                # record fills per categorical column
                for col in categorical_cols:
                    before = int(missing_before.get(col, 0))
                    if before > 0:
                        column_summaries.append({
                            "column": col,
                            "strategy": categorical_strategy,
                            "filled_count": before,
                            "dropped_rows": 0,
                        })

                df_clean[categorical_cols] = df_clean[categorical_cols].fillna(fill_values)
                if categorical_strategy == "mode":
                    cat_action = "Filled missing categorical values using mode."
                else:
                    cat_action = 'Filled missing categorical values with constant "Missing".'
            else:
                cat_action = "No categorical columns to clean."
    else:
        cat_action = f"Unknown categorical strategy '{categorical_strategy}' (no changes)."

    # Save back to global DATASETS
    DATASETS[ACTIVE_DATASET_ID]["df"] = df_clean
    profile = build_dataset_profile(df_clean, filename)

    strategy_desc = f"Numeric: {numeric_strategy}; Categorical: {categorical_strategy}"

    # Build a human-readable summary per column
    column_details_lines = []
    for entry in column_summaries:
        col = entry["column"]
        strat = entry["strategy"]
        filled = entry.get("filled_count", 0)
        dropped = entry.get("dropped_rows", 0)

        if strat.startswith("drop_rows"):
            column_details_lines.append(
                f"Column '{col}': dropped {dropped} rows due to missing values."
            )
        else:
            column_details_lines.append(
                f"Column '{col}': filled {filled} missing values using '{strat}'."
            )

    if column_details_lines:
        details = " ".join(column_details_lines)
    else:
        details = "No missing values found to clean."

    add_audit_event(
        action="Auto-Clean Data",
        details=details,
        strategy=strategy_desc,
    )

    profile["success"] = True
    profile["auto_clean_summary"] = {
        "strategy": strategy_desc,
        "details": details,
        "columns": column_summaries,
    }

    return jsonify(profile), 200


# Endpoint to undo the last auto-cleaning operation by restoring the previous DataFrame snapshot.
@app.route("/api/undo-auto-clean", methods=["POST"])
def undo_auto_clean():
    ds_meta = DATASETS.get(ACTIVE_DATASET_ID)
    if not ds_meta:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    undo_df = ds_meta.get("undo_df")
    undo_label = ds_meta.get("undo_label")

    if undo_df is None:
        return jsonify({"success": False, "error": "No previous state to undo to"}), 400

    # Restore DataFrame
    ds_meta["df"] = undo_df.copy()
    # Clear undo history (single-level undo; you can keep if you want repeated undo)
    ds_meta["undo_df"] = None
    ds_meta["undo_label"] = None
    DATASETS[ACTIVE_DATASET_ID] = ds_meta

    filename = ds_meta.get("filename", "current.csv")
    profile = build_dataset_profile(ds_meta["df"], filename)

    add_audit_event(
        action="Undo Auto-Clean",
        details=f"Dataset restored to snapshot: {undo_label}",
        strategy=None,
    )

    profile["success"] = True
    profile["undo_label"] = undo_label
    return jsonify(profile), 200



@app.route("/api/build-chart", methods=["POST"])
def build_chart():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    chart_type = data.get("chart_type", "line")
    x = data.get("x")
    y = data.get("y")
    agg = data.get("agg")
    target = data.get("target")
    group_by = data.get("group_by")
    time_period = data.get("time_period")
    cat_group_by = data.get("cat_group_by")
    bar_mode = data.get("bar_mode")
    date_column = data.get("date_column")
    date_from = data.get("date_from")
    date_to = data.get("date_to")
    factor = data.get("factor")
    factor_mode = data.get("factor_mode")
    target_source = data.get("target_source")
    target_period = data.get("target_period")
    target_agg = data.get("target_agg")

    # x is required except for gauge
    if chart_type != "gauge" and not x:
        return jsonify({"success": False, "error": "'x' is required"}), 400
    if not y:
        return jsonify({"success": False, "error": "'y' is required"}), 400

    try:
        fig_json = build_chart_figure(
            df,
            chart_type=chart_type,
            x=x,
            y=y,
            agg=agg,
            date_column=date_column,
            date_from=date_from,
            date_to=date_to,
            target=target,
            group_by=group_by,
            time_period=time_period,
            cat_group_by=cat_group_by,
            bar_mode=bar_mode,
            factor=factor,
            factor_mode=factor_mode,
            target_source=target_source,
            target_period=target_period,
            target_agg=target_agg,
            )

        add_audit_event(
            action="Build chart",
            details=(
                f"Chart type={chart_type}, x={x}, y={y}, agg={agg}, "
                f"group_by={group_by}, time_period={time_period}, "
                f"date_column={date_column}, date_from={date_from}, date_to={date_to}"
            ),
            strategy="Chart builder",
        )

        return jsonify({"success": True, "figure_json": fig_json}), 200

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




@app.route("/api/ml/initialize", methods=["POST"])
def ml_initialize():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    target_column = data.get("target_column")
    feature_columns = data.get("feature_columns") or []

    if not target_column:
        return jsonify({"success": False, "error": "target_column is required"}), 400

    if target_column not in df.columns:
        return jsonify({"success": False, "error": f"Target column '{target_column}' not found"}), 400

    if not feature_columns:
        feature_columns = [c for c in df.columns if c != target_column]

    invalid_features = [c for c in feature_columns if c not in df.columns]
    if invalid_features:
        return jsonify({
            "success": False,
            "error": f"Invalid feature columns: {invalid_features}"
        }), 400

    task_type = infer_task_type(df, target_column)

    ml_state = get_ml_state()
    ml_state["config"] = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "task_type": task_type,
    }

    add_audit_event(
        action="ML Initialize",
        details=f"Target={target_column}; features={len(feature_columns)}; task={task_type}",
        strategy="ML pipeline initialization",
    )

    return jsonify({
        "success": True,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "task_type": task_type,
    }), 200


@app.route("/api/ml/preprocessing-plan", methods=["POST"])
def ml_preprocessing_plan():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    target_column = data.get("target_column")
    feature_columns = data.get("feature_columns") or []

    if not target_column:
        return jsonify({"success": False, "error": "target_column is required"}), 400

    if not feature_columns:
        feature_columns = [c for c in df.columns if c != target_column]

    task_type = data.get("task_type") or infer_task_type(df, target_column)

    plan = build_preprocessing_plan(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        task_type=task_type,
    )

    ml_state = get_ml_state()
    ml_state["config"] = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "task_type": task_type,
    }
    ml_state["plan"] = plan

    add_audit_event(
        action="ML Preprocessing Plan",
        details=f"Generated preprocessing plan for target '{target_column}'",
        strategy="Transparent preprocessing recommendation",
    )

    return jsonify({
        "success": True,
        "plan": plan,
    }), 200


@app.route("/api/ml/train", methods=["POST"])
def ml_train():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    print("ML TRAIN PAYLOAD:", data)  # debug

    target_column = data.get("target_column")
    feature_columns = data.get("feature_columns") or []
    task_type = data.get("task_type") or infer_task_type(df, target_column)


    if not target_column:
        return jsonify({"success": False, "error": "target_column is required"}), 400
    
    if target_column not in df.columns:
        return jsonify({"success": False, "error": f"Target column '{target_column}' not found"}), 400
    
    # remove target from features and de-duplicate while preserving order
    feature_columns = [c for c in feature_columns if c and c != target_column]
    feature_columns = list(dict.fromkeys(feature_columns))

    if not feature_columns:
        feature_columns = [c for c in df.columns if c != target_column]

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        return jsonify({"success": False, "error": f"Feature columns not found: {missing_features}"}), 400


    df = df.copy()


    default_models = (
        ["logisticregression", "decisiontreeclassifier"]
        if task_type == "classification"
        else ["linearregression", "ridgeregression", "gradientboostingregressor", "decisiontreeregressor", "kneighborsregressor"]
    )

    config = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "task_type": task_type,
        "numeric_strategy": data.get("numeric_strategy", "median"),
        "categorical_strategy": data.get("categorical_strategy", "most_frequent"),
        "use_scaling": data.get("use_scaling", True),
        "test_size": data.get("test_size", 0.2),
        "random_state": data.get("random_state", 42),
        "models": data.get("models", default_models),
    }

    def generate():
        import threading
        import time
        
        def send_progress(data):
            return f"data: {json.dumps(data, cls=SafeJsonEncoder)}\n\n"
        
        try:
            # Send initial progress
            yield send_progress({"type": "progress", "status": "starting", "percentage": 0})
            
            # Create progress callback
            latest_progress = {"value": None}
            def progress_callback(data):
                latest_progress["value"] = data
            
            # Run training in background thread
            result_container = {}
            def train_in_background():
                try:
                    result_container["output"] = train_models(df, config, progress_callback=progress_callback)
                    result_container["success"] = True
                except Exception as e:
                    import traceback
                    print(f"[train] BACKGROUND ERROR: {e}")
                    traceback.print_exc()
                    result_container["error"] = str(e)
                    result_container["success"] = False
            
            thread = threading.Thread(target=train_in_background)
            thread.start()
            
            # Stream progress updates
            while thread.is_alive():
                if latest_progress["value"]:
                    yield send_progress(latest_progress["value"])
                    latest_progress["value"] = None
                time.sleep(0.1)  # Poll every 100ms
            
            thread.join()
            
            # Send final progress
            yield send_progress({"type": "progress", "status": "complete", "percentage": 100})
            
            # Send final result
            if result_container.get("success"):
                output = result_container["output"]
                
                # Store in ml_state
                ml_state = get_ml_state()
                config["numeric_features"] = output.get("numeric_features", [])
                config["categorical_features"] = output.get("categorical_features", [])
                ml_state["config"] = config
                ml_state["results"] = output["results"]
                ml_state["best_model_key"] = output["best_model_key"]
                ml_state["label_classes"] = output.get("label_classes")
                ml_state["label_encoder"] = output.get("label_encoder")
                ml_state["models"] = {}
                print(f"[train] config feature_columns: {config.get('feature_columns', [])}")
                print(f"[train] config numeric_features: {config.get('numeric_features', [])}")
                print(f"[train] config categorical_features: {config.get('categorical_features', [])}")
                for model_key, model_bundle in output["trained_models"].items():
                    ml_state["models"][model_key] = {
                        "pipeline": model_bundle["pipeline"],
                        "metrics": model_bundle["metrics"],
                        "insights": model_bundle["insights"],
                        "chart_data": model_bundle.get("chart_data"),
                        "label_encoder": output.get("label_encoder"),
                        "datetime_columns": output.get("datetime_columns", []),
                    }
                
                add_audit_event(
                    action="ML Train Models",
                    details=f"Task={task_type}; Trained models: {', '.join(config['models'])}; best={output['best_model_key']}",
                    strategy="Scikit-learn ML pipeline",
                )
                
                yield send_progress({
                    "type": "complete",
                    "success": True,
                    "data": {
                        "task_type": output["task_type"],
                        "target_column": output["target_column"],
                        "feature_columns": output["feature_columns"],
                        "numeric_features": output["numeric_features"],
                        "categorical_features": output["categorical_features"],
                        "train_rows": output["train_rows"],
                        "test_rows": output["test_rows"],
                        "best_model_key": output["best_model_key"],
                        "results": [
                            {
                                "model_key": item["model_key"],
                                "metrics": item["metrics"],
                                "insights": item["insights"],
                                "chart_data": item.get("chart_data"),
                            }
                            for item in output["results"]
                        ],
                        "label_classes": output.get("label_classes"),
                    }
                })
            else:
                yield send_progress({
                    "type": "complete",
                    "success": False,
                    "error": result_container.get("error", "Training failed")
                })
        except Exception as e:
            import traceback
            print(f"[train] GENERATOR ERROR: {e}")
            traceback.print_exc()
            yield send_progress({
                "type": "complete",
                "success": False,
                "error": str(e)
            })
    
    return Response(generate(), mimetype="text/event-stream")
    

@app.route("/api/ml/results", methods=["GET"])
def ml_results():
    ml_state = get_ml_state()
    if ml_state is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    if not ml_state.get("results"):
        return jsonify({"success": False, "error": "No ML training results available"}), 404

    return jsonify({
        "success": True,
        "config": ml_state.get("config"),
        "best_model_key": ml_state.get("best_model_key"),
        "results": ml_state.get("results"),
        "label_classes": ml_state.get("label_classes"),
    }), 200


@app.route("/api/ml/pipeline-state", methods=["GET"])
def ml_pipeline_state():
    ds = DATASETS.get(ACTIVE_DATASET_ID)
    ml_state = ds.get("ml") if ds else None
    ts_state = ds.get("timeseries") if ds else None

    # Standard ML state
    ml_config = None
    ml_has_results = False
    ml_results_data = None
    if ml_state and ml_state.get("config"):
        ml_has_results = bool(ml_state.get("results"))
        if ml_has_results:
            ml_results_data = {
                "task_type": ml_state["config"].get("task_type"),
                "target_column": ml_state["config"].get("target_column"),
                "feature_columns": ml_state["config"].get("feature_columns"),
                "numeric_strategy": ml_state["config"].get("numeric_strategy", "median"),
                "categorical_strategy": ml_state["config"].get("categorical_strategy", "most_frequent"),
                "use_scaling": ml_state["config"].get("use_scaling", True),
                "models": ml_state["config"].get("models", []),
                "train_rows": ml_state.get("train_rows"),
                "test_rows": ml_state.get("test_rows"),
                "best_model_key": ml_state.get("best_model_key"),
                "results": [
                    {
                        "model_key": item["model_key"],
                        "metrics": item["metrics"],
                        "insights": item["insights"],
                        "chart_data": item.get("chart_data"),
                    }
                    for item in ml_state["results"]
                ],
                "label_classes": ml_state.get("label_classes"),
            }
        ml_config = ml_state["config"]

    # Time-series state
    ts_has_results = False
    ts_results_data = None
    if ts_state and ts_state.get("config") and ts_state.get("results"):
        ts_has_results = True
        ts_results = ts_state["results"]
        ts_results_data = {
            "task_type": "timeseries",
            "target_column": ts_state["config"].get("target_column"),
            "datetime_column": ts_state["config"].get("datetime_column"),
            "feature_columns": ts_state["config"].get("feature_columns", []),
            "forecast_horizon": ts_state["config"].get("forecast_horizon", 30),
            "frequency": ts_state["config"].get("frequency", "D"),
            "test_size": ts_state["config"].get("test_size", 0.2),
            "aggregation": ts_state["config"].get("aggregation", "sum"),
            "model_type": ts_state.get("model_type", ["arimax"]),
            "best_model_key": ts_state.get("best_model", "arimax"),
        }

        # Support both old format (dict) and new format (list)
        if isinstance(ts_results, list):
            ts_results_data["results"] = ts_results
            # Backward compat: expose first model's fields at top level
            first = ts_results[0] if ts_results else {}
            ts_results_data["order"] = first.get("order")
            ts_results_data["aic"] = first.get("aic")
            ts_results_data["train_rows"] = first.get("train_rows")
            ts_results_data["test_rows"] = first.get("test_rows")
            ts_results_data["test_metrics"] = first.get("test_metrics", {})
            ts_results_data["forecast_data"] = first.get("forecast_data")
        else:
            # Legacy single-model format
            ts_results_data["results"] = [{
                "model_type": "arimax",
                "model_label": "ARIMAX",
                "test_metrics": ts_results.get("test_metrics", {}),
                "forecast_data": ts_results.get("forecast_data"),
                "train_rows": ts_results.get("train_rows"),
                "test_rows": ts_results.get("test_rows"),
                "order": ts_results.get("order"),
                "aic": ts_results.get("aic"),
            }]
            ts_results_data["order"] = ts_results.get("order")
            ts_results_data["aic"] = ts_results.get("aic")
            ts_results_data["train_rows"] = ts_results.get("train_rows")
            ts_results_data["test_rows"] = ts_results.get("test_rows")
            ts_results_data["test_metrics"] = ts_results.get("test_metrics", {})
            ts_results_data["forecast_data"] = ts_results.get("forecast_data")

    return jsonify({
        "success": True,
        "config": ml_config,
        "plan": ml_state.get("plan") if ml_state else None,
        "has_results": ml_has_results,
        "results_data": ml_results_data,
        "timeseries": {
            "has_results": ts_has_results,
            "results_data": ts_results_data,
        },
    }), 200


@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    ml_state = get_ml_state()
    if not ml_state or not ml_state.get("models"):
        return jsonify({"success": False, "error": "No trained model available"}), 400

    data = request.get_json(silent=True) or {}
    model_key = data.get("model_key") or ml_state.get("best_model_key")
    rows = data.get("rows") or []

    if model_key not in ml_state["models"]:
        return jsonify({"success": False, "error": f"Model '{model_key}' not found"}), 400

    if not rows:
        return jsonify({"success": False, "error": "rows is required"}), 400

    try:
        predictions = predict_rows(ml_state["models"][model_key], rows)

        add_audit_event(
            action="ML Predict",
            details=f"Predicted {len(rows)} row(s) using model '{model_key}'",
            strategy="Stored ML pipeline prediction",
        )

        return jsonify({
            "success": True,
            "model_key": model_key,
            "predictions": predictions,
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    

@app.route("/api/ml/predict-batch", methods=["POST"])
def ml_predict_batch():
    df, filename = get_active_dataset()
    if df is None:
        print("[predict-batch] ERROR: No active dataset")
        return jsonify({"success": False, "error": "No active dataset"}), 400

    ml_state = get_ml_state()
    if not ml_state or not ml_state.get("models"):
        print(f"[predict-batch] ERROR: No trained model available. ml_state={ml_state}, models={ml_state.get('models') if ml_state else None}")
        return jsonify({"success": False, "error": "No trained model available"}), 400

    data = request.get_json(silent=True) or {}
    model_key = data.get("model_key") or ml_state.get("best_model_key")
    group_by = data.get("group_by")

    print(f"[predict-batch] model_key={model_key}, available models={list(ml_state['models'].keys())}")

    if model_key not in ml_state["models"]:
        return jsonify({"success": False, "error": f"Model '{model_key}' not found. Available: {list(ml_state['models'].keys())}"}), 400

    model_bundle = ml_state["models"][model_key]
    config = ml_state.get("config") or {}
    target_column = config.get("target_column", "")
    task_type = config.get("task_type", "regression")

    def generate():
        import numpy as np

        feature_columns = config.get("feature_columns", [])
        numeric_features = config.get("numeric_features", [])
        categorical_features = config.get("categorical_features", [])

        def send_progress(data):
            return f"data: {json.dumps(data, cls=SafeJsonEncoder)}\n\n"

        yield send_progress({"type": "progress", "percentage": 5, "step": "Loading dataset rows..."})

        try:
            feature_cols_in_df = [c for c in feature_columns if c in df.columns]
            rows = df[feature_cols_in_df].to_dict(orient="records")

            yield send_progress({"type": "progress", "percentage": 15, "step": f"Running predictions (0/{len(rows)})..."})

            batch_size = 5000
            all_predictions = []
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                preds = predict_rows(model_bundle, batch)
                all_predictions.extend(preds)
                pct = 15 + int(65 * min((i + batch_size), len(rows)) / len(rows))
                yield send_progress({"type": "progress", "percentage": pct,
                    "step": f"Running predictions ({min(i + batch_size, len(rows))}/{len(rows)})..."})

            yield send_progress({"type": "progress", "percentage": 82, "step": "Computing summary statistics..."})

            numeric_preds = []
            for p in all_predictions:
                try:
                    numeric_preds.append(float(p))
                except (ValueError, TypeError):
                    pass

            preds_arr = np.array(numeric_preds) if numeric_preds else np.array([0.0])
            summary = {
                "mean": float(np.mean(preds_arr)),
                "median": float(np.median(preds_arr)),
                "min": float(np.min(preds_arr)),
                "max": float(np.max(preds_arr)),
                "std": float(np.std(preds_arr)),
            }

            yield send_progress({"type": "progress", "percentage": 88, "step": "Computing distribution..."})

            counts, bin_edges = np.histogram(preds_arr, bins=15)
            distribution = {
                "bin_edges": [float(x) for x in bin_edges],
                "counts": [int(x) for x in counts],
            }

            grouped_data = None
            if group_by and group_by in df.columns:
                yield send_progress({"type": "progress", "percentage": 92, "step": f"Computing grouped averages by {group_by}..."})
                temp_df = df.copy()
                temp_df["_predicted"] = [float(p) if p else 0 for p in all_predictions]
                grouped = temp_df.groupby(group_by)["_predicted"]
                grouped_data = {
                    "group_column": group_by,
                    "groups": [str(x) for x in grouped.groups.keys()],
                    "averages": [float(v) for v in grouped.mean().values],
                    "sums": [float(v) for v in grouped.sum().values],
                    "counts": [int(v) for v in grouped.count().values],
                }

            yield send_progress({"type": "progress", "percentage": 96, "step": "Computing feature sensitivity..."})
            sensitivity = compute_feature_sensitivity(
                model_bundle, feature_columns, numeric_features,
                categorical_features, df
            )

            # Cache predictions for fast re-grouping
            ml_state["cached_predictions"] = all_predictions

            yield send_progress({"type": "progress", "percentage": 100, "step": "Complete!"})

            complete_data = {
                "type": "complete",
                "success": True,
                "target_column": target_column,
                "task_type": task_type,
                "total_rows": len(rows),
                "summary": summary,
                "distribution": distribution,
                "grouped_data": grouped_data,
                "feature_sensitivity": sensitivity,
            }
            yield send_progress(complete_data)

        except Exception as e:
            yield send_progress({"type": "complete", "success": False, "error": str(e)})

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/ml/regroup", methods=["POST"])
def ml_regroup():
    """Re-group cached batch predictions by a different column without re-running the model."""
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    ml_state = get_ml_state()
    if not ml_state or not ml_state.get("models"):
        return jsonify({"success": False, "error": "No trained model available"}), 400

    data = request.get_json(silent=True) or {}
    group_by = data.get("group_by")

    cached_predictions = ml_state.get("cached_predictions")
    if not cached_predictions:
        return jsonify({"success": False, "error": "No cached predictions. Run batch prediction first."}), 400

    if not group_by or group_by not in df.columns:
        return jsonify({"success": False, "error": f"Column '{group_by}' not found"}), 400

    try:
        temp_df = df.copy()
        temp_df["_predicted"] = [float(p) if p else 0 for p in cached_predictions]
        grouped = temp_df.groupby(group_by)["_predicted"]
        grouped_data = {
            "group_column": group_by,
            "groups": [str(x) for x in grouped.groups.keys()],
            "averages": [float(v) for v in grouped.mean().values],
            "sums": [float(v) for v in grouped.sum().values],
            "counts": [int(v) for v in grouped.count().values],
        }

        return jsonify({"success": True, "grouped_data": grouped_data}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ml/predict-whatif", methods=["POST"])
def ml_predict_whatif():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    ml_state = get_ml_state()
    if not ml_state or not ml_state.get("models"):
        return jsonify({"success": False, "error": "No trained model available"}), 400

    data = request.get_json(silent=True) or {}
    model_key = data.get("model_key") or ml_state.get("best_model_key")
    features = data.get("features", {})
    aggregate_all = data.get("aggregate_all", {})

    if model_key not in ml_state["models"]:
        return jsonify({"success": False, "error": f"Model '{model_key}' not found"}), 400

    model_bundle = ml_state["models"][model_key]
    config = ml_state.get("config") or {}
    target_column = config.get("target_column", "")
    task_type = config.get("task_type", "regression")
    feature_columns = config.get("feature_columns", [])
    categorical_features = config.get("categorical_features", [])
    datetime_cols_in_model = model_bundle.get("datetime_columns", [])

    print(f"[predict-whatif] feature_columns={feature_columns}")
    print(f"[predict-whatif] datetime_columns in model={datetime_cols_in_model}")
    print(f"[predict-whatif] features from request={list(features.keys())}")

    try:
        agg_cols = [col for col, is_all in aggregate_all.items() if is_all and col in categorical_features]

        if agg_cols:
            target_agg_col = agg_cols[0]
            unique_values = df[target_agg_col].dropna().unique()
            rows = []
            for val in unique_values:
                row = {}
                for col in feature_columns:
                    if col == target_agg_col:
                        row[col] = str(val)
                    elif col in features:
                        row[col] = features[col]
                    elif col in df.columns:
                        row[col] = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                    else:
                        row[col] = ""
                rows.append(row)
            preds = predict_rows(model_bundle, rows)
            numeric_preds = []
            for p in preds:
                try:
                    numeric_preds.append(float(p))
                except (ValueError, TypeError):
                    pass
            avg_pred = sum(numeric_preds) / len(numeric_preds) if numeric_preds else 0
            return jsonify({
                "success": True,
                "prediction": avg_pred,
                "target_column": target_column,
                "task_type": task_type,
                "aggregate": True,
                "aggregate_column": target_agg_col,
                "individual_predictions": {str(v): float(p) for v, p in zip(unique_values, preds)},
            }), 200
        else:
            row = {}
            for col in feature_columns:
                if col in features:
                    row[col] = features[col]
                elif col in df.columns:
                    modes = df[col].mode()
                    row[col] = modes.iloc[0] if not modes.empty else ""
                else:
                    row[col] = ""
            preds = predict_rows(model_bundle, [row])
            pred_val = float(preds[0]) if preds else 0
            return jsonify({
                "success": True,
                "prediction": pred_val,
                "target_column": target_column,
                "task_type": task_type,
                "aggregate": False,
            }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ml/export-predictions", methods=["GET"])
def ml_export_predictions():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    ml_state = get_ml_state()
    if not ml_state or not ml_state.get("models"):
        return jsonify({"success": False, "error": "No trained model available"}), 400

    model_key = request.args.get("model_key") or ml_state.get("best_model_key")
    if model_key not in ml_state["models"]:
        return jsonify({"success": False, "error": f"Model '{model_key}' not found"}), 400

    model_bundle = ml_state["models"][model_key]
    config = ml_state.get("config") or {}
    target_column = config.get("target_column", "target")
    feature_columns = config.get("feature_columns", [])

    try:
        feature_cols_in_df = [c for c in feature_columns if c in df.columns]
        rows = df[feature_cols_in_df].to_dict(orient="records")
        predictions = predict_rows(model_bundle, rows)

        result_df = df.copy()
        pred_col_name = f"predicted_{target_column}"
        result_df[pred_col_name] = predictions

        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        download_name = f"predictions_{model_key}_{filename.replace('.csv', '')}.csv"

        return Response(
            csv_buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={download_name}"},
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ml/feature-sensitivity", methods=["POST"])
def ml_feature_sensitivity():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    ml_state = get_ml_state()
    if not ml_state or not ml_state.get("models"):
        return jsonify({"success": False, "error": "No trained model available"}), 400

    data = request.get_json(silent=True) or {}
    model_key = data.get("model_key") or ml_state.get("best_model_key")

    if model_key not in ml_state["models"]:
        return jsonify({"success": False, "error": f"Model '{model_key}' not found"}), 400

    model_bundle = ml_state["models"][model_key]
    config = ml_state.get("config") or {}
    feature_columns = config.get("feature_columns", [])
    numeric_features = config.get("numeric_features", [])
    categorical_features = config.get("categorical_features", [])

    try:
        sensitivity = compute_feature_sensitivity(
            model_bundle, feature_columns, numeric_features,
            categorical_features, df
        )
        return jsonify({"success": True, "sensitivity": sensitivity}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# for exporting trained models as pickle
@app.route("/api/ml/export-model", methods=["POST"]) 
def ml_export_model(): 
    ml_state = get_ml_state()  # get current in-memory ML state for the active dataset
    if ml_state is None:  
        return jsonify({"success": False, "error": "No active dataset"}), 400 

    if not ml_state.get("models"):  
        return jsonify({"success": False, "error": "No trained model available"}), 400  

    data = request.get_json(silent=True) or {} 
    model_key = data.get("model_key") or ml_state.get("best_model_key")  # Default to best model if user did not choose one

    if model_key not in ml_state["models"]: 
        return jsonify({"success": False, "error": f"Model '{model_key}' not found"}), 400  

    try: 
        export_bundle = build_export_model_bundle(ml_state, model_key)  
        safe_model_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_key) 
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{safe_model_key}.pkl") 
        temp_file.close() 

        with open(temp_file.name, "wb") as f: 
            pickle.dump(export_bundle, f) 

        add_audit_event( 
            action="ML Export Model",  
            details=f"Exported trained model '{model_key}' as pickle file.",  
            strategy="Pickle model export",  
        ) 

        return send_file(  
            temp_file.name,  
            as_attachment=True, 
            download_name=f"{safe_model_key}_model.pkl", 
            mimetype="application/octet-stream",  
        ) 

    except Exception as e:
        import traceback
        print(f"[ml-export-model] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


    

@app.route("/api/date-range-profile", methods=["POST"])
def date_range_profile():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    date_column = data.get("date_column")

    if not date_column:
        return jsonify({"success": False, "error": "date_column is required"}), 400

    if date_column not in df.columns:
        return jsonify({"success": False, "error": f"Column '{date_column}' not found"}), 400

    try:
        s = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if s.empty:
            return jsonify({"success": False, "error": f"No valid dates found in column '{date_column}'"}), 400

        max_date = s.max().normalize()
        min_date = s.min().normalize()
        default_start = (max_date - pd.DateOffset(months=1)).normalize()

        if default_start < min_date:
            default_start = min_date

        return jsonify({
            "success": True,
            "date_column": date_column,
            "min_date": min_date.strftime("%Y-%m-%d"),
            "max_date": max_date.strftime("%Y-%m-%d"),
            "default_start_date": default_start.strftime("%Y-%m-%d"),
            "default_end_date": max_date.strftime("%Y-%m-%d"),
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== TIME-SERIES ENDPOINTS ====================

@app.route("/api/ml/detect-timeseries", methods=["POST"])
def detect_timeseries():
    """Detect if dataset is suitable for time-series forecasting."""
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    target_column = data.get("target_column")

    if not target_column or target_column not in df.columns:
        return jsonify({"success": False, "error": "target_column is required"}), 400

    datetime_cols = []
    for col in df.columns:
        if col == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                non_null = df[col].notna()
                if non_null.sum() > 0 and parsed[non_null].notna().mean() > 0.5:
                    datetime_cols.append(col)
            except Exception:
                pass

    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_column])
    row_count = len(df)

    return jsonify({
        "success": True,
        "is_timeseries_candidate": len(datetime_cols) > 0 and is_numeric_target and row_count >= 30,
        "datetime_columns": datetime_cols,
        "is_numeric_target": is_numeric_target,
        "row_count": row_count,
        "minimum_required": 30,
    }), 200


@app.route("/api/ml/train-timeseries", methods=["POST"])
def ml_train_timeseries():
    df, filename = get_active_dataset()
    if df is None:
        return jsonify({"success": False, "error": "No active dataset"}), 400

    data = request.get_json(silent=True) or {}
    target_column = data.get("target_column")
    datetime_column = data.get("datetime_column")
    feature_columns = data.get("feature_columns") or []
    forecast_horizon = data.get("forecast_horizon", 30)
    frequency = data.get("frequency", "D")
    test_size = data.get("test_size", 0.2)
    aggregation = data.get("aggregation", "sum")
    model_types = data.get("model_type", ["arimax"])

    if not target_column or target_column not in df.columns:
        return jsonify({"success": False, "error": "target_column is required"}), 400
    if not datetime_column or datetime_column not in df.columns:
        return jsonify({"success": False, "error": "datetime_column is required"}), 400

    config = {
        "target_column": target_column,
        "datetime_column": datetime_column,
        "feature_columns": [c for c in feature_columns if c in df.columns and c != target_column and c != datetime_column],
        "forecast_horizon": forecast_horizon,
        "frequency": frequency,
        "test_size": test_size,
        "aggregation": aggregation,
    }

    model_trainers = {
        "arimax": ("ARIMAX", train_arimax_model),
        "xgboost": ("XGBoost", train_xgboost_model),
    }

    def generate():
        latest_progress = {"value": None}

        def progress_callback(progress_data):
            latest_progress["value"] = progress_data

        result_container = {}

        def train_in_background():
            try:
                all_results = []
                total = len(model_types)
                for idx, mt in enumerate(model_types):
                    if mt not in model_trainers:
                        continue
                    model_label, trainer_fn = model_trainers[mt]

                    def nested_callback(data, _idx=idx, _label=model_label, _total=total):
                        progress_callback(data)

                    try:
                        output = trainer_fn(df, config, progress_callback=nested_callback)
                        output["model_type"] = mt
                        output["model_label"] = model_label
                        all_results.append(output)
                    except Exception as e:
                        import traceback
                        print(f"[train-timeseries] {model_label} FAILED: {e}")
                        traceback.print_exc()

                result_container["output"] = all_results
                result_container["success"] = True
            except Exception as e:
                import traceback
                print(f"[train-timeseries] BACKGROUND ERROR: {e}")
                traceback.print_exc()
                result_container["error"] = str(e)
                result_container["success"] = False

        import threading
        import time

        yield f"data: {json.dumps({'type': 'progress', 'percentage': 0, 'step': 'Starting...'}, cls=SafeJsonEncoder)}\n\n"

        thread = threading.Thread(target=train_in_background)
        thread.start()

        while thread.is_alive():
            if latest_progress["value"]:
                yield f"data: {json.dumps(latest_progress['value'], cls=SafeJsonEncoder)}\n\n"
                latest_progress["value"] = None
            time.sleep(0.2)

        thread.join()

        if latest_progress["value"]:
            yield f"data: {json.dumps(latest_progress['value'], cls=SafeJsonEncoder)}\n\n"

        if result_container.get("success"):
            all_results = result_container["output"]

            if not all_results:
                yield f"data: {json.dumps({'type': 'complete', 'success': False, 'error': 'No valid models trained'}, cls=SafeJsonEncoder)}\n\n"
                return

            # Find best model by R2
            best_result = max(all_results, key=lambda r: r.get("test_metrics", {}).get("r2", -999))

            results_array = []
            for r in all_results:
                results_array.append({
                    "model_type": r["model_type"],
                    "model_label": r["model_label"],
                    "test_metrics": r["test_metrics"],
                    "forecast_data": r["forecast_data"],
                    "train_rows": r["train_rows"],
                    "test_rows": r["test_rows"],
                    "order": r.get("order"),
                    "aic": r.get("aic"),
                })

            # Store state
            ts_state = DATASETS.get(ACTIVE_DATASET_ID, {})
            if "timeseries" not in ts_state:
                ts_state["timeseries"] = {}
            ts_state["timeseries"]["config"] = config
            ts_state["timeseries"]["model_type"] = model_types
            ts_state["timeseries"]["results"] = results_array
            ts_state["timeseries"]["best_model"] = best_result["model_type"]
            ts_state["timeseries"]["model_fits"] = {r["model_type"]: r["model_fit"] for r in all_results}
            DATASETS[ACTIVE_DATASET_ID] = ts_state

            add_audit_event(
                action="Train Time-Series Model",
                details=f"Models={model_types}; target={target_column}; horizon={forecast_horizon}; best={best_result['model_type']}",
                strategy="Multi-model time-series comparison",
            )

            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'data': {
                'task_type': 'timeseries',
                'target_column': target_column,
                'datetime_column': datetime_column,
                'feature_columns': config['feature_columns'],
                'forecast_horizon': forecast_horizon,
                'frequency': frequency,
                'test_size': test_size,
                'aggregation': aggregation,
                'model_type': model_types,
                'results': results_array,
                'best_model_key': best_result['model_type'],
            }}, cls=SafeJsonEncoder)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'complete', 'success': False, 'error': result_container.get('error', 'Training failed')}, cls=SafeJsonEncoder)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/ml/timeseries-info", methods=["GET"])
def ml_timeseries_info():
    ds = DATASETS.get(ACTIVE_DATASET_ID)
    if not ds or "timeseries" not in ds:
        return jsonify({"success": False, "error": "No trained time-series model"}), 404

    ts_state = ds["timeseries"]
    return jsonify({
        "success": True,
        "info": get_timeseries_info(ts_state),
    }), 200


@app.route("/api/ml/forecast", methods=["POST"])
def ml_forecast():
    ds = DATASETS.get(ACTIVE_DATASET_ID)
    if not ds or "timeseries" not in ds:
        return jsonify({"success": False, "error": "No trained time-series model"}), 404

    ts_state = ds["timeseries"]
    model_fits = ts_state.get("model_fits", {})
    model_fit = ts_state.get("model_fit")
    results = ts_state.get("results", [])

    # Support both old format (single model_fit) and new format (model_fits dict)
    if not model_fits and not model_fit:
        return jsonify({"success": False, "error": "Model not available"}), 400

    data = request.get_json(silent=True) or {}
    periods = data.get("periods", ts_state.get("config", {}).get("forecast_horizon", 30))
    model_key = data.get("model_key", ts_state.get("best_model", "arimax"))

    try:
        active_fit = model_fits.get(model_key) or model_fit
        if active_fit is None:
            active_fit = next(iter(model_fits.values()), None) if model_fits else model_fit

        # For XGBoost, return stored forecast from training results
        if model_key == "xgboost" and isinstance(results, list):
            for r in results:
                if r.get("model_type") == "xgboost":
                    stored = r.get("forecast_data", {})
                    return jsonify({
                        "success": True,
                        "forecast": {
                            "values": stored.get("forecast_values", []),
                            "lower_bound": stored.get("lower_bound", []),
                            "upper_bound": stored.get("upper_bound", []),
                            "dates": stored.get("forecast_dates", []),
                        },
                    }), 200

        # For ARIMAX, generate fresh forecast
        frequency = ts_state.get("config", {}).get("frequency", "D")
        # Get last date from the dataset
        import pandas as pd
        last_date = None
        if "df" in ds:
            df = ds["df"]
            datetime_col = ts_state.get("config", {}).get("datetime_column")
            if datetime_col and datetime_col in df.columns:
                last_date = pd.to_datetime(df[datetime_col]).max()
        forecast = generate_forecast(active_fit, periods, frequency=frequency, last_date=last_date)
        return jsonify({
            "success": True,
            "forecast": forecast,
        }), 200
    except Exception as e:
        import traceback
        print(f"[forecast] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    from resource_monitor import start_background_monitor
    start_background_monitor(interval=30, output_file="resource_log.txt")
    app.run(debug=True, port=5000, threaded=True)
    