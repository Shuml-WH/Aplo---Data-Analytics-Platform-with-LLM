import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np


def format_model_label(model_key):
    """Convert model key to human-readable label."""
    labels = {
        "linearregression": "Linear Regression",
        "ridgeregression": "Ridge Regression",
        "gradientboostingregressor": "Gradient Boosting Regressor",
        "decisiontreeregressor": "Decision Tree Regressor",
        "kneighborsregressor": "K-Neighbors Regressor",
        "logisticregression": "Logistic Regression",
        "decisiontreeclassifier": "Decision Tree Classifier",
    }
    return labels.get(model_key, model_key)


def infer_task_type(df: pd.DataFrame, target_column: str) -> str:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    series = df[target_column].dropna()
    if series.empty:
        raise ValueError("Target column contains only missing values.")

    if pd.api.types.is_numeric_dtype(series):
        return "regression"

    return "classification"


def detect_datetime_columns(df, columns):
    """Detect columns that contain datetime-like values."""
    datetime_cols = []
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            non_null = df[col].notna()
            if non_null.sum() > 0 and parsed[non_null].notna().mean() > 0.5:
                datetime_cols.append(col)
        except Exception:
            pass
    return datetime_cols


def decompose_datetimes(df, datetime_cols):
    """Decompose datetime columns into numeric features (year, month, day, day_of_week, hour)."""
    result = df.copy()
    new_cols = []
    for col in datetime_cols:
        dt = pd.to_datetime(result[col], errors="coerce")
        result[f"{col}_year"] = dt.dt.year
        result[f"{col}_month"] = dt.dt.month
        result[f"{col}_day"] = dt.dt.day
        result[f"{col}_day_of_week"] = dt.dt.dayofweek
        if dt.notna().any() and dt.dt.hour.nunique() > 1:
            result[f"{col}_hour"] = dt.dt.hour
            new_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_day_of_week", f"{col}_hour"])
        else:
            new_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_day_of_week"])
        result = result.drop(columns=[col])
    return result, new_cols


def build_preprocessing_plan(df: pd.DataFrame, feature_columns: list, target_column: str, task_type: str) -> dict:
    missing_summary = {
        col: int(df[col].isnull().sum())
        for col in feature_columns
        if col in df.columns
    }

    numeric_features = df[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    target_series = df[target_column]
    target_null_count = int(target_series.isnull().sum())
    target_unique_count = int(target_series.nunique(dropna=True))

    class_distribution = None
    if task_type == "classification":
        class_distribution = {
            str(k): int(v)
            for k, v in target_series.dropna().value_counts().to_dict().items()
        }

    recommended_models = (
        ["logisticregression", "decisiontreeclassifier"] 
        if task_type == "classification" 
        else ["linearregression", "ridgeregression", "gradientboostingregressor", "decisiontreeregressor", "kneighborsregressor"]
    )

    return {
        "task_type": task_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_summary": missing_summary,
        "target_summary": {
            "null_count": target_null_count,
            "unique_count": target_unique_count,
            "class_distribution": class_distribution,
        },
        "recommendations": {
            "numeric_strategy": "median",
            "categorical_strategy": "most_frequent",
            "use_scaling": True,
            "test_size": 0.2,
            "random_state": 42,
            "models": recommended_models,
        },
    }



def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    use_scaling: bool = True,
):
    # Numeric pipeline
    numeric_steps = []
    if numeric_strategy == "drop":
        # Rows with missing values are dropped in train_models before this is called
        # Still need a valid imputer for the pipeline to work
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    else:
        numeric_steps.append(("imputer", SimpleImputer(strategy=numeric_strategy)))
    if use_scaling:
        numeric_steps.append(("scaler", StandardScaler()))

    # Categorical pipeline - encoder is ALWAYS required
    categorical_steps = []
    if categorical_strategy != "drop":
        categorical_steps.append(("imputer", SimpleImputer(strategy=categorical_strategy)))
    categorical_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=categorical_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor


def evaluate_classification_model(model, X_test, y_test, label_encoder=None):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    if label_encoder is not None:
        class_labels = [str(x) for x in label_encoder.classes_.tolist()]
        actual_labels = [str(x) for x in label_encoder.inverse_transform(y_test)]
        predicted_labels = [str(x) for x in label_encoder.inverse_transform(y_pred)]
    else:
        unique_labels = sorted(pd.Series(y_test).unique().tolist())
        class_labels = [str(x) for x in unique_labels]
        actual_labels = [str(x) for x in y_test]
        predicted_labels = [str(x) for x in y_pred]

    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": cm.tolist(),
        },
        "chart_data": {
            "confusion_matrix": {
                "labels": class_labels,
                "matrix": cm.tolist(),
            },
            "actual_vs_predicted": {
                "actual": actual_labels,
                "predicted": predicted_labels,
            },
        },
    }



def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    actual = pd.Series(y_test).astype(float).reset_index(drop=True)
    predicted = pd.Series(y_pred).astype(float).reset_index(drop=True)

    safe_actual = actual.replace([np.inf, -np.inf], np.nan)
    safe_predicted = predicted.replace([np.inf, -np.inf], np.nan)

    valid_mask = safe_actual.notna() & safe_predicted.notna()
    safe_actual = safe_actual[valid_mask].reset_index(drop=True)
    safe_predicted = safe_predicted[valid_mask].reset_index(drop=True)

    if len(safe_actual) == 0:
        raise ValueError("No valid regression evaluation rows remain after filtering NaN/inf values.")

    non_zero_mask = safe_actual != 0
    pct_error = ((safe_predicted[non_zero_mask] - safe_actual[non_zero_mask]) / safe_actual[non_zero_mask]) * 100.0


    return {
        "metrics": {
            "mae": float(mean_absolute_error(safe_actual, safe_predicted)),
            "rmse": float(np.sqrt(mean_squared_error(safe_actual, safe_predicted))),
            "r2": float(r2_score(safe_actual, safe_predicted)),
        },
        "chart_data": {
            "predicted_vs_actual": {
                "actual": safe_actual.round(6).tolist(),
                "predicted": safe_predicted.round(6).tolist(),
            },
            "percentage_error_hist": {
                "values": pct_error.replace([np.inf, -np.inf], np.nan).dropna().round(6).tolist()
            }
        }
    }


def get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        cat_encoder = (
            preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
        )
        cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_names)

    return feature_names


def extract_feature_insights(pipeline, model_key, numeric_features, categorical_features):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = get_transformed_feature_names(
        preprocessor, numeric_features, categorical_features
    )

    # Models with feature_importances_ (tree-based models)
    importance_models = [
        "decisiontreeclassifier", 
        "decisiontreeregressor",
        "gradientboostingregressor", 
        "gradientboostingclassifier"
    ]
    
    if model_key in importance_models and hasattr(model, "feature_importances_"):
        ranked = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"feature": name, "importance": float(score)}
            for name, score in ranked[:15]
        ]

    # Models with coef_ (linear models)
    if model_key == "logisticregression" and hasattr(model, "coef_"):
        if len(model.coef_.shape) == 2 and model.coef_.shape[0] >= 1:
            coef = model.coef_[0]
            ranked = sorted(
                zip(feature_names, coef),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            return [
                {"feature": name, "coefficient": float(score)}
                for name, score in ranked[:15]
            ]

    if model_key in ["linearregression", "ridgeregression"] and hasattr(model, "coef_"):
        coef = model.coef_
        ranked = sorted(
            zip(feature_names, coef),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return [
            {"feature": name, "coefficient": float(score)}
            for name, score in ranked[:15]
        ]

    # K-Neighbors: No feature importance available
    return []


def train_models(df: pd.DataFrame, config: dict, progress_callback=None) -> dict:
    target_column = config["target_column"]
    feature_columns = config["feature_columns"]
    task_type = config["task_type"]

    feature_columns = [c for c in feature_columns if c and c != target_column]
    feature_columns = list(dict.fromkeys(feature_columns))

    if not target_column:
        raise ValueError("Target column is required.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if not feature_columns:
        raise ValueError("At least one feature column is required.")

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in dataset: {missing_features}")

    work_df = df[feature_columns + [target_column]].copy()
    work_df = work_df.loc[:, ~work_df.columns.duplicated()].copy()
    work_df = work_df.dropna(subset=[target_column])

    # Handle "drop" strategy: remove rows with any missing values in features
    numeric_strategy = config.get("numeric_strategy", "median")
    categorical_strategy = config.get("categorical_strategy", "most_frequent")
    if numeric_strategy == "drop" or categorical_strategy == "drop":
        before_count = len(work_df)
        work_df = work_df.dropna(subset=feature_columns)
        after_count = len(work_df)
        print(f"[ml] 'drop' strategy: removed {before_count - after_count} rows with missing values ({before_count} -> {after_count})")

    # Detect and decompose datetime columns into numeric features
    datetime_cols = detect_datetime_columns(work_df, feature_columns)
    if datetime_cols:
        work_df, decomposed_cols = decompose_datetimes(work_df, datetime_cols)
        feature_columns = [c for c in feature_columns if c not in datetime_cols] + decomposed_cols

    if work_df.empty:
        raise ValueError("No rows remain after removing missing target values.")

    X = work_df[feature_columns].copy()
    y_raw = work_df[target_column].copy()



   # force y_raw to be a 1D Series even if duplicate labels slipped through
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0].copy()

 
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    numeric_strategy = config.get("numeric_strategy", "median")
    categorical_strategy = config.get("categorical_strategy", "most_frequent")
    use_scaling = bool(config.get("use_scaling", True))
    test_size = float(config.get("test_size", 0.2))
    random_state = int(config.get("random_state", 42))
    # model_keys = config.get("models", ["logistic_regression", "random_forest"])


    if task_type == "classification":
        model_keys = config.get("models", ["logisticregression", "decisiontreeclassifier"])
    elif task_type == "regression":
        model_keys = config.get("models", ["linearregression", "ridgeregression", "gradientboostingregressor", "decisiontreeregressor", "kneighborsregressor"])
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        use_scaling=use_scaling,
    )

    # debug
    print("target_column:", target_column)
    print("feature_columns contains target:", target_column in feature_columns)
    print("X shape:", X.shape)
    print("y_raw type:", type(y_raw))
    print("y_raw ndim:", getattr(y_raw, "ndim", None))
    print("duplicate columns:", work_df.columns[work_df.columns.duplicated()].tolist())
    print("model_keys from config:", model_keys)


# Classification Modeling
    if task_type == "classification":
        y_raw = y_raw.astype(str)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        class_counts = pd.Series(y).value_counts()
        use_stratify = class_counts.min() >= 2

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if use_stratify else None,
        )


        available_models = {
            "logisticregression": LogisticRegression(max_iter=1000),
            "decisiontreeclassifier": DecisionTreeClassifier(random_state=random_state),
        }

        trained_models = {}
        results = []

        for i, model_key in enumerate(model_keys):
            if model_key not in available_models:
                continue

            # Send progress update
            if progress_callback:
                progress_callback({
                    "type": "progress",
                    "current_model": format_model_label(model_key),
                    "model_index": i + 1,
                    "total_models": len(model_keys),
                    "percentage": int((i / len(model_keys)) * 100),
                    "status": "training"
                })

            estimator = available_models[model_key]

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            print(f"[ml] starting classification model: {model_key}")
            print(f"[ml] train rows={len(X_train)}, test rows={len(X_test)}")

            try:
                if model_key == "logisticregression":
                    print("[ml] logisticregression: fitting pipeline...")
                elif model_key == "decisiontreeclassifier":
                    print("[ml] decisiontreeclassifier: fitting pipeline...")

                pipeline.fit(X_train, y_train)

                print(f"[ml] fit completed for: {model_key}")
                print(f"[ml] evaluating model: {model_key}")


                evaluation = evaluate_classification_model(
                    pipeline,
                    X_test,
                    y_test,
                    label_encoder=label_encoder,
                )
                metrics = evaluation["metrics"]
                chart_data = evaluation["chart_data"]
                
                print(f"[ml] metrics for {model_key}: {metrics}")


                insights = extract_feature_insights(
                    pipeline,
                    model_key,
                    numeric_features,
                    categorical_features,
                )

                trained_models[model_key] = {
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "insights": insights,
                    "chart_data": chart_data,
                }

                results.append({
                    "model_key": model_key,
                    "metrics": metrics,
                    "insights": insights,
                    "chart_data": chart_data,
                })
            except Exception as e:
                print(f"[ml] Warning: {model_key} failed: {e}. Continuing with remaining models.")
                continue

        if not results:
            raise ValueError("No valid models were selected for training.")

        # Send final progress
        if progress_callback:
            progress_callback({
                "type": "progress",
                "current_model": "Complete",
                "model_index": len(model_keys),
                "total_models": len(model_keys),
                "percentage": 100,
                "status": "complete"
            })

        best_model = max(results, key=lambda r: r["metrics"]["accuracy"])

        return {
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "label_classes": label_encoder.classes_.tolist(),
            "trained_models": trained_models,
            "results": results,
            "best_model_key": best_model["model_key"],
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "label_encoder": label_encoder,
            "datetime_columns": datetime_cols,
        }
    



# Regression Modeling
    elif task_type == "regression": 
        y = pd.to_numeric(y_raw, errors="coerce")
        valid_mask = y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        if len(X) == 0:
            raise ValueError("No valid numeric target values remain for regression.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        available_models = {
            "linearregression": LinearRegression(),
            "ridgeregression": Ridge(random_state=random_state),
            "gradientboostingregressor": GradientBoostingRegressor(random_state=random_state),
            "decisiontreeregressor": DecisionTreeRegressor(random_state=random_state),
            "kneighborsregressor": KNeighborsRegressor(),
        }

        trained_models = {}
        results = []

        for i, model_key in enumerate(model_keys):
            if model_key not in available_models:
                continue

            # Send progress update
            if progress_callback:
                progress_callback({
                    "type": "progress",
                    "current_model": format_model_label(model_key),
                    "model_index": i + 1,
                    "total_models": len(model_keys),
                    "percentage": int((i / len(model_keys)) * 100),
                    "status": "training"
                })

            estimator = available_models[model_key]

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            print(f"[ml] starting regression model: {model_key}")
            print(f"[ml] train rows={len(X_train)}, test rows={len(X_test)}")

            try:
                if model_key == "linearregression":
                    print("[ml] linearregression: fitting pipeline...")
                elif model_key == "ridgeregression":
                    print("[ml] ridgeregression: fitting pipeline...")
                elif model_key == "gradientboostingregressor":
                    print("[ml] gradientboostingregressor: fitting pipeline...")
                elif model_key == "decisiontreeregressor":
                    print("[ml] decisiontreeregressor: fitting pipeline...")
                elif model_key == "kneighborsregressor":
                    print("[ml] kneighborsregressor: fitting pipeline...")

                pipeline.fit(X_train, y_train)

                print(f"[ml] fit completed for: {model_key}")
                print(f"[ml] evaluating model: {model_key}")

                evaluation = evaluate_regression_model(pipeline, X_test, y_test)
                metrics = evaluation["metrics"]
                chart_data = evaluation["chart_data"]

                print(f"[ml] metrics for {model_key}: {metrics}")
                

                insights = extract_feature_insights(
                    pipeline,
                    model_key,
                    numeric_features,
                    categorical_features,
                )

                trained_models[model_key] = {
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "insights": insights,
                    "chart_data": chart_data,
                }

                results.append({
                    "model_key": model_key,
                    "metrics": metrics,
                    "insights": insights,
                    "chart_data": chart_data,
                })
            except Exception as e:
                print(f"[ml] Warning: {model_key} failed: {e}. Continuing with remaining models.")
                continue

        if not results:
            raise ValueError("No valid models were selected for training.")

        # Send final progress
        if progress_callback:
            progress_callback({
                "type": "progress",
                "current_model": "Complete",
                "model_index": len(model_keys),
                "total_models": len(model_keys),
                "percentage": 100,
                "status": "complete"
            })

        best_model = max(results, key=lambda r: r["metrics"]["r2"])

        return {
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "label_classes": None, 
            "trained_models": trained_models,
            "results": results,
            "best_model_key": best_model["model_key"],
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "label_encoder": None, 
            "datetime_columns": datetime_cols,
        }

# If not Classification nor Regression
    else: 
        raise ValueError(f"Unsupported task type: {task_type}")






def predict_rows(model_bundle: dict, rows: list) -> list:
    pipeline = model_bundle["pipeline"]
    label_encoder = model_bundle.get("label_encoder")
    datetime_columns = model_bundle.get("datetime_columns", [])

    input_df = pd.DataFrame(rows)

    # Apply datetime decomposition if the model was trained with it
    if datetime_columns:
        input_df, _ = decompose_datetimes(input_df, datetime_columns)

    pred_encoded = pipeline.predict(input_df)

    if label_encoder is not None:
        preds = label_encoder.inverse_transform(pred_encoded)
        return [str(x) for x in preds]

    return [str(x) for x in pred_encoded]


def compute_feature_sensitivity(model_bundle: dict, feature_columns: list,
                                 numeric_features: list, categorical_features: list,
                                 dataset_df: pd.DataFrame) -> dict:
    """Compute per-unit prediction sensitivity for each numeric feature.

    Uses finite differences: perturb each numeric feature by +1 unit,
    measure the change in prediction. Categorical features get sensitivity 0.
    """
    baseline_row = {}
    for col in feature_columns:
        if col in numeric_features and col in dataset_df.columns:
            val = dataset_df[col].median()
            if pd.isna(val):
                val = dataset_df[col].mean()
            if pd.isna(val):
                val = 0.0
            baseline_row[col] = float(val)
        elif col in dataset_df.columns:
            modes = dataset_df[col].mode()
            baseline_row[col] = str(modes.iloc[0]) if not modes.empty else ""
        else:
            baseline_row[col] = 0

    try:
        baseline_pred = float(predict_rows(model_bundle, [baseline_row])[0])
    except Exception as e:
        print(f"[sensitivity] baseline prediction failed: {e}")
        return {col: 0.0 for col in feature_columns}

    sensitivities = {}
    for col in feature_columns:
        if col in numeric_features:
            perturbed_row = baseline_row.copy()
            current_val = perturbed_row[col]
            if pd.isna(current_val):
                sensitivities[col] = 0.0
                continue
            perturbed_row[col] = float(current_val) + 1.0
            try:
                perturbed_pred = float(predict_rows(model_bundle, [perturbed_row])[0])
                sensitivities[col] = perturbed_pred - baseline_pred
            except Exception as e:
                print(f"[sensitivity] perturbed prediction failed for {col}: {e}")
                sensitivities[col] = 0.0
        else:
            sensitivities[col] = 0.0

    print(f"[sensitivity] baseline_pred={baseline_pred}, sensitivities={sensitivities}")
    return sensitivities


