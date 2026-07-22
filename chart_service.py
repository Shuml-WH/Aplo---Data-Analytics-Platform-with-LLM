import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta


# Time period presets for gauge chart
TIME_PERIOD_MAP = {
    "7d": 7,
    "30d": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "last_week": "last_week",
    "last_month": "last_month",
    "last_quarter": "last_quarter",
    "last_year": "last_year",
}

# Resample rules for time grouping
GROUPBY_RESAMPLE = {
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}

# Date format for x-axis labels after grouping
GROUPBY_FORMAT = {
    "daily": "%Y-%m-%d",
    "weekly": "%Y-W%U",
    "monthly": "%Y-%m",
    "quarterly": "%Y-Q%q",
    "yearly": "%Y",
}


def _filter_by_time_period(df, date_column, time_period, reference_date=None):
    """Filter dataframe to only include rows from the last N days before reference_date."""
    if not time_period or time_period == "all" or not date_column:
        return df

    days = TIME_PERIOD_MAP.get(time_period)
    if not days:
        return df

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])

    # Use reference_date (end of date range) instead of now()
    if reference_date:
        ref = pd.to_datetime(reference_date)
    else:
        ref = df[date_column].max()

    cutoff = ref - pd.Timedelta(days=days)
    return df[df[date_column] >= cutoff]


def _get_period_bounds(time_period, reference_date=None):
    """
    Get start/end dates for a named period.
    Returns (start, end) as pd.Timestamp objects.
    reference_date is the "as of" date (e.g., end of dataset date range).
    """
    if reference_date:
        ref = pd.to_datetime(reference_date)
    else:
        ref = pd.Timestamp.now()

    if time_period == "last_week":
        # Previous full week (Mon-Sun)
        this_monday = ref - pd.Timedelta(days=ref.weekday())
        prev_monday = this_monday - pd.Timedelta(weeks=1)
        prev_sunday = prev_monday + pd.Timedelta(days=6)
        return prev_monday, prev_sunday

    elif time_period == "last_month":
        # Previous full month
        first_of_this_month = ref.replace(day=1)
        prev_month_end = first_of_this_month - pd.Timedelta(days=1)
        prev_month_start = prev_month_end.replace(day=1)
        return prev_month_start, prev_month_end

    elif time_period == "last_quarter":
        # Previous full quarter
        current_quarter = (ref.month - 1) // 3  # 0, 1, 2, 3
        if current_quarter == 0:
            # Currently Q1, so previous quarter is Q4 of last year
            prev_q_start = pd.Timestamp(ref.year - 1, 10, 1)
            prev_q_end = pd.Timestamp(ref.year - 1, 12, 31)
        else:
            # Previous quarter in same year
            prev_q_month = (current_quarter - 1) * 3 + 1
            prev_q_start = pd.Timestamp(ref.year, prev_q_month, 1)
            if current_quarter == 1:
                prev_q_end = pd.Timestamp(ref.year, 3, 31)
            elif current_quarter == 2:
                prev_q_end = pd.Timestamp(ref.year, 6, 30)
            else:
                prev_q_end = pd.Timestamp(ref.year, 9, 30)
        return prev_q_start, prev_q_end

    elif time_period == "last_year":
        # Previous full year (Jan 1 - Dec 31)
        prev_year_start = pd.Timestamp(ref.year - 1, 1, 1)
        prev_year_end = pd.Timestamp(ref.year - 1, 12, 31)
        return prev_year_start, prev_year_end

    return None, None


def _compute_previous_period_target(df, date_column, y_col, agg, time_period, factor=1.0, factor_mode="multiply"):
    """
    Compute target from previous period.

    Args:
        df: full dataset
        date_column: name of date column
        y_col: numeric column to aggregate
        agg: aggregation function ('sum', 'mean', 'median')
        time_period: 'last_week', 'last_month', 'last_quarter', 'last_year'
        factor: multiplier or additive value
        factor_mode: 'multiply' or 'add'

    Returns:
        target value (float) or None if no data
    """
    if not date_column or date_column not in df.columns:
        return None
    if not y_col or y_col not in df.columns:
        return None

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])

    # Use dataset's max date as reference, not now()
    ref_date = df[date_column].max()

    start, end = _get_period_bounds(time_period, reference_date=ref_date)
    if start is None:
        return None

    # Filter to previous period
    mask = (df[date_column] >= start) & (df[date_column] <= end)
    df_prev = df[mask]

    if df_prev.empty:
        return None

    # Aggregate
    agg_funcs = {"sum": "sum", "mean": "mean", "median": "median"}
    agg_func = agg_funcs.get(agg, "sum")
    target = float(df_prev[y_col].agg(agg_func))

    # Apply factor
    if factor_mode == "add":
        target = target + factor
    else:
        target = target * factor

    return target


def _apply_groupby(df, x, y_cols, agg, group_by):
    """Resample datetime x-axis by the selected time period, then aggregate y."""
    if not group_by or group_by == "none":
        return df

    if x not in df.columns:
        return df

    resample_freq = GROUPBY_RESAMPLE.get(group_by)
    if not resample_freq:
        return df

    # Convert x to datetime for resampling
    df[x] = pd.to_datetime(df[x], errors="coerce")
    df = df.dropna(subset=[x])
    df = df.set_index(x).sort_index()

    # Build aggregation dict for y columns only
    agg_func = agg or "sum"
    agg_dict = {col: agg_func for col in y_cols if pd.api.types.is_numeric_dtype(df[col])}
    if not agg_dict:
        df = df.reset_index()
        return df

    df = df.resample(resample_freq).agg(agg_dict).dropna(how="all")
    df = df.reset_index()

    # Format x-axis labels for display
    fmt = GROUPBY_FORMAT.get(group_by)
    if fmt and pd.api.types.is_datetime64_any_dtype(df[x]):
        if group_by == "quarterly":
            # Quarter format: 2023-Q4
            df[x] = df[x].apply(lambda d: f"{d.year}-Q{(d.month - 1) // 3 + 1}")
        else:
            df[x] = df[x].dt.strftime(fmt)

    return df


def build_chart_figure(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y,
    agg: str | None = None,
    date_column: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    target: float | None = None,
    group_by: str | None = None,
    time_period: str | None = None,
    cat_group_by: str | None = None,
    bar_mode: str | None = None,
    factor: float | None = None,
    factor_mode: str | None = None,
    target_source: str | None = None,
    target_period: str | None = None,
    target_agg: str | None = None,
):
    """
    df: active dataset
    chart_type: "bar" | "line" | "scatter" | "pie" | "gauge"
    x: column name for x axis (not required for gauge)
    y: column name or list of names for y axis
    agg: optional aggregation: "sum" | "mean" | "median" | None
    date_column: optional global date filter column
    date_from/date_to: optional inclusive range in YYYY-MM-DD
    target: optional range max value for gauge chart
    group_by: optional time grouping: "daily" | "weekly" | "monthly" | "quarterly" | "yearly"
    time_period: optional time period filter for gauge: "7d" | "30d" | "3m" | "6m" | "1y" | "all"
    """

    if isinstance(y, str):
        y_cols = [y]
    else:
        y_cols = list(y)

    # Validate columns (gauge doesn't need x)
    cols_to_validate = ([x] + y_cols) if x else y_cols
    for col in cols_to_validate:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in dataset")

    df_plot = df.copy()

    # Apply global date filter first
    if date_column:
        if date_column not in df_plot.columns:
            raise ValueError(f"Date column '{date_column}' not in dataset")

        df_plot[date_column] = pd.to_datetime(df_plot[date_column], errors="coerce")
        df_plot = df_plot.dropna(subset=[date_column])

        if date_from:
            start_ts = pd.to_datetime(date_from, errors="coerce")
            if pd.isna(start_ts):
                raise ValueError(f"Invalid date_from '{date_from}'")
            df_plot = df_plot[df_plot[date_column] >= start_ts]

        if date_to:
            end_ts = pd.to_datetime(date_to, errors="coerce")
            if pd.isna(end_ts):
                raise ValueError(f"Invalid date_to '{date_to}'")
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            df_plot = df_plot[df_plot[date_column] <= end_ts]

    if df_plot.empty:
        raise ValueError("No data left after applying the selected date filter")

    # For gauge charts, save full data before time filtering (used for Target Max)
    df_gauge_full = df_plot.copy() if chart_type == "gauge" else None

    # Apply time period filter (for gauge) — use date_to as reference if available
    if time_period and time_period != "all":
        date_col_for_period = date_column or x
        reference_date = None
        if date_to:
            reference_date = pd.to_datetime(date_to)
        df_plot = _filter_by_time_period(df_plot, date_col_for_period, time_period, reference_date)
        if df_plot.empty:
            raise ValueError(f"No data left after applying time period filter ({time_period})")

    # Build Plotly figure
    if chart_type == "bar":
        # Auto-aggregate for bar charts
        if not agg:
            numeric_y = [col for col in y_cols if pd.api.types.is_numeric_dtype(df_plot[col])]
            if numeric_y:
                agg = "sum"
                y_cols = numeric_y

        # Apply time grouping
        df_plot = _apply_groupby(df_plot, x, y_cols, agg, group_by)

        # Determine barmode
        barmode = bar_mode if bar_mode in ("stack", "group") else "group"

        # Categorical group by (side-by-side or stacked grouped bars)
        if cat_group_by and cat_group_by in df_plot.columns and len(y_cols) == 1:
            y_col = y_cols[0]
            agg_func = agg or "sum"
            # Group by both x and cat_group_by, then aggregate
            if pd.api.types.is_numeric_dtype(df_plot[y_col]):
                pivot_df = df_plot.groupby([x, cat_group_by], as_index=False).agg({y_col: agg_func})
                fig = px.bar(pivot_df, x=x, y=y_col, color=cat_group_by,
                    title=f"Bar chart for {y_col} vs {x} (grouped by {cat_group_by})",
                    barmode=barmode)
            else:
                fig = px.bar(df_plot, x=x, y=y_cols, title=f"Bar chart for {', '.join(y_cols)} vs {x}",
                    barmode=barmode)
        else:
            # Regular aggregation (no cat_group_by)
            if not group_by or group_by == "none":
                if agg:
                    agg_funcs = {"sum": "sum", "mean": "mean", "median": "median"}
                    if agg not in agg_funcs:
                        raise ValueError(f"Unsupported aggregation '{agg}'")
                    agg_dict = {col: agg_funcs[agg] for col in y_cols if pd.api.types.is_numeric_dtype(df_plot[col])}
                    if agg_dict:
                        df_plot = df_plot.groupby(x, as_index=False).agg(agg_dict)

            fig = px.bar(df_plot, x=x, y=y_cols,
                title=f"Bar chart for {', '.join(y_cols)} vs {x}",
                barmode=barmode)
        
        fig.update_layout(
            xaxis_title=x, yaxis_title=", ".join(y_cols),
            margin=dict(l=40, r=20, t=60, b=80), autosize=True,
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        )

        # When group_by is active, force x-axis to category and add data labels
        if group_by and group_by != "none":
            fig.update_layout(xaxis_type="category")
            for trace in fig.data:
                trace.text = [f"{v:,.0f}" if v >= 1000 else f"{v}" for v in trace.y]
                trace.textposition = "outside"

    elif chart_type == "line":
        # Auto-aggregate for line charts
        if not agg:
            numeric_y = [col for col in y_cols if pd.api.types.is_numeric_dtype(df_plot[col])]
            if numeric_y:
                agg = "sum"
                y_cols = numeric_y

        # Apply time grouping
        df_plot = _apply_groupby(df_plot, x, y_cols, agg, group_by)

        # Additional aggregation if no group_by
        if not group_by or group_by == "none":
            if agg:
                agg_funcs = {"sum": "sum", "mean": "mean", "median": "median"}
                if agg not in agg_funcs:
                    raise ValueError(f"Unsupported aggregation '{agg}'")
                agg_dict = {col: agg_funcs[agg] for col in y_cols if pd.api.types.is_numeric_dtype(df_plot[col])}
                if agg_dict:
                    df_plot = df_plot.groupby(x, as_index=False).agg(agg_dict)

        fig = px.line(df_plot, x=x, y=y_cols,
            title=f"Line chart for {', '.join(y_cols)} vs {x}")
        fig.update_layout(
            xaxis_title=x, yaxis_title=", ".join(y_cols),
            margin=dict(l=40, r=20, t=60, b=80), autosize=True,
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        )

        # When group_by is active, force x-axis to category
        if group_by and group_by != "none":
            fig.update_layout(xaxis_type="category")

    elif chart_type == "scatter":
        fig = px.scatter(df_plot, x=x, y=y_cols[0],
            title=f"Scatter of {y_cols[0]} vs {x}")
        fig.update_layout(
            xaxis_title=x, yaxis_title=y_cols[0],
            margin=dict(l=40, r=20, t=60, b=80), autosize=True,
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        )

    elif chart_type == "pie":
        # Apply time grouping if specified
        df_plot = _apply_groupby(df_plot, x, y_cols, "sum", group_by)

        # Always aggregate by sum for pie
        agg_dict = {y_cols[0]: "sum"}
        df_pie = df_plot.groupby(x, as_index=False).agg(agg_dict)
        fig = px.pie(df_pie, names=x, values=y_cols[0],
                     title=f"Pie chart of {y_cols[0]} by {x}")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=40, r=20, t=60, b=60))

    elif chart_type == "gauge":
        # Aggregate to single value
        if agg:
            agg_funcs = {"sum": "sum", "mean": "mean", "median": "median"}
            value = float(df_plot[y_cols[0]].agg(agg_funcs.get(agg, "sum")))
        else:
            value = float(df_plot[y_cols[0]].sum())

        # Compute target from previous period if requested
        if target_source == "previous_period":
            # Use target_period if provided, otherwise fall back to time_period
            period_for_target = target_period or time_period
            if period_for_target and period_for_target in ("last_week", "last_month", "last_quarter", "last_year"):
                # Find the date column for period comparison
                date_col_for_period = date_column or x
                computed_target = _compute_previous_period_target(
                    df, date_col_for_period, y_cols[0], target_agg or agg or "sum",
                    period_for_target, factor=factor or 1.0, factor_mode=factor_mode or "multiply"
                )
                if computed_target is not None:
                    target = computed_target

        # Target: user-provided or None (no target reference)
        if target is not None:
            target = float(target)
            # Axis must accommodate both value and target
            axis_max = max(value, target)
            # Gauge with target reference (delta + color bands)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                delta={"reference": target},
                gauge={
                    "axis": {"range": [0, axis_max]},
                    "bar": {"color": "#22d3ee"},
                    "steps": [
                        {"range": [0, axis_max * 0.25], "color": "#ef476f"},
                        {"range": [axis_max * 0.25, axis_max * 0.5], "color": "#ffb703"},
                        {"range": [axis_max * 0.5, axis_max * 0.75], "color": "#ffd166"},
                        {"range": [axis_max * 0.75, axis_max], "color": "#06d6a0"},
                    ],
                },
                title={"text": f"{y_cols[0]}"},
            ))
        else:
            # Gauge without target (just number, no delta)
            max_val = float(df_gauge_full[y_cols[0]].max()) if df_gauge_full is not None else value * 1.2
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                gauge={
                    "axis": {"range": [0, max_val]},
                    "bar": {"color": "#22d3ee"},
                },
                title={"text": f"{y_cols[0]}"},
            ))

        fig.update_layout(
            margin=dict(l=30, r=30, t=40, b=20),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eef2"),
        )

    else:
        raise ValueError(f"Unsupported chart_type '{chart_type}'")

    # Return dict so Flask can JSONify easily
    return fig.to_json()
