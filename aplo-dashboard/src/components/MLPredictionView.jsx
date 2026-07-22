import React from 'react';
import PlotModule from "react-plotly.js";

const Plot = PlotModule.default || PlotModule;

const FORMATTED_MODELS = {
    linearregression: "Linear Regression",
    ridgeregression: "Ridge Regression",
    gradientboostingregressor: "Gradient Boosting Regressor",
    decisiontreeregressor: "Decision Tree Regressor",
    kneighborsregressor: "K-Neighbors Regressor",
    logisticregression: "Logistic Regression",
    decisiontreeclassifier: "Decision Tree Classifier",
};

const plotConfig = { responsive: true, displayModeBar: false };
const plotLayout = (overrides) => ({
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#9ca3af", size: 11 },
    margin: { l: 50, r: 20, t: 10, b: 50 },
    ...overrides,
});

function formatModelLabel(modelKey) {
    return FORMATTED_MODELS[modelKey] || modelKey;
}

function HelpIcon({ text }) {
    return (
        <span className="ml-metric-help" style={{ position: "relative", display: "inline-flex" }}>
            ?
            <span className="ml-metric-tooltip" style={{ width: 280, textAlign: "left", lineHeight: 1.5 }}
                  dangerouslySetInnerHTML={{ __html: text }} />
        </span>
    );
}

export default function MLPredictionView({ datasetProfile, onNavigate, initialTab }) {
    const [mlInfo, setMlInfo] = React.useState(null);
    const [loadingInfo, setLoadingInfo] = React.useState(false);
    const [selectedModel, setSelectedModel] = React.useState("");
    const [predictError, setPredictError] = React.useState("");

    const [activeTab, setActiveTab] = React.useState(initialTab || "batch");

    React.useEffect(() => {
        if (initialTab) setActiveTab(initialTab);
    }, [initialTab]);

    const [batchRunning, setBatchRunning] = React.useState(false);
    const [batchProgress, setBatchProgress] = React.useState(0);
    const [batchStep, setBatchStep] = React.useState("");
    const [batchResults, setBatchResults] = React.useState(null);
    const [batchCache, setBatchCache] = React.useState({ predictions: null, modelKey: null });
    const [groupByColumn, setGroupByColumn] = React.useState("");
    const [groupMode, setGroupMode] = React.useState("avg");
    const [showAllGroupColumns, setShowAllGroupColumns] = React.useState(false);
    const [impactMode, setImpactMode] = React.useState("perunit");

    const [whatIfStartingPoint, setWhatIfStartingPoint] = React.useState("average");
    const [whatIfValues, setWhatIfValues] = React.useState({});
    const [whatIfAggregateAll, setWhatIfAggregateAll] = React.useState({});
    const [whatIfResult, setWhatIfResult] = React.useState(null);
    const [whatIfBaselineResult, setWhatIfBaselineResult] = React.useState(null);
    const [whatIfRunning, setWhatIfRunning] = React.useState(false);

    // Forecast tab state
    const [tsInfo, setTsInfo] = React.useState(null);
    const [loadingTsInfo, setLoadingTsInfo] = React.useState(false);
    const [forecastPeriods, setForecastPeriods] = React.useState(30);
    const [forecastPeriodsInput, setForecastPeriodsInput] = React.useState("30");
    const [selectedTsModel, setSelectedTsModel] = React.useState("");
    const [dateRangeStart, setDateRangeStart] = React.useState("");
    const [dateRangeEnd, setDateRangeEnd] = React.useState("");

    React.useEffect(() => { setForecastPeriodsInput(String(forecastPeriods)); }, [forecastPeriods]);
    const [forecastData, setForecastData] = React.useState(null);
    const [forecastRunning, setForecastRunning] = React.useState(false);
    const [forecastError, setForecastError] = React.useState("");

    const debounceRef = React.useRef(null);

    const fetchMlInfo = async () => {
        setLoadingInfo(true);
        setPredictError("");
        try {
            const res = await fetch("http://localhost:5000/api/ml/results");
            const data = await res.json();
            if (!res.ok || !data.success) {
                setMlInfo(null);
                return;
            }
            setMlInfo(data);
            setSelectedModel(data.best_model_key || data.results?.[0]?.model_key || "");
        } catch (err) {
            setMlInfo(null);
        } finally {
            setLoadingInfo(false);
        }
    };

    const fetchTsInfo = async () => {
        setLoadingTsInfo(true);
        setForecastError("");
        try {
            const res = await fetch("http://localhost:5000/api/ml/timeseries-info");
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error(data.error || "No time-series model found.");
            setTsInfo(data.info);
            setForecastPeriods(data.info.forecast_horizon || 30);
        } catch (err) {
            setTsInfo(null);
        } finally {
            setLoadingTsInfo(false);
        }
    };

    const runForecast = async () => {
        if (!tsInfo) { setForecastError("No time-series model available."); return; }
        setForecastRunning(true);
        setForecastError("");
        setForecastData(null);
        try {
            const res = await fetch("http://localhost:5000/api/ml/forecast", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ periods: forecastPeriods })
            });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error(data.error || "Forecast failed");
            setForecastData(data.forecast);
        } catch (err) {
            setForecastError(err.message || String(err));
        } finally {
            setForecastRunning(false);
        }
    };

    const downloadForecastCSV = () => {
        if (!forecastData || !tsInfo) return;
        const headers = ["Date", "Predicted", "Lower Bound", "Upper Bound"];
        const rows = forecastData.values.map((v, i) => [
            `forecast_${i + 1}`,
            v.toFixed(2),
            forecastData.lower_bound[i].toFixed(2),
            forecastData.upper_bound[i].toFixed(2),
        ]);
        const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `forecast_${tsInfo.target_column}_${forecastPeriods}periods.csv`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
    };

    React.useEffect(() => {
        fetchMlInfo();
        fetchTsInfo();
    }, []);

    const trainedFeatureColumns = mlInfo?.config?.feature_columns || [];
    const trainedModels = mlInfo?.results || [];
    const targetColumn = mlInfo?.config?.target_column || "Target";
    const taskType = mlInfo?.config?.task_type || "regression";
    const numericFeatures = mlInfo?.config?.numeric_features || [];
    const categoricalFeatures = mlInfo?.config?.categorical_features || [];
    const datasetColumnProfiles = datasetProfile?.columns_profile || [];

    const getColumnProfile = (colName) =>
        datasetColumnProfiles.find((item) => item.feature === colName) || null;

    const isCategoricalColumn = (colName) => {
        if (categoricalFeatures.includes(colName)) return true;
        if (numericFeatures.includes(colName)) return false;
        const profile = getColumnProfile(colName);
        return profile?.dtype_label !== "Numeric";
    };

    const getCategoricalOptions = (colName) => {
        const profile = getColumnProfile(colName);
        const valuesFromHead = (datasetProfile?.head || [])
            .map((row) => row?.[colName])
            .filter((v) => v !== null && v !== undefined && String(v).trim() !== "")
            .map((v) => String(v));
        const merged = [profile?.top, ...valuesFromHead]
            .filter((v) => v !== null && v !== undefined && String(v).trim() !== "")
            .map((v) => String(v));
        return [...new Set(merged)];
    };

    const getMinMax = (colName) => {
        const profile = getColumnProfile(colName);
        return { min: profile?.min ?? 0, max: profile?.max ?? 100 };
    };

    const getBaselineValues = React.useCallback(() => {
        const values = {};
        trainedFeatureColumns.forEach((col) => {
            if (isCategoricalColumn(col)) {
                const opts = getCategoricalOptions(col);
                if (whatIfStartingPoint === "mode") {
                    const profile = getColumnProfile(col);
                    values[col] = profile?.top || opts[0] || "";
                } else {
                    values[col] = opts[0] || "";
                }
            } else {
                const profile = getColumnProfile(col);
                if (whatIfStartingPoint === "mode") {
                    values[col] = profile?.top ?? profile?.q50 ?? 0;
                } else {
                    values[col] = profile?.q50 ?? profile?.min ?? 0;
                }
            }
        });
        return values;
    }, [trainedFeatureColumns, whatIfStartingPoint, datasetColumnProfiles]);

    React.useEffect(() => {
        if (mlInfo && selectedModel) {
            const baselineVals = getBaselineValues();
            setWhatIfValues(baselineVals);
            setWhatIfAggregateAll({});
            setWhatIfResult(null);
            setWhatIfBaselineResult(null);
            fetch("http://localhost:5000/api/ml/predict-whatif", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_key: selectedModel, features: baselineVals, aggregate_all: {} }),
            })
                .then((r) => r.json())
                .then((data) => { if (data.success) setWhatIfBaselineResult(data); })
                .catch(() => {});
        }
    }, [mlInfo, selectedModel, whatIfStartingPoint]);

    const categoricalColumns = trainedFeatureColumns.filter((col) => isCategoricalColumn(col));

    const autoGroupByColumn = React.useMemo(() => {
        if (groupByColumn && categoricalColumns.includes(groupByColumn)) return groupByColumn;
        const candidates = categoricalColumns.filter((col) => {
            const opts = getCategoricalOptions(col);
            return opts.length >= 2 && opts.length <= 20;
        });
        return candidates[0] || categoricalColumns[0] || "";
    }, [categoricalColumns, groupByColumn, datasetColumnProfiles]);

    const handleWhatIfChange = (col, val) => {
        setWhatIfValues((prev) => ({ ...prev, [col]: val }));
        clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => runWhatIf(), 500);
    };

    const handleWhatIfTypedInput = (col, rawVal) => {
        const num = parseFloat(rawVal.replace(/[$%,]/g, ""));
        if (isNaN(num)) return;
        const { min, max } = getMinMax(col);
        const clamped = Math.max(min, Math.min(max, num));
        setWhatIfValues((prev) => ({ ...prev, [col]: clamped }));
        clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => runWhatIf(), 500);
    };

    const runWhatIf = async () => {
        if (!selectedModel) return;
        setWhatIfRunning(true);
        try {
            const res = await fetch("http://localhost:5000/api/ml/predict-whatif", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_key: selectedModel,
                    features: whatIfValues,
                    aggregate_all: whatIfAggregateAll,
                }),
            });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error(data.error || "Prediction failed");
            setWhatIfResult(data);
        } catch (err) {
            setPredictError(err.message || String(err));
        } finally {
            setWhatIfRunning(false);
        }
    };

    const runBatchPrediction = async (overrideGroupBy) => {
        if (!selectedModel) { setPredictError("Please select a trained model."); return; }
        const groupCol = overrideGroupBy || autoGroupByColumn;
        setBatchRunning(true);
        setBatchProgress(0);
        setBatchStep("Initializing...");
        setPredictError("");

        try {
            const response = await fetch("http://localhost:5000/api/ml/predict-batch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_key: selectedModel, group_by: groupCol }),
            });

            if (!response.ok) {
                let errorMsg = `Server error: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch {}
                console.error("[predict-batch] Error:", errorMsg);
                throw new Error(errorMsg);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const messages = buffer.split("\n\n");
                buffer = messages.pop();
                for (const msg of messages) {
                    const trimmed = msg.trim();
                    if (!trimmed) continue;
                    const dataLines = trimmed.split("\n")
                        .filter((l) => l.startsWith("data: "))
                        .map((l) => l.slice(6))
                        .join("");
                    if (!dataLines) continue;
                    try {
                        const data = JSON.parse(dataLines);
                        if (data.type === "progress") {
                            setBatchProgress(data.percentage);
                            setBatchStep(data.step);
                        } else if (data.type === "complete") {
                            if (data.success) {
                                setBatchResults(data);
                                setGroupByColumn(data.grouped_data?.group_column || autoGroupByColumn);
                            } else {
                                setPredictError(data.error || "Batch prediction failed");
                            }
                        }
                    } catch { }
                }
            }
        } catch (err) {
            setPredictError(err.message || String(err));
        } finally {
            setBatchRunning(false);
        }
    };

    const regroupPredictions = async (newGroupBy) => {
        if (!batchResults || !newGroupBy) return;
        try {
            const res = await fetch("http://localhost:5000/api/ml/regroup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ group_by: newGroupBy }),
            });
            const data = await res.json();
            if (data.success && data.grouped_data) {
                setBatchResults(prev => ({ ...prev, grouped_data: data.grouped_data }));
            }
        } catch (err) {
            console.error("Regroup error:", err);
        }
    };

    const handleGroupByChange = (newGroupBy) => {
        setGroupByColumn(newGroupBy);
        if (batchResults) {
            regroupPredictions(newGroupBy);
        } else {
            runBatchPrediction(newGroupBy);
        }
    };

    const downloadCSV = async () => {
        if (!selectedModel) return;
        try {
            const res = await fetch(`http://localhost:5000/api/ml/export-predictions?model_key=${selectedModel}`);
            if (!res.ok) throw new Error("Download failed");
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `predictions_${selectedModel}.csv`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            setPredictError(err.message || String(err));
        }
    };

    if (!datasetProfile) {
        return (
            <div className="view-container">
                <h2 className="view-title">ML Prediction</h2>
                <div className="card">
                    <p className="text-muted">Upload a dataset first before using the prediction tab.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="view-container">
            <h2 className="view-title">ML Prediction</h2>

            {predictError && !mlInfo && !tsInfo && <p className="error-text small" style={{ marginBottom: 12 }}>Error: {predictError}</p>}

            {!mlInfo && !tsInfo ? (
                <div className="card">
                    <p className="text-muted">Train at least one model in the ML Modeling tab before using prediction.</p>
                </div>
            ) : (
                <>
                    <div style={{ display: "flex", gap: 4, marginBottom: 20, background: "var(--bg-card)", borderRadius: 10, padding: 4, border: "1px solid var(--border-soft)" }}>
                        <button className={`nav-tab ${activeTab === "batch" ? "active" : ""}`}
                            style={{ flex: 1, padding: "10px 16px", border: activeTab === "batch" ? "1px solid var(--accent)" : "1px solid transparent", borderRadius: 8, background: activeTab === "batch" ? "var(--bg-card-soft)" : "transparent", color: activeTab === "batch" ? "var(--accent)" : "var(--text-soft)", cursor: "pointer", fontWeight: 500 }}
                            onClick={() => setActiveTab("batch")}>Batch Prediction</button>
                        <button className={`nav-tab ${activeTab === "whatif" ? "active" : ""}`}
                            style={{ flex: 1, padding: "10px 16px", border: activeTab === "whatif" ? "1px solid var(--accent)" : "1px solid transparent", borderRadius: 8, background: activeTab === "whatif" ? "var(--bg-card-soft)" : "transparent", color: activeTab === "whatif" ? "var(--accent)" : "var(--text-soft)", cursor: "pointer", fontWeight: 500 }}
                            onClick={() => setActiveTab("whatif")}>What-If Scenario</button>
                        <button className={`nav-tab ${activeTab === "timeseries" ? "active" : ""}`}
                            style={{ flex: 1, padding: "10px 16px", border: activeTab === "timeseries" ? "1px solid var(--accent)" : "1px solid transparent", borderRadius: 8, background: activeTab === "timeseries" ? "var(--bg-card-soft)" : "transparent", color: activeTab === "timeseries" ? "var(--accent)" : "var(--text-soft)", cursor: "pointer", fontWeight: 500 }}
                            onClick={() => { setActiveTab("timeseries"); fetchTsInfo(); }}>Time-Series</button>
                    </div>

                    {/* ==================== BATCH PREDICTION TAB ==================== */}
                    {activeTab === "batch" && (
                        <div>
                            {!mlInfo && (
                                <div className="card" style={{ textAlign: "center", padding: "24px" }}>
                                    <p className="text-muted" style={{ marginBottom: "12px" }}>
                                        No regression/classification model trained yet. Train a model in the ML Modeling tab first.
                                    </p>
                                    {onNavigate && (
                                        <button className="btn btn-primary" onClick={() => onNavigate("ml")}>
                                            Go to ML Modeling &rarr;
                                        </button>
                                    )}
                                </div>
                            )}
                            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
                                <button className="btn btn-primary" onClick={() => runBatchPrediction()} disabled={batchRunning || !selectedModel}>
                                    {batchRunning ? "Running..." : "Run Batch Prediction"}
                                </button>
                                <button className="btn btn-ghost" onClick={downloadCSV} disabled={!batchResults || !selectedModel}>
                                    Download CSV
                                </button>
                                <span style={{ fontSize: "0.8rem", color: "var(--text-soft)" }}>
                                    Predicting on {datasetProfile?.shape?.rows ?? "?"} rows
                                </span>
                            </div>

                            {batchRunning && (
                                <div className="progress-wrapper">
                                    <div className="progress-label">
                                        <span>{batchStep}</span>
                                        <span>{batchProgress}%</span>
                                    </div>
                                    <div className="progress-container">
                                        <div className="progress-fill" style={{ width: `${batchProgress}%` }} />
                                    </div>
                                </div>
                            )}

                            {batchResults && (
                                <div>
                                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12, marginBottom: 16 }}>
                                        {[
                                            { label: "Mean", value: `$${batchResults.summary?.mean?.toFixed(2) ?? "-"}` },
                                            { label: "Median", value: `$${batchResults.summary?.median?.toFixed(2) ?? "-"}` },
                                            { label: "Min / Max", value: `$${batchResults.summary?.min?.toFixed(2) ?? "-"} / $${batchResults.summary?.max?.toFixed(2) ?? "-"}` },
                                            { label: "Std Dev", value: `$${batchResults.summary?.std?.toFixed(2) ?? "-"}` },
                                        ].map((s) => (
                                            <div key={s.label} className="stat-card" style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: 14, textAlign: "center" }}>
                                                <div style={{ fontSize: "0.7rem", color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6 }}>{s.label}</div>
                                                <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--accent)" }}>{s.value}</div>
                                            </div>
                                        ))}
                                    </div>

                                    {batchResults.distribution && (
                                        <div className="card">
                                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                                                <h3 style={{ fontSize: "0.95rem" }}>Prediction Distribution</h3>
                                                <HelpIcon text={`This chart shows the distribution of predicted <strong>${targetColumn}</strong> values across all <strong>${batchResults.total_rows}</strong> data instances. Each bar represents how many records fall within that prediction range.`} />
                                            </div>
                                            <div style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: 8 }}>
                                                <Plot
                                                    data={[{
                                                        type: "bar",
                                                        x: batchResults.distribution.bin_edges.slice(0, -1).map((v, i) =>
                                                            `$${Math.round(v)}-$${Math.round(batchResults.distribution.bin_edges[i + 1])}`
                                                        ),
                                                        y: batchResults.distribution.counts,
                                                        marker: { color: "rgba(34, 211, 238, 0.6)", line: { color: "var(--accent)", width: 1 } },
                                                    }]}
                                                    layout={plotLayout({
                                                        xaxis: { title: `Predicted ${targetColumn} per record`, gridcolor: "var(--border-soft)" },
                                                        yaxis: { title: "Number of records", gridcolor: "var(--border-soft)" },
                                                        bargap: 0.05,
                                                    })}
                                                    style={{ width: "100%", height: 300 }}
                                                    config={plotConfig}
                                                    useResizeHandler
                                                />
                                            </div>
                                        </div>
                                    )}

                                    {batchResults.grouped_data && (
                                        <div className="card">
                                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                    <h3 style={{ fontSize: "0.95rem" }}>
                                                        {groupMode === "avg" ? "Grouped Averages" : "Grouped Totals"}
                                                    </h3>
                                                    <HelpIcon text={groupMode === "avg"
                                                        ? `This chart shows the <strong>average</strong> predicted <strong>${targetColumn}</strong> for each <strong>${batchResults.grouped_data.group_column}</strong> group.`
                                                        : `This chart shows the <strong>total sum</strong> of predicted <strong>${targetColumn}</strong> for each <strong>${batchResults.grouped_data.group_column}</strong> group.`} />
                                                </div>
                                                <div className="pred-toggle-group">
                                                    <button className={`pred-toggle-btn ${groupMode === "avg" ? "active" : ""}`} onClick={() => setGroupMode("avg")}>Avg</button>
                                                    <button className={`pred-toggle-btn ${groupMode === "sum" ? "active" : ""}`} onClick={() => setGroupMode("sum")}>Sum</button>
                                                </div>
                                            </div>
                                            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
                                                <div className="form-group" style={{ margin: 0, minWidth: 200 }}>
                                                    <label className="small ml-prediction-label">Group by</label>
                                                    <select className="ml-prediction-field" value={groupByColumn || batchResults.grouped_data.group_column}
                                                        onChange={(e) => handleGroupByChange(e.target.value)}>
                                                        {categoricalColumns.map((col) => (
                                                            <option key={col} value={col}>{col}{col === autoGroupByColumn ? " (auto-suggested)" : ""}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                            </div>
                                            <div className="pred-collapsible-header" onClick={() => setShowAllGroupColumns(!showAllGroupColumns)}>
                                                <span className="arrow">{showAllGroupColumns ? "\u25BC" : "\u25B6"}</span> More grouping options
                                            </div>
                                            {showAllGroupColumns && (
                                                <div className="pred-collapsible-body open">
                                                    <div className="pred-radio-group">
                                                        {categoricalColumns.map((col) => (
                                                            <span key={col}
                                                                className={`pred-radio-pill ${groupByColumn === col ? "active" : ""}`}
                                                                onClick={() => handleGroupByChange(col)}>{col}</span>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                            <div style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: 8 }}>
                                                <Plot
                                                    data={[{
                                                        type: "bar",
                                                        x: batchResults.grouped_data.groups,
                                                        y: groupMode === "avg" ? batchResults.grouped_data.averages : batchResults.grouped_data.sums,
                                                        marker: {
                                                            color: (groupMode === "avg" ? batchResults.grouped_data.averages : batchResults.grouped_data.sums)
                                                                .map((v, _, arr) => v === Math.max(...arr) ? "#06d6a0" : "rgba(34, 211, 238, 0.6)"),
                                                        },
                                                        text: (groupMode === "avg" ? batchResults.grouped_data.averages : batchResults.grouped_data.sums)
                                                            .map((v) => groupMode === "avg" ? `$${v.toFixed(2)}` : `$${v.toLocaleString()}`),
                                                        textposition: "outside",
                                                        textfont: { color: "var(--text-main)", size: 11 },
                                                    }]}
                                                    layout={plotLayout({
                                                        xaxis: { gridcolor: "var(--border-soft)" },
                                                        yaxis: {
                                                            title: groupMode === "avg"
                                                                ? `Avg predicted ${targetColumn} per record ($)`
                                                                : `Total sum of predicted ${targetColumn} ($)`,
                                                            gridcolor: "var(--border-soft)",
                                                            rangemode: "tozero",
                                                            autorange: "tight",
                                                        },
                                                        bargap: 0.3,
                                                    })}
                                                    style={{ width: "100%", height: 300 }}
                                                    config={plotConfig}
                                                    useResizeHandler
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* ==================== WHAT-IF SCENARIO TAB ==================== */}
                    {activeTab === "whatif" && (
                        <div>
                            {!mlInfo && (
                                <div className="card" style={{ textAlign: "center", padding: "24px" }}>
                                    <p className="text-muted" style={{ marginBottom: "12px" }}>
                                        No model trained yet. Train a model in the ML Modeling tab first.
                                    </p>
                                    {onNavigate && (
                                        <button className="btn btn-primary" onClick={() => onNavigate("ml")}>
                                            Go to ML Modeling &rarr;
                                        </button>
                                    )}
                                </div>
                            )}
                            <div className="card" style={{ marginBottom: 16, opacity: mlInfo ? 1 : 0.5, pointerEvents: mlInfo ? "auto" : "none" }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                                    <h4 style={{ fontSize: "0.9rem", margin: 0 }}>Starting Point</h4>
                                    <HelpIcon text={`Choose the starting feature values for your scenario.<br><br><strong>Average Row:</strong> Uses the mean of numeric columns and the most common category — represents the "typical" record.<br><br><strong>Mode Row:</strong> Uses the most frequent value for every column — represents the most common real record in your data.<br><br>When you move a slider, you're asking <em>"what if this feature were different?"</em> For example, increasing price from $52 to $72 means "what would the predicted ${targetColumn} be if we set this feature $20 higher?" The comparison cards show the baseline (starting values) vs. your adjusted scenario, so you can see the exact impact of each change.`} />
                                </div>
                                <div className="form-group" style={{ margin: 0, minWidth: 220 }}>
                                    <label className="small ml-prediction-label">Start from</label>
                                    <select className="ml-prediction-field" value={whatIfStartingPoint} onChange={(e) => setWhatIfStartingPoint(e.target.value)}>
                                        <option value="average">Average Row (dataset mean)</option>
                                        <option value="mode">Mode Row (most frequent values)</option>
                                    </select>
                                </div>
                            </div>

                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, opacity: mlInfo ? 1 : 0.5, pointerEvents: mlInfo ? "auto" : "none" }}>
                                <div style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 10, padding: 16 }}>
                                    <h4 style={{ fontSize: "0.8rem", color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 12 }}>Adjust Features</h4>
                                    {trainedFeatureColumns.map((col) => {
                                        const isCat = isCategoricalColumn(col);
                                        if (isCat) {
                                            const opts = getCategoricalOptions(col);
                                            return (
                                                <div key={col} className="form-group ml-prediction-form-group" style={{ marginBottom: 12 }}>
                                                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                                                        <label className="small ml-prediction-label" style={{ margin: 0 }}>{col}</label>
                                                    </div>
                                                    <select className="ml-prediction-field" value={whatIfValues[col] ?? ""}
                                                        onChange={(e) => {
                                                            const val = e.target.value;
                                                            if (val === "__ALL__") {
                                                                setWhatIfAggregateAll((prev) => ({ ...prev, [col]: true }));
                                                                setWhatIfValues((prev) => ({ ...prev, [col]: "__ALL__" }));
                                                            } else {
                                                                setWhatIfAggregateAll((prev) => ({ ...prev, [col]: false }));
                                                                handleWhatIfChange(col, val);
                                                            }
                                                        }}>
                                                        <option value="__ALL__">All (average across all categories)</option>
                                                        {opts.map((v) => <option key={v} value={v}>{v}</option>)}
                                                    </select>
                                                </div>
                                            );
                                        }
                                        const { min, max } = getMinMax(col);
                                        const val = whatIfValues[col] ?? 0;
                                        return (
                                            <div key={col} className="form-group ml-prediction-form-group" style={{ marginBottom: 12 }}>
                                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                                    <label className="small ml-prediction-label" style={{ margin: 0 }}>{col}</label>
                                                    <span style={{ fontSize: "0.8rem", color: "var(--accent)", fontWeight: 600 }}>{val}</span>
                                                </div>
                                                <div className="pred-slider-row">
                                                    <input type="range" min={min} max={max} step={max > 100 ? 1 : 0.1}
                                                        value={val} onChange={(e) => handleWhatIfChange(col, parseFloat(e.target.value))} />
                                                    <input type="text" className="pred-typed-input"
                                                        value={typeof val === "number" ? val.toFixed(max > 100 ? 0 : 2) : val}
                                                        onChange={(e) => handleWhatIfTypedInput(col, e.target.value)} />
                                                </div>
                                                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-soft)", marginTop: 2 }}>
                                                    <span>{min}</span><span>{max}</span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                    <button className="btn btn-primary" style={{ width: "100%", marginTop: 8 }} onClick={runWhatIf} disabled={whatIfRunning || !selectedModel}>
                                        {whatIfRunning ? "Predicting..." : "Run Scenario"}
                                    </button>
                                </div>

                                <div style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 10, padding: 16 }}>
                                    <h4 style={{ fontSize: "0.8rem", color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 12 }}>Results</h4>
                                    <div className="pred-comparison-grid">
                                        <div className="pred-comparison-card baseline">
                                            <div className="card-label">Baseline</div>
                                            <div className="card-value">{whatIfBaselineResult?.prediction != null ? `$${whatIfBaselineResult.prediction.toFixed(2)}` : "\u2014"}</div>
                                            <div className="card-delta" style={{ color: "var(--text-soft)" }}>{whatIfStartingPoint === "average" ? "Average row" : "Mode row"}</div>
                                        </div>
                                        <div className="pred-comparison-card whatif">
                                            <div className="card-label">Predicted</div>
                                            <div className="card-value">
                                                {whatIfResult?.prediction != null ? `$${whatIfResult.prediction.toFixed(2)}` : "\u2014"}
                                                {whatIfResult?.prediction != null && whatIfBaselineResult?.prediction != null && whatIfBaselineResult.prediction !== 0 ? (() => {
                                                    const pct = ((whatIfResult.prediction - whatIfBaselineResult.prediction) / Math.abs(whatIfBaselineResult.prediction)) * 100;
                                                    if (Math.abs(pct) < 0.01) return null;
                                                    return <span style={{ fontSize: "0.8rem", fontWeight: 500, marginLeft: 6, color: pct > 0 ? "var(--green)" : "var(--red)" }}> ({pct > 0 ? "+" : ""}{pct.toFixed(1)}%)</span>;
                                                })() : null}
                                            </div>
                                            <div className="card-delta">
                                                {whatIfResult?.prediction != null && whatIfBaselineResult?.prediction != null ? (() => {
                                                    const delta = whatIfResult.prediction - whatIfBaselineResult.prediction;
                                                    if (Math.abs(delta) < 0.01) return <span style={{ color: "var(--text-soft)", fontSize: "0.8rem" }}>No change</span>;
                                                    const pct = whatIfBaselineResult.prediction !== 0 ? ((delta) / Math.abs(whatIfBaselineResult.prediction)) * 100 : null;
                                                    return (
                                                        <span className={delta > 0 ? "pred-delta-positive" : "pred-delta-negative"} style={{ fontSize: "0.85rem", fontWeight: 600 }}>
                                                            {delta > 0 ? "\u25B2 +" : "\u25BC "}${Math.abs(delta).toFixed(2)}
                                                            {pct !== null && Math.abs(pct) >= 0.01 ? ` (${pct > 0 ? "+" : ""}${pct.toFixed(1)}%)` : ""}
                                                        </span>
                                                    );
                                                })() : whatIfResult?.aggregate && whatIfResult?.individual_predictions ? (
                                                    <span style={{ fontSize: "0.75rem", color: "var(--text-soft)" }}>
                                                        Avg of {Object.keys(whatIfResult.individual_predictions).length} categories
                                                    </span>
                                                ) : null}
                                            </div>
                                        </div>
                                    </div>

                                    {whatIfResult?.aggregate && whatIfResult?.individual_predictions && (
                                        <div style={{ background: "var(--bg-body)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: 8, marginTop: 12 }}>
                                            <Plot
                                                data={[{
                                                    type: "bar",
                                                    x: Object.keys(whatIfResult.individual_predictions),
                                                    y: Object.values(whatIfResult.individual_predictions),
                                                    marker: { color: "rgba(34, 211, 238, 0.6)" },
                                                    text: Object.values(whatIfResult.individual_predictions).map((v) => `$${v.toFixed(2)}`),
                                                    textposition: "outside",
                                                    textfont: { color: "var(--text-main)", size: 10 },
                                                }]}
                                                layout={plotLayout({
                                                    xaxis: { gridcolor: "var(--border-soft)" },
                                                    yaxis: { title: `Predicted ${targetColumn}`, gridcolor: "var(--border-soft)", autorange: "tight" },
                                                    bargap: 0.3, margin: { l: 50, r: 20, t: 10, b: 60 },
                                                })}
                                                style={{ width: "100%", height: 200 }}
                                                config={plotConfig}
                                                useResizeHandler
                                            />
                                        </div>
                                    )}

                                    {batchResults?.feature_sensitivity && (
                                        <div style={{ background: "var(--bg-body)", border: "1px solid var(--border-soft)", borderRadius: 10, padding: 16, marginTop: 12 }}>
                                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                    <h4 style={{ fontSize: "0.8rem", color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.5px", margin: 0 }}>Feature Impact</h4>
                                                    <HelpIcon text={impactMode === "perunit"
                                                        ? "Shows how much the prediction changes per 1 unit of each feature. Reveals which features have the most leverage."
                                                        : "Shows the total impact of your specific adjustments on the prediction. Hover each bar to see the exact change."} />
                                                </div>
                                                <div className="pred-toggle-group">
                                                    <button className={`pred-toggle-btn ${impactMode === "perunit" ? "active" : ""}`} onClick={() => setImpactMode("perunit")}>Per-Unit</button>
                                                    <button className={`pred-toggle-btn ${impactMode === "total" ? "active" : ""}`} onClick={() => setImpactMode("total")}>Total</button>
                                                </div>
                                            </div>
                                            <div style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: 8 }}>
                                                {impactMode === "perunit" ? (
                                                    <Plot
                                                        data={[{
                                                            type: "bar",
                                                            y: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])),
                                                            x: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])).map((k) => batchResults.feature_sensitivity[k]),
                                                            orientation: "h",
                                                            marker: { color: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])).map((k) => batchResults.feature_sensitivity[k] >= 0 ? "#06d6a0" : "#f97373") },
                                                            text: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])).map((k) => {
                                                                const v = batchResults.feature_sensitivity[k];
                                                                return v === 0 ? "categorical" : `${v >= 0 ? "+" : ""}$${v.toFixed(2)}/unit`;
                                                            }),
                                                            textposition: "inside",
                                                            insidetextanchor: "end",
                                                            textfont: { color: "#0e2a4a", size: 10 },
                                                        }]}
                                                        layout={plotLayout({
                                                            xaxis: { title: "Impact per 1 unit ($)", gridcolor: "var(--border-soft)", zeroline: true, zerolinecolor: "var(--border-strong)" },
                                                            yaxis: { gridcolor: "var(--border-soft)", autorange: "reversed" },
                                                            bargap: 0.3, margin: { l: 140, r: 20, t: 10, b: 40 }, height: 1200,
                                                        })}
                                                        style={{ width: "100%", height: 1300 }}
                                                        config={plotConfig}
                                                        useResizeHandler
                                                    />
                                                ) : (
                                                    <Plot
                                                        data={[{
                                                            type: "bar",
                                                            y: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])),
                                                            x: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])).map((k) => {
                                                                const sens = batchResults.feature_sensitivity[k];
                                                                const baseVal = whatIfValues[k] ?? 0;
                                                                const baseNum = typeof baseVal === "number" ? baseVal : 0;
                                                                return sens * baseNum;
                                                            }),
                                                            orientation: "h",
                                                            marker: { color: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])).map((k) => {
                                                                const sens = batchResults.feature_sensitivity[k];
                                                                const baseVal = whatIfValues[k] ?? 0;
                                                                const baseNum = typeof baseVal === "number" ? baseVal : 0;
                                                                return (sens * baseNum) >= 0 ? "#06d6a0" : "#f97373";
                                                            })},
                                                            text: Object.keys(batchResults.feature_sensitivity).sort((a, b) => Math.abs(batchResults.feature_sensitivity[b]) - Math.abs(batchResults.feature_sensitivity[a])).map((k) => {
                                                                const sens = batchResults.feature_sensitivity[k];
                                                                const baseVal = whatIfValues[k] ?? 0;
                                                                const baseNum = typeof baseVal === "number" ? baseVal : 0;
                                                                const impact = sens * baseNum;
                                                                return impact === 0 ? "no change" : `${impact >= 0 ? "+" : ""}$${impact.toFixed(2)}`;
                                                            }),
                                                            textposition: "inside",
                                                            insidetextanchor: "end",
                                                            textfont: { color: "#0e2a4a", size: 10 },
                                                        }]}
                                                        layout={plotLayout({
                                                            xaxis: { title: "Total Impact ($)", gridcolor: "var(--border-soft)", zeroline: true, zerolinecolor: "var(--border-strong)" },
                                                            yaxis: { gridcolor: "var(--border-soft)", autorange: "reversed" },
                                                            bargap: 0.3, margin: { l: 140, r: 20, t: 10, b: 40 },
                                                        })}
                                                        style={{ width: "100%", height: 1000 }}
                                                        config={plotConfig}
                                                        useResizeHandler
                                                    />
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* ==================== TIME-SERIES TAB ==================== */}
                    {activeTab === "timeseries" && (
                        <div>
                            {!tsInfo && !loadingTsInfo && (
                                <div className="card" style={{ textAlign: "center", padding: "24px" }}>
                                    <p className="text-muted" style={{ marginBottom: "12px" }}>
                                        No time-series model trained yet.
                                    </p>
                                    {onNavigate && (
                                        <button className="btn btn-primary" onClick={() => onNavigate("ml")}>
                                            Go to ML Modeling &rarr;
                                        </button>
                                    )}
                                </div>
                            )}

                            {loadingTsInfo && (
                                <div className="card">
                                    <p className="text-muted">Loading time-series model info...</p>
                                </div>
                            )}

                            {tsInfo && (() => {
                                const fd = tsInfo.forecast_data;
                                const metrics = tsInfo.test_metrics || {};
                                const totalRows = (tsInfo.train_rows || 0) + (tsInfo.test_rows || 0);

                                // Get available models from tsInfo.results
                                const tsModels = tsInfo.results || [];
                                const hasMultipleModels = tsModels.length > 1;

                                // Get active model result - default to best model
                                const bestModelKey = tsInfo.best_model_key || (tsModels[0]?.model_type);
                                const activeResult = hasMultipleModels
                                    ? tsModels.find(r => r.model_type === (selectedTsModel || bestModelKey)) || tsModels.find(r => r.model_type === bestModelKey) || tsModels[0]
                                    : tsModels[0] || {};
                                const activeFd = activeResult.forecast_data || fd;
                                const activeMetrics = activeResult.test_metrics || metrics;

                                // Filter dates by range
                                const filterByRange = (dates, values) => {
                                    if (!dateRangeStart && !dateRangeEnd) return { dates, values };
                                    const filteredDates = [];
                                    const filteredValues = [];
                                    for (let i = 0; i < dates.length; i++) {
                                        const d = dates[i];
                                        if (dateRangeStart && d < dateRangeStart) continue;
                                        if (dateRangeEnd && d > dateRangeEnd) continue;
                                        filteredDates.push(d);
                                        filteredValues.push(values[i]);
                                    }
                                    return { dates: filteredDates, values: filteredValues };
                                };

                                return (
                                <div>
                                    {/* Model switcher & date range */}
                                    <div className="card" style={{ marginBottom: 16 }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                                            {hasMultipleModels && (
                                                <div className="form-group" style={{ margin: 0, minWidth: 150 }}>
                                                    <label className="small ml-prediction-label">Model</label>
                                                    <select className="ml-prediction-field" value={selectedTsModel || bestModelKey} onChange={(e) => setSelectedTsModel(e.target.value)}>
                                                        {tsModels.map((r) => (
                                                            <option key={r.model_type} value={r.model_type}>{r.model_label || r.model_type}{r.model_type === bestModelKey ? " (best)" : ""}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                            )}
                                            <div className="form-group" style={{ margin: 0, minWidth: 140 }}>
                                                <label className="small ml-prediction-label">From</label>
                                                <input type="date" className="ml-prediction-field" value={dateRangeStart} onChange={(e) => setDateRangeStart(e.target.value)} style={{ fontSize: "0.8rem" }} />
                                            </div>
                                            <div className="form-group" style={{ margin: 0, minWidth: 140 }}>
                                                <label className="small ml-prediction-label">To</label>
                                                <input type="date" className="ml-prediction-field" value={dateRangeEnd} onChange={(e) => setDateRangeEnd(e.target.value)} style={{ fontSize: "0.8rem" }} />
                                            </div>
                                            {(dateRangeStart || dateRangeEnd) && (
                                                <button className="btn btn-ghost" onClick={() => { setDateRangeStart(""); setDateRangeEnd(""); }}>Clear</button>
                                            )}
                                            <div style={{ fontSize: "0.8rem", color: "var(--text-soft)", marginLeft: "auto" }}>
                                                {activeResult.order && <>Order: <span className="model-badge">({activeResult.order?.join(",")})</span>{" | "}</>}
                                                Horizon: <span className="model-badge">{activeFd.forecast_dates?.length || tsInfo.forecast_horizon} periods</span>
                                            </div>
                                            <button className="btn btn-ghost" onClick={fetchTsInfo} disabled={loadingTsInfo}>
                                                {loadingTsInfo ? "Refreshing..." : "Refresh"}
                                            </button>
                                        </div>
                                    </div>

                                    {forecastError && <p className="error-text small" style={{ marginBottom: 12 }}>Error: {forecastError}</p>}

                                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "10px", marginBottom: 16 }}>
                                        {[
                                            { label: "MAPE", value: activeMetrics.mape != null ? activeMetrics.mape.toFixed(1) + "%" : "-",
                                              tip: "Mean Absolute Percentage Error. Lower is better. <10% is excellent, 10-20% is good." },
                                            { label: "RMSE", value: activeMetrics.rmse != null && activeMetrics.target_mean != null && activeMetrics.target_mean > 0 ? ((activeMetrics.rmse / activeMetrics.target_mean) * 100).toFixed(1) + "%" : (activeMetrics.rmse != null ? activeMetrics.rmse.toLocaleString(undefined, {maximumFractionDigits:0}) : "-"),
                                              tip: "Root Mean Squared Error as % of mean — penalizes large errors more." },
                                            { label: "R\u00B2", value: activeMetrics.r2 != null ? activeMetrics.r2.toFixed(3) : "-",
                                              tip: "Coefficient of determination. 1.0 = perfect. >0.8 is strong." },
                                        ].map((m) => (
                                            <div key={m.label} className="card" style={{ padding: "10px 12px", textAlign: "center" }}>
                                                <div style={{ fontSize: "0.7rem", color: "var(--text-soft)" }}>{m.label}
                                                    <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                                                    <span className="ml-metric-tooltip" dangerouslySetInnerHTML={{ __html: m.tip }} />
                                                    </span>
                                                </div>
                                                <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent)", marginTop: 2 }}>{m.value}</div>
                                            </div>
                                        ))}
                                    </div>

                                    {activeFd && (() => {
                                        // Filter data by date range
                                        const trainFiltered = filterByRange(activeFd.train_dates, activeFd.train_values);
                                        const testFiltered = filterByRange(activeFd.test_dates, activeFd.test_values);
                                        const testPredFiltered = activeFd.test_predicted ? filterByRange(activeFd.test_dates, activeFd.test_predicted) : null;
                                        const forecastFiltered = filterByRange(activeFd.forecast_dates, activeFd.forecast_values);
                                        const upperFiltered = filterByRange(activeFd.forecast_dates, activeFd.upper_bound);
                                        const lowerFiltered = filterByRange(activeFd.forecast_dates, activeFd.lower_bound);

                                        // Show last 7 test predictions
                                        const lastN = 7;
                                        const testPredSlice = testPredFiltered && testPredFiltered.dates.length > 0
                                            ? { dates: testPredFiltered.dates.slice(-lastN), values: testPredFiltered.values.slice(-lastN) }
                                            : null;
                                        const testActualSlice = testFiltered.dates.length > 0
                                            ? { dates: testFiltered.dates.slice(-lastN), values: testFiltered.values.slice(-lastN) }
                                            : null;

                                        return (
                                    <div className="card" style={{ marginBottom: 16 }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                                            <h3 style={{ fontSize: "0.95rem" }}>Actual vs Forecast &mdash; {tsInfo.target_column}</h3>
                                            <span style={{ fontSize: "0.75rem", color: "var(--text-soft)" }}>
                                                {totalRows.toLocaleString()} data points ({tsInfo.train_rows?.toLocaleString()} train + {tsInfo.test_rows?.toLocaleString()} test)
                                            </span>
                                        </div>
                                        <div style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: 8 }}>
                                            <Plot
                                                data={[
                                                    { x: trainFiltered.dates, y: trainFiltered.values, type: "scatter", mode: "lines", name: "Training (actual)", line: { color: "#22d3ee", width: 1.5 } },
                                                    { x: testFiltered.dates, y: testFiltered.values, type: "scatter", mode: "lines", name: "Test (actual)", line: { color: "#06d6a0", width: 1.5 } },
                                                    testActualSlice && { x: testActualSlice.dates, y: testActualSlice.values, type: "scatter", mode: "lines", name: "Test (actual, last 7)", line: { color: "#06d6a0", width: 2.5 } },
                                                    testPredSlice && { x: testPredSlice.dates, y: testPredSlice.values, type: "scatter", mode: "lines+markers", name: "Model prediction (last 7)", line: { color: "#a78bfa", width: 2, dash: "dot" }, marker: { size: 5 } },
                                                    { x: forecastFiltered.dates, y: forecastFiltered.values, type: "scatter", mode: "lines", name: "Forecast", line: { color: "#f59e0b", width: 2, dash: "dash" } },
                                                    {
                                                        x: upperFiltered.dates.concat([...lowerFiltered.dates].reverse()),
                                                        y: upperFiltered.values.concat([...lowerFiltered.values].reverse()),
                                                        type: "scatter", fill: "toself",
                                                        fillcolor: "rgba(245, 158, 11, 0.12)",
                                                        line: { color: "rgba(245, 158, 11, 0.25)", width: 0.8 },
                                                        name: "95% confidence"
                                                    },
                                                ].filter(Boolean)}
                                                layout={plotLayout({
                                                    xaxis: { title: "Date", gridcolor: "var(--border-soft)" },
                                                    yaxis: { title: tsInfo.target_column, gridcolor: "var(--border-soft)" },
                                                    hovermode: "x unified",
                                                    legend: { orientation: "h", y: -0.25, x: 0.5, xanchor: "center" },
                                                })}
                                                style={{ width: "100%", height: 420 }}
                                                config={plotConfig}
                                                useResizeHandler
                                            />
                                        </div>
                                    </div>
                                    )})()}

                                    <div className="card" style={{ marginBottom: 16 }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                                            <div className="form-group" style={{ margin: 0, minWidth: 200 }}>
                                                <label className="small ml-prediction-label">Forecast Periods</label>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                    <input
                                                        type="range"
                                                        min={7}
                                                        max={365}
                                                        value={forecastPeriods}
                                                        onChange={(e) => setForecastPeriods(parseInt(e.target.value))}
                                                        style={{ flex: 1 }}
                                                    />
                                                    <input
                                                        type="number"
                                                        min={7}
                                                        max={365}
                                                        value={forecastPeriodsInput}
                                                        onChange={(e) => setForecastPeriodsInput(e.target.value)}
                                                        onBlur={() => {
                                                            const v = parseInt(forecastPeriodsInput);
                                                            if (!isNaN(v) && v >= 7 && v <= 365) setForecastPeriods(v);
                                                            else setForecastPeriodsInput(String(forecastPeriods));
                                                        }}
                                                        onKeyDown={(e) => {
                                                            if (e.key === "Enter") {
                                                                const v = parseInt(forecastPeriodsInput);
                                                                if (!isNaN(v) && v >= 7 && v <= 365) setForecastPeriods(v);
                                                                else setForecastPeriodsInput(String(forecastPeriods));
                                                            }
                                                        }}
                                                        style={{ width: "60px", padding: "4px 8px", background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: "6px", color: "var(--accent)", fontWeight: 700, fontSize: "0.9rem", textAlign: "center" }}
                                                    />
                                                    <span style={{ fontSize: "0.8rem", color: "var(--text-soft)" }}>days</span>
                                                </div>
                                            </div>
                                            <button className="btn btn-primary" onClick={runForecast} disabled={forecastRunning || !tsInfo} style={{ marginTop: 16 }}>
                                                {forecastRunning ? "Forecasting..." : "Generate Forecast"}
                                            </button>
                                            <button className="btn btn-ghost" onClick={downloadForecastCSV} disabled={!forecastData} style={{ marginTop: 16 }}>
                                                Download CSV
                                            </button>
                                        </div>
                                    </div>

                                    {forecastData && (
                                        <div>
                                            <div className="card" style={{ marginBottom: 16 }}>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                                                    <h3 style={{ fontSize: "0.95rem" }}>Forecast &mdash; {tsInfo.target_column}</h3>
                                                </div>
                                                <div style={{ background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", borderRadius: 8, padding: 8 }}>
                                                    <Plot
                                                        data={[
                                                            {
                                                                x: forecastData.dates && forecastData.dates.length > 0
                                                                    ? forecastData.dates
                                                                    : Array.from({length: forecastData.values.length}, (_, i) => `T+${i + 1}`),
                                                                y: forecastData.values,
                                                                type: "scatter",
                                                                mode: "lines",
                                                                name: "Forecast",
                                                                line: { color: "#f59e0b", width: 2, dash: "dash" },
                                                            },
                                                            {
                                                                x: forecastData.dates && forecastData.dates.length > 0
                                                                    ? forecastData.dates.concat([...forecastData.dates].reverse())
                                                                    : Array.from({length: forecastData.upper_bound.length}, (_, i) => `T+${i + 1}`).concat(
                                                                        Array.from({length: forecastData.lower_bound.length}, (_, i) => `T+${i + 1}`).reverse()
                                                                      ),
                                                                y: forecastData.upper_bound.concat(forecastData.lower_bound.slice().reverse()),
                                                                type: "scatter",
                                                                fill: "toself",
                                                                fillcolor: "rgba(245, 158, 11, 0.2)",
                                                                line: { color: "rgba(245, 158, 11, 0)" },
                                                                name: "95% confidence",
                                                            },
                                                        ]}
                                                        layout={plotLayout({
                                                            xaxis: { title: "Forecast Period", gridcolor: "var(--border-soft)" },
                                                            yaxis: { title: `Predicted ${tsInfo.target_column}`, gridcolor: "var(--border-soft)" },
                                                            hovermode: "x unified",
                                                            legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center" },
                                                        })}
                                                        style={{ width: "100%", height: 350 }}
                                                        config={plotConfig}
                                                        useResizeHandler
                                                    />
                                                </div>
                                            </div>

                                            <div className="card">
                                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                                                    <h3 style={{ fontSize: "0.95rem" }}>Next {forecastData.values.length} periods forecast</h3>
                                                    <span style={{ fontSize: "0.75rem", color: "#f59e0b", fontWeight: 600 }}>95% confidence</span>
                                                </div>
                                                <div style={{ maxHeight: 300, overflowY: "auto" }}>
                                                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
                                                        <thead>
                                                            <tr>
                                                                <th style={{ textAlign: "left", padding: "10px 12px", borderBottom: "1px solid var(--border-soft)", color: "var(--text-soft)" }}>Period</th>
                                                                <th style={{ textAlign: "left", padding: "10px 12px", borderBottom: "1px solid var(--border-soft)", color: "var(--text-soft)" }}>Predicted</th>
                                                                <th style={{ textAlign: "left", padding: "10px 12px", borderBottom: "1px solid var(--border-soft)", color: "var(--text-soft)" }}>Lower Bound</th>
                                                                <th style={{ textAlign: "left", padding: "10px 12px", borderBottom: "1px solid var(--border-soft)", color: "var(--text-soft)" }}>Upper Bound</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {forecastData.values.map((v, i) => {
                                                                const isEarly = i < Math.ceil(forecastData.values.length / 3);
                                                                const periodLabel = forecastData.dates && forecastData.dates[i]
                                                                    ? forecastData.dates[i]
                                                                    : `T+${i + 1}`;
                                                                return (
                                                                <tr key={i} style={{ background: isEarly ? "rgba(6,214,160,0.06)" : "rgba(34,211,238,0.04)" }}>
                                                                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--border-soft)", color: "var(--text-soft)" }}>{periodLabel}</td>
                                                                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--border-soft)", color: "#f59e0b", fontWeight: 600 }}>{v?.toFixed(2)}</td>
                                                                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--border-soft)", color: "var(--text-soft)" }}>{forecastData.lower_bound[i]?.toFixed(2)}</td>
                                                                    <td style={{ padding: "8px 12px", borderBottom: "1px solid var(--border-soft)", color: "var(--text-soft)" }}>{forecastData.upper_bound[i]?.toFixed(2)}</td>
                                                                </tr>
                                                                );
                                                            })}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                                );
                            })()}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
