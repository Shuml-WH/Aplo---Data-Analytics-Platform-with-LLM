import React from 'react';
import Plotly from 'plotly.js-dist-min';

const darkThemeLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e8eef2" },
    margin: { l: 50, r: 20, t: 50, b: 80 },
    legend: { orientation: "h", y: -0.25, x: 0.5, xanchor: "center" },
};


export default function ChartBuilderView(props) {

    const columns = props.columns || [];
    const [tiles, setTiles] = React.useState([]);
    const [globalDateColumn, setGlobalDateColumn] = React.useState("");
    const [globalDateFrom, setGlobalDateFrom] = React.useState("");
    const [globalDateTo, setGlobalDateTo] = React.useState("");
    const [isLoadingDefaultRange, setIsLoadingDefaultRange] = React.useState(false);

    const datasetProfile = props.datasetProfile || null;
    const generatedChart = props.generatedChart || null;
    const onChartConsumed = props.onChartConsumed || (() => {});


    // default date range is 3 months window, from the latest date of the dataset
    const loadDefaultDateRange = async (dateColumn) => {
        if (!dateColumn) {
            setGlobalDateFrom("");
            setGlobalDateTo("");
            return;
        }

        try {
            setIsLoadingDefaultRange(true);

            const res = await fetch("/api/date-range-profile", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ date_column: dateColumn })
            });

            const data = await res.json();

            if (!res.ok || !data.success) {
            throw new Error(data.error || "Failed to load date range profile");
            }

            setGlobalDateFrom(data.default_start_date || "");
            setGlobalDateTo(data.default_end_date || "");

        } catch (err) {
            console.error("Error loading default date range", err);
            alert("Failed to initialize date range: " + (err.message || String(err)));
        } finally {
            setIsLoadingDefaultRange(false);
        }
    };


    const handleGlobalDateColumnChange = async (value) => {
        setGlobalDateColumn(value);
        await loadDefaultDateRange(value);
    };


    const inferDefaultDateColumn = React.useCallback(() => {
        const cols = datasetProfile?.columns || columns || [];
        if (!cols.length) return "";

        const byDtype = cols.find(col => {
            const dtype = String(col.dtype || "").toLowerCase();
            return dtype.includes("date") || dtype.includes("time");
        });
        if (byDtype) return byDtype.name;

        const byName = cols.find(col => {
            const name = String(col.name || "").toLowerCase();
            return (
                name.includes("date") ||
                name.includes("time") ||
                name.includes("month") ||
                name.includes("year") ||
                name.includes("day")
            );
        });
        return byName?.name || "";
    }, [datasetProfile, columns]);

    const inferDefaultNumericColumn = React.useCallback(() => {
        const cols = datasetProfile?.columns || columns || [];
        if (!cols.length) return "";

        const byDtype = cols.find(col => {
            const dtype = String(col.dtype || "").toLowerCase();
            return dtype.includes("int") || dtype.includes("float") || dtype.includes("num");
        });
        return byDtype?.name || columns[0]?.name || "";
    }, [datasetProfile, columns]);

    const inferDefaultCategoricalColumn = React.useCallback(() => {
        const cols = datasetProfile?.columns || columns || [];
        if (!cols.length) return "";

        const byName = cols.find(col => {
            const name = String(col.name || "").toLowerCase();
            return (
                name.includes("status") ||
                name.includes("category") ||
                name.includes("type") ||
                name.includes("region") ||
                name.includes("product")
            );
        });
        if (byName) return byName.name;

        const byDtype = cols.find(col => {
            const dtype = String(col.dtype || "").toLowerCase();
            return dtype.includes("object") || dtype.includes("string") || dtype.includes("category");
        });
        return byDtype?.name || columns[0]?.name || "";
    }, [datasetProfile, columns]);

    React.useEffect(() => {
        if (globalDateColumn) return;
        if (!columns.length) return;

        const inferred = inferDefaultDateColumn();
        if (inferred) {
            handleGlobalDateColumnChange(inferred);
        }
    }, [globalDateColumn, columns.length, inferDefaultDateColumn]);


    const getDefaultXForChartType = (chartType) => {
        if (chartType === "Line Chart" || chartType === "Bar Chart") {
            return inferDefaultDateColumn();
        }
        if (chartType === "Scatter Plot") {
            return inferDefaultNumericColumn();
        }
        if (chartType === "Pie Chart") {
            return inferDefaultCategoricalColumn();
        }
        // Gauge doesn't need x-axis
        return "";
    };

    const addTile = () => {
        const newTile = {
            id: Date.now(),
            isEditing: true,
            chartType: "",
            xAxis: "",
            yAxes: [],
            allowMultiY: false,
            agg: "sum",
            tileSize: 2,
            gaugeTarget: "",
            gaugeTargetSource: "previous_period",
            gaugeTargetPeriod: "last_year",
            gaugeTargetAgg: "sum",
            gaugeFactorMode: "multiply",
            gaugeFactor: "1.0",
            groupBy: "none",
            timePeriod: "all",
            catGroupBy: "",
            barMode: "group",
        };

        setTiles(prev => [...prev, newTile]);
    };

    const updateTile = (id, field, value) => {
        setTiles(prev =>
            prev.map(t => t.id === id ? { ...t, [field]: value } : t));
    };

    const handleChartTypeSelect = (id, chartType) => {
        const defaultX = getDefaultXForChartType(chartType);
        setTiles(prev =>
            prev.map(t => t.id === id ? { ...t, chartType, xAxis: defaultX } : t));
    };

    const toggleYAxisColumn = (id, colName) => {
        setTiles(prev =>
            prev.map(t => {
                if (t.id !== id) return t;
                const yAxes = t.yAxes || [];
                const next = yAxes.includes(colName)
                    ? yAxes.filter(c => c !== colName)
                    : [...yAxes, colName];
                return { ...t, yAxes: next };
            }));
    };

    const saveTile = async (id, tileOverride = null) => {
        const tile = tileOverride || tiles.find(t => t.id === id);
        if (!tile) return;

        setTiles(prev =>
            prev.map(t => t.id === id ? { ...t, isEditing: false } : t)
        );

        const chartTypeMap = {
            "Bar Chart": "bar",
            "Line Chart": "line",
            "Scatter Plot": "scatter",
            "Pie Chart": "pie",
            "Gauge Chart": "gauge",
        };
        const chart_type = chartTypeMap[tile.chartType] || "bar";

        const x = tile.chartType === "Gauge Chart" ? null : tile.xAxis;
        const ys = tile.yAxes && tile.yAxes.length > 0 ? tile.yAxes : [];
        const agg = tile.chartType === "Scatter Plot" ? null : (tile.agg || "sum");
        const target = tile.chartType === "Gauge Chart" && tile.gaugeTargetSource === "manual" && tile.gaugeTarget
            ? Number(tile.gaugeTarget)
            : null;
        const group_by = ["Bar Chart", "Line Chart", "Pie Chart"].includes(tile.chartType)
            ? (tile.groupBy || "none")
            : null;
        const cat_group_by = tile.chartType === "Bar Chart" && tile.catGroupBy
            ? tile.catGroupBy
            : null;
        const bar_mode = tile.chartType === "Bar Chart" && tile.barMode
            ? tile.barMode
            : null;
        const time_period = tile.chartType === "Gauge Chart"
            ? (tile.timePeriod || "all")
            : null;
        const factor = tile.chartType === "Gauge Chart" && tile.gaugeFactor
            ? Number(tile.gaugeFactor)
            : null;
        const factor_mode = tile.chartType === "Gauge Chart"
            ? (tile.gaugeFactorMode || "multiply")
            : null;
        const target_source = tile.chartType === "Gauge Chart"
            ? (tile.gaugeTargetSource || "previous_period")
            : null;
        const target_period = tile.chartType === "Gauge Chart" && tile.gaugeTargetSource === "previous_period"
            ? (tile.gaugeTargetPeriod || "last_year")
            : null;
        const target_agg = tile.chartType === "Gauge Chart" && tile.gaugeTargetSource === "previous_period"
            ? (tile.gaugeTargetAgg || "sum")
            : null;

        try {
            const res = await fetch("/api/build-chart", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    x,
                    y: ys,
                    chart_type,
                    agg,
                    target,
                    group_by,
                    cat_group_by,
                    bar_mode,
                    time_period,
                    factor,
                    factor_mode,
                    target_source,
                    target_period,
                    target_agg,
                    date_column: globalDateColumn || null,
                    date_from: globalDateFrom || null,
                    date_to: globalDateTo || null
                })
            });

            const data = await res.json();
            if (!data.success) {
                console.error("Chart build error:", data.error);
                alert("Failed to build chart: " + data.error);
                return;
            }

            const fig = JSON.parse(data.figure_json);
            fig.layout = { ...darkThemeLayout, ...fig.layout };

            const divId = `chart-tile-${id}`;
            const chartDiv = document.getElementById(divId);
            if (!chartDiv) {
                console.warn("Chart div not found:", divId);
                return;
            }

            Plotly.newPlot(divId, fig.data, fig.layout, { responsive: true });
            setTiles(prev =>
                prev.map(t =>
                    t.id === id ? { ...t, isWide: true } : t
                )
            );

        } catch (err) {
            console.error("Error calling /api/build-chart", err);
            alert("Error building chart. See console for details.");
        }
    };

    const deleteTile = (id) => {
        setTiles(prev => prev.filter(t => t.id !== id));
    };

    const moveTile = (id, direction) => {
        setTiles(prev => {
            const idx = prev.findIndex(t => t.id === id);
            if (idx === -1) return prev;
            const targetIdx = direction === 'left' ? idx - 1 : idx + 1;
            if (targetIdx < 0 || targetIdx >= prev.length) return prev;
            const next = [...prev];
            [next[idx], next[targetIdx]] = [next[targetIdx], next[idx]];
            return next;
        });
    };

    React.useEffect(() => {
        const visibleTiles = tiles.filter(t => !t.isEditing);
        visibleTiles.forEach(t => {
            saveTile(t.id, { ...t, isEditing: false });
        });
    }, [globalDateColumn, globalDateFrom, globalDateTo]);

    // Resize Plotly charts when tile size changes
    const prevTileSizesRef = React.useRef({});
    React.useEffect(() => {
        const changed = tiles.some(t => {
            const prev = prevTileSizesRef.current[t.id];
            return prev !== undefined && prev !== t.tileSize;
        });
        if (!changed) {
            tiles.forEach(t => { prevTileSizesRef.current[t.id] = t.tileSize; });
            return;
        }
        tiles.forEach(t => { prevTileSizesRef.current[t.id] = t.tileSize; });

        // Use double rAF to wait for layout reflow
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                tiles.forEach(t => {
                    if (!t.isEditing) {
                        const divId = `chart-tile-${t.id}`;
                        const chartDiv = document.getElementById(divId);
                        if (chartDiv && chartDiv.data && Plotly?.Plots) {
                            Plotly.Plots.resize(chartDiv);
                        }
                    }
                });
            });
        });
    }, [tiles.map(t => t.tileSize).join(",")]);

    // Handle chart generated from chatbot
    React.useEffect(() => {
        if (!generatedChart) return;

        const chartTypeMap = {
            "bar": "Bar Chart",
            "line": "Line Chart",
            "scatter": "Scatter Plot",
            "pie": "Pie Chart",
            "gauge": "Gauge Chart",
        };

        const newTile = {
            id: Date.now(),
            isEditing: false,
            chartType: chartTypeMap[generatedChart.chart_type] || "Bar Chart",
            xAxis: generatedChart.x,
            yAxes: generatedChart.y || [],
            allowMultiY: (generatedChart.y || []).length > 1,
            agg: generatedChart.chart_type === "scatter" ? null : (generatedChart.agg || "sum"),
            tileSize: 2,
            gaugeTarget: generatedChart.target || "",
            groupBy: generatedChart.group_by || "none",
            timePeriod: generatedChart.time_period || "all",
        };

        setTiles(prev => [...prev, newTile]);

        setTimeout(() => {
            const divId = `chart-tile-${newTile.id}`;
            const chartDiv = document.getElementById(divId);
            if (chartDiv && generatedChart.figure_json) {
                const fig = typeof generatedChart.figure_json === "string"
                    ? JSON.parse(generatedChart.figure_json)
                    : generatedChart.figure_json;
                fig.layout = { ...darkThemeLayout, ...fig.layout };
                Plotly.newPlot(divId, fig.data, fig.layout, { responsive: true });
            }
        }, 100);

        onChartConsumed();
    }, [generatedChart]);

    const isGauge = tile => tile.chartType === "Gauge Chart";
    const isPie = tile => tile.chartType === "Pie Chart";
    const isScatter = tile => tile.chartType === "Scatter Plot";
    const needsXAxis = tile => !isGauge(tile);
    const needsAgg = tile => !isPie(tile) && !isScatter(tile);
    const needsGroupBy = tile => ["Bar Chart", "Line Chart", "Pie Chart"].includes(tile.chartType);
    const needsCategoricalGroupBy = tile => tile.chartType === "Bar Chart";

    const handleClearDashboard = () => {
        if (!window.confirm("Are you sure? This will remove all chart panels.")) return;
        setTiles([]);
        setGlobalDateColumn("");
        setGlobalDateFrom("");
        setGlobalDateTo("");
    };

    return (
    <div className="view-container">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
            <h2 className="view-title" style={{ margin: 0 }}>Data Visualizer</h2>
            <button
                className="btn btn-ghost"
                onClick={handleClearDashboard}
            >
                Clear Dashboard
            </button>
        </div>

        <div
        className="card"
        style={{
            marginBottom: "16px",
            padding: "12px",
            display: "flex",
            gap: "12px",
            alignItems: "end",
            flexWrap: "wrap"
        }}
        >
            <div className="form-group" style={{ minWidth: "220px", flex: "1 1 220px" }}>
                <label>Date column:</label><br />
                <select
                value={globalDateColumn}
                onChange={(e) => handleGlobalDateColumnChange(e.target.value)}
                >
                <option value="">None</option>
                {columns.map(col => (
                    <option key={col.name} value={col.name}>
                    {col.name}
                    </option>
                ))}
                </select>
            </div>

            <div className="form-group" style={{ minWidth: "160px" }}>
                <label>From:</label><br />
                <input
                type="date"
                value={globalDateFrom}
                onChange={(e) => setGlobalDateFrom(e.target.value)}
                disabled={!globalDateColumn || isLoadingDefaultRange}
                />
            </div>

            <div className="form-group" style={{ minWidth: "160px" }}>
                <label>To:</label><br />
                <input
                type="date"
                value={globalDateTo}
                onChange={(e) => setGlobalDateTo(e.target.value)}
                disabled={!globalDateColumn || isLoadingDefaultRange}
                />
            </div>
        </div>


        <div className="tile-grid">

        {tiles.map(tile => (

            <div key={tile.id}
            className={`card chart-tile chart-tile--size-${tile.tileSize || 2}`}
            >

            {tile.isEditing ? (
                <div className="tile-config">

                <div className="card-header">
                    <h4>Configure Chart</h4>
                    <button
                    className="btn btn-ghost"
                    onClick={() => deleteTile(tile.id)}>X</button>
                </div>

                {/* Step 1: Chart Type - always visible */}
                <div className="form-group">
                    <label>Chart Type</label>
                    <div className="chart-type-buttons">
                        {["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Gauge Chart"].map(type => (
                            <button
                                key={type}
                                className={`chart-type-btn ${tile.chartType === type ? "active" : ""}`}
                                onClick={() => handleChartTypeSelect(tile.id, type)}
                            >
                                {type}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Step 2: Axis Configuration - only after chart type selected */}
                {tile.chartType && (
                    <>
                        {/* Y-Axis / Values / Metric - shown first */}
                        <div className="form-group">
                            <label>{isPie(tile) ? "Values" : isGauge(tile) ? "Metric" : "Y-Axis"}</label><br/>

                            {!tile.allowMultiY ? (
                                <select
                                    value={tile.yAxes[0] || ""}
                                    onChange={(e) => {
                                        const val = e.target.value;
                                        updateTile(tile.id, 'yAxes', val ? [val] : []);
                                    }}
                                >
                                    <option value="">Select column...</option>
                                    {columns.map(col => (
                                        <option key={col.name} value={col.name}>
                                            {col.name}
                                        </option>
                                    ))}
                                </select>
                            ) : (
                                <div className="y-axis-checkbox-popup">
                                    <div className="y-axis-checkbox-list">
                                        {columns.map(col => (
                                            <label key={col.name} className="y-axis-checkbox-item">
                                                <input
                                                    type="checkbox"
                                                    checked={(tile.yAxes || []).includes(col.name)}
                                                    onChange={() => toggleYAxisColumn(tile.id, col.name)}
                                                />
                                                {col.name}
                                            </label>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {!isGauge(tile) && (
                            <label className="multi-y-toggle">
                                <input
                                    type="checkbox"
                                    checked={tile.allowMultiY}
                                    onChange={(e) => {
                                        const checked = e.target.checked;
                                        if (!checked) {
                                            const first = tile.yAxes?.[0] || "";
                                            updateTile(tile.id, 'yAxes', first ? [first] : []);
                                        }
                                        updateTile(tile.id, 'allowMultiY', checked);
                                    }}
                                />
                                Compare multiple series
                            </label>
                            )}
                        </div>

                        {/* X-Axis / Labels - hidden for gauge */}
                        {needsXAxis(tile) && (
                        <div className="form-group">
                            <label>{isPie(tile) ? "Labels" : "X-Axis"}</label><br/>
                            <select
                                value={tile.xAxis}
                                onChange={(e) => updateTile(tile.id, 'xAxis', e.target.value)}
                            >
                                <option value="">Select column...</option>
                                {columns.map(col => (
                                    <option key={col.name} value={col.name}>
                                        {col.name}
                                    </option>
                                ))}
                            </select>
                        </div>
                        )}

                        {/* Aggregation - not applicable for scatter, pie, gauge */}
                        {needsAgg(tile) && (
                        <div className="form-group">
                            <label>Aggregation:</label><br/>
                            <select
                                value={tile.agg || "sum"}
                                onChange={(e) => updateTile(tile.id, 'agg', e.target.value)}
                            >
                                <option value="sum">Sum</option>
                                <option value="mean">Mean</option>
                                <option value="median">Median</option>
                            </select>
                        </div>
                        )}

                        {/* Group by - for bar, line, pie */}
                        {needsGroupBy(tile) && (
                        <div className="form-group">
                            <label>Group by (time):</label><br/>
                            <select
                                value={tile.groupBy || "none"}
                                onChange={(e) => updateTile(tile.id, 'groupBy', e.target.value)}
                            >
                                <option value="none">None</option>
                                <option value="daily">Daily</option>
                                <option value="weekly">Weekly</option>
                                <option value="monthly">Monthly</option>
                                <option value="quarterly">Quarterly</option>
                                <option value="yearly">Yearly</option>
                            </select>
                        </div>
                        )}

                        {/* Categorical Group By - for bar chart only */}
                        {needsCategoricalGroupBy(tile) && (
                        <div className="form-group">
                            <label>Group bars by (category):</label><br/>
                            <select
                                value={tile.catGroupBy || ""}
                                onChange={(e) => updateTile(tile.id, 'catGroupBy', e.target.value)}
                            >
                                <option value="">None (single series)</option>
                                {columns.filter(col => col.name !== tile.xAxis && !tile.yAxes?.includes(col.name)).map(col => (
                                    <option key={col.name} value={col.name}>
                                        {col.name}
                                    </option>
                                ))}
                            </select>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                                Group bars by another category for side-by-side comparison
                            </div>
                        </div>
                        )}

                        {/* Bar Mode - cluster vs stack - for bar chart with cat_group_by or multi-Y */}
                        {tile.chartType === "Bar Chart" && (tile.catGroupBy || tile.allowMultiY) && (
                        <div className="form-group">
                            <label>Bar Mode:</label><br/>
                            <select
                                value={tile.barMode || "group"}
                                onChange={(e) => updateTile(tile.id, 'barMode', e.target.value)}
                            >
                                <option value="group">Clustered (side-by-side)</option>
                                <option value="stack">Stacked</option>
                            </select>
                        </div>
                        )}

                        {/* Time Period - only for gauge chart */}
                        {isGauge(tile) && (
                        <div className="form-group">
                            <label>Time Period:</label><br/>
                            <select
                                value={tile.timePeriod || "all"}
                                onChange={(e) => updateTile(tile.id, 'timePeriod', e.target.value)}
                            >
                                <option value="all">All Data</option>
                                <option value="last_week">Last Week</option>
                                <option value="last_month">Last Month</option>
                                <option value="last_quarter">Last Quarter</option>
                                <option value="last_year">Last Year</option>
                                <option value="7d">Last 7 Days</option>
                                <option value="30d">Last 30 Days</option>
                                <option value="3m">Last 3 Months</option>
                                <option value="6m">Last 6 Months</option>
                                <option value="1y">Last Year (Rolling)</option>
                            </select>
                        </div>
                        )}

                        {/* Gauge Target Source - only for gauge chart */}
                        {isGauge(tile) && (
                        <div className="form-group">
                            <label>Target Source:</label><br/>
                            <div style={{ display: "flex", gap: "12px", marginTop: "4px" }}>
                                <label style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer" }}>
                                    <input
                                        type="radio"
                                        name={`target-source-${tile.id}`}
                                        checked={tile.gaugeTargetSource === "previous_period"}
                                        onChange={() => updateTile(tile.id, 'gaugeTargetSource', "previous_period")}
                                    />
                                    Previous Period
                                </label>
                                <label style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer" }}>
                                    <input
                                        type="radio"
                                        name={`target-source-${tile.id}`}
                                        checked={tile.gaugeTargetSource === "manual"}
                                        onChange={() => updateTile(tile.id, 'gaugeTargetSource', "manual")}
                                    />
                                    Manual
                                </label>
                            </div>
                        </div>
                        )}

                        {/* Target Period - only when Previous Period selected */}
                        {isGauge(tile) && tile.gaugeTargetSource === "previous_period" && (
                        <div className="form-group">
                            <label>Target Period:</label><br/>
                            <select
                                value={tile.gaugeTargetPeriod || "last_year"}
                                onChange={(e) => updateTile(tile.id, 'gaugeTargetPeriod', e.target.value)}
                            >
                                <option value="last_week">Last Week</option>
                                <option value="last_month">Last Month</option>
                                <option value="last_quarter">Last Quarter</option>
                                <option value="last_year">Last Year</option>
                            </select>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                                Target = This period's value as baseline
                            </div>
                        </div>
                        )}

                        {/* Target Aggregation - only when Previous Period selected */}
                        {isGauge(tile) && tile.gaugeTargetSource === "previous_period" && (
                        <div className="form-group">
                            <label>Target Aggregation:</label><br/>
                            <select
                                value={tile.gaugeTargetAgg || "sum"}
                                onChange={(e) => updateTile(tile.id, 'gaugeTargetAgg', e.target.value)}
                            >
                                <option value="sum">Sum</option>
                                <option value="mean">Average</option>
                                <option value="median">Median</option>
                            </select>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                                How to aggregate the target period's values
                            </div>
                        </div>
                        )}

                        {/* Factor Mode & Value - only for gauge chart with previous_period */}
                        {isGauge(tile) && tile.gaugeTargetSource === "previous_period" && (
                        <div className="form-group">
                            <label>Factor:</label><br/>
                            <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
                                <select
                                    value={tile.gaugeFactorMode || "multiply"}
                                    onChange={(e) => updateTile(tile.id, 'gaugeFactorMode', e.target.value)}
                                    style={{ flex: "0 0 auto", width: "auto" }}
                                >
                                    <option value="multiply">Multiply (x)</option>
                                    <option value="add">Add (+)</option>
                                </select>
                                <input
                                    type="number"
                                    value={tile.gaugeFactor || "1.0"}
                                    onChange={(e) => updateTile(tile.id, 'gaugeFactor', e.target.value)}
                                    step="0.1"
                                    min="0"
                                    style={{ flex: 1, width: "100%", padding: "7px 9px", borderRadius: "8px", border: "1px solid var(--border-soft)", backgroundColor: "#020617", color: "var(--text-main)" }}
                                />
                            </div>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                                {tile.gaugeFactorMode === "add"
                                    ? "Target = Previous Period value + Factor"
                                    : "Target = Previous Period value x Factor (e.g., 1.2 = 120%)"}
                            </div>
                        </div>
                        )}

                        {/* Manual Target Input - only for gauge chart with manual target */}
                        {isGauge(tile) && tile.gaugeTargetSource === "manual" && (
                        <div className="form-group">
                            <label>Target Value:</label>
                            <input
                                type="number"
                                className="form-select"
                                value={tile.gaugeTarget || ""}
                                onChange={(e) => updateTile(tile.id, 'gaugeTarget', e.target.value)}
                                placeholder="Enter target value"
                            />
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                                Enter a numeric target value
                            </div>
                        </div>
                        )}
                    </>
                )}

                {/* Generate Chart Button */}
                {tile.chartType && tile.yAxes.length > 0 && (needsXAxis(tile) ? tile.xAxis : true) && (
                    <div style={{ marginTop: "auto" }}>
                        <button
                            className="btn btn-primary full-width"
                            onClick={() => saveTile(tile.id)}
                        >Generate Chart
                        </button>
                    </div>
                )}
                </div>
            ) : (

                <div className="tile-view">
                <div className="card-header">
                    <h4>{tile.chartType}</h4>
                    <div className="header-actions">
                    <select
                        value={tile.tileSize || 2}
                        onChange={(e) => updateTile(tile.id, "tileSize", Number(e.target.value))}
                        style={{ marginRight: "8px", fontSize: "0.75rem", padding: "4px 6px" }}
                    >
                        <option value={1}>1/3</option>
                        <option value={2}>2/3</option>
                        <option value={3}>Full</option>
                    </select>

                    {(() => {
                        const idx = tiles.findIndex(t => t.id === tile.id);
                        const isFirst = idx === 0;
                        const isLast = idx === tiles.length - 1;
                        return (
                            <span className="tile-move-arrows">
                                <button
                                    className="tile-arrow-btn"
                                    onClick={() => moveTile(tile.id, 'left')}
                                    disabled={isFirst}
                                    title="Move left"
                                >&#9664;</button>
                                <button
                                    className="tile-arrow-btn"
                                    onClick={() => moveTile(tile.id, 'right')}
                                    disabled={isLast}
                                    title="Move right"
                                >&#9654;</button>
                            </span>
                        );
                    })()}

                    <button
                        className="btn btn-ghost"
                        onClick={() => updateTile(tile.id, 'isEditing', true)}
                    >
                        Edit
                    </button>
                    <button
                        className="btn btn-ghost"
                        onClick={() => deleteTile(tile.id)}
                    >
                        Delete
                    </button>
                    </div>
                </div>

                <div className="chart-placeholder">
                    <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                    <div>
                        {isGauge(tile)
                            ? tile.yAxes?.[0] || ""
                            : (tile.yAxes && tile.yAxes.length > 0 ? tile.yAxes.join(", ") : "")
                              + " by " + tile.xAxis
                        }
                        {!isPie(tile) && !isGauge(tile) && tile.agg ? ` (agg: ${tile.agg})` : ""}
                        {tile.groupBy && tile.groupBy !== "none" ? ` (grouped: ${tile.groupBy})` : ""}
                        {isGauge(tile) && tile.timePeriod && tile.timePeriod !== "all" ? ` (${tile.timePeriod})` : ""}
                    </div>
                    <div
                        id={`chart-tile-${tile.id}`}
                        style={{ width: "100%", height: "100%" }}
                    >
                    </div>
                    </div>
                </div>
                </div>
            )}
            </div>
        ))}

        <div
            className="card chart-tile add-tile"
            onClick={addTile}
        >
            <div className="add-tile-inner">
            <h1>+</h1>
            <h3>Add Chart Panel</h3>
            </div>
        </div>

        </div>
    </div>
    )};
