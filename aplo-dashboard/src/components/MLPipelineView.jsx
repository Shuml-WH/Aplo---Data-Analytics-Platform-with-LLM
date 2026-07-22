import React from 'react';
import PlotModule from "react-plotly.js";

const Plot = PlotModule.default || PlotModule;
console.log("Plot import resolved:", Plot);

export default function MLPipelineView({ datasetProfile, onNavigate }) {

    const columns = datasetProfile?.columns || [];
    const initialTarget = columns[0]?.name || "";
    const initialFeatures = [];

    const [activeStep, setActiveStep] = React.useState("Step 1 Data & Target");
    const [targetColumn, setTargetColumn] = React.useState(initialTarget);
    const [featureColumns, setFeatureColumns] = React.useState(initialFeatures);
    const [taskType, setTaskType] = React.useState("");
    const [plan, setPlan] = React.useState(null);
    const [trainResult, setTrainResult] = React.useState(null);
    const [loadingPlan, setLoadingPlan] = React.useState(false);
    const [training, setTraining] = React.useState(false);
    const [mlError, setMlError] = React.useState("");
    const [numericStrategy, setNumericStrategy] = React.useState("median");
    const [categoricalStrategy, setCategoricalStrategy] = React.useState("most_frequent");
    const [useScaling, setUseScaling] = React.useState(true);
    const [selectedModels, setSelectedModels] = React.useState([]);
    const [selectedEvalModel, setSelectedEvalModel] = React.useState("");
    const [selectedEvalChart, setSelectedEvalChart] = React.useState("metrics");
    const [exportingModel, setExportingModel] = React.useState(false);
    const [trainingProgress, setTrainingProgress] = React.useState(null);
    const [trainingStep, setTrainingStep] = React.useState("");

    // Time-series specific state
    const [datetimeColumn, setDatetimeColumn] = React.useState("");
    const [forecastHorizon, setForecastHorizon] = React.useState(30);
    const [tsFrequency, setTsFrequency] = React.useState("D");
    const [tsTrainResult, setTsTrainResult] = React.useState(null);
    const [tsTraining, setTsTraining] = React.useState(false);
    const [tsTrainingProgress, setTsTrainingProgress] = React.useState(0);
    const [tsTrainingStep, setTsTrainingStep] = React.useState("");
    const [tsForecastData, setTsForecastData] = React.useState(null);
    const [tsTestSize, setTsTestSize] = React.useState(0.2);
    const [tsAggregation, setTsAggregation] = React.useState("sum");
    const [tsSelectedModels, setTsSelectedModels] = React.useState(["arimax", "xgboost"]);
    const [tsSelectedEvalModel, setTsSelectedEvalModel] = React.useState("");
    const [forecastHorizonInput, setForecastHorizonInput] = React.useState(String(forecastHorizon));
    const [tsTestSizeInput, setTsTestSizeInput] = React.useState(String(Math.round(tsTestSize * 100)));

    React.useEffect(() => { setForecastHorizonInput(String(forecastHorizon)); }, [forecastHorizon]);
    React.useEffect(() => { setTsTestSizeInput(String(Math.round(tsTestSize * 100))); }, [tsTestSize]);

    const hasStep1Input = !!targetColumn && featureColumns.length > 0;

    const isDatetimeColumn = (colName) => {
        const col = datasetProfile?.columns?.find(c => c.name === colName);
        if (!col) return false;
        return col.is_datetime === true;
    };
    const hasPlan = !!plan;
    const hasTrainingResult = !!trainResult;



    // React.useEffect(() => {
    // if (!columns.length) {
    //     setTargetColumn("");
    //     setFeatureColumns([]);
    //     setTaskType("");
    //     setSelectedModels([]);
    //     setPlan(null);
    //     setTrainResult(null);
    //     setActiveStep("Step 1 Data & Target");
    //     return;
    // }

    // const firstColumn = columns[0]?.name || "";
    // setTargetColumn(firstColumn);
    // setFeatureColumns(columns.slice(1).map(col => col.name));
    // setTaskType("");
    // setSelectedModels([]);
    // setPlan(null);
    // setTrainResult(null);
    // setActiveStep("Step 1 Data & Target");
    // }, [datasetProfile]);

    React.useEffect(() => {
    if (!targetColumn) return;
    setFeatureColumns(prev => prev.filter(col => col !== targetColumn));
    }, [targetColumn]);

    React.useEffect(() => {
        if (!trainResult?.results?.length) return;
        setSelectedEvalModel(trainResult.best_model_key || trainResult.results[0]?.model_key || "");
    }, [trainResult]);

    // Restore pipeline state from backend on mount
    React.useEffect(() => {
        const restoreState = async () => {
            try {
                const res = await fetch("/api/ml/pipeline-state");
                const data = await res.json();
                if (!data.success) return;

                // Restore standard ML state
                if (data.config) {
                    setTargetColumn(data.config.target_column || "");
                    setFeatureColumns(data.config.feature_columns || []);
                    setTaskType(data.config.task_type || "");
                    setNumericStrategy(data.config.numeric_strategy || "median");
                    setCategoricalStrategy(data.config.categorical_strategy || "most_frequent");
                    setUseScaling(data.config.use_scaling ?? true);
                    setSelectedModels(data.config.models || []);
                    if (data.has_results && data.results_data) {
                        setTrainResult(data.results_data);
                        setActiveStep("Step 5 Model Evaluation & Saving");
                    } else if (data.plan) {
                        setPlan(data.plan);
                        setActiveStep("Step 3 Feature Scaling");
                    }
                }

                // Restore time-series state
                if (data.timeseries?.has_results && data.timeseries?.results_data) {
                    const tsData = data.timeseries.results_data;
                    setTaskType("timeseries");
                    setTargetColumn(tsData.target_column || "");
                    setDatetimeColumn(tsData.datetime_column || "");
                    setFeatureColumns(tsData.feature_columns || []);
                    setForecastHorizon(tsData.forecast_horizon || 30);
                    setTsFrequency(tsData.frequency || "D");
                    setTsTestSize(tsData.test_size || 0.2);
                    setTsAggregation(tsData.aggregation || "sum");
                    setTsSelectedModels(tsData.model_type || ["arimax"]);
                    setTsSelectedEvalModel(tsData.best_model_key || tsData.results?.[0]?.model_type || "arimax");
                    setTsTrainResult(tsData);
                    setTsForecastData(tsData.forecast_data || null);
                    setActiveStep("Step 3 Evaluation");
                }
            } catch (err) {
                console.error("Failed to restore pipeline state:", err);
            }
        };
        restoreState();
    }, []);





    // const pipelineDiagram = ` : now uses real clickable step blocks through data-step
    //   <div data-step="Step 1: Raw Data" style="padding: 10px; border: 1px dashed gray; text-align: center; cursor: pointer; background: #9c9a9a; margin-bottom: 10px;">[ Step 1: Raw Data ]</div>
    //   <div style="text-align: center; margin-bottom: 10px;">↓</div>
    //   <div data-step="Step 2: Handle Missing Values" style="padding: 10px; border: 2px solid blue; text-align: center; cursor: pointer; background: #05d9fa; margin-bottom: 10px;">[ Step 2: Handle Missing Values ]</div>
    //   <div style="text-align: center; margin-bottom: 10px;">↓</div>
    //   <div data-step="Step 3: Feature Scaling" style="padding: 10px; border: 1px dashed gray; text-align: center; cursor: pointer; background: #9c9a9a; margin-bottom: 10px;">[ Step 3: Feature Scaling ]</div>
    //   <div style="text-align: center; margin-bottom: 10px;">↓</div>
    //   <div data-step="Step 4: Model Training" style="padding: 10px; border: 1px dashed gray; text-align: center; cursor: pointer; background: #9c9a9a;">[ Step 4: Model Training ]</div>
    // `;

    // const handleDiagramClick = (event) => { 
    //   const step = event.target.getAttribute("data-step");
    //   if (step) setActiveStep(step);
    // };

    function getDefaultModelsForTask(task) {
    return task === "regression"
        ? ["linearregression", "ridgeregression", "gradientboostingregressor", "decisiontreeregressor", "kneighborsregressor"]
        : ["logisticregression", "decisiontreeclassifier"];
    }

    const handleFeatureToggle = (colName) => { 
    setFeatureColumns(prev =>
        prev.includes(colName)
        ? prev.filter(c => c !== colName)
        : [...prev, colName]
    );
    };

    const fetchPreprocessingPlan = async () => { 
    if (!targetColumn) {
        setMlError("Please select a target column.");
        return;
    }

    setLoadingPlan(true);
    setMlError("");
    setPlan(null);
    setTrainResult(null);

    try {
        const initRes = await fetch("/api/ml/initialize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            target_column: targetColumn,
            feature_columns: featureColumns
        })
        });

        const initData = await initRes.json();
        if (!initRes.ok || !initData.success) {
        throw new Error(initData.error || "Failed to initialize ML pipeline.");
        }

        setTaskType(initData.task_type || "");
        setSelectedModels(getDefaultModelsForTask(initData.task_type));

        const planRes = await fetch("/api/ml/preprocessing-plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            target_column: targetColumn,
            feature_columns: featureColumns,
            task_type: initData.task_type
        })
        });

        const planData = await planRes.json();
        if (!planRes.ok || !planData.success) {
        throw new Error(planData.error || "Failed to load preprocessing plan.");
        }

        setPlan(planData.plan);
        setNumericStrategy(planData.plan?.recommendations?.numeric_strategy || "median");
        setCategoricalStrategy(planData.plan?.recommendations?.categorical_strategy || "most_frequent");
        setUseScaling(Boolean(planData.plan?.recommendations?.use_scaling));
        setActiveStep("Step 2 Missing Values");
    } catch (err) {
        console.error("ML preprocessing plan error:", err);
        setMlError(err.message || String(err));
    } finally {
        setLoadingPlan(false);
    }
    };

    const handleTrainModels = async () => { 
    if (!targetColumn) {
        setMlError("Please select a target column.");
        return;
    }

 


    if (!featureColumns.length) {
        setMlError("Please select at least one feature column.");
        return;
    }

    if (!selectedModels.length) {
        setMlError("Please select at least one model.");
        return;
    }

    setTraining(true);
    setMlError("");
    setTrainResult(null);
    setTrainingProgress(0);
    setTrainingStep("Initializing...");

    try {
        const response = await fetch("/api/ml/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            target_column: targetColumn,
            feature_columns: featureColumns,
            task_type: taskType || "classification",
            numeric_strategy: numericStrategy,
            categorical_strategy: categoricalStrategy,
            use_scaling: useScaling,
            models: selectedModels
        })
        });

        if (!response.ok) {
            let errorMsg = `Server error: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg;
            } catch { /* response may not be JSON */ }
            throw new Error(errorMsg);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Process complete SSE messages (delimited by \n\n)
            const messages = buffer.split('\n\n');
            buffer = messages.pop(); // Keep incomplete last chunk in buffer
            
            for (const msg of messages) {
                const trimmed = msg.trim();
                if (!trimmed) continue;
                
                // Handle multi-line data fields
                const dataLines = trimmed.split('\n')
                    .filter(l => l.startsWith('data: '))
                    .map(l => l.slice(6))
                    .join('');
                
                if (!dataLines) continue;
                
                try {
                    const data = JSON.parse(dataLines);
                    
                    if (data.type === 'progress') {
                        setTrainingProgress(data.percentage);
                        setTrainingStep(`Training ${data.current_model} (${data.model_index}/${data.total_models})`);
                    } else if (data.type === 'complete') {
                        if (data.success) {
                            console.log("trainResult", data.data);
                            setTrainResult(data.data);
                            setActiveStep("Step 4 Model Training");
                        } else {
                            throw new Error(data.error || "Failed to train models.");
                        }
                    }
                } catch (parseErr) {
                    console.error("SSE parse error:", parseErr, "Raw:", dataLines.substring(0, 200));
                }
            }
        }
    } catch (err) {
        console.error("ML train error:", err);
        setMlError(err.message || String(err));
    } finally {
        setTraining(false);
        setTrainingProgress(null);
        setTrainingStep("");
    }
    };


    const handleTrainTimeseries = async () => {
        if (!targetColumn) { setMlError("Please select a target column."); return; }
        if (!datetimeColumn) { setMlError("Please select a datetime column."); return; }

        setTsTraining(true);
        setMlError("");
        setTsTrainResult(null);
        setTsForecastData(null);
        setTsTrainingProgress(0);
        setTsTrainingStep("Initializing...");

        try {
            const response = await fetch("/api/ml/train-timeseries", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    target_column: targetColumn,
                    datetime_column: datetimeColumn,
                    feature_columns: featureColumns,
                    forecast_horizon: forecastHorizon,
                    frequency: tsFrequency,
                    test_size: tsTestSize,
                    aggregation: tsAggregation,
                    model_type: tsSelectedModels,
                })
            });

            if (!response.ok) {
                let errorMsg = `Server error: ${response.status}`;
                try { const d = await response.json(); errorMsg = d.error || errorMsg; } catch {}
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
                        .filter(l => l.startsWith("data: "))
                        .map(l => l.slice(6))
                        .join("");
                    if (!dataLines) continue;
                    try {
                        const data = JSON.parse(dataLines);
                        if (data.type === "progress") {
                            setTsTrainingProgress(data.percentage);
                            setTsTrainingStep(data.step);
                        } else                         if (data.type === "complete") {
                            if (data.success) {
                                setTsTrainResult(data.data);
                                setTsForecastData(data.data.results?.[0]?.forecast_data || data.data.forecast_data);
                                setTsSelectedEvalModel(data.data.best_model_key || data.data.results?.[0]?.model_type || "arimax");
                                setActiveStep("Step 3 Evaluation");
                            } else {
                                throw new Error(data.error || "Training failed");
                            }
                        }
                    } catch (e) {
                        if (e.message && !e.message.includes("JSON")) throw e;
                    }
                }
            }
        } catch (err) {
            console.error("Timeseries train error:", err);
            setMlError(err.message || String(err));
        } finally {
            setTsTraining(false);
            setTsTrainingProgress(null);
            setTsTrainingStep("");
        }
    };


    const handleExportModel = async () => {
        if (!selectedEvalModel && !trainResult?.best_model_key) { 
            setMlError("No trained model selected for export.");
            return; 
        }

        setExportingModel(true); 
        setMlError(""); 

        try { 
            const modelKey = selectedEvalModel || trainResult?.best_model_key; 
            const res = await fetch("/api/ml/export-model", { 
            method: "POST", 
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify({ model_key: modelKey }), 
            });

            if (!res.ok) {
            let errorMessage = "Failed to export model."; 
            try {
                const errData = await res.json();
                errorMessage = errData.error || errorMessage; 
            } catch { //ignore JSON parse errors for non-JSON response bodies
            }
            throw new Error(errorMessage);
            } 

            const blob = await res.blob(); // read pickle file as binary blob
            const downloadUrl = window.URL.createObjectURL(blob); 
            const link = document.createElement("a"); // create temporary anchor element for file download
            const safeModelKey = (modelKey || "trained_model").replace(/[^a-z0-9_-]/gi, "_");
            link.href = downloadUrl;
            link.download = `${safeModelKey}_model.pkl`; 
            document.body.appendChild(link);
            link.click();
            link.remove(); 
            window.URL.revokeObjectURL(downloadUrl); // free temporary object URL memory
        } catch (err) { 
            console.error("Model export error:", err); 
            setMlError(err.message || String(err));
        } finally {
            setExportingModel(false); 
        } 
    };


    const handleRestartPipeline = () => {
        if (!window.confirm("Are you sure? This will clear all pipeline progress.")) return;
        setActiveStep("Step 1 Data & Target");
        setTargetColumn(columns[0]?.name || "");
        setFeatureColumns(columns.slice(1).map(col => col.name));
        setTaskType("");
        setPlan(null);
        setTrainResult(null);
        setMlError("");
        setNumericStrategy("median");
        setCategoricalStrategy("most_frequent");
        setUseScaling(true);
        setSelectedModels([]);
        setSelectedEvalModel("");
        setSelectedEvalChart("metrics");
        setTrainingProgress(null);
        setTrainingStep("");
        setDatetimeColumn("");
        setForecastHorizon(30);
        setTsFrequency("D");
        setTsTrainResult(null);
        setTsForecastData(null);
        setTsTestSize(0.2);
        setTsAggregation("sum");
        setTsSelectedModels(["arimax", "xgboost"]);
        setTsSelectedEvalModel("");
    };

    const hasTsStep1Input = !!targetColumn && !!datetimeColumn;
    const hasTsTrainResult = !!tsTrainResult;

    const tsStepStatus = {
        step1: hasTsStep1Input ? "Completed" : "Pending",
        step2: hasTsStep1Input ? "Ready" : "Locked",
        step3: hasTsTrainResult ? "Completed" : (hasTsStep1Input ? "Ready" : "Locked"),
    };

    const stepStatus = {
        step1: hasStep1Input ? "Completed" : "Pending",
        step2: hasPlan ? "Completed" : (hasStep1Input ? "Ready" : "Pending"),
        step3: hasPlan ? "Ready" : "Locked",
        step4: hasTrainingResult ? "Completed" : (hasPlan ? "Ready" : "Locked"),
        step5: hasTrainingResult ? "Ready" : "Locked",
    };

    const pipelineSteps = taskType === "timeseries" ? [
    {
        key: "Step 1 Data & Target",
        title: "Step 1",
        subtitle: "Data & Target",
        status: tsStepStatus.step1,
        summary: [
        `Target: ${targetColumn || "-"}`,
        `Datetime: ${datetimeColumn || "-"}`,
        `Exogenous: ${featureColumns.length}`,
        `Task: Time-Series Forecast`
        ]
    },
    {
        key: "Step 2 Modeling & Training",
        title: "Step 2",
        subtitle: "Modeling & Training",
        status: tsStepStatus.step2,
        summary: [
        `Model: ARIMAX`,
        `Horizon: ${forecastHorizon} periods`,
        `Frequency: ${tsFrequency}`,
        `Test split: ${Math.round(tsTestSize * 100)}%`
        ]
    },
    {
        key: "Step 3 Evaluation",
        title: "Step 3",
        subtitle: "Evaluation",
        status: tsStepStatus.step3,
        summary: [
        `Order: ${tsTrainResult?.order ? `(${tsTrainResult.order.join(",")})` : "-"}`,
        `MAPE: ${tsTrainResult?.test_metrics?.mape != null ? tsTrainResult.test_metrics.mape.toFixed(1) + "%" : "-"}`,
        `${hasTsTrainResult ? "Forecast ready" : "Not trained"}`
        ]
    }
    ] : [
    {
        key: "Step 1 Data & Target",
        title: "Step 1",
        subtitle: "Data & Target",
        status: stepStatus.step1,
        summary: [
        `Target: ${targetColumn || "-"}`,
        `Features: ${featureColumns.length}`,
        `Task: ${taskType || "To be inferred"}`
        ]
    },
    {
        key: "Step 2 Missing Values",
        title: "Step 2",
        subtitle: "Missing Values",
        status: stepStatus.step2,
        summary: [
        `Numeric: ${numericStrategy}`,
        `Categorical: ${categoricalStrategy}`,
        `Missing columns: ${plan?.missing_summary ? Object.keys(plan.missing_summary).length : 0}`
        ]
    },
    {
        key: "Step 3 Feature Scaling",
        title: "Step 3",
        subtitle: "Feature Scaling",
        status: stepStatus.step3,
        summary: [
        `Scaling: ${useScaling ? "Enabled" : "Disabled"}`,
        `Numeric features: ${plan?.numeric_features?.length || 0}`,
        `${plan ? "Preprocessing ready" : "Awaiting plan"}`
        ]
    },
    {
        key: "Step 4 Model Training",
        title: "Step 4",
        subtitle: "Model Training",
        status: stepStatus.step4,
        summary: [
        `Models selected: ${selectedModels.length}`,
        `Best model: ${trainResult?.best_model_key || "-"}`,
        `${trainResult ? "Training complete" : "Not trained"}`
        ]
    },

    {
        key: "Step 5 Model Evaluation & Saving",
        title: "Step 5",
        subtitle: "Evaluation & Saving",
        status: stepStatus.step5,
        summary: [
        `Charts: ${hasTrainingResult ? "Available" : "Locked"}`,
        `Selected model: ${selectedEvalModel || trainResult?.best_model_key || "-"}`,
        `${hasTrainingResult ? "Ready for review" : "Train a model first"}`
        ]
    }
    
    ];

    const plotConfig = {
    responsive: true,
    displayModeBar: true,
    };

    function formatModelLabel(modelKey) {
    const map = {
        linearregression: "Linear Regression",
        ridgeregression: "Ridge Regression",
        gradientboostingregressor: "Gradient Boosting Regressor",
        decisiontreeregressor: "Decision Tree Regressor",
        kneighborsregressor: "K-Neighbors Regressor",
        logisticregression: "Logistic Regression",
        decisiontreeclassifier: "Decision Tree Classifier",
    };
    return map[modelKey] || modelKey;
    }

    function renderMetricsChart(results, taskType) {
    if (!Array.isArray(results) || results.length === 0) {
        return <p className="text-muted">No metrics available.</p>;
    }

    const labels = results.map(item => formatModelLabel(item.model_key));

    if (taskType === "regression") {
        return (
        <Plot
            data={[
            {
                type: "bar",
                name: "MAE",
                x: labels,
                y: results.map(item => Number(item.metrics?.mae ?? 0)),
                marker: { color: "#ef476f" },
            },
            {
                type: "bar",
                name: "RMSE",
                x: labels,
                y: results.map(item => Number(item.metrics?.rmse ?? 0)),
                marker: { color: "#ffb703" },
            },
            ]}
            layout={{
            title: { text: "Regression Error Comparison" },
            barmode: "group",
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#e8eef2" },
            margin: { l: 50, r: 20, t: 50, b: 60 },
            xaxis: { title: { text: "Model" } },
            yaxis: { title: { text: "Error Value" } },
            }}
            style={{ width: "100%", height: "420px" }}
            config={plotConfig}
            useResizeHandler
        />
        );
    }

    const metricKeys = ["accuracy", "precision", "recall", "f1"];

    const traces = metricKeys.map((metricKey) => ({
        type: "bar",
        name: metricKey.toUpperCase(),
        x: labels,
        y: results.map(item => Number(item.metrics?.[metricKey] ?? 0)),
    }));

    return (
        <Plot
        data={traces}
        layout={{
            title: { text: "Model Metrics Comparison" },
            barmode: "group",
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#e8eef2" },
            margin: { l: 50, r: 20, t: 50, b: 60 },
            xaxis: { title: { text: "Model" } },
            yaxis: { title: { text: "Score" } },
        }}
        style={{ width: "100%", height: "420px" }}
        config={plotConfig}
        useResizeHandler
        />
    );
    }

    // // For testing and debuging the Plotly issue
    // function renderMetricsChart(results, taskType) {
    // if (!Array.isArray(results) || results.length === 0) {
    //     return <p className="text-muted">No metrics available.</p>;
    // }

    // const labels = results.map(item => formatModelLabel(item.model_key));
    // const metricKeys =
    //     taskType === "regression"
    //     ? ["r2", "mae", "rmse"]
    //     : ["accuracy", "precision", "recall", "f1"];

    // const traces = metricKeys.map((metricKey) => ({
    //     type: "bar",
    //     name: metricKey.toUpperCase(),
    //     x: labels,
    //     y: results.map(item => Number(item.metrics?.[metricKey] ?? 0)),
    // }));

    // return (
    //     <Plot
    //     data={traces}
    //     layout={{ title: { text: "Model Metrics Comparison" }, barmode: "group" }}
    //     style={{ width: "100%", height: "420px" }}
    //     config={{ responsive: true, displayModeBar: true }}
    //     useResizeHandler
    //     />
    // );
    // }

    function renderRegressionR2Cards(results, bestModelKey) {
    if (!Array.isArray(results) || results.length === 0) return null;

    return (
        <div
        style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "12px",
            marginBottom: "16px",
        }}
        >
        {results.map((item) => {
            const isBest = item.model_key === bestModelKey;
            const isSelected = item.model_key === selectedEvalModel;
            return (
            <div
                key={item.model_key}
                onClick={() => setSelectedEvalModel(item.model_key)}
                style={{
                border: isSelected ? "2px solid #22d3ee" : isBest ? "2px solid #06d6a0" : "1px solid #3a4654",
                borderRadius: "8px",
                padding: "12px",
                background: isSelected ? "rgba(34, 211, 238, 0.1)" : "#101820",
                cursor: "pointer",
                transition: "all 0.2s ease",
                }}
            >
                <div style={{ fontSize: "0.85rem", color: "#9fb3c8", marginBottom: "6px" }}>
                {formatModelLabel(item.model_key)}
                </div>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#ffffff" }}>
                {item.metrics?.r2 != null ? Number(item.metrics.r2).toFixed(3) : "-"}
                </div>
                <div style={{ fontSize: "0.8rem", color: isSelected ? "#22d3ee" : isBest ? "#06d6a0" : "#9fb3c8", marginTop: "4px" }}>
                {isSelected ? "Selected" : isBest ? "Best R² model" : "R² score"}
                </div>
            </div>
            );
        })}
        </div>
    );
    }



    function renderPredictedActualChart(activeModel) {
    const chart = activeModel?.chart_data?.predicted_vs_actual;
    if (!chart?.actual?.length || !chart?.predicted?.length) {
        return <p className="text-muted">No predicted-vs-actual data available.</p>;
    }

    const allVals = [...chart.actual, ...chart.predicted].map(Number).filter(v => !Number.isNaN(v));
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);

    return (
        <Plot
        data={[
            {
            type: "scatter",
            mode: "markers",
            x: chart.actual,
            y: chart.predicted,
            name: "Predictions",
            marker: {
                color: "#05d9fa",
                size: 1.5,
                opacity: 0.35,
            },
            },
            {
            type: "scatter",
            mode: "lines",
            x: [minVal, maxVal],
            y: [minVal, maxVal],
            name: "Ideal line",
            line: {
                color: "#ffb703",
                dash: "dash",
            },
            },
        ]}
        layout={{
            title: { text: "Predicted vs Actual" },
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#e8eef2" },
            margin: { l: 60, r: 20, t: 50, b: 60 },
            xaxis: { title: { text: "Actual" } },
            yaxis: { title: { text: "Predicted" } },
        }}
        style={{ width: "100%", height: "420px" }}
        config={plotConfig}
        useResizeHandler
        />
    );
    }

    function renderErrorHistogram(activeModel) {
    const values = activeModel?.chart_data?.percentage_error_hist?.values || [];
    if (!values.length) {
        return <p className="text-muted">No error distribution data available.</p>;
    }

    return (
        <Plot
        data={[
            {
            type: "histogram",
            x: values,
            marker: { color: "#ef476f" },
            nbinsx: 20,
            name: "Percentage Error",
            },
        ]}
        layout={{
            title: { text: "Prediction Error Distribution (%)" },
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#e8eef2" },
            margin: { l: 60, r: 20, t: 50, b: 60 },
            xaxis: { title: { text: "Percentage Error" } },
            yaxis: { title: { text: "Count" } },
        }}
        style={{ width: "100%", height: "420px" }}
        config={plotConfig}
        useResizeHandler
        />
    );
    }

    function renderFeatureChart(activeModel) {
    const insights = Array.isArray(activeModel?.insights) ? activeModel.insights.slice(0, 10) : [];
    if (!insights.length) {
        return <p className="text-muted">No feature insights available.</p>;
    }

    const isImportance = insights[0]?.importance != null;
    const xVals = insights.map(item => Number(isImportance ? item.importance : item.coefficient));
    const yVals = insights.map(item => item.feature);

    return (
                <div> {/* edited: wrap chart with a container so we can add explanatory tooltip text above it */}
        {!isImportance && (
            <div style={{ marginBottom: "10px", color: "#9fb3c8", fontSize: "0.85rem" }}> {/* edited: add coefficient help row above chart */}
            <span className="ml-metric-label-with-help"> {/* edited: reuse existing label+tooltip layout */}
                Coefficient {/* edited: add label for explanation */}
                <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                <span className="ml-metric-tooltip">
                    Coefficient shows how strongly a feature affects the prediction in linear models. A positive value means the prediction tends to increase as the feature increases, while a negative value means it tends to decrease. A larger absolute value usually means a stronger influence.
                </span> {/* edited: add coefficient explanation tooltip */}
                </span>
            </span>
            </div>
        )}
        <Plot
        data={[
            {
            type: "bar",
            orientation: "h",
            x: xVals.slice().reverse(),
            y: yVals.slice().reverse(),
            marker: {
                color: isImportance ? "#06d6a0" : "#a78bfa",
            },
            name: isImportance ? "Importance" : "Coefficient",
            },
        ]}
        layout={{
            title: { text: isImportance ? "Top Feature Importances" : "Top Feature Coefficients" },
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#e8eef2" },
            margin: { l: 180, r: 20, t: 50, b: 50 },
            xaxis: { title: { text: isImportance ? "Importance" : "Coefficient" } },
            yaxis: { automargin: true },
        }}
        style={{ width: "100%", height: "440px" }}
        config={plotConfig}
                    useResizeHandler
                    />
                </div>
    );
    }

    function renderConfusionMatrixChart(activeModel) {
    const cm = activeModel?.chart_data?.confusion_matrix;
    if (!cm?.matrix?.length || !cm?.labels?.length) {
        return <p className="text-muted">No confusion matrix data available.</p>;
    }

    return (
        <Plot
        data={[
            {
            type: "heatmap",
            z: cm.matrix,
            x: cm.labels,
            y: cm.labels,
            colorscale: "Blues",
            showscale: true,
            },
        ]}
        layout={{
            title: { text: "Confusion Matrix" },
            autosize: true,
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#e8eef2" },
            margin: { l: 70, r: 20, t: 50, b: 70 },
            xaxis: { title: { text: "Predicted" } },
            yaxis: { title: { text: "Actual" }, autorange: "reversed" },
        }}
        style={{ width: "100%", height: "420px" }}
        config={plotConfig}
        useResizeHandler
        />
    );
    }

    const renderStepDetails = () => { 
    if (!activeStep) {
        return <p className="text-muted">Click a step in the pipeline to view and edit its parameters.</p>;
    }

    // ==================== TIME-SERIES STEPS ====================
    if (taskType === "timeseries") {
        if (activeStep === "Step 1 Data & Target") {
            const datetimeCols = columns.filter(c => c.is_datetime);
            const numericCols = columns.filter(c => {
                const dtype = (c.dtype || "").toLowerCase();
                return (dtype.startsWith("int") || dtype.startsWith("float") || dtype.startsWith("double") || dtype.startsWith("number"));
            });
            return (
            <div>
                <h5 className="step-title">
                Step 1 Data & Target <span className="badge-status">Active</span>
                </h5>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Date / Time Column</label><br />
                <select value={datetimeColumn} onChange={(e) => setDatetimeColumn(e.target.value)}>
                    <option value="">-- Select datetime column --</option>
                    {datetimeCols.map(col => (
                    <option key={col.name} value={col.name}>{col.name}</option>
                    ))}
                    {datetimeCols.length === 0 && columns.filter(c => c.name !== targetColumn).map(col => (
                    <option key={col.name} value={col.name}>{col.name} (will attempt parse)</option>
                    ))}
                </select>
                {datetimeColumn && (
                    <div style={{ fontSize: "11px", color: "#8b949e", marginTop: "4px" }}>
                    Column will be resampled to regular frequency for ARIMAX
                    </div>
                )}
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Target Column (what to forecast)</label><br />
                <select value={targetColumn} onChange={(e) => { setTargetColumn(e.target.value); setFeatureColumns(prev => prev.filter(c => c !== e.target.value)); }}>
                    <option value="">-- Select target --</option>
                    {numericCols.map(col => (
                    <option key={col.name} value={col.name}>{col.name}</option>
                    ))}
                </select>
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Exogenous Feature Columns (optional)</label>
                <div style={{ fontSize: "0.82rem", color: "#94a3b8", marginBottom: "6px" }}>
                    External columns that may help predict the target.
                </div>
                <div style={{ display: "flex", gap: "8px", marginBottom: "6px" }}>
                    <button type="button" className="btn btn-ghost" style={{ fontSize: "0.75rem", padding: "2px 8px" }}
                        onClick={() => setFeatureColumns(columns.filter(c => c.name !== targetColumn && c.name !== datetimeColumn).map(c => c.name))}>
                        Select All
                    </button>
                    <button type="button" className="btn btn-ghost" style={{ fontSize: "0.75rem", padding: "2px 8px" }}
                        onClick={() => setFeatureColumns([])}>
                        Select None
                    </button>
                </div>
                <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                    gap: "6px",
                }}>
                    {columns
                    .filter(col => col.name !== targetColumn && col.name !== datetimeColumn)
                    .map(col => (
                        <label key={col.name}>
                        <input
                            type="checkbox"
                            checked={featureColumns.includes(col.name)}
                            onChange={() => {
                            setFeatureColumns(prev =>
                                prev.includes(col.name)
                                ? prev.filter(c => c !== col.name)
                                : [...prev, col.name]
                            );
                            }}
                        />{" "}
                        {col.name}
                        </label>
                    ))}
                </div>
                </div>



                <div className="card" style={{ padding: "10px", marginTop: "10px" }}>
                <p><strong>Rows:</strong> {datasetProfile?.shape?.rows ?? "-"}</p>
                <p><strong>Columns:</strong> {datasetProfile?.shape?.columns ?? "-"}</p>
                <p><strong>Datetime columns detected:</strong> {datetimeCols.length}</p>
                </div>

                <div style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
                <button
                    className="btn btn-primary"
                    onClick={() => {
                    if (!datetimeColumn) { setMlError("Please select a datetime column."); return; }
                    if (!targetColumn) { setMlError("Please select a target column."); return; }
                    setMlError("");
                    setActiveStep("Step 2 Modeling & Training");
                    }}
                    disabled={!datetimeColumn || !targetColumn}
                >
                    Continue to Modeling & Training
                </button>
                </div>
            </div>
            );
        }

        if (activeStep === "Step 2 Modeling & Training") {
            return (
            <div>
                <h5 className="step-title">
                Step 2 Modeling & Training <span className="badge-status">Active</span>
                </h5>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Model Type
                    <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                    <span className="ml-metric-tooltip" style={{ width: 280 }}>
                        Select which time-series models to train. <strong>ARIMAX</strong> uses native time-series modeling with confidence intervals. <strong>XGBoost</strong> auto-generates lag and date features for higher accuracy.
                    </span>
                    </span>
                </label>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    {[
                        { key: "arimax", label: "ARIMAX", desc: "ARIMA + eXogenous regressors — native confidence intervals" },
                        { key: "xgboost", label: "XGBoost", desc: "Gradient Boosting with auto-generated lag & date features" },
                    ].map((m) => (
                        <label key={m.key} style={{ display: "flex", alignItems: "flex-start", gap: "10px", padding: "10px 12px", background: tsSelectedModels.includes(m.key) ? "rgba(34,211,238,0.08)" : "#0f1a2e", border: `1px solid ${tsSelectedModels.includes(m.key) ? "rgba(34,211,238,0.4)" : "#1e2d3d"}`, borderRadius: "8px", cursor: "pointer", transition: "all 0.2s" }}>
                            <input
                                type="checkbox"
                                checked={tsSelectedModels.includes(m.key)}
                                onChange={() => {
                                    setTsSelectedModels(prev =>
                                        prev.includes(m.key)
                                            ? prev.filter(k => k !== m.key)
                                            : [...prev, m.key]
                                    );
                                }}
                                style={{ marginTop: "2px", accentColor: "#22d3ee" }}
                            />
                            <div>
                                <span style={{ fontSize: "0.85rem", fontWeight: 500, color: "#e8eef2" }}>{m.label}</span>
                                <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: "2px" }}>{m.desc}</div>
                            </div>
                        </label>
                    ))}
                </div>
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Forecast Horizon
                    <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                    <span className="ml-metric-tooltip" style={{ width: 260 }}>
                        How many future periods to predict. For example, <strong>30 with Daily</strong> means forecasting the next 30 days. Longer horizons produce wider confidence bands (more uncertainty).
                    </span>
                    </span>
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <input
                    type="range"
                    min={7}
                    max={365}
                    value={forecastHorizon}
                    onChange={(e) => setForecastHorizon(parseInt(e.target.value))}
                    style={{ flex: 1 }}
                    />
                    <input
                    type="number"
                    min={7}
                    max={365}
                    value={forecastHorizonInput}
                    onChange={(e) => setForecastHorizonInput(e.target.value)}
                    onBlur={() => {
                        const v = parseInt(forecastHorizonInput);
                        if (!isNaN(v) && v >= 7 && v <= 365) setForecastHorizon(v);
                        else setForecastHorizonInput(String(forecastHorizon));
                    }}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            const v = parseInt(forecastHorizonInput);
                            if (!isNaN(v) && v >= 7 && v <= 365) setForecastHorizon(v);
                            else setForecastHorizonInput(String(forecastHorizon));
                        }
                    }}
                    style={{ width: "60px", padding: "4px 8px", background: "#0f1a2e", border: "1px solid #1e2d3d", borderRadius: "6px", color: "#22d3ee", fontWeight: 700, fontSize: "0.9rem", textAlign: "center" }}
                    />
                    <span style={{ fontSize: "0.8rem", color: "#64748b" }}>periods</span>
                </div>
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Frequency
                    <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                    <span className="ml-metric-tooltip" style={{ width: 260 }}>
                        The time interval between data points. Choose <strong>Daily</strong> if your data has one row per day, <strong>Weekly</strong> for weekly aggregates, or <strong>Monthly</strong> for monthly totals. The data is resampled to this frequency.
                    </span>
                    </span>
                </label>
                <select value={tsFrequency} onChange={(e) => setTsFrequency(e.target.value)}>
                    <option value="D">Daily</option>
                    <option value="W">Weekly</option>
                    <option value="M">Monthly</option>
                </select>
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Aggregation Method
                    <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                    <span className="ml-metric-tooltip" style={{ width: 300 }}>
                        How to combine multiple rows within each time period into one value.<br/><br/>
                        <strong>Sum</strong> — adds all values in the period. Best for revenue, sales count, traffic, or any metric where the total matters.<br/><br/>
                        <strong>Mean</strong> — averages all values in the period. Best for rates, scores, percentages, or any metric where the average matters.<br/><br/>
                        This applies to the target column and all numeric exogenous features equally.
                    </span>
                    </span>
                </label>
                <select value={tsAggregation} onChange={(e) => setTsAggregation(e.target.value)}>
                    <option value="sum">Sum (total per period)</option>
                    <option value="mean">Mean (average per period)</option>
                </select>
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Train / Test Split
                    <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                    <span className="ml-metric-tooltip" style={{ width: 260 }}>
                        The percentage of data held out for testing. Unlike regular ML, time-series <strong>never shuffles</strong> data — the last N% of rows (by date) are used for evaluation. For example, <strong>20%</strong> means the most recent 20% of your data tests the model.
                    </span>
                    </span>
                </label>
                <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                    <input
                    type="range"
                    min={10}
                    max={40}
                    step={5}
                    value={Math.round(tsTestSize * 100)}
                    onChange={(e) => setTsTestSize(parseInt(e.target.value) / 100)}
                    style={{ flex: 1 }}
                    />
                    <input
                    type="number"
                    min={10}
                    max={40}
                    step={5}
                    value={tsTestSizeInput}
                    onChange={(e) => setTsTestSizeInput(e.target.value)}
                    onBlur={() => {
                        const v = parseInt(tsTestSizeInput);
                        if (!isNaN(v) && v >= 10 && v <= 40) setTsTestSize(v / 100);
                        else setTsTestSizeInput(String(Math.round(tsTestSize * 100)));
                    }}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            const v = parseInt(tsTestSizeInput);
                            if (!isNaN(v) && v >= 10 && v <= 40) setTsTestSize(v / 100);
                            else setTsTestSizeInput(String(Math.round(tsTestSize * 100)));
                        }
                    }}
                    style={{ width: "55px", padding: "4px 8px", background: "#0f1a2e", border: "1px solid #1e2d3d", borderRadius: "6px", color: "#22d3ee", fontWeight: 700, fontSize: "0.9rem", textAlign: "center" }}
                    />
                    <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "#22d3ee" }}>%</span>
                </div>
                <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: "4px" }}>Last {Math.round(tsTestSize * 100)}% of chronological rows held out for testing (no shuffle)</div>
                </div>

                <div style={{ background: "#0f1a2e", border: "1px solid #1e2d3d", borderRadius: "8px", padding: "12px", marginTop: "12px" }}>
                <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginBottom: "8px" }}>Forecast Summary</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", fontSize: "0.8rem" }}>
                    <div><span style={{ color: "#64748b" }}>Models: </span><span style={{ color: "#e8eef2" }}>{tsSelectedModels.map(m => m === "arimax" ? "ARIMAX" : "XGBoost").join(", ") || "None"}</span></div>
                    <div><span style={{ color: "#64748b" }}>Horizon: </span><span style={{ color: "#e8eef2" }}>{forecastHorizon} periods</span></div>
                    <div><span style={{ color: "#64748b" }}>Aggregation: </span><span style={{ color: "#e8eef2" }}>{tsAggregation === "sum" ? "Sum" : "Mean"}</span></div>
                    <div><span style={{ color: "#64748b" }}>Frequency: </span><span style={{ color: "#e8eef2" }}>{tsFrequency === "D" ? "Daily" : tsFrequency === "W" ? "Weekly" : "Monthly"}</span></div>
                </div>
                </div>

                {mlError && (
                <p className="error-text small" style={{ marginTop: "10px" }}>Error: {mlError}</p>
                )}

                {tsTraining && tsTrainingProgress !== null && (
                <div className="progress-wrapper">
                    <div className="progress-label">
                    <span>{tsTrainingStep}</span>
                    <span>{tsTrainingProgress}%</span>
                    </div>
                    <div className="progress-container">
                    <div className="progress-fill" style={{ width: `${tsTrainingProgress}%` }} />
                    </div>
                    {tsSelectedModels.length > 0 && (
                        <div style={{ marginTop: "10px", display: "flex", flexDirection: "column", gap: "6px" }}>
                            {tsSelectedModels.map((modelKey, idx) => {
                                const modelLabel = modelKey === "arimax" ? "ARIMAX" : "XGBoost";
                                const isCompleted = tsTrainingProgress >= ((idx + 1) / tsSelectedModels.length) * 100;
                                const isCurrent = tsTrainingStep && tsTrainingStep.includes(modelLabel);
                                return (
                                    <div key={modelKey} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.8rem" }}>
                                        <span style={{
                                            width: "18px", height: "18px", borderRadius: "50%",
                                            display: "flex", alignItems: "center", justifyContent: "center",
                                            background: isCompleted ? "#06d6a0" : isCurrent ? "#22d3ee" : "var(--bg-card-soft)",
                                            color: isCompleted || isCurrent ? "#000" : "var(--text-muted)",
                                            fontSize: "0.7rem", fontWeight: 600, flexShrink: 0
                                        }}>
                                            {isCompleted ? "\u2713" : idx + 1}
                                        </span>
                                        <span style={{ color: isCompleted ? "#06d6a0" : isCurrent ? "#22d3ee" : "var(--text-muted)", fontWeight: isCurrent ? 600 : 400 }}>
                                            {modelLabel}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
                )}

                {forecastHorizon > 180 && (
                <div style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: "8px", padding: "10px 12px", marginTop: "12px", fontSize: "0.8rem", color: "#f59e0b" }}>
                    Long forecast horizon ({forecastHorizon} periods). Confidence bands will widen significantly. For best accuracy, consider keeping the horizon under 180 periods.
                </div>
                )}

                <div style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
                <button className="btn btn-secondary" onClick={() => setActiveStep("Step 1 Data & Target")}>Back to Data</button>
                <button className="btn btn-primary" onClick={handleTrainTimeseries} disabled={tsTraining || tsSelectedModels.length === 0}>
                    {tsTraining ? "Training..." : `Train ${tsSelectedModels.length > 1 ? tsSelectedModels.length + " Models" : tsSelectedModels[0] === "arimax" ? "ARIMAX" : "XGBoost"}`}
                </button>
                </div>
            </div>
            );
        }

        if (activeStep === "Step 3 Evaluation") {
            if (!hasTsTrainResult) {
            return (
                <div>
                <h5 className="step-title">
                    Step 3 Evaluation <span className="badge-status">Active</span>
                </h5>
                <p className="text-muted">Train the model in Step 2 to view forecast results.</p>
                <div style={{ marginTop: "14px" }}>
                    <button className="btn btn-secondary" onClick={() => setActiveStep("Step 2 Modeling & Training")}>Back to Model Config</button>
                </div>
                </div>
            );
            }

            const tsResults = tsTrainResult.results || [];
            const activeResult = tsResults.find(r => r.model_type === tsSelectedEvalModel) || tsResults[0];
            const activeMetrics = activeResult?.test_metrics || {};

            return (
            <div>
                <h5 className="step-title">
                Step 3 Evaluation <span className="badge-status">Completed</span>
                </h5>

                {tsResults.length > 1 && (
                <div style={{ marginBottom: "16px" }}>
                    <label className="small" style={{ marginBottom: "6px", display: "block" }}>Select Model to View</label>
                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                    {tsResults.map((r) => {
                        const isBest = r.model_type === tsTrainResult.best_model_key;
                        const isActive = r.model_type === tsSelectedEvalModel;
                        return (
                            <button
                                key={r.model_type}
                                onClick={() => setTsSelectedEvalModel(r.model_type)}
                                style={{
                                    padding: "8px 16px", borderRadius: "8px", border: `1px solid ${isActive ? "#22d3ee" : "#1e2d3d"}`,
                                    background: isActive ? "rgba(34,211,238,0.1)" : "#0f1a2e", color: isActive ? "#22d3ee" : "#94a3b8",
                                    cursor: "pointer", fontSize: "0.85rem", fontWeight: isActive ? 600 : 400, transition: "all 0.2s"
                                }}
                            >
                                {r.model_label || r.model_type}
                                {isBest && <span style={{ marginLeft: "6px", fontSize: "0.7rem", background: "rgba(6,214,160,0.15)", color: "#06d6a0", padding: "1px 6px", borderRadius: "4px" }}>BEST</span>}
                            </button>
                        );
                    })}
                    </div>
                </div>
                )}

                {tsResults.length > 1 && (
                <div style={{ background: "#0f1a2e", border: "1px solid #1e2d3d", borderRadius: "8px", padding: "14px", marginBottom: "16px" }}>
                    <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginBottom: "10px" }}>Model Comparison</div>
                    <table style={{ width: "100%", fontSize: "0.8rem", borderCollapse: "collapse" }}>
                    <thead>
                        <tr style={{ color: "#64748b", borderBottom: "1px solid #1e2d3d" }}>
                        <th style={{ textAlign: "left", padding: "6px 8px" }}>Model</th>
                        <th style={{ textAlign: "right", padding: "6px 8px" }}>R&sup2;</th>
                        <th style={{ textAlign: "right", padding: "6px 8px" }}>MAPE</th>
                        <th style={{ textAlign: "right", padding: "6px 8px" }}>RMSE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tsResults.map((r) => {
                        const m = r.test_metrics || {};
                        const isBest = r.model_type === tsTrainResult.best_model_key;
                        return (
                            <tr key={r.model_type} style={{ color: isBest ? "#06d6a0" : "#e8eef2", borderBottom: "1px solid #1e2d3d" }}>
                            <td style={{ padding: "6px 8px", fontWeight: isBest ? 600 : 400 }}>
                                {r.model_label || r.model_type}
                                {isBest && <span style={{ marginLeft: "6px", fontSize: "0.65rem", color: "#06d6a0" }}>(best)</span>}
                            </td>
                            <td style={{ textAlign: "right", padding: "6px 8px" }}>{m.r2 != null ? m.r2.toFixed(3) : "-"}</td>
                            <td style={{ textAlign: "right", padding: "6px 8px" }}>{m.mape != null ? m.mape.toFixed(1) + "%" : "-"}</td>
                            <td style={{ textAlign: "right", padding: "6px 8px" }}>{m.rmse != null && m.target_mean != null && m.target_mean > 0 ? ((m.rmse / m.target_mean) * 100).toFixed(1) + "%" : (m.rmse != null ? m.rmse.toLocaleString(undefined, {maximumFractionDigits:0}) : "-")}</td>
                            </tr>
                        );
                        })}
                    </tbody>
                    </table>
                </div>
                )}

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px", marginBottom: "16px" }}>
                {[
                    { label: "MAPE", value: activeMetrics.mape != null ? activeMetrics.mape.toFixed(1) + "%" : "-", good: activeMetrics.mape != null && activeMetrics.mape < 10,
                      tip: "Mean Absolute Percentage Error. Lower is better. <10% is excellent, 10-20% is good." },
                    { label: "RMSE", value: activeMetrics.rmse != null && activeMetrics.target_mean != null && activeMetrics.target_mean > 0 ? ((activeMetrics.rmse / activeMetrics.target_mean) * 100).toFixed(1) + "%" : (activeMetrics.rmse != null ? activeMetrics.rmse.toLocaleString(undefined, {maximumFractionDigits:0}) : "-"), good: true,
                      tip: "Root Mean Squared Error as % of mean — penalizes large errors more." },
                    { label: "R\u00B2", value: activeMetrics.r2 != null ? activeMetrics.r2.toFixed(3) : "-", good: activeMetrics.r2 != null && activeMetrics.r2 > 0.8,
                      tip: "Coefficient of determination. 1.0 = perfect. >0.8 is strong." },
                    { label: "Horizon", value: tsTrainResult.forecast_horizon + " periods", good: true,
                      tip: "How many future periods were forecast." },
                ].map((m) => (
                    <div key={m.label} style={{ background: "#0f1a2e", border: "1px solid #1e2d3d", borderRadius: "8px", padding: "14px", textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem", color: "#64748b" }}>{m.label}
                        <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                        <span className="ml-metric-tooltip" dangerouslySetInnerHTML={{ __html: m.tip }} />
                        </span>
                    </div>
                    <div style={{ fontSize: "1.4rem", fontWeight: 700, color: m.good ? "#06d6a0" : "#22d3ee", marginTop: "4px" }}>{m.value}</div>
                    </div>
                ))}
                </div>

                <div style={{ background: "#0f1a2e", border: "1px solid #1e2d3d", borderRadius: "8px", padding: "14px", marginBottom: "16px", textAlign: "center" }}>
                    <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                    Detailed forecast chart and data table available in <strong style={{ color: "#22d3ee" }}>ML Prediction</strong> &rarr; <strong style={{ color: "#22d3ee" }}>Time-Series</strong> tab
                    </div>
                </div>

                <div style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
                <button className="btn btn-secondary" onClick={() => setActiveStep("Step 2 Modeling & Training")}>Back to Model Config</button>
                <button className="btn btn-primary" onClick={() => onNavigate && onNavigate("predict", "timeseries")} style={{ marginLeft: "auto" }}>
                    Proceed to ML Prediction &rarr;
                </button>
                </div>
            </div>
            );
        }
    }

    // ==================== STANDARD ML STEPS ====================
    if (activeStep === "Step 1 Data & Target") {
        return (
            <div>
            <h5 className="step-title">
                Step 1 Data & Target <span className="badge-status">Active</span>
            </h5>

            <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Target Column</label><br />
                <select
                value={targetColumn}
                onChange={(e) => {
                    const newTarget = e.target.value;
                    setTargetColumn(newTarget);
                    setFeatureColumns(prev => prev.filter(col => col !== newTarget));
                }}
                >
                {columns.map(col => (
                    <option key={col.name} value={col.name}>
                    {col.name}
                    </option>
                ))}
                </select>
                {isDatetimeColumn(targetColumn) && (
                    <div style={{ fontSize: "11px", color: "#8b949e", marginTop: "4px" }}>
                        ⓘ This column will be decomposed into year, month, day, day_of_week during training
                    </div>
                )}
            </div>

            <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Feature Columns</label>
                <div style={{ marginTop: "8px", marginBottom: "6px" }}>
                    <label style={{ fontSize: "0.85rem", cursor: "pointer" }}>
                        <input
                            type="checkbox"
                            checked={featureColumns.length === columns.filter(c => c.name !== targetColumn).length}
                            onChange={(e) => {
                                if (e.target.checked) {
                                    setFeatureColumns(columns.filter(c => c.name !== targetColumn).map(c => c.name));
                                } else {
                                    setFeatureColumns([]);
                                }
                            }}
                        />{" "}
                        Select All
                    </label>
                </div>
                <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                gap: "6px",
                }}>
                {columns
                    .filter(col => col.name !== targetColumn)
                    .map(col => (
                    <label key={col.name}>
                        <input
                        type="checkbox"
                        checked={featureColumns.includes(col.name)}
                        onChange={() => handleFeatureToggle(col.name)}
                        />{" "}
                        {col.name}
                        {isDatetimeColumn(col.name) && (
                            <span style={{ fontSize: "10px", color: "#8b949e", marginLeft: "4px" }}>
                                (datetime → year, month, day, day_of_week)
                            </span>
                        )}
                    </label>
                    ))}
                </div>
            </div>

            <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Task Type</label><br />
                <div style={{
                    padding: "8px 12px",
                    borderRadius: "8px",
                    border: "1px solid var(--border-soft)",
                    background: "#020617",
                    color: taskType ? "var(--accent)" : "var(--text-muted)",
                    fontSize: "0.85rem",
                    fontWeight: taskType ? "600" : "400"
                }}>
                    {taskType ? (taskType === "regression" ? "Regression (predicting a numeric value)" : "Classification (predicting a category)") : "Will be inferred after generating preprocessing plan"}
                </div>
            </div>

            <div className="card" style={{ padding: "10px", marginTop: "10px" }}>
                <p><strong>Rows:</strong> {datasetProfile?.shape?.rows ?? "-"}</p>
                <p><strong>Columns:</strong> {datasetProfile?.shape?.columns ?? "-"}</p>
                <p><strong>Selected Features:</strong> {featureColumns.length}</p>
            </div>

            <div style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
                <button
                className="btn btn-primary"
                onClick={fetchPreprocessingPlan}
                disabled={loadingPlan}
                >
                {loadingPlan ? "Loading Plan..." : "Generate Preprocessing Plan"}
                </button>
            </div>
            </div>
        );
        }
    

    if (activeStep === "Step 2 Missing Values") {
        return (
        <div>
            <h5 className="step-title">
            Step 2 Missing Values <span className="badge-status">Active</span>
            </h5>

            <div className="form-group" style={{ marginBottom: "12px" }}>
            <label className="small">Numeric Missing Strategy</label><br />
            <select
                value={numericStrategy}
                onChange={(e) => setNumericStrategy(e.target.value)}
            >
                <option value="median">Median (fill with middle value)</option>
                <option value="mean">Mean (fill with average value)</option>
                <option value="drop">Ignore row (remove rows with missing values)</option>
                </select>
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
                <label className="small">Aggregation Method
                    <span className="ml-metric-help" style={{ marginLeft: "4px" }}>?
                    <span className="ml-metric-tooltip" style={{ width: 300 }}>
                        How to combine multiple rows within each time period into one value.<br/><br/>
                        <strong>Sum</strong> — adds all values in the period. Best for revenue, sales count, traffic, or any metric where the total matters.<br/><br/>
                        <strong>Mean</strong> — averages all values in the period. Best for rates, scores, percentages, or any metric where the average matters.<br/><br/>
                        This applies to the target column and all numeric exogenous features equally.
                    </span>
                    </span>
                </label>
                <select value={tsAggregation} onChange={(e) => setTsAggregation(e.target.value)}>
                    <option value="sum">Sum (total per period)</option>
                    <option value="mean">Mean (average per period)</option>
                </select>
                </div>

                <div className="form-group" style={{ marginBottom: "12px" }}>
            <label className="small">Categorical Missing Strategy</label><br />
            <select
                value={categoricalStrategy}
                onChange={(e) => setCategoricalStrategy(e.target.value)}
            >
                <option value="most_frequent">Most Frequent (fill with most common value)</option>
                <option value="constant">Constant (fill with a fixed value like "Missing")</option>
                <option value="drop">Ignore row (remove rows with missing values)</option>
            </select>
            </div>

            {!plan ? (
            <p className="text-muted">
                Generate the preprocessing plan first to inspect missing values.
            </p>
            ) : (
            <div>
                <p><strong>Detected Missing Values</strong></p>
                {Object.keys(plan?.missing_summary || {}).length === 0 ? (
                <p className="text-muted">No missing values detected in selected features.</p>
                ) : (
                <ul style={{ paddingLeft: "18px" }}>
                    {Object.entries(plan.missing_summary).map(([col, count]) => (
                    <li key={col}>{col}: {count}</li>
                    ))}
                </ul>
                )}

                <div className="card" style={{ padding: "10px", marginTop: "10px" }}>
                <p><strong>Recommended numeric strategy:</strong> {plan?.recommendations?.numeric_strategy || "-"}</p>
                <p><strong>Recommended categorical strategy:</strong> {plan?.recommendations?.categorical_strategy || "-"}</p>
                </div>
            </div>
            )}

            <div style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
            <button
                className="btn btn-primary"
                onClick={() => setActiveStep("Step 3 Feature Scaling")}
                disabled={!plan}
            >
                Continue to Feature Scaling
            </button>
            </div>
        </div>
        );
    }



    if (activeStep === "Step 3 Feature Scaling") {
        return (
        <div>
            <h5 className="step-title">
            Step 3 Feature Scaling <span className="badge-status">Active</span>
            </h5>

            <div style={{ marginBottom: "12px" }}>
            <label>
                <input
                type="checkbox"
                checked={useScaling}
                onChange={(e) => setUseScaling(e.target.checked)}
                />{" "}
                Enable Feature Scaling (StandardScaler)
            </label>
            </div>

            <p><strong>Numeric Features:</strong> {plan?.numeric_features?.join(", ") || "None"}</p>

            <div style={{ padding: "10px", borderRadius: "8px", background: "var(--bg-card-soft)", border: "1px solid var(--border-soft)", marginBottom: "12px" }}>
                <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", margin: 0 }}>
                    <strong>What is feature scaling?</strong> Feature scaling ensures that columns with naturally large values (e.g., revenue in thousands) don't dominate columns with smaller values (e.g., conversion rate between 0-1). Without scaling, a model might unfairly favor features just because their numbers are bigger, even if they aren't more important. Scaling puts all features on a comparable scale so the model can learn from each one fairly.
                </p>
            </div>

            <div style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
            <button
                className="btn btn-primary"
                onClick={() => setActiveStep("Step 4 Model Training")}
                disabled={!plan}
            >
                Continue to Model Training
            </button>
            </div>
        </div>
        );
    }

    if (activeStep === "Step 4 Model Training") {
    return (
        <div>
        <h5 className="step-title">
            Step 4 Model Training <span className="badge-status">Active</span>
        </h5>

        <div className="form-group" style={{ marginBottom: "12px" }}>
            <label className="small">Models</label>
            <div style={{ lineHeight: 1.8, marginTop: "8px" }}>
            {(taskType === "regression"
                ? [
                    ["linearregression", "Linear Regression"],
                    ["ridgeregression", "Ridge Regression"],
                    ["gradientboostingregressor", "Gradient Boosting Regressor"],
                    ["decisiontreeregressor", "Decision Tree Regressor"],
                    ["kneighborsregressor", "K-Neighbors Regressor"]
                ]
                : [
                    ["logisticregression", "Logistic Regression"],
                    ["decisiontreeclassifier", "Decision Tree Classifier"]
                ]
            ).map(([modelKey, label]) => (
                <label key={modelKey} style={{ display: "block" }}>
                <input
                    type="checkbox"
                    checked={selectedModels.includes(modelKey)}
                    onChange={() =>
                    setSelectedModels(prev =>
                        prev.includes(modelKey)
                        ? prev.filter(m => m !== modelKey)
                        : [...prev, modelKey]
                    )
                    }
                />{" "}
                {label}
                </label>
            ))}
            </div>
        </div>

        <div style={{ marginTop: "14px", display: "flex", gap: "8px" }}>
            <button
            className="btn btn-primary"
            onClick={handleTrainModels}
            disabled={training || !plan}
            >
            {training ? "Training..." : "Train Models"}
            </button>
        </div>

        {training && trainingProgress !== null && (
            <div className="progress-wrapper">
                <div className="progress-label">
                    <span>{trainingStep}</span>
                    <span>{trainingProgress}%</span>
                </div>
                <div className="progress-container">
                    <div className="progress-fill" style={{ width: `${trainingProgress}%` }} />
                </div>
                {selectedModels.length > 0 && (
                    <div style={{ marginTop: "10px", display: "flex", flexDirection: "column", gap: "6px" }}>
                        {selectedModels.map((modelKey, idx) => {
                            const modelLabel = formatModelLabel(modelKey);
                            const isCompleted = trainingProgress >= ((idx + 1) / selectedModels.length) * 100;
                            const isCurrent = trainingStep && trainingStep.includes(modelLabel);
                            return (
                                <div key={modelKey} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.8rem" }}>
                                    <span style={{
                                        width: "18px", height: "18px", borderRadius: "50%",
                                        display: "flex", alignItems: "center", justifyContent: "center",
                                        background: isCompleted ? "#06d6a0" : isCurrent ? "#22d3ee" : "var(--bg-card-soft)",
                                        color: isCompleted || isCurrent ? "#000" : "var(--text-muted)",
                                        fontSize: "0.7rem", fontWeight: 600, flexShrink: 0
                                    }}>
                                        {isCompleted ? "\u2713" : idx + 1}
                                    </span>
                                    <span style={{ color: isCurrent ? "var(--accent)" : isCompleted ? "#06d6a0" : "var(--text-muted)" }}>
                                        {modelLabel}
                                    </span>
                                    {isCurrent && <span style={{ color: "var(--accent)", fontSize: "0.75rem" }}>Training...</span>}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        )}

        {!trainResult ? (
            <p className="text-muted" style={{ marginTop: "12px" }}> {/* edited: keep pre-training helper text */}
                Train the selected models to view performance summary.
            </p>
        ) : (
            <div style={{ marginTop: "14px" }}> {/* edited: keep result container but remove duplicate per-model metric details */}
                <p><strong>Best Model: </strong> {formatModelLabel(trainResult?.best_model_key) || "-"}</p> {/* edited: retain best model summary only */}
                <p><strong>Train Rows: </strong> {trainResult?.train_rows ?? "-"}</p> {/* edited: retain train row count summary */}
                <p><strong>Test Rows: </strong> {trainResult?.test_rows ?? "-"}</p> {/* edited: retain test row count summary */}

                <div style={{ marginTop: "14px", display: "flex", gap: "8px", flexWrap: "wrap" }}> {/* edited: add action row for next step */}
                    <button
                        className="btn btn-primary"
                        onClick={() => setActiveStep("Step 5 Model Evaluation & Saving")}
                    >
                        Continue to Evaluation & Saving
                    </button>
                </div>
            </div>
        )}
        </div>
    );
    }

    if (activeStep === "Step 5 Model Evaluation & Saving") {
    const results = trainResult?.results || [];
    const activeModel =
        results.find((item) => item.model_key === selectedEvalModel) || results[0] || null;

    if (!trainResult || results.length === 0) {
        return (
        <div className="pipeline-step-card">
            <h3>Model Evaluation & Saving</h3>
            <p>Train at least one model to view evaluation charts and export options.</p>
        </div>
        );
    }

    return (
    <div className="pipeline-step-card">
        <div className="step-card-header">
            <div>
            <h3>Model Evaluation & Saving</h3>
            <p>Review trained model performance and inspect evaluation charts.</p>
            </div>
        </div>

        <div className="ml-eval-summary-grid">
            <div className="ml-mini-stat">
            <span className="ml-mini-stat-label">Best Model:  </span>
            <strong>{formatModelLabel(trainResult.best_model_key) || "-"}</strong> {/* show friendly best model name instead of raw model key */}
            </div>

            <div className="ml-mini-stat">
            <span className="ml-mini-stat-label">Train Rows:  </span>
            <strong>{trainResult.train_rows ?? "-"}</strong>
            </div>

            <div className="ml-mini-stat">
            <span className="ml-mini-stat-label">Test Rows:  </span>
            <strong>{trainResult.test_rows ?? "-"}</strong>
            </div>
        </div>
        <br />

        <div className="ml-eval-toolbar">
            <label>
            Model
            <select
                value={selectedEvalModel}
                onChange={(e) => setSelectedEvalModel(e.target.value)}
            >
                {results.map((item) => (
                    <option key={item.model_key} value={item.model_key}>
                        {formatModelLabel(item.model_key)} {/* show friendly model name in dropdown */}
                    </option>
                ))}
            </select>
            </label>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginTop: "12px", marginBottom: "12px" }}>
            <button className={`btn btn-secondary ${selectedEvalChart === "metrics" ? "active" : ""}`}
             onClick={() => setSelectedEvalChart("metrics")}>
                Metrics
            </button>

            {trainResult.task_type === "regression" && (
                <button className={`btn btn-secondary ${selectedEvalChart === "predictedActual" ? "active" : ""}`}
                 onClick={() => setSelectedEvalChart("predictedActual")}>
                Actual vs Predicted
                </button>
            )}

            <button className={`btn btn-secondary ${selectedEvalChart === "importance" ? "active" : ""}`}
             onClick={() => setSelectedEvalChart("importance")}>
                Feature Insights
            </button>

            {trainResult.task_type === "classification" && (
                <button className={`btn btn-secondary ${selectedEvalChart === "confusionMatrix" ? "active" : ""}`}
                 onClick={() => setSelectedEvalChart("confusionMatrix")}>
                Confusion Matrix
                </button>
            )}
            </div>

            {trainResult.task_type === "regression" &&
            renderRegressionR2Cards(results, trainResult.best_model_key)}

            <div className="ml-chart-panel">
                {selectedEvalChart === "metrics" && renderMetricsChart(results, trainResult.task_type)}

                {selectedEvalChart === "predictedActual" && renderPredictedActualChart(activeModel)}

                {selectedEvalChart === "errorPct" && renderErrorHistogram(activeModel)}

                {selectedEvalChart === "confusionMatrix" && renderConfusionMatrixChart(activeModel)}

                {selectedEvalChart === "importance" && renderFeatureChart(activeModel)}
            </div>
        </div>

        <div className="ml-eval-metrics-panel">
            <h4>Selected Model Metrics</h4>

            {trainResult.task_type === "regression" ? (
            <div className="ml-metrics-grid">
                <div className="ml-metric-box"> {/* edited: keep existing metric card container */}
                <span className="ml-metric-label-with-help"> {/* edited: wrap metric label with tooltip trigger */}
                    R² {/* edited: metric label */}
                    <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                        <span className="ml-metric-tooltip">R² shows how well the model explains the variation in the target variable. Higher is generally better.</span> {/* edited: add tooltip description */}
                    </span>
                </span>
                <strong>
                    {activeModel?.metrics?.r2 != null ? Number(activeModel.metrics.r2).toFixed(3) : "-"}
                </strong>
                </div>

                <div className="ml-metric-box"> {/* edited: keep existing metric card container */}
                <span className="ml-metric-label-with-help"> {/* edited: wrap metric label with tooltip trigger */}
                    MAE {/* edited: metric label */}
                    <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                        <span className="ml-metric-tooltip">MAE is the average absolute difference between predicted and actual values. Lower is better.</span> {/* edited: add tooltip description */}
                    </span>
                </span>
                <strong>
                    {activeModel?.metrics?.mae != null ? Number(activeModel.metrics.mae).toFixed(3) : "-"}
                </strong>
                </div>

                <div className="ml-metric-box"> {/* edited: keep existing metric card container */}
                <span className="ml-metric-label-with-help"> {/* edited: wrap metric label with tooltip trigger */}
                    RMSE {/* edited: metric label */}
                    <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                        <span className="ml-metric-tooltip">RMSE measures prediction error and gives more weight to larger mistakes. Lower is better.</span> {/* edited: add tooltip description */}
                    </span>
                </span>
                <strong>
                    {activeModel?.metrics?.rmse != null ? Number(activeModel.metrics.rmse).toFixed(3) : "-"}
                </strong>
                </div>
            </div>
            ) : (
            <div className="ml-metrics-grid">
                <div className="ml-metric-box"> {/* edited: keep existing metric card container */}
                <span className="ml-metric-label-with-help"> {/* edited: wrap metric label with tooltip trigger */}
                    Accuracy {/* edited: metric label */}
                    <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                        <span className="ml-metric-tooltip">Accuracy is the proportion of all predictions that are correct.</span> {/* edited: add tooltip description */}
                    </span>
                </span>
                <strong>
                    {activeModel?.metrics?.accuracy != null ? Number(activeModel.metrics.accuracy).toFixed(3) : "-"}
                </strong>
                </div>

                <div className="ml-metric-box"> {/* edited: keep existing metric card container */}
                <span className="ml-metric-label-with-help"> {/* edited: wrap metric label with tooltip trigger */}
                    Precision {/* edited: metric label */}
                    <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                        <span className="ml-metric-tooltip">Precision shows, out of all predicted positive cases, how many were actually positive.</span> {/* edited: add tooltip description */}
                    </span>
                </span>
                <strong>
                    {activeModel?.metrics?.precision != null ? Number(activeModel.metrics.precision).toFixed(3) : "-"}
                </strong>
                </div>

                <div className="ml-metric-box"> {/* edited: keep existing metric card container */}
                <span className="ml-metric-label-with-help"> {/* edited: wrap metric label with tooltip trigger */}
                    Recall {/* edited: metric label */}
                    <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                        <span className="ml-metric-tooltip">Recall shows, out of all actual positive cases, how many the model correctly identified.</span> {/* edited: add tooltip description */}
                    </span>
                </span>
                <strong>
                    {activeModel?.metrics?.recall != null ? Number(activeModel.metrics.recall).toFixed(3) : "-"}
                </strong>
                </div>

                <div className="ml-metric-box"> {/* edited: keep existing metric card container */}
                <span className="ml-metric-label-with-help"> {/* edited: wrap metric label with tooltip trigger */}
                    F1 {/* edited: metric label */}
                    <span className="ml-metric-help">ⓘ {/* edited: add hover help icon */}
                        <span className="ml-metric-tooltip">F1 is a balanced score that combines precision and recall into one metric.</span> {/* edited: add tooltip description */}
                    </span>
                </span>
                <strong>
                    {activeModel?.metrics?.f1 != null ? Number(activeModel.metrics.f1).toFixed(3) : "-"}
                </strong>
                </div>
            </div>
            )}
        </div>


        <div className="ml-save-panel">
        <h4>Save / Export</h4> 
        <p>Download the selected trained model as a pickle file for later reuse.</p> 

        <div className="form-group" style={{ marginTop: "10px" }}> 
            <label className="small">Model to Save</label> 
            <br /> 
            <select
            value={selectedEvalModel || trainResult?.best_model_key || ""} // bind dropdown to current selected evaluation model
            onChange={(e) => setSelectedEvalModel(e.target.value)} 
            >
            {results.map((item) => ( 
                <option key={item.model_key} value={item.model_key}> 
                {formatModelLabel(item.model_key)} 
                </option>
            ))}
            </select>
        </div>

        <div style={{ marginTop: "12px", display: "flex", gap: "8px", flexWrap: "wrap" }}> 
            <button
            className="btn btn-primary"
            onClick={handleExportModel} // trigger backend pickle export
            disabled={exportingModel || !results.length} 
            >
            {exportingModel ? "Saving..." : "Save as Pickle (.pkl)"} 
            </button>
        </div>

        <p className="text-muted" style={{ marginTop: "10px" }}>
            The exported file includes the fitted preprocessing pipeline and trained estimator for the selected model.
        </p>
        </div>

    </div>
    );
    }

    return <p className="text-muted">No step details available.</p>;
    };


// function getDefaultModelsForTask(task) {
//   return task === "regression"
//     ? ["linear_regression", "random_forest_regressor"]
//     : ["logistic_regression", "random_forest"];
// };

    return (
    <div className="view-container">
        <h2 className="view-title">Transparent Data Cleansing & Pre-Processing Pipeline</h2>

        {!datasetProfile ? (
        <div className="card">
            <p className="text-muted">
            Upload a dataset first before using the ML modeling pipeline.
            </p>
        </div>
        ) : (
        <div>
            {/* Task Type Selector */}
            <div className="card" style={{ marginBottom: "16px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
                <label className="small" style={{ fontWeight: 600 }}>Task Type</label>
                <div style={{ display: "flex", gap: "4px", background: "#0a1628", borderRadius: "8px", padding: "4px", border: "1px solid #1e2d3d" }}>
                <button
                    className={`nav-tab ${taskType !== "timeseries" ? "active" : ""}`}
                    style={{ padding: "8px 16px", fontSize: "0.82rem" }}
                    onClick={() => { setTaskType(""); setPlan(null); setTrainResult(null); setActiveStep("Step 1 Data & Target"); }}
                >
                    Regression / Classification
                </button>
                <button
                    className={`nav-tab ${taskType === "timeseries" ? "active" : ""}`}
                    style={{ padding: "8px 16px", fontSize: "0.82rem" }}
                    onClick={() => { setTaskType("timeseries"); setPlan(null); setTrainResult(null); setActiveStep("Step 1 Data & Target"); }}
                >
                    Time-Series Forecast
                </button>
                </div>

            </div>
            </div>

        <div className="row row-gap">
            {/* REVISED: left card is now the main navigation + summary */}
            <div className="card flex-1">
            <h4>Pipeline Flow</h4>
            <p className="text-muted small">
                Click a step to configure or review that part of the ML workflow.
            </p>

            <div className="pipeline-diagram">
                {pipelineSteps.map(step => (
                <div
                    key={step.key}
                    onClick={() => setActiveStep(step.key)}
                    style={{
                    border: activeStep === step.key ? "2px solid #05d9fa" : "1px solid #666",
                    background: activeStep === step.key ? "#1d2a35" : "#9c9a9a",
                    color: activeStep === step.key ? "#fff" : "#111",
                    padding: "12px",
                    marginBottom: "12px",
                    borderRadius: "8px",
                    cursor: "pointer"
                    }}
                >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                    <strong>{step.title}</strong>
                    <span className="small">{step.status}</span>
                    </div>
                    <div style={{ fontSize: "0.95rem", marginBottom: "8px" }}>{step.subtitle}</div>
                    <ul style={{ margin: 0, paddingLeft: "18px", fontSize: "0.85rem" }}>
                    {step.summary.map((item, idx) => <li key={idx}>{item}</li>)}
                    </ul>
                </div>
                ))}
            </div>

            <button
                className="btn btn-ghost"
                onClick={handleRestartPipeline}
                style={{ marginTop: "12px", width: "100%" }}
            >
                Restart Pipeline
            </button>

            {mlError && (
                <p className="error-text small" style={{ marginTop: "10px" }}>
                Error: {mlError}
                </p>
            )}
            </div>

            {/* REVISED: right card becomes the working panel */}
            <div className="card flex-1 step-details-card">
            <h4>Step Details</h4>
            {renderStepDetails()}
            </div>
        </div>
        </div>
        )}
    </div>
    );
}