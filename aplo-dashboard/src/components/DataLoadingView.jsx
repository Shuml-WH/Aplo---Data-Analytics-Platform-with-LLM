import React from 'react';

export default function DataLoadingView({ datasetProfile, setDatasetProfile }) {

    // Upload UI and errors
    const [uploadError, setUploadError] = React.useState("");
    const [isUploading, setIsUploading] = React.useState(false);
    const fileInputRef = React.useRef(null);

    // View toggles
    const [showJsonDebug, setShowJsonDebug] = React.useState(false);
    const [showInfoRaw, setShowInfoRaw] = React.useState(false);

    // Audit trails
    const [auditEvents, setAuditEvents] = React.useState([]);
    const [auditError, setAuditError] = React.useState(""); 

    // Auto-clean strategies
    const [numericStrategy, setNumericStrategy] = React.useState("median");
    const [categoricalStrategy, setCategoricalStrategy] = React.useState("mode");
    const [isAutoCleaning, setIsAutoCleaning] = React.useState(false);
    const [autoCleanSummary, setAutoCleanSummary] = React.useState(null);
    const [canUndoAutoClean, setCanUndoAutoClean] = React.useState(false);


    React.useEffect(() => {
    console.log("[state] datasetProfile changed:", datasetProfile);
    }, [datasetProfile]);


    React.useEffect(() => {
    async function fetchExistingProfile() {
        try {
        console.log("[init] fetching existing dataset profile");
        const res = await fetch("/api/dataset-profile");
        
        if (!res.ok) {
            // 404 means "no active dataset", which is fine for a fresh start
            console.warn("[init] no active dataset or error status:", res.status);
            return;
        }

        const data = await res.json();
        if (data.success) {
            console.log("[init] loaded existing datasetProfile:", data);
            setDatasetProfile(data);
            setUploadError("");
        } else {
            console.warn("[init] dataset-profile returned success=false:", data.error);
        }
        } catch (err) {
        console.error("[init] failed to fetch dataset-profile:", err);
        }
    }

    async function fetchAuditTrail() {
        try {
            const res = await fetch("/api/audit-trail");
            if (!res.ok) {
            console.warn("[audit] status:", res.status);
            return;
            }
            const data = await res.json();
            if (data.success && Array.isArray(data.events)) {
            setAuditEvents(data.events);
            setAuditError("");
            } else {
            setAuditError(data.error || "Failed to load audit trail.");
            }
        } catch (err) {
            console.error("[audit] failed to fetch:", err);
            setAuditError("Network error while loading audit trail.");
        }
        }

    fetchExistingProfile();
    fetchAuditTrail();

}, []);



    // use fetch() API to communicate with Flask
    const handleFileUpload = async (event) => {
    const file = event.target.files[0];  // event.target.files is a FileList of chosen files, files[0] is the first selected file
    console.log("[upload] handleFileUpload fired, file:", file);  // Debug log to confirm file selection and event firing

    if (!file) {
        console.log("[upload] no file found on input");  // Debug log to check if file is missing
        return;
    }

    setUploadError("");
    setIsUploading(true);

    const inputEl = event.target;
    const formData = new FormData();
    formData.append("file", file);   // "file" should match the key expected by Flask's request.files.get("file")

    try {
        console.log("[upload] sending fetch to /api/upload-csv");
        const response = await fetch("/api/upload-csv", {
            method: "POST",
            body: formData
        });

        console.log("[upload] response status:", response.status);

        let data;
        try {
            data = await response.json();
        } catch (e) {
            console.error("[upload] response.json() failed:", e);
            // throw new Error(`Server returned non-JSON response (status ${response.status})`);
        }

        console.log("[upload] parsed JSON:", data);

        if (!response.ok || !data.success) {
            throw new Error(data.error || `Upload failed with status ${response.status}`);
        }

        console.log("[upload] setting datasetProfile now");
        setDatasetProfile(data);
        
        try {
            const resAudit = await fetch("/api/audit-trail");
            const auditData = await resAudit.json();
            if (resAudit.ok && auditData.success && Array.isArray(auditData.events)) {
            setAuditEvents(auditData.events);
            setAuditError("");
            }
        } catch (e) {
            console.warn("[audit] refresh failed after upload:", e);
        }


        setUploadError("");
        alert("CSV uploaded and processed by backend successfully!");
        } 
        catch (error) {
        console.error("[upload] error object:", error);
        console.error("[upload] error name:", error.name);
        console.error("[upload] error message:", error.message);

        setUploadError(error.message || String(error));
        alert("Upload failed: " + (error.message || String(error)));
        } 
        finally {
        setIsUploading(false);
        inputEl.value = "";
        }
    };


    const handleResetDataset = async () => {
    if (!window.confirm("Reset current dataset and clear the view?")) {
        return;
    }

    try {
        const res = await fetch("/api/reset-dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
        });

        const data = await res.json();
        if (res.ok && data.success) {
        console.log("[reset] success:", data.message);
        setDatasetProfile(null);
        setUploadError("");
        alert("Dataset has been reset.");
        } else {
        console.error("[reset] failed:", data);
        alert("Failed to reset dataset: " + (data.error || "Unknown error"));
        }
    } catch (err) {
        console.error("[reset] network error:", err);
        alert("Failed to reset dataset (network error).");
    }
    };

    
    const dataHealth = React.useMemo(() => {
    if (!datasetProfile?.shape || !datasetProfile?.null_counts) return null;

    const totalCells =
        datasetProfile.shape.rows * datasetProfile.shape.columns;
    const missingCells = Object.values(datasetProfile.null_counts).reduce(
        (sum, v) => sum + Number(v || 0),
        0
    );

    const missingPct = totalCells > 0 ? (missingCells / totalCells) * 100 : 0;
    // 99.9% completeness -> 99, 100% -> 100, 0% -> 0
    const score = Math.max(0, Math.floor(100 - missingPct));

    // display value with a minimum of 0.01% when there is any missing
    const displayedMissingPct = missingPct > 0 && missingPct < 0.01 ? 0.01 : missingPct;


    let worstColumn = null;
    let worstMissing = 0;
    const columnsWithMissing = [];

    // FIXED: use countRaw, then derive count
    for (const [col, countRaw] of Object.entries(
        datasetProfile.null_counts
    )) {
        const count = Number(countRaw || 0);
        if (count > 0) {
        columnsWithMissing.push({ name: col, count });
        if (count > worstMissing) {
            worstMissing = count;
            worstColumn = col;
        }
        }
    }

    columnsWithMissing.sort((a, b) => b.count - a.count);

    return {
        score,
        missingPct,
        displayedMissingPct,
        worstColumn,
        worstMissing,
        columnsWithMissing,
    };
    }, [datasetProfile]);


    const handleAutoClean = async () => {
    if (!datasetProfile) return;

    if (!window.confirm("Apply auto-clean to the current dataset? This will modify the data in memory.")) {
        return;
    }

    setIsAutoCleaning(true);
    try {
        const res = await fetch("/api/auto-clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            numeric_strategy: numericStrategy,
            categorical_strategy: categoricalStrategy,
        }),
        });

        const data = await res.json();
        if (!res.ok || !data.success) {
        throw new Error(data.error || `Auto-clean failed with status ${res.status}`);
        }

        // Update dataset profile with cleaned data
        setDatasetProfile(data);

        // capture short summary if present
        if (data.auto_clean_summary) {
        setAutoCleanSummary(data.auto_clean_summary);
        setCanUndoAutoClean(true);
        }

        // Refresh audit trail
        try {
        const resAudit = await fetch("/api/audit-trail");
        const auditData = await resAudit.json();
        if (resAudit.ok && auditData.success && Array.isArray(auditData.events)) {
            setAuditEvents(auditData.events);
            setAuditError("");
        }
        } catch (err) {
        console.warn("[audit] refresh failed after auto-clean:", err);
        }

        alert("Auto-clean completed and dataset profile updated.");
    } catch (err) {
        console.error("[auto-clean] error:", err);
        alert("Auto-clean failed: " + (err.message || String(err)));
    } finally {
        setIsAutoCleaning(false);
    }
    };


    const [isUndoing, setIsUndoing] = React.useState(false);
    {/* call Flask API for undoing the last auto-clean action */}
    const handleUndoAutoClean = async () => {
        if (!canUndoAutoClean) return;

        if (!window.confirm("Undo the last Auto-Clean and restore the previous dataset state?")) {
        return;
        }

        setIsUndoing(true);
        try {
        const res = await fetch("/api/undo-auto-clean", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });

        const data = await res.json();
        console.log("[undo-auto-clean] response:", res.status, data);

        if (!res.ok || data.success === false) {
            throw new Error(data.error || `Undo failed with status ${res.status}`);
        }

        setDatasetProfile(data);
        setCanUndoAutoClean(false);  // single-level undo

        // Optionally clear last auto-clean summary or show undo label
        setAutoCleanSummary(null);

        // Refresh audit trail
        try {
            const resAudit = await fetch("/api/audit-trail");
            const auditData = await resAudit.json();
            if (resAudit.ok && auditData.success && Array.isArray(auditData.events)) {
            setAuditEvents(auditData.events);
            setAuditError("");
            }
        } catch (err) {
            console.warn("[audit] refresh failed after undo:", err);
        }

        alert("Undo completed. Dataset restored to previous state.");
        } catch (err) {
        console.error("[undo-auto-clean] error:", err);
        alert("Undo failed: " + (err.message || String(err)));
        } finally {
        setIsUndoing(false);
        }
    };




    return (
    <div className="view-container">
        <h2 className="view-title">Data Loading & Preview</h2>
        
        <div className="row row-gap">
        <div className="card flex-1 data-source-card">
            <h4>Data Source</h4>
            <div className="inline-controls">
            <input
                type="file"
                id="csvUpload"
                accept=".csv"
                ref={fileInputRef}            
                onChange={handleFileUpload}
                style={{ display: "none" }}
            />
            <button
                className="btn btn-primary"
                onClick={() => {
                if (fileInputRef.current) {
                    fileInputRef.current.click();   
                }
                }}
            >
                {isUploading ? "Uploading..." : "Upload CSV"}
            </button>
            <button className="btn btn-ghost" disabled>Connect API</button>

            {/* Dataset reset button */}
            <button
                className="btn btn-ghost"
                type="button"
                onClick={handleResetDataset}
            >
                Reset Dataset
            </button>

            </div>
            
            {/* dynamic data shape from Flask response */}
            <p className="text-muted small">
            Current Dataset: {
                datasetProfile?.shape
                ? `${datasetProfile.filename} (${datasetProfile.shape.rows} rows, ${datasetProfile.shape.columns} columns)`
                : "No dataset uploaded yet"
            }
            </p>

            {/* Toggle button between table and JSON */}
            <div style={{ marginBottom: "8px", display: "flex", gap: "8px", alignItems: "center" }}>

            {datasetProfile && (
                <span className="text-muted small">
                View mode: {showJsonDebug ? "Raw JSON" : "Table (first 50 records)"}
                </span>
            )}
            <button
                type="button"
                className="btn btn-ghost"
                disabled={!datasetProfile}
                onClick={() => setShowJsonDebug(!showJsonDebug)}
            >
                {showJsonDebug ? "Show Table Preview" : "Show Raw JSON Debug"}
            </button>
            </div>

            {/* Conditional rendering: table vs JSON debug */}
            {!datasetProfile ? (
            <p className="text-muted small">Upload a CSV file to see a preview here.</p>
            ) : showJsonDebug ? (
            <pre className="debug-pre" style={{ maxHeight: "300px", overflowY: "auto" }}>
                {JSON.stringify(datasetProfile, null, 2)}
            </pre>
            ) : (
            datasetProfile.head && Array.isArray(datasetProfile.head) && datasetProfile.head.length > 0 ? (
                <div style={{ maxHeight: "300px", overflowY: "auto", border: "1px solid #ddd", borderRadius: "4px" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
                    <thead style={{ backgroundColor: "#9c9a9a" }}>
                    <tr>
                        {datasetProfile.head.length > 0 && Object.keys(datasetProfile.head[0]).map((key) => (
                        <th
                            key={key}
                            style={{
                            padding: "6px",
                            borderBottom: "1px solid #ddd",
                            textAlign: "left",
                            position: "sticky",
                            top: 0,
                            backgroundColor: "#9c9a9a"
                            }}
                        >
                            {key}
                        </th>
                        ))}
                    </tr>
                    </thead>
                    <tbody>
                    {datasetProfile.head.slice(0, 50).map((row, rowIndex) => (
                        <tr key={rowIndex}>

                        {Object.values(row).map((value, colIndex) => (
                            <td key={colIndex} style={{ padding: "6px", borderBottom: "1px solid #eee" }}>
                            {value !== null && value !== undefined ? String(value) : "NaN"}
                            </td>
                        ))}

                        </tr>
                    ))}
                    </tbody>
                </table>
                </div>
            ) : (
                <p className="text-muted small">No preview rows available.</p>
            )
            )}
            
            {/* Displays error message on the UI if the Flask upload fails */}
            {uploadError && <p className="error-text small">Error: {uploadError}</p>}
        </div>

        <div className="card flex-1 data-health-card">
            <h4>Data Health Score</h4>
            <h2 className="health-score">{dataHealth ? `${dataHealth.score}/100` : "--/100"}</h2>

            {!dataHealth ? (
            <p className="text-muted">
                Upload a dataset to calculate health score.
            </p>
            ) : dataHealth.columnsWithMissing.length === 0 ? (
            <p className="text-muted">
                No missing values detected in any column.
            </p>
            ) : (
            <>
                <p className="text-muted small">
                Total missing cells:{" "}
                {dataHealth.columnsWithMissing.reduce(
                    (sum, c) => sum + c.count,
                    0
                )}{" "}
                ({dataHealth.displayedMissingPct.toFixed(2)}% of all cells)
                </p>
                <p className="text-muted small">
                Columns with missing values:
                </p>
                <ul className="text-muted small" style={{ paddingLeft: "16px", margin: 0 }}>
                {dataHealth.columnsWithMissing.map((c) => (
                    <li key={c.name}>
                    {c.name}: {c.count} missing
                    </li>
                ))}
                </ul>
            </>
            )}

        </div>
        </div>


        {/* lower half (audit + pre-processing + auto-clean) */}
        <div className="row row-gap">
        {/* Left: tall Audit Trail card */}
        <div className="card audit-card">
            <h4>Data Audit Trail</h4>

            {/* Audit Trail Dynamic Rendering */}
            <div>
                {auditError && (
                <p className="error-text small">{auditError}</p>
                )}

                {auditEvents.length === 0 ? (
                <p className="text-muted small">
                    No audit events yet. Upload a dataset or apply transformations to see the trail.
                </p>
                ) : (
                <ul className="audit-list">
                        {auditEvents.map((evt) => (
                        <li key={evt.id}>
                            <span className="small">
                            <strong>{evt.action}</strong>
                            {evt.strategy && (
                                <> &nbsp; <em>({evt.strategy})</em></>
                            )}
                            {" — "}{evt.timestamp}
                            </span>
                            {evt.details && (
                            <div className="small text-muted">
                                {evt.details}
                            </div>
                            )}
                        </li>
                        ))}
                    </ul>
                )}
            </div>

        </div>

        {/* Right: stack pre-processing + auto-clean */}
        <div className="audit-right-column flex-1">
            {/* Interactive Column Pre-processing (existing card) */}
            <div className="card">
            <div className="card-header">
                <h4>Data Pre-processing</h4>
            </div>
            {/* existing pre-processing content here */}
            {!datasetProfile?.columns ? (
                <div style={{ textAlign: "center", padding: "40px", color: "#666" }}>
                <p>Upload a CSV file to view data profile.</p>
                </div>
            ) : (
                <div className="profiling-scroll">
                {/* Columns Overview */}
                <h5 className="profiling-section-title">Columns Overview</h5>
                <div className="columns-overview-tags">
                    {datasetProfile.columns.map((col, idx) => (
                    <span key={idx} className="column-tag">
                        <strong>{col.name}</strong> <em>({col.dtype})</em>
                    </span>
                    ))}
                </div>

                {/* DataFrame Info */}
                <h5 className="profiling-section-title">DataFrame Info</h5>

                {!datasetProfile.columns_profile || datasetProfile.columns_profile.length === 0 ? (
                    <p className="text-muted small">No column profile available.</p>
                ) : (
                    <div className="dataframe-info-wrapper">
                    <table className="dataframe-info-table">
                        <thead className="dataframe-info-thead">
                        <tr>
                            {[
                            "Feature",
                            "No. of non-null record",
                            "Data type",
                            "Min",
                            "25% Quartile",
                            "50% Quartile",
                            "75% Quartile",
                            "Max",
                            "Most Frequent Occurred Value",
                            ].map((label) => {
                                const isLeft = label === "Feature" || label === "Data type";

                                let extraClass = "";
                                if (label === "Feature") {
                                extraClass = "dataframe-info-th--feature";
                                } else if (label === "Most Frequent Occurred Value") {
                                extraClass = "dataframe-info-th--top";
                                }

                                return (
                                <th
                                    key={label}
                                    className={
                                    "dataframe-info-th " +
                                    (isLeft ? "dataframe-info-th-left" : "dataframe-info-th-right") +
                                    " " +
                                    extraClass
                                    }
                                >
                                    {label}
                                </th>
                                );                                
                            })}
                        </tr>
                        </thead>

                        <tbody>
                        {datasetProfile.columns_profile.map((col) => (
                            <tr key={col.feature}>
                            <td className="dataframe-info-td dataframe-info-td--feature">
                                {col.feature}
                            </td>
                            <td className="dataframe-info-td dataframe-info-td-right">
                                {col.non_null}
                            </td>
                            <td className="dataframe-info-td dataframe-info-td-right">
                                {col.dtype_label}
                            </td>
                            {["min", "q25", "q50", "q75", "max"].map((key) => {
                                const val = col[key];
                                return (
                                <td
                                    key={key}
                                    className="dataframe-info-td dataframe-info-td-right"
                                >
                                    {val === null || val === undefined
                                    ? "-"
                                    : typeof val === "number" && !Number.isInteger(val)
                                        ? val.toFixed(2)
                                        : String(val)}
                                </td>
                                );
                            })}

                                <td className="dataframe-info-td dataframe-info-td-right dataframe-info-td--top">
                                {col.top == null || col.top === ""
                                    ? "-"
                                    : String(col.top)}
                                </td>
                            </tr>
                        ))}
                        </tbody>
                        
                    </table>
                    </div>
                )}

                {/* Toggle + collapsible raw df.info() */}
                {datasetProfile.info_text && (
                    <div className="info-debug-toggle">
                    <button
                        type="button"
                        className="btn btn-ghost"
                        style={{ padding: "4px 8px", fontSize: "11px" }}
                        onClick={() => setShowInfoRaw(!showInfoRaw)}
                    >
                        {showInfoRaw ? "Hide raw df.info()" : "Show raw df.info() (debug)"}
                    </button>

                    {showInfoRaw && (
                        <pre className="debug-pre info-debug-pre">
                        {datasetProfile.info_text}
                        </pre>
                    )}
                    </div>
                )}

                </div>
            )}
            </div>

            {/* New Auto-Clean Data card, placed below pre-processing */}
            <div className="card auto-clean-card">
            <div className="card-header">
                <h4>Auto-Clean Data</h4>
                <span className="text-muted small">
                Configure how to handle missing values
                </span>
            </div>

                {autoCleanSummary && (
                <div className="text-muted small" style={{ marginTop: "4px" }}>
                    <div>
                    <strong>Last auto-clean:</strong> {autoCleanSummary.strategy}
                    </div>
                    <ul style={{ paddingLeft: "16px", margin: "4px 0 0 0" }}>
                    {autoCleanSummary.columns.map((c) => (
                        <li key={c.column}>
                        {c.column}:{" "}
                        {c.strategy.startsWith("drop_rows")
                            ? `dropped ${c.dropped_rows} rows`
                            : `filled ${c.filled_count} values using '${c.strategy}'`}
                        </li>
                    ))}
                    </ul>
                </div>
                )}

            {!datasetProfile ? (
                <p className="text-muted small">
                Upload a dataset to enable auto-clean options.
                </p>
            ) : (
                <div className="auto-clean-body">
                <div className="form-group">
                    <label className="small">Global strategy for numeric columns:</label><br />
                    <select
                    value={numericStrategy}
                    onChange={(e) => setNumericStrategy(e.target.value)}
                    >
                    <option value="median">Median (recommended)</option>
                    <option value="mean">Mean</option>
                    <option value="zero">Fill with 0</option>
                    <option value="drop">Drop rows with missing numeric values</option>
                    </select>
                </div>

                <div className="form-group">
                    <label className="small">Global strategy for categorical columns:</label><br />
                    <select
                    value={categoricalStrategy}
                    onChange={(e) => setCategoricalStrategy(e.target.value)}
                    >
                    <option value="mode">Most frequent (mode)</option>
                    <option value="constant">Fill with "Missing"</option>
                    <option value="drop">Drop rows with missing categorical values</option>
                    </select>
                </div>

                <p className="text-muted small">
                    The system will apply these strategies to columns with missing values
                    and update the dataset profile and audit trail.
                </p>

                <div style={{ marginTop: "10px", display: "flex", gap: "8px" }}>
                    <button
                    className="btn btn-primary"
                    disabled={!datasetProfile || isAutoCleaning}
                    onClick={handleAutoClean}
                    >
                    {isAutoCleaning ? "Auto-Cleaning..." : "Apply Auto-Clean"}
                    </button>
                    <button
                        className="btn btn-ghost"
                        disabled={!canUndoAutoClean || isUndoing}
                        onClick={handleUndoAutoClean}
                    >
                        {isUndoing ? "Undoing..." : "Undo Last Auto-Clean"}
                    </button>
                </div>
                </div>
            )}
            </div>
        </div>
        </div>


    </div>
    );

}