import React from 'react';

export default function ChatSidebar({ onNavigate, onChartGenerated }) {

    const [messages, setMessages] = React.useState([
    { role: "bot", text: "What would you like to ask?" }
    ]);
    const [inputText, setInputText] = React.useState("");
    const [loading, setLoading] = React.useState(false);

    async function sendMessage(event) {
    event.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || loading) return;

    setMessages(prev => [
        ...prev, { role: "user", text: trimmed },
        { role: "bot", text: "" }
    ]);
    setInputText("");
    setLoading(true);

    try {
        const response = await fetch("/api/chat-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed }),
        });

        if (!response.ok || !response.body) {
        const fallback = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: trimmed }),
        });
        const data = await fallback.json();
        setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
            role: "bot",
            text: data.reply || data.error || "Error contacting AI.",
            chartData: data.chart || null,
            };
            return updated;
        });
        return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");

        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.startsWith("data:")) continue;

            const rawChunk = line.slice(5);
            const chunk = rawChunk;

            // Check for chart marker
            const chartMatch = chunk.match(/\[CHART\](.*?)\[\/CHART\]/s);
            if (chartMatch) {
                try {
                    const chartData = JSON.parse(chartMatch[1]);
                    setMessages(prev => {
                        const updated = [...prev];
                        const lastIndex = updated.length - 1;
                        if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
                            updated[lastIndex] = {
                                ...updated[lastIndex],
                                chartData: chartData,
                            };
                        }
                        return updated;
                    });
                } catch (e) {
                    console.error("Failed to parse chart data:", e);
                }
                continue;
            }

            // Check for usage/token marker
            const usageMatch = chunk.match(/\[USAGE\](.*?)\[\/USAGE\]/s);
            if (usageMatch) {
                try {
                    const usage = JSON.parse(usageMatch[1]);
                    setMessages(prev => {
                        const updated = [...prev];
                        const lastIndex = updated.length - 1;
                        if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
                            updated[lastIndex] = {
                                ...updated[lastIndex],
                                usage: usage,
                            };
                        }
                        return updated;
                    });
                } catch (e) {
                    console.error("Failed to parse usage data:", e);
                }
                continue;
            }

            const textChunk = chunk
            .replace(/\\n/g, "\n")
            .replace(/\\r/g, "\r")
            .replace(/\\t/g, "  ");

            if (!textChunk) continue;

            setMessages(prev => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            if (lastIndex < 0 || updated[lastIndex].role !== "bot") {
                return updated;
            }

            const currentText = updated[lastIndex].text || "";

            // Skip <chart>...</chart> blocks during streaming
            if (updated[lastIndex]._inChartBlock) {
                if (/<\s*\/\s*chart\s*>/.test(textChunk)) {
                    updated[lastIndex] = { ...updated[lastIndex], _inChartBlock: false };
                }
                return updated;
            }
            if (/<\s*chart[\s>]/.test(textChunk) && !/<\s*\/\s*chart\s*>/.test(textChunk)) {
                updated[lastIndex] = { ...updated[lastIndex], _inChartBlock: true };
                return updated;
            }
            // Skip chunks that contain both opening and closing chart tags
            if (/<\s*chart\s*>/.test(textChunk) && /<\s*\/\s*chart\s*>/.test(textChunk)) {
                return updated;
            }

            // Simply concatenate - the LLM tokenizer includes leading spaces in tokens
            let newText = currentText + textChunk;

            updated[lastIndex] = {
                ...updated[lastIndex],
                text: newText,
            };
            return updated;
            });
        }
        }

        // Process any remaining buffer content (e.g. the last [USAGE] event)
        if (buffer.trim()) {
            const leftover = buffer.trim();
            if (leftover.startsWith("data:")) {
                const chunk = leftover.slice(5).trimStart();

                const chartMatch = chunk.match(/\[CHART\](.*?)\[\/CHART\]/s);
                if (chartMatch) {
                    try {
                        const chartData = JSON.parse(chartMatch[1]);
                        setMessages(prev => {
                            const updated = [...prev];
                            const lastIndex = updated.length - 1;
                            if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
                                updated[lastIndex] = { ...updated[lastIndex], chartData };
                            }
                            return updated;
                        });
                    } catch (e) {
                        console.error("Failed to parse chart data:", e);
                    }
                }

                const usageMatch = chunk.match(/\[USAGE\](.*?)\[\/USAGE\]/s);
                if (usageMatch) {
                    try {
                        const usage = JSON.parse(usageMatch[1]);
                        setMessages(prev => {
                            const updated = [...prev];
                            const lastIndex = updated.length - 1;
                            if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
                                updated[lastIndex] = { ...updated[lastIndex], usage };
                            }
                            return updated;
                        });
                    } catch (e) {
                        console.error("Failed to parse usage data:", e);
                    }
                }
            }
        }

    } catch (err) {
        console.error("chat-stream error:", err);
        setMessages(prev => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        if (lastIndex >= 0 && updated[lastIndex].role === "bot") {
            updated[lastIndex] = {
            ...updated[lastIndex],
            text: "Connection error while streaming response.",
            };
        }
        return updated;
        });
    } finally {
        setMessages(prev => normalizeFinalBotMessage(prev));
        setLoading(false);
    }
    }

    function normalizeFinalBotMessage(messages) {
    const updated = [...messages];
    const lastIndex = updated.length - 1;
    if (lastIndex < 0 || updated[lastIndex].role !== "bot") {
        return updated;
    }

    let text = updated[lastIndex].text || "";

    // Remove chart markers from displayed text
    text = text.replace(/\[CHART\][^\[]*\[\/CHART\]/g, "").trim();
    // Remove usage markers
    text = text.replace(/\[USAGE\][^\[]*\[\/USAGE\]/g, "").trim();
    // Remove <chart>...</chart> tags (with optional spaces inside tags)
    text = text.replace(/<\s*chart\s*>.*?<\s*\/\s*chart\s*>/gs, "").trim();
    // Remove markdown code blocks (```...```)
    text = text.replace(/```[\s\S]*?```/g, "").trim();
    // Remove trailing chart/visualization description sentences (safety net)
    text = text.replace(
        /\s*(This|The|A|An)\s+(bar|line|scatter|pie|gauge|chart|graph|plot|visualization|visualisation)\s*(chart|graph|plot)?\s*(that\s+)?(shows|displays|illustrates|represents|depicts|demonstrates|highlights|reveals|indicates|summarizes|summarises|compares|breaks\s+down|illustrates).*/gs,
        ""
    ).trim();

    text = text
        .replace(/\\n/g, "\n")
        .replace(/\\t/g, "  ");

    // Fix spaces around underscores (column names like "sales_revenue_usd")
    text = text.replace(/\s+_/g, "_");   // remove space before underscore
    text = text.replace(/_\s+/g, "_");   // remove space after underscore

    text = text
        .replace(/\s+([,.;:!?])/g, "$1")
        .replace(/\$\s+(\d)/g, "$$$1")
        .replace(/\s+%/g, "%")
        .replace(/\b([A-Za-z]+)\s+'s\b/g, "$1's")
        .replace(/(\d),\s+(\d{3})/g, "$1,$2")
        .replace(/(\d)\.\s+(\d)/g, "$1.$2")
        .replace(/(\*\*[^*]+?\*\*):?/g, "$1:\n");

    text = text.replace(
        /\s*Follow\s*-up questions:\s*/i,
        "\n\nFollow-up questions:\n"
    );

    const parts = text.split(/Follow-up questions:\n/i);
    if (parts.length === 2) {
        const header = "Follow-up questions:\n";
        let tail = parts[1];
        tail = tail.replace(/\s*\u2022\s+/g, "\n\u2022 ");
        text = parts[0] + header + tail;
    }

    updated[lastIndex] = { ...updated[lastIndex], text, _inChartBlock: false };
    return updated;
    }



    return (
    <div className="chat-sidebar card">
        <h3 className="card-title">AI Data Assistant Chatbot</h3>
        <div className="chat-box">
        {messages.map((msg, index) => (
            <div key={index} className={`chat-message-row ${msg.role === "user" ? "align-right" : "align-left"}`}>
            <div className={`chat-bubble ${
            msg.role === "user" ? "chat-bubble-user" : "chat-bubble-bot"
            }`}>
                <strong>{msg.role === "user" ? "You" : "AI"}:</strong>
                {msg.role === "bot" && msg.usage && (
                    <span style={{ fontWeight: "normal", fontSize: "0.7rem", color: "#999", marginLeft: "6px" }}>
                        input: {msg.usage.input_tokens} tokens | output: {msg.usage.output_tokens} tokens
                    </span>
                )}
                <br/>
                {msg.text}
                {msg.chartData && (
                    <div style={{ marginTop: "8px" }}>
                        <button
                            className="btn btn-primary"
                            style={{ fontSize: "0.75rem", padding: "4px 10px" }}
                            onClick={() => onChartGenerated && onChartGenerated(msg.chartData)}
                        >
                            View Chart in Dashboard
                        </button>
                    </div>
                )}
            </div>
            </div>
        ))}
        {loading && <div className="chat-typing"><em>AI chatbot is typing...</em></div>}
        </div>
        
        <form onSubmit={sendMessage} className="chat-input-row">
        <input 
            type="text" 
            placeholder="Ask a question..." 
            className="chat-input" 
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={loading}
        />
        <button type="submit" disabled={loading} className="btn btn-primary">Send</button>
        </form>
    </div>
    );

}
