import React from 'react';
import ChatSidebar from './components/ChatSidebar';
import DataLoadingView from './components/DataLoadingView';
import ChartBuilderView from './components/ChartBuilderView';
import MLPipelineView from './components/MLPipelineView';
import MLPredictionView from './components/MLPredictionView';

export default function App() {
      const [activeTab, setActiveTab] = React.useState("data");
      const [datasetProfile, setDatasetProfile] = React.useState(null);
      const [generatedChart, setGeneratedChart] = React.useState(null);
      const [chatOpen, setChatOpen] = React.useState(true);
      const [predictInitialTab, setPredictInitialTab] = React.useState(null);

      const handleNavigate = (tab, subTab) => {
          if (tab === "predict" && subTab) {
              setPredictInitialTab(subTab);
          }
          setActiveTab(tab);
      };

      const handleChartGenerated = (chartData) => {
        setGeneratedChart(chartData);
        setActiveTab("charts");
      };

      return (
        <div className="app-root">
          <header className="app-header">
          <div className="header-left">
            <h1 className="app-title">Aplo Data Analytics Platform</h1>
          </div>
            
          <div className="header-center">
            <div className="nav-tabs-bar">
              <button 
                className = {`nav-tab ${activeTab === "data" ? "active" : ""}`}
                onClick={() => setActiveTab("data")}>
                {activeTab === "data" ? "▶ Data Loading & Preview" : "Data Loading & Preview"}
              </button>
              <button 
                className = {`nav-tab ${activeTab === "charts" ? "active" : ""}`}
                onClick={() => setActiveTab("charts")}>
                {activeTab === "charts" ? "▶ Dashboard Data Visualizer" : "Data Visualizer Dashboard"}
              </button>
              <button 
                className = {`nav-tab ${activeTab === "ml" ? "active" : ""}`}
                onClick={() => setActiveTab("ml")}>
                {activeTab === "ml" ? "▶ ML Modeling" : "ML Modeling"}
              </button>
              <button 
                className = {`nav-tab ${activeTab === "predict" ? "active" : ""}`} 
                onClick={() => setActiveTab("predict")}> 
                {activeTab === "predict"? "▶ ML Prediction" : "ML Prediction"}
              </button>
              
            </div>
          </div>

          <div className="header-right">
            <button
              className={`chat-toggle-btn ${chatOpen ? "active" : ""}`}
              onClick={() => setChatOpen(prev => !prev)}
              title={chatOpen ? "Hide AI Chatbot" : "Show AI Chatbot"}
            >
              {chatOpen ? "AI Chat \u2715" : "AI Chat"}
            </button>
          <div className="user-pill">User: Admin</div>
          </div>

          </header>

          <main className="dashboard-main">
          <div className="dashboard-body">
            <section className="dashboard-left">
              {activeTab === "data" && <DataLoadingView
                datasetProfile={datasetProfile}
                setDatasetProfile={setDatasetProfile}
              />}     
              {/* if activeTab == "data", render <DataLoadingView>, else do nth  */}
                
              <div style={{ display: activeTab === 'charts' ? 'block' : 'none' }}>
                <ChartBuilderView
                  columns={datasetProfile?.columns || [] } datasetProfile={datasetProfile}
                  generatedChart={generatedChart}
                  onChartConsumed={() => setGeneratedChart(null)}
                />
              </div>

              <div style={{ display: activeTab === 'ml' ? 'block' : 'none' }}>
                <MLPipelineView
                  key={datasetProfile?.filename || 'no-dataset'}
                  datasetProfile={datasetProfile}
                  onNavigate={handleNavigate}
                />
              </div>

              {activeTab === "predict" && (
                <MLPredictionView datasetProfile={datasetProfile} onNavigate={handleNavigate} initialTab={predictInitialTab} />
              )}
            </section>

            <aside className={`dashboard-right ${chatOpen ? "chat-open" : "chat-closed"}`}>
            <ChatSidebar
              onNavigate={setActiveTab}
              onChartGenerated={handleChartGenerated}
            />
            </aside>
          </div>
          </main>
        </div>
      );
}