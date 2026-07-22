# User Manual: Aplo Data Analytics Platform

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Uploading a Dataset](#3-uploading-a-dataset)
4. [Dataset Pre-processing](#4-dataset-pre-processing)
5. [Data Cleaning](#5-data-cleaning)
6. [Data Visualizer](#6-data-visualizer)
7. [Machine Learning (ML)](#7-machine-learning-ml)
8. [Time-Series Forecasting](#8-time-series-forecasting)
9. [Audit Trail](#9-audit-trail)
10. [AI Chat Assistant](#10-ai-chat-assistant)
11. [Troubleshooting](#11-troubleshooting)
12. [Appendix A: Screenshot Checklist](#appendix-a-screenshot-checklist)
13. [Appendix B: Sample Dataset Reference](#appendix-b-sample-dataset-reference)

---

## 1. Introduction

### 1.1 What is Aplo?

Aplo is a web-based data analytics platform designed for non-technical users. It allows you to upload datasets, explore data, clean missing values, build interactive charts, train machine learning models, and forecast time-series trends — all through a user-friendly interface powered by an AI chat assistant.

### 1.2 Key Capabilities

| Feature | Description |
|---------|-------------|
| **CSV Upload** | Upload CSV files and get instant dataset pre-processing |
| **Data Cleaning** | Auto-fill or drop missing values with one click |
| **Data Visualizer** | Create bar, line, scatter, pie, and gauge charts |
| **Machine Learning** | Train classification and regression models automatically |
| **Time-Series Forecasting** | Forecast future values using ARIMAX and XGBoost models |
| **AI Chat Assistant** | Ask questions in natural language and get answers with charts |

### 1.3 Prerequisites

**Option A — Docker:**
- **Docker Desktop** installed and running (see [What is Docker?](#what-is-docker) below)
- ~3.5 GB free disk space for the Ollama model
- ~1 GB free disk space for Docker images

**Option B — Local Installation:**
- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **Ollama** (for AI chat features) with the `llama3.2` model installed

---

## 2. Getting Started

You have two ways to run the Aplo platform: **Docker** (recommended for simplicity) or **Local Installation** (for development).

### What is Docker?

Docker packages each component (backend, frontend, AI model) into isolated "containers" that run consistently on any machine. Instead of manually installing Python, Node.js, Ollama, and all dependencies, you install just one tool — Docker Desktop — and run a single command to start everything.

**Download Docker Desktop:** [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

- **Windows:** Run the installer, ensure **WSL2** is selected during setup, then restart
- **macOS:** Run the installer and follow the prompts
- **Linux:** Docker Desktop is optional — install `docker` and `docker compose` via your package manager, e.g. `sudo apt install docker.io docker-compose-v2`

After installation, open Docker Desktop and wait for the engine to start (the whale icon in your system tray or menu bar should be running).

### 2.1 Option A: Docker (Recommended)

**Step 1: Start the platform**

Open a terminal in the `Development_code` directory and run:

```bash
cd Development_code
docker compose up -d
```

On the first run, Docker will:
1. Build the backend image (Python 3.13)
2. Build the frontend image (Node 22 → Nginx)
3. Start the Ollama LLM server
4. Automatically pull the `llama3.2` model (~2 GB download)

This takes a few minutes on first launch.

**Step 2: Open the app**

Navigate to `http://localhost:3000` in your browser.

**Step 3: Stop the platform**

```bash
docker compose down
```

To also remove data volumes (model + uploads):

```bash
docker compose down -v
```

### 2.2 Option B: Local Installation

**Step 1: Install Python dependencies**

```bash
cd Development_code
pip install -r requirements.txt
```

**Step 2: Install Node.js dependencies**

```bash
cd Development_code/aplo-dashboard
npm install
```

**Step 3: Install and start Ollama**

Download Ollama from [https://ollama.ai](https://ollama.ai), then pull the required model:

```bash
ollama pull llama3.2
```

### 2.3 Starting the App (Local)

You need to start **two terminals** — one for the backend, one for the frontend.

**Terminal 1 — Flask Backend:**

```bash
cd Development_code
python app.py
```

The backend will start at `http://localhost:5000`.

![Terminal running Flask backend](screenshots/ch02_flask_terminal.png)

**Terminal 2 — Vite Frontend:**

```bash
cd Development_code/aplo-dashboard
npm run dev
```

The frontend will start at `http://localhost:5173`.

![Terminal running Vite frontend](screenshots/ch02_vite_terminal.png)

### 2.4 Opening the App

- **Docker method:** Navigate to `http://localhost:3000`
- **Local method:** Navigate to `http://localhost:5173`

You should see the Aplo home page.

![Aplo home page](screenshots/ch02_home_page.png)

---

## 3. Uploading a Dataset

### 3.1 Supported Formats

Aplo accepts **CSV files only**. The file should:
- Have a `.csv` extension
- Contain a header row (column names in the first row)
- Use UTF-8 or Latin-1 encoding

### 3.2 How to Upload

1. Click the **Upload** button on the navigation bar or home page.
2. Click **Choose File**. For sample dataset, choose `marketing_sales_dataset.csv`
3. Click **Upload**.

![Upload dialog on the Data Loading page](screenshots/ch03_upload_success_0.png)

### 3.3 After Upload

Once the file is uploaded, Aplo will:
- Load the dataset into memory
- Display a **dataset profile** with row count, column count, data types, and a preview of the first 5 rows
- Store the dataset for use in charts, ML, and forecasting

![Upload success showing dataset profile](screenshots/ch03_upload_success.png)

### 3.4 Example

Using the sample file `marketing_sales_dataset.csv`:
- **Rows:** 60,000
- **Columns:** 23 (id, date, region, sales_channel, product_category, customer_segment, season, marketing_budget_usd, ad_spend_online_usd, ad_spend_offline_usd, num_promotions, discount_percentage, num_sales_representatives, customer_age, customer_satisfaction_score, competitor_price_index, website_traffic, conversion_rate, email_open_rate, social_media_followers, days_since_last_purchase, num_previous_purchases, sales_revenue_usd)


## 3.5 Audit Trail

Aplo maintains a log of all actions performed on the dataset. This is useful for tracking what changes were made and in what order.

### 3.5.1 Viewing the Audit Trail

Click **Audit Trail** in the navigation bar to view the log.

Data Audit Trail example:

![Audit trail panel showing action log](screenshots/ch09_audit_trail.png)

### 3.5.2 Tracked Actions

| Action | Description |
|--------|-------------|
| Uploaded CSV | File uploaded and loaded |
| Auto-Clean Data | Missing values filled or dropped |
| Undo Auto-Clean | Reverted to previous state |
| Build Chart | Chart created |
| ML Initialize | ML target and features configured |
| ML Preprocessing Plan | Preprocessing plan generated |
| ML Train Models | Models trained |
| ML Predict | Predictions generated |
| Chat query | AI chat question asked |
---

###
## 4. Dataset Pre-processing

After uploading, Aplo automatically generates a dataset profile. You can also view it by clicking **Dataset Profile** in the navigation.

### 4.1 Data Shape

The profile shows the total number of rows and columns.

![Dataset profile panel showing shape and column info](screenshots/ch04_dataset_profile.png)

### 4.2 Column Information

For each column, Aplo displays:
- **Feature name**
- **Data type** (Numeric / Non-numeric)
- **Non-null count**
- **Min, Q25, Median, Q75, Max** (for numeric columns)
- **Top value** (most frequent value)

### 4.3 Summary Statistics

A statistics table shows descriptive statistics for all columns including mean, standard deviation, min, max, and quartiles.

![Summary statistics table](screenshots/ch04_summary_stats.png)

### 4.4 Summary Statistics Detail

The summary statistics include null value counts, quartiles, and distribution information for each column. This helps you decide on a cleaning strategy.

![Summary statistics table detail](screenshots/ch04_summary_stats_1.png)

---

## 5. Data Cleaning

Real-world datasets often contain missing values. Aplo provides an **Auto-Clean** feature to handle them quickly.

### 5.1 Auto-Clean Options

Click **Auto-Clean** to open the cleaning dialog. You can choose separate strategies for numeric and categorical columns:

| Strategy | Numeric Columns | Categorical Columns |
|----------|----------------|---------------------|
| **Fill** | Mean, Median, or Zero | Mode (most frequent) or Constant ("Missing") |
| **Drop** | Remove rows with any missing numeric values | Remove rows with any missing categorical values |

![Auto-clean dialog with strategy dropdowns](screenshots/ch05_auto_clean_dialog.png)

### 5.2 Example

Using `marketing_sales_dataset.csv`:
- Numeric strategy: **Median**
- Categorical strategy: **Mode**

After clicking **Clean**, Aplo will:
1. Fill missing numeric values with the column median
2. Fill missing categorical values with the column mode
3. Show a summary of what was cleaned


![Auto-clean action ](screenshots/ch05_auto_clean_action.png)

![Auto-clean result summary](screenshots/ch05_auto_clean_result.png)

![Auto-clean result summary 2](screenshots/ch05_auto_clean_result_2.png)

### 5.3 Undo Auto-Clean

If you want to revert the cleaning, click **Undo Auto-Clean**. This restores the dataset to its state before the last clean operation.

![Undo auto-clean button](screenshots/ch05_undo_clean.png)

![Undo auto-clean result](screenshots/ch05_undo_clean_result.png)

---

## 6. Data Visualizer

Aplo currently supports five chart types: **Bar**, **Line**, **Scatter**, **Pie**, and **Gauge**. Charts can be built manually through the Data Visualizer UI or generated by the AI Chat Assistant.

### 6.1 Data Visualizer UI

Open the Data Visualizer from the navigation bar. You will see the following options:

![Chart builder UI with chart type selector](screenshots/ch06_chart_builder_ui.png)

| Parameter | Description |
|-----------|-------------|
| **Chart Type** | Bar, Line, Scatter, Pie, or Gauge |
| **X Axis** | Column for the horizontal axis (not required for Gauge) |
| **Y Axis** | Column(s) for the vertical axis / values |
| **Aggregation** | Sum, Mean, Median, or None (raw data) |
| **Group By** | Time grouping: Daily, Weekly, Monthly, Quarterly, Yearly |
| **Date Column** | Optional date filter column |
| **Date From / To** | Optional date range filter |
| **Target** | Target max value (Gauge charts only) |

### 6.1.1 Chart Reordering

Each chart tile has arrow buttons (◀ ▶) that appear when you hover over the tile. These allow you to reorder charts on the dashboard:

- **◀ (Left arrow):** Moves the chart one position to the left
- **▶ (Right arrow):** Moves the chart one position to the right

The left arrow is disabled on the first chart, and the right arrow is disabled on the last chart. You can also change the chart size using the dropdown selector (1/3, 2/3, or Full width).

### 6.2 Bar Chart

**Example:** Total sales revenue by product category.

- Chart type: **Bar**
- Y: `sales_revenue_usd`
- X: `product_category`
- Aggregation: **sum**

![Bar chart of sales revenue by product category](screenshots/ch06_bar_chart.png)

### 6.3 Scatter Plot

**Example:** Relationship between marketing budget and sales revenue.

- Chart type: **Scatter**
- Y: `sales_revenue_usd`
- X: `marketing_budget_usd`
- Aggregation: **None** (raw data points)

![Scatter plot of marketing budget vs sales revenue](screenshots/ch06_scatter_plot.png)

### 6.3 Line Chart

**Example:** Monthly sales revenue trend over time.

- Chart type: **Line**
- Y: `sales_revenue_usd`
- X: `date`
- Aggregation: **sum**
- Group by: **Daily**
###
After Generate Chart, set:
- Width (top right corner dropdown selection): `Full`

![Line chart of monthly sales revenue](screenshots/ch06_line_chart.png)

### 6.4 Pie Chart

**Example:** Sales revenue distribution by region.

- Chart type: **Pie**
- Value: `sales_revenue_usd`
- Labels: `region`
###
After Generate Chart, set:
- Width (top right corner dropdown selection): `1/3`
- Click `◀` until the chart card is placed at the right to the scatter plot

![Pie chart of sales revenue by region](screenshots/ch06_pie_chart.png)



### 6.5 Gauge Chart

**Example:** Total sales revenue against a target of 50,000,000.

- Chart type: **Gauge**
- Metric: `sales_revenue_usd`
- Aggregation: **sum**
- Target: `Previous Period`
- Target Period: `Last Month`
- Target Aggregation: `sum`
- Factor: `Multiply 1.0`
###
After Generate Chart, set:
- Width (top right corner dropdown selection): `1/3`
- Click `◀` until the chart card is placed as the first chart


![Gauge chart with target](screenshots/ch06_gauge_chart.png)


The dashboard visualizer of the above sample demo will appear as follows:

![dashboard](screenshots/dashboard.png)

![dashboard 2](screenshots/dashboard_2.png)

![dashboard 3](screenshots/dashboard_3.png)


---

## 7. Machine Learning (ML)

Aplo provides an end-to-end machine learning pipeline. It automatically detects whether your problem is **classification** or **regression** based on the target column data type, then trains multiple models and compares their performance.

### 7.1 Classification vs Regression

| Target Column Type | Task Type | Example |
|-------------------|-----------|---------|
| Non-numeric (text, category) | Classification | Predicting `product_category` |
| Numeric (integer, float) | Regression | Predicting `sales_revenue_usd` |

### 7.2 Available Models

**Classification:**
- Logistic Regression
- Decision Tree Classifier

**Regression:**
- Linear Regression
- Ridge Regression
- Gradient Boosting Regressor
- Decision Tree Regressor
- K-Neighbors Regressor

### 7.3 Training Workflow

**Step 1: Initialize ML**

Select the **target column** (what you want to predict) and optionally choose which **feature columns** to use as inputs.

![ML initialize dialog with target and feature selection](screenshots/ch07_ml_features_selected.png)

**Example selections:**
- Target column: `sales_revenue_usd`
- Feature columns: All other columns except `id`

**Step 2: Preprocessing Plan**

Aplo analyzes the dataset and recommends preprocessing strategies:

![Preprocessing plan output](screenshots/ch07_preprocessing_plan.png)

The plan includes:
- Numeric features and recommended imputation strategy
- Categorical features and recommended encoding strategy
- Target column summary (null count, unique count, class distribution)


**Step 3: Feature Scaling**

Click **Enabling Feature Scaling** for Feature Scaling. The system will automatically apply standard score scaling to each of the data features, so that variation of values of each data features can be captured, or else if the values of a data feature is large, it makes other feature trivial and hence affect its actual significance in modeling.

![Feature plan output](screenshots/ch07_feature_scaling.png)


**Step 4: Train Models**

Click **Train** to start training. Progress is shown in real-time as each model is trained.

![Training progress bar](screenshots/ch07_training_progress.png)

![Training progress bar - in progress](screenshots/ch07_training_in_progress.png)

**Step 5: View Results**

After training completes, Aplo shows a comparison of all trained models with their evaluation metrics.

![Model results comparison table](screenshots/ch07_model_results.png)

![Model results comparison table 2](screenshots/ch07_model_results_2.png)

### 7.4 Evaluation Metrics

**Classification Metrics in the system include:**

| Metric | Description |
|--------|-------------|
| **Accuracy** | Correct predictions / total predictions |
| **Precision** | True positives / predicted positives | 
| **Recall** | True positives / actual positives |
| **F1 Score** | Harmonic mean of precision and recall | 


###
**Regression Metrics:**

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (lower is better) | 
| **RMSE** | Root Mean Squared Error (lower is better) | 
| **R²** | How well the model explains variance |




### 7.5 Predicted vs Actual (Regression)

For regression tasks, a scatter plot compares predicted values against actual values.
By clicking on the `Actual vs Predicted` button, a scatter graph will illustrate the spread of points that shows the differences between actual value and the predicted value in the dataset. Ideally, if the model is perfect, all point should align to the diagonal ldeal line. Thus, by comparing the scatter plots across different models, users will be able to tell the performance of each of the models trained.

![Feature importance chart](screenshots/ch07_feature_importance.png)

![Predicted vs actual scatter plot](screenshots/ch07_predicted_vs_actual.png)


### 7.6 Feature Importance

By clicking on the `Feature Insights` button
For tree-based and linear models, Aplo shows which features had the most influence on predictions, base on the trained model. With higher the importance / absolute coefficient value of the data feature is, the more this data feature would influence the predicted output value.

![Model results comparison table 2](screenshots/ch07_feature_importance.png)

### 7.7 Making Predictions

After training a model, navigate to the **ML Prediction** tab from the top navigation bar. The prediction page provides two modes: **Batch Prediction** (run predictions on the entire dataset) and **What-If Scenario** (predict a single outcome by adjusting feature values).



#### 7.7.1 Batch Prediction

Batch Prediction runs your trained model against every row in the dataset and displays a summary of all predicted values.

**Steps:**

1. Ensure you have trained at least one model in the **ML Modeling** tab.
2. Click **ML Prediction** in the navigation bar.
3. If a model is trained, the page shows the model name and available controls.
4. (Optional) Select a **Group By** column to break predictions into categories.
5. Click **Run Batch Prediction**.

![Run Batch Prediction button](screenshots/ch07_ml_prediction.png)


The results appear as a distribution chart showing the spread of predicted values across the dataset with summary statistic table.

**Batch prediction summary table:**

| Statistic | Description |
|-----------|-------------|
| **Count** | Total number of predictions made |
| **Mean** | Average predicted value |
| **Median** | Middle value of predictions |
| **Min / Max** | Range of predicted values |
| **Std Dev** | Spread of predictions |

![Batch prediction distribution chart](screenshots/ch07_batch_prediction_chart.png)


![Batch prediction distribution chart_2](screenshots/ch07_batch_prediction_summary_2.png)

#### 7.7.2 What-If Scenario

The What-If Scenario lets you manually enter feature values and get an instant single-row prediction. This is useful for testing how changes in input affect the outcome.

**Steps:**

1. Navigate to the **What-If Scenario** section by clicking the button on the ML Prediction page.
2. Enter values for each feature field. Required fields are marked with an asterisk (*).
3. Click **Predict**.

![What-If Scenario form with input fields](screenshots/ch07_whatif_form.png)

The prediction result appears below the form.



**Example input for prediction:**

| Feature | Value |
|---------|-------|
| `marketing_budget_usd` | 5000 |
| `ad_spend_online_usd` | 2000 |
| `ad_spend_offline_usd` | 1000 |
| `num_promotions` | 5 |
| `discount_percentage` | 15 |
| `website_traffic` | 10000 |
| `conversion_rate` | 0.10 |

The model returns a predicted `sales_revenue_usd` value based on the entered features.

![What-If prediction result](screenshots/ch07_whatif_result.png)

![What-If prediction result_2](screenshots/ch07_whatif_result_2.png)

![What-If prediction result_3](screenshots/ch07_whatif_result_3.png)

---

## 8. Time-Series Forecasting

Aplo supports time-series forecasting using two models: **ARIMAX** (statistical) and **XGBoost** (machine learning). Both automatically handle date-based features and generate forecasts with confidence intervals.

### 8.1 Configuration

Navigate back to **ML Modeling** tab, Open the Time-Series panel and in this case, configure the following:



| Parameter | Description | Example |
|-----------|-------------|---------|
| **Datetime Column** | Column containing dates | `date` |
| **Target Column** | Column to forecast | `sales_revenue_usd` |
|**Exogenous Feature Columns** | other columns that may help predict the target | `Select None` |

![Time-series config dialog](screenshots/ch08_timeseries_config.png)


| Parameter | Description | Example |
|-----------|-------------|---------|
| **Frequency** | Data granularity | `monthly` |
| **Forecast Horizon** | Number of periods to forecast | `12` |
| **Aggregation Method** | Method to combine multiple rows within each time period into one value | `Sum` |
| **Test Size** | Portion of data for testing | `0.2` (20%) |

![Time-series config dialog 2](screenshots/ch08_timeseries_config_2.png)


### 8.2 To Train the Model

Two Time-Series prediction machine learning models are currently available in Aplo, ARIMAX and XGBoost.


Click 'Train Model' to starting the training.

![Time-series Training](screenshots/ch08_timeseries_training1.png)



### 8.4 Test Metrics

After training, Aplo shows evaluation metrics on the test set:

![Test metrics (MAPE, MAE, RMSE, R²)](screenshots/ch08_test_metrics.png)

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MAPE** | Mean Absolute Percentage Error | < 10% excellent, < 20% good |
| **MAE** | Mean Absolute Error | Lower is better |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **R²** | Coefficient of determination | > 0.8 |

Click **Proceed to ML Prediction** button to navigate to prediction. 

---

### 8.4 Forecast Output

The forecast chart displays the evaluation metrics of the corresponding model and its predicted values for the upcoming time periods

![Forecast chart with train, test, and forecast](screenshots/ch08_forecast_chart.png)

In the **Forecast Periods** section, you may generate the forecast table by setting a forecast period and clicking the **Generate Forecast** button. In this example, set the forecast period to `12`, and the forecast table will show the forecast sales revenue of the next 12 months. 

This shows a prediction table with the forecast value, its lower and upper bound within the models' confidence level. 


![Forecast table](screenshots/ch08_forecast_table.png)

![Forecast table 2](screenshots/ch08_forecast_table_2.png)

---



## 10. AI Chat Assistant

The AI Chat Assistant allows you to ask questions about your dataset in natural language. It can answer analytical questions, explain data patterns, and generate charts on request.

### 10.1 Asking Questions

By default the Chatbot chat panel is on the right side of the UI.
Click the **AI Chat** button at the top right corner of the UI in case the chat panel is hidden. Type your question and press Enter.

![Chat panel with user question](screenshots/ch10_chat_question.png)

### 10.2 Example: Analytical Question

**Question:** `What is the average sales revenue?`

**Expected Response:** The AI will compute and return the average value of `sales_revenue_usd` across all rows.

![AI response with average sales revenue](screenshots/ch10_chat_response.png)

### 10.3 Example: Chart Request

**Question:** `Create a bar chart of total sales revenue by product category`

**Expected Response:** The AI will generate a bar chart showing `sales_revenue_usd` summed by `product_category`.

![AI response with embedded chart](screenshots/ch10_chat_response_chart.png)

![AI response with embedded chart 2](screenshots/ch10_chat_response_chart_2.png)

### 10.4 Example Queries

| Query Type | Example |
|-----------|---------|
| Summary | "How many rows are in the dataset?" |
| Average | "What is the average marketing budget?" |
| Total | "What is the total sales revenue by region?" |
| Maximum | "Which product category has the highest revenue?" |
| Minimum | "What is the lowest customer satisfaction score?" |
| Distribution | "How many sales are in each region?" |
| Trend | "Show me monthly sales revenue over time" |
| Chart | "Create a pie chart of revenue by sales channel" |
| ML guidance | "Which columns should I use for prediction?" |

### 10.5 Tips

- Be specific about column names when possible
- For charts, explicitly say "create a chart", "draw a graph", or "show me a plot"
- The AI understands common analytical terms: average, sum, total, count, maximum, minimum, trend, comparison

---

## 11. Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "No active dataset" | No CSV uploaded yet | Upload a CSV file first |
| "Only CSV files are allowed" | Tried to upload non-CSV file | Convert your file to CSV format |
| "Target column not found" | Selected target doesn't exist in dataset | Check column names in dataset profile |
| "Insufficient data" (Time-Series) | Less than 30 rows after filtering | Use a larger dataset or remove strict filters |
| "No trained model available" | Trying to predict before training | Train an ML model first |
| Chat not responding | Ollama not running | Start Ollama: `ollama serve` |
| "model not found" | llama3.2 not pulled | Run: `ollama pull llama3.2` |
| Port 5000 in use | Another process using the port | Kill the process or change the port in `app.py` |
| Port 5173 in use | Another Vite instance running | Kill existing Vite process or use `--port` flag |
| **Docker: Upload fails "Cannot read properties of undefined"** | Nginx `client_max_body_size` too small for large CSV | Ensure you have the latest `nginx.conf` with `client_max_body_size 50m`, then rebuild frontend |
| **Docker: Backend remains unhealthy** | Flask can't connect to Ollama | Check `docker compose logs backend` — ensure `OLLAMA_BASE_URL=http://ollama:11434` is set |
| **Docker: pull-model fails "could not connect"** | Model pull container can't reach Ollama server | Ensure `OLLAMA_HOST=http://ollama:11434` env var is set on pull-model service |
| **Docker: Port 11434 already in use** | Local Ollama conflicts with Docker | Local Ollama takes priority — remove port mapping in `docker-compose.yml` (internal networking works without it) |
| **Docker: Containers not starting** | Docker Desktop not running or WSL issue | Open Docker Desktop, ensure it's running, then try `wsl --shutdown` and retry |

### Getting Help

If you encounter issues not listed above:
1. **Docker:** Run `docker compose logs <service>` (e.g., `docker compose logs backend`) to see container logs
2. **Local:** Check the terminal output for error messages
3. Verify all dependencies are installed (`pip install ...` and `npm install`)
4. Ensure Ollama is running and the model is pulled
5. Try resetting the dataset and starting over

---

## Appendix A: Sample Dataset Reference

### marketing_sales_dataset.csv
Can be assessed from Kaggle Open Source Dataset page: https://www.kaggle.com/datasets/abdelfattahibrahim/marketing-sales-dataset?resource=download

| Column | Type | Description |
|--------|------|-------------|
| `id` | Numeric | Unique record identifier |
| `date` | Datetime | Transaction date |
| `region` | Categorical | City/region (Riyadh, Dubai, Cairo, etc.) |
| `sales_channel` | Categorical | Sales channel (Online, Retail Store, Wholesale, Direct Sales) |
| `product_category` | Categorical | Product type (Electronics, Cosmetics, Clothing, Food & Beverage, etc.) |
| `customer_segment` | Categorical | Customer type (New, Regular, Corporate) |
| `season` | Categorical | Season (Q1, Q2, Q3, Q4) |
| `marketing_budget_usd` | Numeric | Total marketing budget in USD |
| `ad_spend_online_usd` | Numeric | Online advertising spend |
| `ad_spend_offline_usd` | Numeric | Offline advertising spend |
| `num_promotions` | Numeric | Number of active promotions |
| `discount_percentage` | Numeric | Discount percentage offered |
| `num_sales_representatives` | Numeric | Number of sales reps assigned |
| `customer_age` | Numeric | Customer age |
| `customer_satisfaction_score` | Numeric | Satisfaction score (1-5) |
| `competitor_price_index` | Numeric | Competitor price comparison index |
| `website_traffic` | Numeric | Website visit count |
| `conversion_rate` | Numeric | Conversion rate (0-1) |
| `email_open_rate` | Numeric | Email open rate (0-1) |
| `social_media_followers` | Numeric | Social media follower count |
| `days_since_last_purchase` | Numeric | Days since last customer purchase |
| `num_previous_purchases` | Numeric | Total previous purchases by customer |
| `sales_revenue_usd` | Numeric | Total sales revenue in USD |

---

###
*End of User Manual*
