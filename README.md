# 📊 AI-Powered Data Analyst

An AI-powered data analyst that lets you upload CSV files and interact with your data using plain English.

Ask questions, generate charts, run SQL or Pandas analysis, detect anomalies, and get clear insights—all without writing code.

Built for the **Digital Back Office AI Engineer Assignment**.

---

## ✨ Features

- 📂 Upload one or more CSV files
- 💬 Ask questions in plain English
- 📊 Generate interactive charts
- 🐼 Execute Pandas analysis
- 🗄️ Generate and run SQL queries
- 🚨 Detect anomalies in the data
- 📈 Automatic dataset summary
- 💾 Export chat history as a Markdown report
- 🔄 Supports follow-up questions using conversation history

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **LLM:** Groq API
- **Data Analysis:** Pandas
- **SQL Engine:** DuckDB
- **Visualization:** Plotly
- **Testing:** Pytest

---

## 📁 Project Structure

```
app/
├── main.py
├── config.py
├── core/
│   ├── data_manager.py
│   ├── llm_client.py
│   ├── tools.py
│   ├── sandbox.py
│   ├── sql_engine.py
│   ├── chart_engine.py
│   ├── anomaly.py
│   └── report_export.py
└── utils/
    └── logger.py
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd ai-data-analyst
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file.

```env
GROQ_API_KEY=your_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

### 5. Run the application

```bash
streamlit run app/main.py
```

Open:

```
http://localhost:8501
```

---

## 💡 Example Questions

- What is the total revenue?
- Which region has the highest sales?
- Show monthly revenue trends.
- Which products are underperforming?
- Generate SQL for the top 5 customers.
- Detect anomalies in the revenue column.
- Create a bar chart for product sales.

---

## 🧪 Run Tests

```bash
pytest
```

---

## ⚙️ Configuration

| Variable | Description |
|-----------|-------------|
| GROQ_API_KEY | Groq API Key |
| GROQ_MODEL | Model name |
| MAX_TOKENS | Maximum response tokens |
| MAX_AGENT_TURNS | Maximum tool execution turns |
| ENABLE_CACHE | Enable response caching |
| LOG_LEVEL | Logging level |

---

## 🏗️ How It Works

1. Upload one or more CSV files.
2. Ask questions in plain English.
3. The AI decides which tool to use.
4. The tool analyzes the uploaded data.
5. Results are returned with charts and explanations.

---

## 🔮 Future Improvements

- User authentication
- Forecasting and prediction
- More chart types
- Better caching
- Database support
- Dashboard customization

---

## 👨‍💻 Author

**Himasri Pithani**

Built as part of the **Digital Back Office AI Engineer Assignment**.
