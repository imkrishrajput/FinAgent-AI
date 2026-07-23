# FinAgent AI

## Project Overview

FinAgent AI is a multi-agent finance assistant that analyzes personal financial statements and transaction data. The application accepts PDF bank statements and CSV transaction files, extracts financial information, performs expense analysis, generates financial reports, and provides conversational financial insights using a Large Language Model (LLM).

The project is built using Streamlit for the user interface and Groq API for fast LLM inference.

---

## Features

- Upload PDF bank statements
- Upload CSV transaction files
- Automatic transaction extraction
- Financial data analysis
- Expense categorization
- Financial summary generation
- Interactive dashboard
- Conversational finance assistant
- AI-generated financial reports

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web Application |
| Groq API | LLM Inference |
| Pandas | Data Processing |
| PyMuPDF | PDF Text Extraction |
| Matplotlib | Data Visualization |
| python-dotenv | Environment Variable Management |

---

## Project Structure

```text
FinAgent-AI/
│
├── src/
│   ├── app.py
│   ├── analyzer.py
│   ├── csv_reader.py
│   ├── pdf_reader.py
│   ├── transaction_extractor.py
│   ├── file_reader_agent.py
│   ├── document_processing_agent.py
│   ├── financial_report_agent.py
│   ├── conversational_finance_agent.py
│   ├── FinancialAnalysisAgent.py
│   ├── llm_client.py
│   ├── model_manager.py
│   ├── models.py
│   └── retriever.py
│
├── assets/
├── data/
├── docs/
├── sample_files/
│
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/FinAgent-AI.git
```

### 2. Navigate to the Project

```bash
cd FinAgent-AI
```

### 3. Create a Virtual Environment

Windows

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

PowerShell

```bash
.\venv\Scripts\Activate.ps1
```

Command Prompt

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Application

```bash
streamlit run src/app.py
```

The application will open in your browser.

---

## Supported Input Formats

- PDF Financial Statements
- CSV Transaction Files

---

## Output

The application provides:

- Transaction extraction
- Expense categorization
- Financial analysis
- Budget insights
- AI-generated financial reports
- Conversational financial assistance

---

## Dataset

This repository does **not** include large datasets or personal financial records.

Users can test the application using their own PDF statements or CSV transaction files.

---

## Future Enhancements

- Support additional banking formats
- Improved transaction categorization
- Export reports as PDF
- Advanced financial analytics
- Cloud deployment

---

## Author

Krish Rajput