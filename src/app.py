
# When executed with plain `python app.py`, Streamlit's runtime context and session
# state are not available which leads to many warnings and broken behavior.
# Detect common Streamlit environment variables and show a helpful message + exit
# when none are present. This allows users who accidentally run `python app.py`
# to get clear guidance while still allowing `streamlit run app.py` to work.


import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from file_reader_agent import process_file
from analyzer import analyze_csv
from document_processing_agent import DocumentProcessingAgent
from financial_report_agent import FinancialReportAgent
# from FinancialAnalysisAgent import FinancialAnalysisAgent
from CalculationEngine import FinancialEngine
from conversational_finance_agent import ConversationalFinanceAgent

# Instantiate agents (RAG/UI removed — using the original simple flow)
agent = DocumentProcessingAgent()
report_agent = FinancialReportAgent()
conversation_agent = ConversationalFinanceAgent()
# analysis_agent = FinancialAnalysisAgent()

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="FinAgent AI",
    page_icon="💰",
    layout="wide"
)

if "processed" not in st.session_state:
    st.session_state.processed = False

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "transactions" not in st.session_state:
    st.session_state.transactions = None

if "finance_answer" not in st.session_state:
    st.session_state.finance_answer = ""

# ---------------------------------------
# Visual identity: a "ledger / passbook" theme.
# Ink navy + brass accent on a cool paper background, monospace figures
# for money (tabular numerals), dashed "perforation" motifs tying back
# to bank-passbook / receipt stationery.
# ---------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #12283D;
        --ink-soft: #2A4257;
        --paper: #F2F5F1;
        --paper-2: #E6EBE4;
        --brass: #B8863B;
        --green: #2E7D5B;
        --rust: #B34B3C;
        --line: #C9D1C6;
    }

    .stApp {
        background: var(--paper);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
        letter-spacing: -0.01em;
    }

    /* Sidebar as the passbook cover */
    section[data-testid="stSidebar"] {
        background: var(--ink);
    }
    section[data-testid="stSidebar"] * {
        color: var(--paper) !important;
    }
    section[data-testid="stSidebar"] .stTextInput input {
        background: var(--ink-soft);
        border: 1px solid var(--brass);
        color: var(--paper) !important;
    }

    /* Buttons: stamped-ledger look */
    .stButton > button {
        background: var(--ink);
        color: var(--paper);
        border: 1px solid var(--ink);
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .stButton > button:hover {
        background: var(--brass);
        border-color: var(--brass);
        color: var(--ink);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--brass);
        border-color: var(--brass);
        color: var(--ink);
        width: 100%;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.92rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--brass) !important;
        border-bottom-color: var(--brass) !important;
    }

    /* Hero banner */
    .hero {
        position: relative;
        overflow: hidden;
        background: var(--ink);
        border-radius: 10px;
        padding: 1.5rem 1.75rem 1.5rem 2.1rem;
        margin-bottom: 1.4rem;
    }
    .hero::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 14px;
        background-image: radial-gradient(circle, var(--paper) 1.6px, transparent 1.8px);
        background-size: 12px 12px;
        opacity: 0.45;
    }
    .hero h1 {
        color: var(--paper);
        margin: 0;
        font-size: 1.9rem;
    }
    .hero p {
        color: var(--paper-2);
        margin: 0.35rem 0 0 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }

    /* Ledger metric cards */
    .ledger-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 4px dashed var(--brass);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
    }
    .ledger-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #6B7770;
        margin-bottom: 0.3rem;
    }
    .ledger-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: var(--ink);
        word-break: break-word;
    }
    .tone-pos .ledger-value { color: var(--green); }
    .tone-neg .ledger-value { color: var(--rust); }
    .tone-brass .ledger-value { color: var(--brass); }

    /* Dashed section divider, ledger-stub style */
    .ledger-divider {
        border-top: 1px dashed var(--line);
        margin: 1rem 0 1.2rem 0;
    }

    /* Empty state invitation */
    .empty-state {
        border: 1px dashed var(--line);
        border-radius: 10px;
        padding: 2.2rem;
        text-align: center;
        background: #ffffff;
    }
    .empty-state h3 { margin-top: 0; }

    /* Tabular figures inside tables */
    [data-testid="stDataFrame"] * {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ledger_card(label, value, tone="neutral"):
    """Render one ledger-style metric card as an HTML snippet."""
    return f"""
    <div class="ledger-card tone-{tone}">
        <div class="ledger-label">{label}</div>
        <div class="ledger-value">{value}</div>
    </div>
    """


def render_metric_row(items, columns=4):
    """items: list of dicts with keys label, value, and optional tone
    ('pos' / 'neg' / 'brass' / 'neutral')."""
    cols = st.columns(columns)
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                ledger_card(item["label"], item["value"], item.get("tone", "neutral")),
                unsafe_allow_html=True,
            )


def section_divider():
    st.markdown('<div class="ledger-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------
# Sidebar: identity + upload (kept as the app's control panel)
# ---------------------------------------
with st.sidebar:
    st.markdown("### 💰 FinAgent AI")
    st.caption("Your statement, read like a ledger.")

    name = st.text_input("Enter your name")

    if st.button("Submit"):
        if name.strip():
            st.success(f"Welcome {name}!")
        else:
            st.warning("Please enter your name.")

    st.markdown("---")
    st.markdown("**Upload statement**")
    uploaded_file = st.file_uploader(
        "Upload PDF or CSV",
        type=["pdf", "csv"],
        label_visibility="collapsed",
    )

# ---------------------------------------
# Hero header
# ---------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💰 FinAgent AI</h1>
        <p>Upload a Google Pay, PhonePe or Paytm statement (PDF/CSV) and get a full
        ledger-style breakdown of your income, spending, and habits.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------
# Process File
# ---------------------------------------
if uploaded_file is not None:

    st.success("✅ File Uploaded Successfully")

    st.subheader("📄 File Information")
    render_metric_row(
        [
            {"label": "Filename", "value": uploaded_file.name, "tone": "neutral"},
            {"label": "File Type", "value": uploaded_file.type, "tone": "neutral"},
            {"label": "File Size", "value": f"{uploaded_file.size} bytes", "tone": "neutral"},
        ],
        columns=3,
    )
    section_divider()

    # Agent 1
    result = process_file(uploaded_file)

    # =====================================================
    # CSV
    # =====================================================
    if result["type"] == "csv":

        df = result["data"]

        st.subheader("📊 CSV Data")

        st.dataframe(df)

        if st.button("Analyze CSV"):

            analysis = analyze_csv(df)

            st.header("📈 CSV Analysis")

            st.write("Rows:", analysis["rows"])
            st.write("Columns:", analysis["columns"])

            st.write("Column Names")
            st.write(analysis["column_names"])

            st.write("Missing Values")
            st.write(analysis["missing_values"])

            st.subheader("Summary Statistics")
            st.dataframe(analysis["summary"])

            # Financial Summary
            if "total" in analysis:

                st.subheader("💰 Financial Summary")

                render_metric_row(
                    [
                        {"label": "Total", "value": f"₹ {analysis['total']:.2f}", "tone": "brass"},
                        {"label": "Average", "value": f"₹ {analysis['average']:.2f}", "tone": "neutral"},
                        {"label": "Maximum", "value": f"₹ {analysis['maximum']:.2f}", "tone": "pos"},
                        {"label": "Minimum", "value": f"₹ {analysis['minimum']:.2f}", "tone": "neg"},
                    ],
                    columns=4,
                )

                # Chart
                st.subheader("📊 Financial Chart")

                fig, ax = plt.subplots()

                ax.bar(
                    ["Total", "Average", "Maximum", "Minimum"],
                    [
                        analysis["total"],
                        analysis["average"],
                        analysis["maximum"],
                        analysis["minimum"]
                    ]
                )

                ax.set_ylabel("Amount")
                ax.set_title("Financial Summary")

                st.pyplot(fig)


    # =====================================================
    # PDF
    # =====================================================
    elif result["type"] == "pdf":
        text = result.get("data", "")

        if result.get("error"):
            st.error(result["error"])

        st.subheader("📄 Extracted PDF Text")
        if text and text.strip():
            preview_text = text[:10000] + ("..." if len(text) > 10000 else "")
            with st.expander("Show extracted text"):
                st.code(preview_text)
        else:
            st.info("No extractable text was found in the uploaded PDF. It may be a scanned or corrupted document.")

        process_clicked = st.button(
            "🚀 Process Statement",
            use_container_width=True
        )

        if process_clicked:

            with st.spinner("Analyzing..."):

                output = agent.process(
                    document_text=text,
                    filename=uploaded_file.name
                )

                transactions = output["transactions"]

                engine = FinancialEngine(transactions)

                analysis = engine.run()

                st.session_state.analysis = analysis
                st.session_state.transactions = transactions
                st.session_state.processed = True

        if st.session_state.get("processed", False):

            analysis = st.session_state.analysis

            validation = analysis.get("validation", {})
            summary = analysis.get("financial_summary", {})
            patterns = analysis.get("spending_patterns", {})
            chart_data = analysis.get("chart_data", {})

            tab_overview, tab_charts, tab_breakdown, tab_top, tab_report = st.tabs(
                ["📋 Overview", "📊 Charts", "🧾 Breakdown", "🔥 Top Transactions", "📝 Report & Ask AI"]
            )

            # -------------------------------------------------
            # Overview tab
            # -------------------------------------------------
            with tab_overview:
                st.markdown("#### Validation Summary")
                render_metric_row(
                    [
                        {"label": "Total Transactions", "value": validation.get("total_transactions", 0), "tone": "neutral"},
                        {"label": "Valid Transactions", "value": validation.get("valid_transactions", 0), "tone": "pos"},
                        {"label": "Invalid Transactions", "value": validation.get("invalid_transactions", 0),
                         "tone": "neg" if validation.get("invalid_transactions", 0) else "neutral"},
                        {"label": "Missing Amount", "value": validation.get("missing_amount", 0),
                         "tone": "neg" if validation.get("missing_amount", 0) else "neutral"},
                    ]
                )

                section_divider()
                st.markdown("#### Financial Dashboard")
                net_balance = summary.get('net_balance', 0)
                render_metric_row(
                    [
                        {"label": "Total Income", "value": f"₹ {summary.get('total_income', 0):,.2f}", "tone": "pos"},
                        {"label": "Total Expense", "value": f"₹ {summary.get('total_expense', 0):,.2f}", "tone": "neg"},
                        {"label": "Net Balance", "value": f"₹ {net_balance:,.2f}", "tone": "pos" if net_balance >= 0 else "neg"},
                        {"label": "Transactions", "value": summary.get("total_transactions", 0), "tone": "neutral"},
                    ]
                )

                render_metric_row(
                    [
                        {"label": "Highest Spending Category", "value": patterns.get("highest_spending_category", "N/A"), "tone": "brass"},
                        {"label": "Most Active Month", "value": patterns.get("most_active_month", "N/A"), "tone": "neutral"},
                        {"label": "Most Active Day", "value": patterns.get("most_active_day", "N/A"), "tone": "neutral"},
                        {"label": "Average Transaction Value", "value": f"₹ {patterns.get('average_transaction_value', 0):,.2f}", "tone": "neutral"},
                    ]
                )

                ratio = summary.get('income_expense_ratio')
                render_metric_row(
                    [
                        {"label": "Savings %", "value": f"{summary.get('saving_percentage', 0):.2f}%", "tone": "pos"},
                        {"label": "Expense %", "value": f"{summary.get('expense_percentage', 0):.2f}%", "tone": "neg"},
                        {"label": "Income/Expense Ratio", "value": str(ratio) if ratio is not None else "N/A", "tone": "neutral"},
                        {"label": "Largest Expense", "value": f"₹ {summary.get('largest_expense', 0):,.2f}", "tone": "neg"},
                    ]
                )

            # -------------------------------------------------
            # Charts tab
            # -------------------------------------------------
            with tab_charts:
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.write("### Expense by Category")
                    category_data = chart_data.get("pie_chart", [])
                    if category_data:
                        labels = [item["label"] for item in category_data]
                        values = [item["value"] for item in category_data]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
                        ax.set_title("Expense by Category")
                        st.pyplot(fig)
                    else:
                        st.info("No category breakdown available.")

                with chart_col2:
                    st.write("### Income vs Expense")
                    bar_data = chart_data.get("bar_chart", [])
                    if bar_data:
                        bar_df = pd.DataFrame(bar_data)
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.bar(bar_df["label"], bar_df["income"], label="Income")
                        ax.bar(bar_df["label"], bar_df["expense"], bottom=0, alpha=0.7, label="Expense")
                        ax.set_ylabel("Amount (₹)")
                        ax.set_title("Monthly Income and Expense")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.info("No income/expense trend data available.")

                section_divider()
                st.write("### Daily Spending Trend")
                spending_trend = analysis.get("daily_trends", {}).get("spending_trend", [])
                if spending_trend:
                    trend_df = pd.DataFrame(spending_trend).set_index("date")
                    st.line_chart(trend_df)
                else:
                    st.info("No daily trend data available.")

            # -------------------------------------------------
            # Breakdown tab
            # -------------------------------------------------
            with tab_breakdown:
                st.markdown("#### Periodic Summary")
                week_df = pd.DataFrame(analysis.get("weekly_summary", []))
                month_df = pd.DataFrame(analysis.get("monthly_summary", []))
                if not week_df.empty:
                    st.write("### Weekly Summary")
                    st.dataframe(week_df, use_container_width=True)
                else:
                    st.info("No weekly summary available.")

                if not month_df.empty:
                    st.write("### Monthly Summary")
                    st.dataframe(month_df, use_container_width=True)
                else:
                    st.info("No monthly summary available.")

                section_divider()
                st.markdown("#### 💳 Payment Method Analysis")
                payment_data = analysis.get("payment_analysis", {})
                payment_df = pd.DataFrame(
                    list(payment_data.items()), columns=["Payment Method", "Amount"]
                ).sort_values(by="Amount", ascending=False)
                if not payment_df.empty:
                    st.dataframe(payment_df, use_container_width=True)
                else:
                    st.info("No payment method data available.")

                st.markdown("#### 🧾 Category Analysis")
                category_df = pd.DataFrame(
                    list(analysis.get("category_analysis", {}).items()),
                    columns=["Category", "Amount"],
                ).sort_values(by="Amount", ascending=False)
                if not category_df.empty:
                    st.dataframe(category_df, use_container_width=True)
                else:
                    st.info("No category data available.")

                st.markdown("#### 🏷 Merchant Analysis")
                merchant_df = pd.DataFrame(
                    list(analysis.get("merchant_analysis", {}).items()),
                    columns=["Merchant", "Amount"],
                ).sort_values(by="Amount", ascending=False)
                if not merchant_df.empty:
                    st.dataframe(merchant_df.head(10), use_container_width=True)
                else:
                    st.info("No merchant data available.")

            # -------------------------------------------------
            # Top Transactions tab
            # -------------------------------------------------
            with tab_top:
                st.markdown("#### 🔥 Top 5 Expenses")
                expense_df = pd.DataFrame(analysis.get("top_5_expenses", []))
                if not expense_df.empty:
                    expense_df.index = expense_df.index + 1
                    expense_df.rename(
                        columns={
                            "merchant": "Merchant",
                            "amount": "Amount (₹)",
                            "date": "Date",
                        },
                        inplace=True,
                    )
                    st.dataframe(expense_df, use_container_width=True)
                else:
                    st.info("No expense transactions found.")

                st.markdown("#### 💵 Top 5 Income")
                income_df = pd.DataFrame(analysis.get("top_5_income", []))
                if not income_df.empty:
                    income_df.index = income_df.index + 1
                    income_df.rename(
                        columns={
                            "merchant": "Merchant",
                            "amount": "Amount (₹)",
                            "date": "Date",
                        },
                        inplace=True,
                    )
                    st.dataframe(income_df, use_container_width=True)
                else:
                    st.info("No income transactions found.")

                section_divider()
                st.markdown("#### 💡 Budget Recommendations")
                budget_recs = analysis.get("budget_recommendations") or []
                if budget_recs:
                    for recommendation in budget_recs:
                        st.success("✅ " + recommendation)
                else:
                    st.info("No budget recommendations available.")

            # -------------------------------------------------
            # Report & Ask AI tab
            # -------------------------------------------------
            with tab_report:
                st.markdown("#### 📝 AI Financial Report")

                report_input = {
                    "summary": summary,
                    "patterns": patterns,
                    "statistics": {
                        "saving_percentage": summary.get('saving_percentage'),
                        "expense_percentage": summary.get('expense_percentage'),
                        "income_expense_ratio": summary.get('income_expense_ratio'),
                        "largest_income": summary.get('largest_income'),
                        "largest_expense": summary.get('largest_expense'),
                        "average_income": summary.get('average_income'),
                        "average_expense": summary.get('average_expense'),
                        "average_transaction_value": summary.get('average_transaction_value'),
                    },
                    "recommendations": analysis.get('budget_recommendations') or [],
                }

                try:
                    financial_report_text = report_agent.generate(report_input)
                except Exception as e:
                    st.error(f"Financial Report generation failed: {e}")
                    financial_report_text = None

                if financial_report_text:
                    st.text_area("Financial Report", value=financial_report_text, height=260)

                section_divider()
                st.markdown("#### 💬 Ask FinAgent AI")

                question = st.text_input(
                    "Ask a question about your finances",
                    placeholder="Example: Where did I spend the most money?",
                    key="finance_question"
                )

                if st.button("Ask FinAgent"):

                    if question.strip():

                        with st.spinner("Thinking..."):

                            try:

                                answer = conversation_agent.ask(
                                    financial_analysis=st.session_state.analysis,
                                    question=question
                                )

                                st.session_state.finance_answer = answer

                            except Exception as e:
                                st.session_state.finance_answer = f"Error: {e}"

                    else:
                        st.warning("Please enter a question.")

                if st.session_state.finance_answer:
                    st.markdown("### 🤖 FinAgent AI")
                    st.info(st.session_state.finance_answer)

    # =====================================================
    # Unsupported
    # =====================================================
    else:

        st.error("Unsupported File Type")

else:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No statement loaded yet</h3>
            <p>Upload a PDF or CSV UPI/bank statement from the sidebar to see your
            transaction ledger, spending breakdown, and ask FinAgent AI about your money.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )