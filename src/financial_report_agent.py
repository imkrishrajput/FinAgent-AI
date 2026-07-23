"""
Agents.financial_report_agent
=============================
A small wrapper "agent" that converts structured analysis data produced by
the FinancialEngine (or equivalent) into a human-readable financial report.

Responsibilities (strictly):
- Accept structured data (no calculations) and produce a formatted report.
- Validate presence of required fields and return friendly errors (raise
  ValueError for the caller to show to the user).
- NEVER calculate or modify numbers; only formats and echoes the supplied
  analysis.

This file implements a single class `FinancialReportAgent` with one public
method: `generate(report_data: dict) -> str`.

No external services, no LLM calls, no DBs.
"""
from typing import Any, Dict, List, Optional


class FinancialReportAgent:
    """Generate a professional financial report from structured analysis.

    The agent follows the strict rules: it only formats supplied data and
    never invents or recalculates values.
    """

    def __init__(self) -> None:
        """Initialize the agent. Stateless by design."""
        pass

    def generate(self, report_data: Dict[str, Any]) -> str:
        """Create a human-readable report from structured input.

        Expected input formats supported (either form is accepted):
        - Preferred keys: {"summary", "patterns", "statistics", "recommendations"}
        - Alternate keys: {"financial_summary", "spending_patterns", "budget_recommendations"}

        The function does NOT compute any numbers. If required values are
        missing it raises ValueError with a clear message for the caller.

        Returns:
            A multi-line string report ready for display.
        """
        if not isinstance(report_data, dict):
            raise ValueError("Report input must be a dictionary.")

        # Flexible mapping: support both preferred and alternate keys
        summary = report_data.get("summary") or report_data.get("financial_summary")
        patterns = report_data.get("patterns") or report_data.get("spending_patterns")
        statistics = report_data.get("statistics")
        recommendations = report_data.get("recommendations") or report_data.get("budget_recommendations") or []

        # Basic validation: we only require a summary and patterns to produce a report
        if not summary or not isinstance(summary, dict):
            raise ValueError("Missing or invalid 'summary' data required for the financial report.")
        if not patterns or not isinstance(patterns, dict):
            raise ValueError("Missing or invalid 'patterns' data required for the financial report.")

        # Build a minimal statistics dict from summary where appropriate, but DO NOT compute
        stats = {}
        if statistics and isinstance(statistics, dict):
            stats = statistics
        else:
            # Take commonly available fields from summary (if present) but do not compute anything
            possible = [
                "saving_percentage",
                "expense_percentage",
                "income_expense_ratio",
                "largest_income",
                "largest_expense",
                "average_income",
                "average_expense",
                "average_transaction_value",
            ]
            for key in possible:
                if key in summary:
                    stats[key] = summary.get(key)

        # Safely read values without changing them
        total_income = summary.get("total_income")
        total_expense = summary.get("total_expense")
        net_balance = summary.get("net_balance")
        total_txns = summary.get("total_transactions")

        highest_category = patterns.get("highest_spending_category")
        highest_spending_amount = patterns.get("highest_spending_amount")
        most_frequent_merchant = patterns.get("most_frequent_merchant")
        most_active_day = patterns.get("most_active_day")
        most_active_month = patterns.get("most_active_month")
        average_txn = patterns.get("average_transaction_value") or stats.get("average_transaction_value")

        saving_percentage = stats.get("saving_percentage")
        expense_percentage = stats.get("expense_percentage")
        income_expense_ratio = stats.get("income_expense_ratio")

        # Start assembling the report lines. Use only supplied values (may be None).
        lines: List[str] = []
        lines.append("FINANCIAL ANALYSIS REPORT")
        lines.append("")

        # 1. Income & Expense Summary
        lines.append("1. Income & Expense Summary")
        lines.append(f"   • Total Income: ₹ {total_income:,.2f}" if total_income is not None else "   • Total Income: N/A")
        lines.append(f"   • Total Expense: ₹ {total_expense:,.2f}" if total_expense is not None else "   • Total Expense: N/A")
        lines.append(f"   • Net Balance: ₹ {net_balance:,.2f}" if net_balance is not None else "   • Net Balance: N/A")
        lines.append(f"   • Total Transactions: {total_txns}" if total_txns is not None else "   • Total Transactions: N/A")
        lines.append("")

        # 2. Financial Overview
        lines.append("2. Financial Overview")
        if total_txns is not None and total_income is not None and total_expense is not None and net_balance is not None:
            lines.append(
                f"   This statement contains {total_txns} financial transactions. During this period the total income was ₹{total_income:,.2f} while total expenses were ₹{total_expense:,.2f}, resulting in a net balance of ₹{net_balance:,.2f}."
            )
        else:
            lines.append("   A complete numeric summary was not provided. Review available fields above.")
        lines.append("")

        # 3. Spending Pattern
        lines.append("3. Spending Pattern")
        lines.append(f"   • Highest Spending Category: {highest_category if highest_category is not None else 'N/A'}")
        lines.append(f"   • Highest Spending Amount: ₹ {highest_spending_amount:,.2f}" if highest_spending_amount is not None else "   • Highest Spending Amount: N/A")
        lines.append(f"   • Most Frequent Merchant: {most_frequent_merchant if most_frequent_merchant is not None else 'N/A'}")
        lines.append(f"   • Most Active Day: {most_active_day if most_active_day is not None else 'N/A'}")
        lines.append(f"   • Most Active Month: {most_active_month if most_active_month is not None else 'N/A'}")
        lines.append(f"   • Average Transaction Value: ₹ {average_txn:,.2f}" if average_txn is not None else "   • Average Transaction Value: N/A")
        lines.append("")
        lines.append("   These values describe where and when most activity occurred and the typical transaction size. They are taken directly from the analysis output.")
        lines.append("")

        # 4. Financial Health
        lines.append("4. Financial Health")
        lines.append(f"   • Saving Percentage: {saving_percentage:.2f}%" if saving_percentage is not None else "   • Saving Percentage: N/A")
        lines.append(f"   • Expense Percentage: {expense_percentage:.2f}%" if expense_percentage is not None else "   • Expense Percentage: N/A")
        lines.append(f"   • Income/Expense Ratio: {income_expense_ratio}" if income_expense_ratio is not None else "   • Income/Expense Ratio: N/A")
        lines.append("")
        lines.append("   Interpret these values according to standard guidance: higher saving % and income/expense ratio > 1 indicate healthier cash flow. This text does not compute or modify the supplied numbers.")
        lines.append("")

        # 5. Key Insights (4-6 insights from supplied data)
        lines.append("5. Key Insights")
        # Use only supplied fields to create insights
        if total_txns is not None:
            lines.append(f"   • The user completed {total_txns} transactions during the period.")
        if total_income is not None and total_expense is not None:
            lines.append(f"   • Income exceeded expenses by ₹ {net_balance:,.2f}." if net_balance is not None else "   • Income and expense totals were provided.")
        if highest_spending_amount is not None:
            lines.append(f"   • The highest spending amount was ₹ {highest_spending_amount:,.2f}.")
        if most_frequent_merchant is not None:
            lines.append(f"   • The most frequent merchant was {most_frequent_merchant}.")
        if most_active_month is not None:
            lines.append(f"   • The most active month was {most_active_month}.")
        if average_txn is not None:
            lines.append(f"   • The average transaction value was ₹ {average_txn:,.2f}.")
        lines.append("")

        # 6. Recommendations - rewritten into natural language
        lines.append("6. Recommendations")
        if recommendations:
            for rec in recommendations:
                lines.append(f"   • {rec}")
        else:
            lines.append("   No financial recommendations are available.")
        lines.append("")

        # 7. Overall Financial Status (derived only by simple rules that do not change numbers)
        lines.append("7. Overall Financial Status")
        # Determine a status label using only supplied statistics (no numeric calculations beyond comparisons)
        status = "Average"
        try:
            if income_expense_ratio is None:
                status = "Average"
            elif income_expense_ratio < 1:
                status = "Needs Improvement"
            elif saving_percentage is not None and income_expense_ratio is not None and saving_percentage >= 25 and income_expense_ratio >= 1.2:
                status = "Good"
            elif saving_percentage is not None and income_expense_ratio is not None and saving_percentage >= 20 and income_expense_ratio >= 1.0:
                status = "Good"
            else:
                status = "Average"
        except Exception:
            status = "Average"

        lines.append(f"   • Overall Financial Status: {status}")
        lines.append("")
        lines.append("Report generated from the analyzed transaction data.")

        return "\n".join(lines)
