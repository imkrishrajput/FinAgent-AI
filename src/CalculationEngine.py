"""
financial_engine.py
=====================
Pure-Python Financial Engine for FinAgent AI.

This module is the calculation heart of the application. It receives the
normalized transaction JSON produced by Agent 1 (document_processing_agent.py)
and performs every financial calculation using ONLY the Python standard
library — no LLM, no pandas, no third-party API calls of any kind.

Pipeline position:
    file → document_processing_agent.py (Agent 1: extract + normalize)
         → financial_engine.py          (pure-Python calculations)
         → app.py                       (Streamlit display / charts)

Design notes for a college viva:
    - `ParsedTransaction` is a small dataclass that stores a cleaned,
      typed version of each raw transaction (parsed date, numeric amount,
      derived category, normalized payment bucket, validity flag). Every
      calculation downstream works off this cleaned list instead of the
      messy raw dicts, so each function stays simple.
    - `FinancialEngine` is the single public entry point. Each of its
      "analysis" methods (financial_summary, category_analysis, ...)
      is independent, so you can call just the one you need, or call
      `run()` to get everything at once as one dictionary.
    - No pandas: grouping is done with `collections.defaultdict` /
      `collections.Counter`, which are standard-library equivalents of
      a pandas `groupby`.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Allowed spending/income categories. Every transaction is mapped to
#: exactly one of these (never more than one).
CATEGORY_KEYS: List[str] = [
    "Food", "Travel", "Shopping", "Bills", "Recharge", "Fuel",
    "Healthcare", "Entertainment", "Education", "Investment",
    "Salary", "Transfer", "Others",
]

#: Keyword → category lookup. Matching is done against the merchant name
#: and description in lowercase. This is intentionally simple (substring
#: matching) so it is easy to explain and easy to extend.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "food", "dominos", "pizza", "starbucks", "eatery"],
    "Travel": ["uber", "ola", "irctc", "flight", "indigo", "makemytrip", "railway", "train", "bus", "cab", "metro", "dmrc"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "shopping", "mall", "store", "mart"],
    "Bills": ["electricity", "water bill", "bill payment", "broadband", "wifi", "gas bill", "rent", "utility"],
    "Recharge": ["recharge", "airtel", "jio", "vodafone", " vi ", "dth"],
    "Fuel": ["petrol", "diesel", "fuel", "hpcl", "iocl", "bpcl", "shell", "petroleum"],
    "Healthcare": ["hospital", "pharmacy", "medical", "clinic", "apollo", "medplus", "doctor", "health"],
    "Entertainment": ["netflix", "prime video", "hotstar", "spotify", "bookmyshow", "movie", "cinema", "gaana"],
    "Education": ["udemy", "coursera", "school", "college", "tuition", "course", "exam fee", "university"],
    "Investment": ["mutual fund", "zerodha", "groww", "sip", "stocks", "upstox", "investment", "shares"],
    "Salary": ["salary", "payroll", "wages", "stipend"],
    "Transfer": ["transfer", "sent to", "received from", "p2p", "neft", "imps", "rtgs"],
}

#: Payment-method raw string → normalized bucket.
PAYMENT_METHOD_BUCKETS: Dict[str, List[str]] = {
    "UPI": ["upi"],
    "Card": ["card", "credit card", "debit card"],
    "Wallet": ["wallet", "paytm wallet", "amazon pay", "phonepe wallet"],
    "Bank": ["bank", "netbanking", "net banking"],
    "Cash": ["cash"],
}

#: Transaction types the engine understands. Anything else is "Unknown".
VALID_TRANSACTION_TYPES = {"Income", "Expense", "Transfer", "Refund"}

#: Date formats accepted from Agent 1. Extend this list if new sources
#: use a different date format.
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",

    "%d %b, %Y",
    "%d %B, %Y",

    "%b %d, %Y",
    "%B %d, %Y",
]


# ---------------------------------------------------------------------------
# Small standalone helper functions
# ---------------------------------------------------------------------------

def safe_float(value: Any) -> Optional[float]:
    """
    Safely convert a value to float.

    Why it exists:
        Raw transaction amounts might be strings, None, or already
        numeric. We never want a bad value to crash the whole engine.

    Receives:
        value: Any raw value (str, int, float, None, ...).

    Returns:
        The float value, or None if it could not be converted.

    Complexity:
        Time O(1), Space O(1).
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse a date string into a `datetime` object, trying several
    common formats.

    Why it exists:
        Different statement providers (Google Pay, PhonePe, Paytm, CSV)
        format dates differently. This centralizes date parsing so every
        other function can assume a real `datetime` or `None`.

    Receives:
        date_str: The raw date string (or None).

    Returns:
        A `datetime` object at midnight of that date, or None if the
        string is missing or unparsable.

    Complexity:
        Time O(k) where k = number of formats tried (constant, ~6).
        Space O(1).
    """
    if not date_str or not isinstance(date_str, str):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def categorize_transaction(merchant: str, description: str, transaction_type: str) -> str:
    """
    Assign exactly one category to a transaction based on merchant name
    and description keywords.

    Why it exists:
        Agent 1 does not categorize transactions — that is a financial
        decision, not an extraction one, so it belongs in the engine.

    Receives:
        merchant: The merchant name (may be empty).
        description: The transaction description (may be empty).
        transaction_type: "Income", "Expense", "Transfer", "Refund", etc.

    Returns:
        One category string from CATEGORY_KEYS. Falls back to "Salary"
        for unmatched income, "Transfer" for unmatched transfers, and
        "Others" for everything else that has no keyword match.

    Complexity:
        Time O(k) where k = total number of keywords (constant, small).
        Space O(1).
    """
    text = f"{merchant or ''} {description or ''}".lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category

    if transaction_type == "Income":
        return "Salary"
    if transaction_type == "Transfer":
        return "Transfer"
    return "Others"


def normalize_payment_method(raw_method: Optional[str]) -> str:
    """
    Normalize a raw payment-method string into one of the fixed buckets.

    Why it exists:
        Providers describe payment methods inconsistently
        ("UPI", "GPay UPI", "upi payment"...). This gives a single
        consistent bucket for reporting.

    Receives:
        raw_method: The raw payment method string (or None).

    Returns:
        One of "UPI", "Card", "Wallet", "Bank", "Cash", or "Others".

    Complexity:
        Time O(1) (constant number of buckets/keywords).
        Space O(1).
    """
    if not raw_method or not isinstance(raw_method, str):
        return "Others"
    text = raw_method.lower()
    for bucket, keywords in PAYMENT_METHOD_BUCKETS.items():
        if any(keyword in text for keyword in keywords):
            return bucket
    return "Others"


def week_key(dt: datetime) -> str:
    """
    Build a sortable ISO-week key like "2026-W23" from a datetime.

    Why it exists:
        Grouping transactions "by week" needs a consistent, sortable
        key — plain `datetime` objects aren't directly usable as
        dictionary group labels for display.

    Receives:
        dt: A datetime object.

    Returns:
        A string in the form "<ISO year>-W<ISO week number>".

    Complexity:
        Time O(1), Space O(1).
    """
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def month_key(dt: datetime) -> str:
    """
    Build a sortable month key like "2026-06" from a datetime.

    Why it exists:
        Same reasoning as `week_key`, but grouped by calendar month.

    Receives:
        dt: A datetime object.

    Returns:
        A string in the form "YYYY-MM".

    Complexity:
        Time O(1), Space O(1).
    """
    return dt.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Cleaned transaction representation
# ---------------------------------------------------------------------------

@dataclass
class ParsedTransaction:
    """
    A cleaned, typed version of one raw transaction dict.

    Why it exists:
        Every calculation in this engine needs a parsed date, a numeric
        amount, a derived category, and a normalized payment bucket.
        Computing these once per transaction (here) instead of
        repeatedly inside every analysis function keeps the rest of the
        engine simple and fast.
    """
    date: Optional[datetime]
    merchant: str
    description: str
    amount: Optional[float]
    currency: str
    transaction_type: str
    payment_method: str
    reference_number: Optional[str]
    source: Optional[str]
    category: str = field(default="Others")
    is_valid: bool = field(default=False)


def build_parsed_transaction(raw_txn: Dict[str, Any]) -> ParsedTransaction:
    """
    Convert one raw transaction dict (from Agent 1) into a ParsedTransaction.

    Why it exists:
        Centralizes all the "cleaning" logic (date parsing, type
        checking, categorizing, validity checking) in one place.

    Receives:
        raw_txn: A single transaction dict from Agent 1's output.

    Returns:
        A ParsedTransaction with derived category, payment bucket, and
        a validity flag.

    Complexity:
        Time O(1) (a handful of constant-time helper calls).
        Space O(1).
    """
    merchant = raw_txn.get("merchant") or "Unknown"
    description = raw_txn.get("description") or ""
    date_obj = parse_date(raw_txn.get("date"))
    amount = safe_float(raw_txn.get("amount"))
    currency = raw_txn.get("currency") or "INR"

    raw_type = raw_txn.get("transaction_type")
    transaction_type = raw_type if raw_type in VALID_TRANSACTION_TYPES else "Unknown"

    payment_method = normalize_payment_method(raw_txn.get("payment_method"))
    category = categorize_transaction(merchant, description, transaction_type)

    is_valid = date_obj is not None and amount is not None and transaction_type != "Unknown"

    return ParsedTransaction(
        date=date_obj,
        merchant=merchant,
        description=description,
        amount=amount,
        currency=currency,
        transaction_type=transaction_type,
        payment_method=payment_method,
        reference_number=raw_txn.get("reference_number"),
        source=raw_txn.get("source"),
        category=category,
        is_valid=is_valid,
    )


# ---------------------------------------------------------------------------
# The Financial Engine
# ---------------------------------------------------------------------------

class FinancialEngine:
    """
    The pure-Python calculation engine.

    Usage:
        engine = FinancialEngine(agent1_output["transactions"])
        result = engine.run()   # one clean dictionary with everything

    Why a class (not one big function):
        The engine caches the cleaned/parsed transactions once in
        `__init__`, and every analysis method reuses that same list.
        This avoids re-parsing dates/amounts repeatedly and lets each
        method stay small and independently callable/testable.
    """

    def __init__(self, transactions: List[Dict[str, Any]]):
        """
        Receives:
            transactions: The raw transaction list from Agent 1's output
                (i.e. `agent1_output["transactions"]`).

        Complexity:
            Time O(n) to parse all n transactions once.
            Space O(n) to store the parsed list.
        """
        self.raw_transactions = transactions or []
        self.parsed: List[ParsedTransaction] = [
            build_parsed_transaction(txn) for txn in self.raw_transactions
        ]
        self.valid: List[ParsedTransaction] = [t for t in self.parsed if t.is_valid]

    # -- Step 1: Validation --------------------------------------------

    def validate_transactions(self) -> Dict[str, int]:
        """
        Count valid vs invalid transactions and break down invalid
        reasons.

        Why it exists:
            Downstream calculations should only trust valid data; this
            gives visibility into how much data was usable.

        Receives:
            (self.parsed, already computed in __init__)

        Returns:
            Dict with total/valid/invalid counts and specific missing-
            field counts.

        Complexity:
            Time O(n), Space O(1).
        """
        missing_amount = sum(1 for t in self.parsed if t.amount is None)
        missing_date = sum(1 for t in self.parsed if t.date is None)
        invalid_type = sum(1 for t in self.parsed if t.transaction_type == "Unknown")

        return {
            "total_transactions": len(self.parsed),
            "valid_transactions": len(self.valid),
            "invalid_transactions": len(self.parsed) - len(self.valid),
            "missing_amount": missing_amount,
            "missing_date": missing_date,
            "invalid_transaction_type": invalid_type,
        }

    # -- Step 2: Core financial summary ---------------------------------

    def financial_summary(self) -> Dict[str, Any]:
        """
        Calculate the core financial numbers: totals, averages, extremes,
        and ratios.

        Why it exists:
            This is the single most-requested block of numbers for any
            finance dashboard.

        Receives:
            (self.valid)

        Returns:
            Dict with total_income, total_expense, net_balance, counts,
            averages, largest income/expense, and percentage ratios.

        Complexity:
            Time O(n), Space O(1) (excluding the small fixed-size output).
        """
        incomes = [t.amount for t in self.valid if t.transaction_type == "Income" and t.amount]
        expenses = [abs(t.amount) for t in self.valid if t.transaction_type == "Expense" and t.amount]

        total_income = sum(incomes)
        total_expense = sum(expenses)
        net_balance = total_income - total_expense

        average_income = statistics.mean(incomes) if incomes else 0
        average_expense = statistics.mean(expenses) if expenses else 0

        all_values = [abs(t.amount) for t in self.valid if t.amount is not None]
        average_transaction_value = statistics.mean(all_values) if all_values else 0

        income_expense_ratio = round(total_income / total_expense, 2) if total_expense else None
        saving_percentage = round((net_balance / total_income) * 100, 2) if total_income else 0
        expense_percentage = round((total_expense / total_income) * 100, 2) if total_income else 0

        return {
            "total_income": round(total_income, 2),
            "total_expense": round(total_expense, 2),
            "net_balance": round(net_balance, 2),
            "income_transactions": len(incomes),
            "expense_transactions": len(expenses),
            "total_transactions": len(self.valid),
            "average_income": round(average_income, 2),
            "average_expense": round(average_expense, 2),
            "average_transaction_value": round(average_transaction_value, 2),
            "largest_income": round(max(incomes), 2) if incomes else 0,
            "largest_expense": round(max(expenses), 2) if expenses else 0,
            "income_expense_ratio": income_expense_ratio,
            "saving_percentage": saving_percentage,
            "expense_percentage": expense_percentage,
        }

    # -- Step 3: Category analysis ---------------------------------------

    def category_analysis(self) -> Dict[str, float]:
        """
        Sum expense amounts per category.

        Why it exists:
            Answers "where is the money going" at a glance.

        Receives:
            (self.valid)

        Returns:
            Dict of {category_name: total_expense_amount}, always
            containing all CATEGORY_KEYS (0 if unused).

        Complexity:
            Time O(n), Space O(1) (fixed number of categories).
        """
        totals: Dict[str, float] = {key: 0.0 for key in CATEGORY_KEYS}
        for t in self.valid:
            if t.transaction_type == "Expense" and t.amount is not None:
                totals[t.category] += abs(t.amount)
        return {key: round(value, 2) for key, value in totals.items()}

    # -- Step 4: Merchant analysis ----------------------------------------

    def merchant_analysis(self) -> Dict[str, float]:
        """
        Sum expense amounts per merchant.

        Why it exists:
            Used for "merchant-wise spending" and the most-frequent-
            merchant insight.

        Receives:
            (self.valid)

        Returns:
            Dict of {merchant_name: total_expense_amount}, sorted from
            highest spend to lowest.

        Complexity:
            Time O(n log n) (due to sorting), Space O(m) where m =
            number of distinct merchants.
        """
        totals: Dict[str, float] = defaultdict(float)
        for t in self.valid:
            if t.transaction_type == "Expense" and t.amount is not None:
                totals[t.merchant] += abs(t.amount)

        sorted_totals = dict(
            sorted(totals.items(), key=lambda item: item[1], reverse=True)
        )
        return {merchant: round(amount, 2) for merchant, amount in sorted_totals.items()}

    # -- Step 5: Payment method analysis -----------------------------------

    def payment_analysis(self) -> Dict[str, float]:
        """
        Sum expense amounts per payment method bucket
        (UPI / Card / Wallet / Bank / Cash / Others).

        Why it exists:
            Shows which payment channel the user relies on most.

        Receives:
            (self.valid)

        Returns:
            Dict of {payment_method: total_expense_amount}.

        Complexity:
            Time O(n), Space O(1) (fixed number of buckets).
        """
        buckets = ["UPI", "Card", "Wallet", "Bank", "Cash", "Others"]
        totals: Dict[str, float] = {bucket: 0.0 for bucket in buckets}
        for t in self.valid:
            if t.amount is not None:
                totals[t.payment_method] += abs(t.amount)
        return {key: round(value, 2) for key, value in totals.items()}

    # -- Step 6: Weekly / monthly summaries ---------------------------------

    def weekly_summary(self) -> List[Dict[str, Any]]:
        """
        Group valid transactions by ISO week and total income/expense
        per week.

        Why it exists:
            Powers "weekly trend" views without pandas' groupby.

        Receives:
            (self.valid)

        Returns:
            List of dicts, one per week, sorted chronologically:
            {"week": "2026-W23", "income": .., "expense": ..,
             "net_balance": .., "transaction_count": ..}

        Complexity:
            Time O(n log w) where w = number of distinct weeks
            (sorting the week keys). Space O(w).
        """
        return self._period_summary(key_func=week_key, label_name="week")

    def monthly_summary(self) -> List[Dict[str, Any]]:
        """
        Group valid transactions by calendar month and total
        income/expense per month.

        Why it exists:
            Powers "monthly trend" views, the most common finance-app
            chart.

        Receives:
            (self.valid)

        Returns:
            List of dicts, one per month, sorted chronologically:
            {"month": "2026-06", "income": .., "expense": ..,
             "net_balance": .., "transaction_count": ..}

        Complexity:
            Time O(n log m) where m = number of distinct months.
            Space O(m).
        """
        return self._period_summary(key_func=month_key, label_name="month")

    def _period_summary(self, key_func, label_name: str) -> List[Dict[str, Any]]:
        """
        Shared grouping logic used by both weekly_summary and
        monthly_summary, to avoid duplicating the same loop twice.

        Why it exists:
            Weekly and monthly summaries only differ in how the group
            key is computed — this keeps that logic DRY.

        Receives:
            key_func: A function(datetime) -> str, e.g. week_key or
                month_key.
            label_name: The key name to use in the output dict
                ("week" or "month").

        Returns:
            Sorted list of per-period summary dicts.

        Complexity:
            Time O(n + p log p) where p = number of distinct periods.
            Space O(p).
        """
        groups: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"income": 0.0, "expense": 0.0, "count": 0}
        )

        for t in self.valid:
            if t.date is None or t.amount is None:
                continue
            key = key_func(t.date)
            groups[key]["count"] += 1
            if t.transaction_type == "Income":
                groups[key]["income"] += t.amount
            elif t.transaction_type == "Expense":
                groups[key]["expense"] += abs(t.amount)

        result = []
        for key in sorted(groups.keys()):
            income = groups[key]["income"]
            expense = groups[key]["expense"]
            result.append({
                label_name: key,
                "income": round(income, 2),
                "expense": round(expense, 2),
                "net_balance": round(income - expense, 2),
                "transaction_count": int(groups[key]["count"]),
            })
        return result

    # -- Step 7: Daily trends -----------------------------------------------

    def daily_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build day-by-day income, expense, and net-spending trend lists.

        Why it exists:
            Powers line-chart style "trend over time" views.

        Receives:
            (self.valid)

        Returns:
            Dict with three chronologically sorted lists:
            {"income_trend": [{"date": .., "value": ..}, ...],
             "expense_trend": [...],
             "spending_trend": [...]}  # spending_trend = expense - income (net outflow)

        Complexity:
            Time O(n + d log d) where d = number of distinct days.
            Space O(d).
        """
        daily: Dict[str, Dict[str, float]] = defaultdict(lambda: {"income": 0.0, "expense": 0.0})

        for t in self.valid:
            if t.date is None or t.amount is None:
                continue
            day_key = t.date.strftime("%Y-%m-%d")
            if t.transaction_type == "Income":
                daily[day_key]["income"] += t.amount
            elif t.transaction_type == "Expense":
                daily[day_key]["expense"] += abs(t.amount)

        sorted_days = sorted(daily.keys())
        income_trend = [{"date": d, "value": round(daily[d]["income"], 2)} for d in sorted_days]
        expense_trend = [{"date": d, "value": round(daily[d]["expense"], 2)} for d in sorted_days]
        spending_trend = [
            {"date": d, "value": round(daily[d]["expense"] - daily[d]["income"], 2)}
            for d in sorted_days
        ]

        return {
            "income_trend": income_trend,
            "expense_trend": expense_trend,
            "spending_trend": spending_trend,
        }

    # -- Step 8: Spending pattern insights ------------------------------------

    def spending_patterns(self) -> Dict[str, Any]:

        categories = self.category_analysis()
        highest_category = max(categories, key=categories.get) if categories else ""
        highest_amount = categories.get(highest_category, 0)

        merchant_counter = Counter(
            t.merchant for t in self.valid if t.merchant and t.merchant != "Unknown"
        )
        most_frequent_merchant = (
            merchant_counter.most_common(1)[0][0]
            if merchant_counter else ""
        )

        weekday_counter = Counter(
            t.date.strftime("%A")
            for t in self.valid
            if t.date
        )
        most_active_day = (
            weekday_counter.most_common(1)[0][0]
            if weekday_counter else ""
        )

        month_counter = Counter(
            t.date.strftime("%B %Y")
            for t in self.valid
            if t.date
        )
        most_active_month = (
            month_counter.most_common(1)[0][0]
            if month_counter else ""
        )

        average_transaction_value = (
            round(
                sum(abs(t.amount) for t in self.valid if t.amount is not None) / len(self.valid),
                2,
            )
            if self.valid else 0
        )

        return {
            "highest_spending_category": highest_category,
            "highest_spending_amount": highest_amount,
            "most_frequent_merchant": most_frequent_merchant,
            "most_active_day": most_active_day,
            "most_active_month": most_active_month,
            "average_transaction_value": average_transaction_value,
            "top_5_expenses": self.top_expenses(5),
            "top_5_income": self.top_income(5),
        }

    # -- Step 9: Top transactions ----------------------------------------------

    def top_expenses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N largest expense transactions.

        Why it exists:
            Highlights the biggest individual outflows for review.

        Receives:
            limit: How many transactions to return (default 5).

        Returns:
            List of dicts: {"merchant", "amount", "date", "description"},
            sorted from largest to smallest.

        Complexity:
            Time O(n log n) (sorting), Space O(n).
        """
        return self._top_transactions("Expense", limit)

    def top_income(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N largest income transactions.

        Why it exists:
            Highlights the biggest individual inflows for review.

        Receives:
            limit: How many transactions to return (default 5).

        Returns:
            List of dicts: {"merchant", "amount", "date", "description"},
            sorted from largest to smallest.

        Complexity:
            Time O(n log n) (sorting), Space O(n).
        """
        return self._top_transactions("Income", limit)

    def _top_transactions(self, transaction_type: str, limit: int) -> List[Dict[str, Any]]:
        """
        Shared logic for top_expenses/top_income to avoid duplication.

        Receives:
            transaction_type: "Expense" or "Income".
            limit: How many results to return.

        Returns:
            Sorted (descending by absolute amount) list of transaction
            summaries.

        Complexity:
            Time O(n log n), Space O(n).
        """
        matching = [t for t in self.valid if t.transaction_type == transaction_type and t.amount is not None]
        matching.sort(key=lambda t: abs(t.amount), reverse=True)

        top = matching[:limit]
        return [
            {
                "merchant": t.merchant,
                "amount": round(abs(t.amount), 2),
                "date": t.date.strftime("%Y-%m-%d") if t.date else None,
                "description": t.description,
            }
            for t in top
        ]

    # -- Step 10: Chart-ready data ----------------------------------------------

    def chart_data(self) -> Dict[str, Any]:
        """
        Prepare chart-ready data structures for Streamlit (no chart
        objects are created here — only plain data).

        Why it exists:
            Keeps chart-formatting logic in one place so `app.py` can
            hand this straight to `st.bar_chart`, `st.line_chart`,
            `st.plotly_chart`, etc. without any further processing.

        Receives:
            (results of the other analysis methods)

        Returns:
            Dict with pie_chart, bar_chart, line_chart, weekly_chart,
            monthly_chart, category_chart, and merchant_chart, each a
            list of {"label"/"date", "value"} points.

        Complexity:
            Time O(n) (reuses already-computed summaries), Space O(k)
            where k = number of distinct groups/points.
        """
        categories = self.category_analysis()
        merchants = self.merchant_analysis()
        weekly = self.weekly_summary()
        monthly = self.monthly_summary()
        trends = self.daily_trends()

        pie_chart = [{"label": cat, "value": amt} for cat, amt in categories.items() if amt > 0]
        category_chart = [{"label": cat, "value": amt} for cat, amt in categories.items()]
        merchant_chart = [
            {"label": merchant, "value": amt}
            for merchant, amt in list(merchants.items())[:10]  # top 10 merchants only
        ]
        bar_chart = monthly_bar = [
            {"label": m["month"], "income": m["income"], "expense": m["expense"]}
            for m in monthly
        ]
        line_chart = trends["spending_trend"]
        weekly_chart = [
            {"label": w["week"], "income": w["income"], "expense": w["expense"]}
            for w in weekly
        ]
        monthly_chart = monthly_bar
        payment_method = self.payment_analysis()
        payment_method_chart = [
            {"label": method, "value": amount}
            for method, amount in payment_method.items()
        ]
        daily_trend = trends["spending_trend"]
 
        return {
            "pie_chart": pie_chart,
            "bar_chart": bar_chart,
            "line_chart": line_chart,
            "weekly_chart": weekly_chart,
            "monthly_chart": monthly_chart,
            "category_chart": category_chart,
            "merchant_chart": merchant_chart,
            "payment_method_chart": payment_method_chart,
            "daily_trend": daily_trend,
        }



    # -- Step 11: Budget Recommendations ----------------------------------------

    def budget_recommendations(self) -> List[str]:
        """
        Generate budget recommendations based on spending patterns.

        Why it exists:
            Provides actionable insights for the user to improve their
            financial health based on their spending patterns.

        Returns:
            List of recommendation strings.

        Complexity:
            Time O(1), Space O(1).
        """
        recommendations = []
        summary = self.financial_summary()
        categories = self.category_analysis()

        # Recommendation 1: Savings rate
        saving_percentage = summary.get("saving_percentage", 0)
        if saving_percentage < 20:
            recommendations.append(
                "Your savings rate is below 20%. Consider reducing non-essential expenses."
            )
        elif saving_percentage > 50:
            recommendations.append(
                "Great! You're saving over 50% of your income. Keep up the good work!"
            )

        # Recommendation 2: Highest spending category
        if categories:
            highest_category = max(categories.items(), key=lambda x: x[1])
            if highest_category[1] > summary.get("total_expense", 0) * 0.4:
                recommendations.append(
                    f"Your {highest_category[0]} spending is {highest_category[1]:,.2f}, which is 40%+ of total expenses. "
                    f"Consider reviewing this category."
                )

        # Recommendation 3: Income to Expense Ratio
        income_expense_ratio = summary.get("income_expense_ratio")
        if income_expense_ratio and income_expense_ratio < 1.5:
            recommendations.append(
                "Your income to expense ratio is less than 1.5:1. Try to increase your income or reduce expenses."
            )

        # Recommendation 4: Transaction count
        if summary.get("total_transactions", 0) > 100:
            avg_txn = summary.get("average_transaction_value", 0)
            if avg_txn < 200:
                recommendations.append(
                    "You have many small transactions. Consider consolidating purchases to reduce transaction frequency."
                )

        if not recommendations:
            recommendations.append("Your spending looks healthy! Keep monitoring your finances.")

        return recommendations

    # -- Step 12: Financial Report --------------------------------------------------

    def financial_report(self) -> str:
        """
        Generate a text-based financial report summarizing key metrics.

        Why it exists:
            Provides a human-readable summary of the financial analysis
            in narrative form for easy interpretation.

        Returns:
            A formatted text report as a multi-line string.

        Complexity:
            Time O(1), Space O(1).
        """
        summary = self.financial_summary()
        patterns = self.spending_patterns()

        total_income = summary.get("total_income", 0)
        total_expense = summary.get("total_expense", 0)
        net_balance = summary.get("net_balance", 0)
        total_txns = summary.get("total_transactions", 0)

        highest_category = patterns.get("highest_spending_category", "Unknown")
        highest_merchant = patterns.get("most_frequent_merchant", "Unknown")
        highest_spending_amount = patterns.get("highest_spending_amount", 0)

        report_lines = [
            "FINANCIAL ANALYSIS REPORT",
            "",
            "1. Income & Expense Summary",
            f"   - Total Income: \u20b9 {total_income:,.2f}",
            f"   - Total Expense: \u20b9 {total_expense:,.2f}",
            f"   - Net Balance: \u20b9 {net_balance:,.2f}",
            f"   - Total Transactions: {total_txns}",
            "",
            "2. Financial Overview",
            f"   This statement contains {total_txns} financial transactions. During this period the total income was \u20b9{total_income:,.2f} while total expenses were \u20b9{total_expense:,.2f}, resulting in a net balance of \u20b9{net_balance:,.2f}.",
            "",
            "3. Spending Pattern",
            f"   - Highest Spending Category: {highest_category}",
            f"   - Highest Spending Amount: \u20b9 {highest_spending_amount:,.2f}",
            f"   - Most Frequent Merchant: {highest_merchant}",
            f"   - Most Active Day: {patterns.get('most_active_day', 'Unknown')}",
            f"   - Most Active Month: {patterns.get('most_active_month', 'Unknown')}",
            f"   - Average Transaction Value: \u20b9 {summary.get('average_transaction_value', 0):,.2f}",
            "",
            "   The highest spending category and merchant demonstrate where the largest portion of expenses were concentrated.",
            "",
            "4. Financial Health",
            f"   - Saving Percentage: {summary.get('saving_percentage', 0):.2f}%",
            f"   - Expense Percentage: {summary.get('expense_percentage', 0):.2f}%",
            f"   - Income/Expense Ratio: {summary.get('income_expense_ratio', 'N/A')}",
            "",
            "   The savings percentage reflects how much income remains after expenses. The income/expense ratio indicates whether the cash flow is positive.",
            "",
            "5. Key Insights",
            f"   - The user completed {total_txns} transactions during the period.",
            f"   - Income exceeded expenses by \u20b9 {net_balance:,.2f}.",
            f"   - The highest spending amount was \u20b9 {summary.get('largest_expense', 0):,.2f}.",
            f"   - The most frequent merchant was {highest_merchant}.",
            f"   - The most active month was {patterns.get('most_active_month', 'Unknown')}",
            f"   - The average transaction value was \u20b9 {summary.get('average_transaction_value', 0):,.2f}.",
            "",
            "6. Recommendations",
        ]

        recommendations = self.budget_recommendations()
        if recommendations:
            for recommendation in recommendations:
                report_lines.append(f"   - {recommendation}")
        else:
            report_lines.append("   No financial recommendations are available.")

        report_lines.extend([
            "",
            "7. Overall Financial Status",
        ])

        saving_percentage = summary.get('saving_percentage', 0)
        income_expense_ratio = summary.get('income_expense_ratio')
        if income_expense_ratio is None:
            status = "Average"
        elif income_expense_ratio < 1:
            status = "Needs Improvement"
        elif saving_percentage >= 25 and income_expense_ratio >= 1.2:
            status = "Good"
        elif saving_percentage >= 20 and income_expense_ratio >= 1.0:
            status = "Good"
        else:
            status = "Average"

        report_lines.extend([
            f"   - Overall Financial Status: {status}",
            "",
            "Report generated from the analyzed transaction data.",
        ])

        return "\n".join(report_lines)


    # -- Final entry point: run everything ---------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run every analysis step and combine the results into the single
        clean dictionary `app.py` should consume.

        Why it exists:
            This is the one public method most callers will use — it
            saves `app.py` from having to call ten methods separately.

        Receives:
            (nothing extra — uses everything computed in __init__)

        Returns:
            {
                "validation": {...},
                "financial_summary": {...},
                "category_analysis": {...},
                "merchant_analysis": {...},
                "payment_analysis": {...},
                "weekly_summary": [...],
                "monthly_summary": [...],
                "daily_trends": {...},
                "spending_patterns": {...},
                "chart_data": {...},
                "top_5_expenses": [...],
                "top_5_income": [...],
            }

        Complexity:
            Time O(n log n) overall (dominated by the sorts inside
            merchant_analysis / top_expenses / top_income).
            Space O(n) for the parsed transaction cache plus O(k) for
            each small summary structure.
        """
        return {
            "validation": self.validate_transactions(),
            "financial_summary": self.financial_summary(),
            "category_analysis": self.category_analysis(),
            "merchant_analysis": self.merchant_analysis(),
            "payment_analysis": self.payment_analysis(),
            "weekly_summary": self.weekly_summary(),
            "monthly_summary": self.monthly_summary(),
            "daily_trends": self.daily_trends(),
            "spending_patterns": self.spending_patterns(),
            "chart_data": self.chart_data(),
            "top_5_expenses": self.top_expenses(5),
            "top_5_income": self.top_income(5),
            "budget_recommendations": self.budget_recommendations(),
            "financial_report": self.financial_report(),
        }


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    sample_transactions = [
        {"date": "2026-06-01", "merchant": "Swiggy", "description": "Food order",
         "amount": -450, "currency": "INR", "transaction_type": "Expense",
         "payment_method": "UPI", "reference_number": "REF001", "source": "sample.csv"},
        {"date": "2026-06-02", "merchant": "Salary", "description": "Monthly salary",
         "amount": 55000, "currency": "INR", "transaction_type": "Income",
         "payment_method": "Bank", "reference_number": "REF002", "source": "sample.csv"},
        {"date": "2026-06-05", "merchant": "DMRC", "description": "Metro recharge",
         "amount": -120, "currency": "INR", "transaction_type": "Expense",
         "payment_method": "UPI", "reference_number": "REF003", "source": "sample.csv"},
        {"date": "2026-06-08", "merchant": "Amazon", "description": "Shopping",
         "amount": -2500, "currency": "INR", "transaction_type": "Expense",
         "payment_method": "Card", "reference_number": "REF004", "source": "sample.csv"},
    ]

    engine = FinancialEngine(sample_transactions)
    print(json.dumps(engine.run(), indent=2))