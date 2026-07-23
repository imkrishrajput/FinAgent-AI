import re
import pandas as pd


def extract_transactions(text):

    transactions = []

    # Find amounts like ₹450 or Rs.450
    amounts = re.findall(r"(?:₹|Rs\.?)\s?([\d,]+(?:\.\d{2})?)", text)

    # Find dates like 01/06/2026 or 01-06-2026
    dates = re.findall(r"\d{2}[/-]\d{2}[/-]\d{4}", text)

    length = min(len(amounts), len(dates))

    for i in range(length):

        transactions.append(
            {
                "Date": dates[i],
                "Amount": float(amounts[i].replace(",", ""))
            }
        )

    return pd.DataFrame(transactions)