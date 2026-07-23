import pandas as pd


def analyze_csv(df):

    numeric_columns = df.select_dtypes(include="number").columns

    analysis = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum(),
        "summary": df.describe(include="all")
    }

    if len(numeric_columns) > 0:

        amount_col = numeric_columns[0]

        analysis["total"] = df[amount_col].sum()
        analysis["average"] = df[amount_col].mean()
        analysis["maximum"] = df[amount_col].max()
        analysis["minimum"] = df[amount_col].min()

    return analysis