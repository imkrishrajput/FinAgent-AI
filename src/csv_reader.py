import pandas as pd

def read_csv_file(uploaded_file):
    """
    Reads an uploaded CSV file and returns a Pandas DataFrame.
    """
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(uploaded_file)
