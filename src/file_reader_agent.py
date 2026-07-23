from pdf_reader import extract_pdf_to_text
from csv_reader import read_csv_file


def process_file(uploaded_file):
    """Read the uploaded file and return structured output for the app."""

    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            text = extract_pdf_to_text(uploaded_file)
            return {
                "type": "pdf",
                "data": text,
            }

        if uploaded_file.name.lower().endswith(".csv"):
            uploaded_file.seek(0)
            df = read_csv_file(uploaded_file)
            return {
                "type": "csv",
                "data": df,
            }

        return {
            "type": "unknown",
            "data": None,
        }
    except Exception as exc:
        return {
            "type": uploaded_file.name.split('.')[-1].lower() if uploaded_file.name else "unknown",
            "data": None,
            "error": str(exc),
        }
