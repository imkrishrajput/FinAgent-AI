import fitz  # PyMuPDF


def extract_pdf_to_text(uploaded_file):
    """
    Extract text from an uploaded PDF file.

    Returns an empty string for scanned PDFs or invalid content,
    but does not crash the application.
    """
    try:
        uploaded_file.seek(0)
        pdf = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )
    except Exception as exc:
        raise ValueError(f"Failed to read PDF file: {exc}")

    text = ""
    for page in pdf:
        text += page.get_text()
    pdf.close()

    if not text.strip():
        raise ValueError(
            "The PDF contains no extractable text. It may be a scanned or image-only PDF."
        )

    return text
