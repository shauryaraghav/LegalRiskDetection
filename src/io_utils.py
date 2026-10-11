"""File reading helpers shared by the Streamlit pages."""
import pdfplumber


def read_upload(uploaded) -> str:
    """Read text from an uploaded PDF or TXT file object (with a .name attribute)."""
    if uploaded.name.lower().endswith(".pdf"):
        pages = []
        with pdfplumber.open(uploaded) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")      # None on image-only pages
        return "\n".join(pages)
    return uploaded.read().decode("utf-8", errors="ignore")


import re as _re


def md_escape(text: str) -> str:
    """Escape Markdown so untrusted contract text can't render links, images or formatting."""
    return _re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$&])", r"\\\1", str(text))
