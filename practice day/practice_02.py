from pathlib import Path
from pypdf import PdfReader
import re

pdf_path = Path("data/papers/2005.11401v4.pdf")

reader = PdfReader(str(pdf_path))
raw_page = reader.pages[4].extract_text()

def clean_text(text: str) -> str:
    # Collapse multiple spaces/newlines into single spaces where safe
    text = re.sub(r"[ \t]+", " ", text)

    # Fix ligature artifacts pypdf sometimes mangles (ﬁ, ﬂ as separate glyphs)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    return text

cleaned = clean_text(raw_page)

print("--- BEFORE (first 500 chars) ---")
print(raw_page[:500])
print("\n--- AFTER (first 500 chars) ---")
print(cleaned[:500])