# converter.py —— 通用格式转换
import os

def to_markdown(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".md":
        return open(path, encoding="utf-8").read()
    if ext == ".txt":
        return open(path, encoding="utf-8", errors="ignore").read()
    if ext == ".pdf":
        import pymupdf4llm
        return pymupdf4llm.to_markdown(path)
    if ext in (".docx", ".doc"):
        from docling.document_converter import DocumentConverter
        return DocumentConverter().convert(path).document.export_to_markdown()
    raise ValueError(f"不支持的格式: {ext}")