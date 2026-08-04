"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def convert_legal_docs() -> list[Path]:
    """Convert every PDF/DOC/DOCX in the legal landing directory."""
    input_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    converter = MarkItDown()
    converted = []
    for filepath in sorted(input_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        print(f"Converting: {filepath.name}")
        result = converter.convert(str(filepath))
        content = (
            "---\n"
            "source_kind: official\n"
            f"source_file: {filepath.name}\n"
            "doc_type: ielts_writing_source\n"
            "---\n\n"
            f"# {filepath.stem.replace('-', ' ').title()}\n\n"
            + result.text_content
        )
        converted.append(_write_markdown(output_dir / f"{filepath.stem}.md", content))
    return converted


def convert_news_articles() -> list[Path]:
    """Convert JSON web records while preserving source and crawl metadata."""
    input_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    converted = []
    for filepath in sorted(input_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        content = data.get("content_markdown") or data.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Missing article content: {filepath}")
        header = (
            "---\n"
            f"title: {data.get('title', 'Unknown')}\n"
            "doc_type: ielts_writing_web_resource\n"
            f"source_kind: {data.get('source_kind', 'unknown')}\n"
            f"source_url: {data.get('url', 'N/A')}\n"
            f"date_crawled: {data.get('date_crawled', 'N/A')}\n"
            "---\n\n"
            f"# {data.get('title', 'Unknown')}\n\n"
        )
        converted.append(_write_markdown(output_dir / f"{filepath.stem}.md", header + content))
    return converted


def convert_all() -> None:
    print("Task 3: Convert to Markdown")
    legal = convert_legal_docs()
    news = convert_news_articles()
    print(f"Done: {len(legal)} legal documents, {len(news)} web resources")


if __name__ == "__main__":
    convert_all()
