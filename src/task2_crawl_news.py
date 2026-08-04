"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://ielts.org/news-and-insights/ielts-writing-band-descriptors-and-key-assessment-criteria",
    "https://ielts.org/take-a-test/preparation-resources/sample-test-questions",
    "https://ielts.org/take-a-test/preparation-resources/sample-test-questions/academic-test",
    "https://ielts.org/take-a-test/preparation-resources/sample-test-questions/general-training-test",
    "https://ielts.org/take-a-test/test-types/ielts-academic-test/ielts-academic-format-writing",
]


class _VisibleTextParser(HTMLParser):
    """Small dependency-free extractor for official static HTML pages."""

    SKIPPED_TAGS = {"script", "style", "noscript", "svg", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag in self.SKIPPED_TAGS:
            self._skip_depth += 1
        if tag in {"p", "li", "h1", "h2", "h3", "br"} and self._skip_depth == 0:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in self.SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3"} and self._skip_depth == 0:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)

    def result(self) -> tuple[str, str]:
        lines = []
        for part in " ".join(self.text_parts).splitlines():
            text = " ".join(part.split())
            if text and (not lines or lines[-1] != text):
                lines.append(text)
        return " ".join(self.title_parts).strip(), "\n\n".join(lines)


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_page(url: str, timeout: int = 45) -> dict:
    response = requests.get(
        url,
        headers={"User-Agent": "K4-RAG-IELTS-Collector/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    parser = _VisibleTextParser()
    parser.feed(response.text)
    title, content = parser.result()
    if len(content) < 200:
        raise ValueError(f"Page content is too short; likely blocked or JS-only: {url}")
    return {
        "url": url,
        "title": title or url,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
        "source_kind": "official",
        "topic": "ielts_writing",
    }


async def crawl_article(url: str) -> dict:
    """Fetch one public page and return the stable JSON record contract."""
    return await asyncio.to_thread(_extract_page, url)


async def crawl_all() -> list[Path]:
    """Collect every configured page, failing loudly instead of fabricating text."""
    setup_directory()
    paths = []
    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        destination = DATA_DIR / f"article_{index:02d}.json"
        destination.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(destination)
        print(f"Saved: {destination.name}")
    return paths


if __name__ == "__main__":
    asyncio.run(crawl_all())
