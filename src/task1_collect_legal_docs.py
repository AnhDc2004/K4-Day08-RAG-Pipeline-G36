"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

DOCUMENTS = [
    {
        "url": "https://takeielts.britishcouncil.org/sites/default/files/ielts_writing_band_descriptors.pdf",
        "filename": "ielts-writing-band-descriptors.pdf",
    },
    {
        "url": "https://ielts.org/cdn/computer-delivered-sample-tests-academic-writing/ielts-academic-writing-example-responses-to-parts-1-and-2-with-band-scores-and-examiner-comments.pdf",
        "filename": "ielts-academic-writing-examiner-comments.pdf",
    },
    {
        "url": "https://ielts.org/cdn/computer-delivered-sample-tests-general-training-writing/ielts-general-training-writing-example-responses-to-parts-1-and-2-with-band-scores-and-examiner-comments.pdf",
        "filename": "ielts-general-training-writing-examiner-comments.pdf",
    },
]


def setup_directory() -> None:
    """Create the landing directory if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def download_file(url: str, filename: str, timeout: int = 60) -> Path:
    """Download one PDF atomically and reject an HTML/error response."""
    setup_directory()
    destination = DATA_DIR / filename
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    response = requests.get(
        url,
        headers={"User-Agent": "K4-RAG-IELTS-Collector/1.0"},
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        raise ValueError(f"Expected PDF but received HTML: {url}")

    with temporary.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                output.write(chunk)

    if temporary.stat().st_size <= 1024 or temporary.read_bytes()[:4] != b"%PDF":
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is not a valid PDF: {url}")

    temporary.replace(destination)
    print(f"Saved: {destination.name} ({destination.stat().st_size:,} bytes)")
    return destination


def collect_documents() -> list[Path]:
    """Download all configured documents and return their local paths."""
    setup_directory()
    paths = []
    for document in DOCUMENTS:
        destination = DATA_DIR / document["filename"]
        if destination.exists() and destination.stat().st_size > 1024:
            print(f"Exists: {destination.name}")
            paths.append(destination)
            continue
        paths.append(download_file(document["url"], document["filename"]))
    return paths


if __name__ == "__main__":
    collect_documents()
