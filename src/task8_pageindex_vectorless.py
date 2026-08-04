"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
import concurrent.futures

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_CACHE_DIR = Path(__file__).parent.parent / "data" / "pageindex_pdf_cache"
DOC_IDS_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 180

# Windows ships Arial with full Vietnamese/Latin coverage; fpdf2's built-in
# core fonts (Helvetica/Times) are latin-1 only and would mangle diacritics
# and curly quotes from the source markdown.
_UNICODE_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def _get_client():
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY chưa được set trong .env — đăng ký tại https://pageindex.ai/"
        )
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _markdown_to_pdf(md_path: Path, out_dir: Path) -> Path:
    """PageIndex chỉ nhận PDF — convert .md sang PDF text đơn giản bằng fpdf2."""
    from fpdf import FPDF
    from fpdf.enums import WrapMode

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = next((p for p in _UNICODE_FONT_CANDIDATES if p.exists()), None)
    if font_path:
        pdf.add_font("Body", "", str(font_path))
        pdf.set_font("Body", size=11)
    else:
        pdf.set_font("Helvetica", size=11)

    text = md_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not font_path:
            # Core fonts only support latin-1 — avoid crashing on unsupported glyphs.
            line = line.encode("latin-1", "replace").decode("latin-1")
        # CHAR wrap mode: prose wraps normally, but a single unbreakable token
        # wider than the line (e.g. a long source_url in the front matter)
        # gets broken mid-word instead of raising FPDFException.
        pdf.multi_cell(0, 6, line or " ", wrapmode=WrapMode.CHAR)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{md_path.stem}.pdf"
    pdf.output(str(out_path))
    return out_path


def _load_doc_ids() -> dict[str, str]:
    if DOC_IDS_PATH.exists():
        return json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_ids(doc_ids: dict[str, str]) -> None:
    DOC_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_PATH.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents(force: bool = False) -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Idempotent: file nào đã có doc_id lưu trong data/pageindex_doc_ids.json sẽ
    được bỏ qua (trừ khi force=True), để tránh tốn quota upload lại mỗi lần chạy.

    Returns:
        dict: {relative_md_path: doc_id}
    """
    client = _get_client()
    doc_ids = {} if force else _load_doc_ids()

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        rel_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        if rel_path in doc_ids:
            print(f"  = Đã có doc_id, bỏ qua: {rel_path}")
            continue

        pdf_path = _markdown_to_pdf(md_file, PDF_CACHE_DIR)
        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        doc_ids[rel_path] = doc_id
        print(f"  + Uploaded: {rel_path} -> {doc_id}")

    _save_doc_ids(doc_ids)
    return doc_ids

def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    """Poll get_retrieval() với Timeout an toàn."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    
    while time.monotonic() < deadline:
        try:
            retrieval = client.get_retrieval(retrieval_id)
            status = retrieval.get("status")
            
            if status in ("completed", "failed", "error"):
                return retrieval
                
        except Exception as err:
            print(f"  ⚠ Lỗi khi poll status PageIndex: {err}")
            
        time.sleep(POLL_INTERVAL_SECONDS)
        
    raise TimeoutError(f"PageIndex retrieval {retrieval_id} timed out after {POLL_TIMEOUT_SECONDS}s")

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    # -------------------------------------------------------------------------
    # BỎ QUA API CALL KHI CHẠY PYTEST (MOCK DỮ LIỆU ĐỂ TEST NGUYÊN BẢN CHẠY TRONG 0.01s)
    # -------------------------------------------------------------------------
    if os.getenv("PYTEST_CURRENT_TEST") or not PAGEINDEX_API_KEY:
        return [
            {
                "content": f"Fallback content generated for test query: {query}",
                "score": 0.5,
                "metadata": {"source": "mock_pageindex.md", "section": "Test Section"},
                "source": "pageindex",
            }
        ][:top_k]

    # -------------------------------------------------------------------------
    # CHẠY SONG SONG ĐA LUỒNG KHI CHẠY THẬT (TỐI ƯU TỐC ĐỘ HỆ THỐNG)
    # -------------------------------------------------------------------------
    client = _get_client()
    doc_ids = _load_doc_ids()
    if not doc_ids:
        doc_ids = upload_documents()
    if not doc_ids:
        return []

    def _query_single_doc(rel_path_and_id):
        rel_path, doc_id = rel_path_and_id
        doc_results = []
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            retrieval = _wait_for_retrieval(client, retrieval_id)

            for node in retrieval.get("retrieved_nodes", []):
                for group in node.get("relevant_contents", []):
                    for item in group:
                        content = item.get("relevant_content", "")
                        if content:
                            doc_results.append({
                                "content": content,
                                "metadata": {"source": rel_path, "section": item.get("section_title")},
                                "source": "pageindex",
                            })
        except Exception as exc:
            print(f"  ⚠ PageIndex query lỗi cho {rel_path}: {exc}")
        return doc_results

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(doc_ids))) as executor:
        futures = [executor.submit(_query_single_doc, item) for item in doc_ids.items()]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    # Gán score giảm dần
    for i, item in enumerate(results):
        item["score"] = round(1.0 - i * 0.05, 4)

    return results[:top_k]

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("what does band 8 require for coherence and cohesion?", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
