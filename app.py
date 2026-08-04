"""
IELTS Writing Assistant — RAG Chatbot.

Trợ lý tra cứu tiêu chí chấm điểm IELTS Writing (Band Descriptors) và phân tích
bài luận mẫu Band 8.0+.

Role 4 (Frontend & Chatbot Developer) — Streamlit UI + Task 10 (Generation có Citation).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"

# Nhãn hiển thị cho từng nguồn retrieval (Task 9 trả về trong key 'source')
RETRIEVAL_LABELS = {
    "hybrid": ("✅", "Hybrid (Semantic + BM25 → RRF)"),
    "pageindex": ("🔁", "PageIndex Fallback (vectorless)"),
    "semantic_only": ("⚠️", "Semantic-only — Task 9 chưa implement"),
    "none": ("❌", "Không tìm thấy tài liệu liên quan"),
}

SOURCE_KIND_LABELS = {
    "official": "🏛️ Official (British Council / IDP / ielts.org)",
    "synthetic": "🧪 Synthetic — bài mẫu mô phỏng, KHÔNG do examiner chấm",
    "mixed": "🗂️ Mixed",
    "unknown": "❔ Unknown",
}

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="IELTS Writing Assistant — RAG Chatbot",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# HELPERS
# =============================================================================

def render_sources(sources: list[dict]) -> None:
    """Hiển thị danh sách tài liệu tham khảo, số thứ tự khớp citation trong câu trả lời."""
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=False):
        st.caption("Số [n] dưới đây khớp với số trong citation của câu trả lời.")
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata") or {}
            source_name = meta.get("source") or meta.get("section") or "Unknown"
            doc_type = meta.get("type", "unknown")
            source_kind = meta.get("source_kind", "unknown")
            score = src.get("score", 0.0)

            st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
            st.caption(SOURCE_KIND_LABELS.get(source_kind, f"❔ {source_kind}"))
            if source_kind == "synthetic":
                st.warning(
                    "Đoạn này thuộc bộ bài luận mô phỏng do nhóm tự sinh để test RAG — "
                    "band score trong đó không phải điểm IELTS chính thức.",
                    icon="🧪",
                )
            if meta.get("source_url"):
                st.caption(f"🔗 {meta['source_url']}")
            st.text(src.get("content", "")[:400] + ("..." if len(src.get("content", "")) > 400 else ""))
            st.divider()


def render_retrieval_badge(retrieval_source: str, model: str) -> None:
    """Hiển thị nhánh retrieval đã chạy (hybrid / fallback) và model đã sinh câu trả lời."""
    key = retrieval_source.split()[0] if retrieval_source else "none"
    icon, label = RETRIEVAL_LABELS.get(key, ("ℹ️", retrieval_source))
    st.caption(f"{icon} Retrieval: **{label}** · 🤖 Model: `{model}`")


def pipeline_status() -> list[tuple[str, bool, str]]:
    """Kiểm tra nhanh các thành phần pipeline đã sẵn sàng chưa (hiện trên sidebar)."""
    checks = []

    has_key = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    checks.append(("API key (LLM)", has_key, "Thiếu .env → copy .env.example thành .env"))

    md_files = list(STANDARDIZED_DIR.rglob("*.md")) if STANDARDIZED_DIR.exists() else []
    checks.append((f"Corpus markdown ({len(md_files)} file)", bool(md_files), "Chạy Task 3"))

    checks.append(("Vector store chroma_db/", CHROMA_DIR.exists(), "Chạy Task 4 để index"))

    # Chỉ kiểm tra import được module — hàm còn raise NotImplementedError sẽ lộ ra
    # khi thực sự gọi (và được báo trong khung chat).
    for task_name, module_path, func_name in [
        ("Task 9 retrieve()", "src.task9_retrieval_pipeline", "retrieve"),
        ("Task 10 generation", "src.task10_generation", "generate_with_citation"),
    ]:
        try:
            module = __import__(module_path, fromlist=[func_name])
            ready = callable(getattr(module, func_name))
        except Exception:
            ready = False
        checks.append((task_name, ready, "Chưa import được module"))

    return checks


# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("📝 IELTS Writing Assistant")
    st.caption(
        "Tra cứu tiêu chí chấm điểm IELTS Writing (Task Achievement, Coherence & Cohesion, "
        "Lexical Resource, Grammatical Range & Accuracy) và phân tích bài luận mẫu Band 8.0+."
    )

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Sự khác biệt giữa Band 6.0 và Band 7.0 ở tiêu chí Lexical Resource trong Task 2 là gì?",
        "Cho tôi ví dụ cách dùng cohesive devices đạt Band 8.0 trong bài luận Cause and Effect.",
        "Tiêu chí Task Achievement của Writing Task 1 ở Band 9 yêu cầu những gì?",
        "Coherence and Cohesion Band 7 khác Band 8 ở điểm nào?",
        "Examiner nhận xét gì về lỗi ngữ pháp trong bài mẫu Band 6.5?",
        "Grammatical Range and Accuracy: Band 5 mắc những lỗi điển hình nào?",
    ]
    for i, s in enumerate(suggestions):
        if st.button(s, use_container_width=True, key=f"sug_{i}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider(
        "Số chunks retrieval (top_k)", 3, 10, 5,
        help="Nhiều chunk hơn = nhiều evidence hơn nhưng dễ bị lost-in-the-middle.",
    )
    use_memory = st.toggle(
        "Nhớ ngữ cảnh hội thoại (follow-up)", value=True,
        help="Gửi lại các lượt hỏi đáp trước cho LLM để trả lời câu hỏi tiếp nối.",
    )
    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    with st.expander("🔍 Trạng thái pipeline"):
        for name, ready, hint in pipeline_status():
            st.markdown(f"{'✅' if ready else '❌'} {name}")
            if not ready:
                st.caption(f"↳ {hint}")

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption(
        "Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback "
        "→ Reordering (front + back[::-1]) → LLM Generation có Citation"
    )

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("📝 IELTS Writing Assistant")
st.caption("Hỏi đáp về tiêu chí chấm điểm IELTS Writing & bài luận mẫu — câu trả lời luôn kèm trích dẫn nguồn")

if not st.session_state.messages:
    st.info(
        "Chọn một câu hỏi gợi ý ở sidebar, hoặc nhập câu hỏi của bạn. "
        "Trợ lý chỉ trả lời dựa trên tài liệu trong `data/standardized/ielts/` — "
        "nếu không đủ căn cứ, nó sẽ nói rõ là không xác minh được.",
        icon="👋",
    )

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("retrieval_source"):
                render_retrieval_badge(msg["retrieval_source"], msg.get("model", "n/a"))
            render_sources(msg.get("sources", []))

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi về tiêu chí chấm điểm hoặc bài luận mẫu IELTS...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Lịch sử TRƯỚC câu hỏi hiện tại (dùng cho follow-up)
    chat_history = (
        [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        if use_memory else []
    )

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu tài liệu IELTS và tổng hợp câu trả lời..."):
            retrieval_source = "none"
            model = "n/a"
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(
                    query, top_k=top_k, chat_history=chat_history
                )
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", "none")
                model = response.get("model", "n/a")

            except NotImplementedError as e:
                answer = (
                    "⚠️ **Pipeline chưa hoàn thiện.** Một task trong chuỗi retrieval "
                    f"chưa được implement: `{e}`.\n\n"
                    "Xem panel **Trạng thái pipeline** ở sidebar để biết phần còn thiếu."
                )
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            st.markdown(answer)
            render_retrieval_badge(retrieval_source, model)
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "model": model,
    })
