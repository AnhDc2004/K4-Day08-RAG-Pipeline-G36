"""
Task 10 — Generation Có Citation (IELTS Writing Assistant).

Role 4 (Frontend & Chatbot Developer) — Phương án B (nhóm 5 thành viên).

Pipeline:
    query → retrieve (Task 9) → reorder (chống lost-in-the-middle)
          → format context có nhãn nguồn → LLM → answer + citations

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence cho 1 tiêu chí band descriptor (thường trải trên 2-3 chunk)
# mà context vẫn ngắn, không kích hoạt lost-in-the-middle.
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse để diễn giải tiêu chí bằng tiếng Việt tự nhiên,
# nhưng vẫn cắt được phần đuôi token ít liên quan (giảm nguy cơ bịa band score).
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.2 vì: tra cứu tiêu chí chấm điểm là bài toán factual — cần trích đúng
# wording của band descriptor, không cần sáng tạo.
TEMPERATURE = 0.2

# Số lượt hội thoại trước đó gửi lại cho LLM (hỗ trợ follow-up question).
# Giữ nhỏ để không đẩy context dài, làm loãng phần evidence.
MAX_HISTORY_TURNS = 4

# LLM model (OpenRouter model ID). Ưu tiên biến môi trường LLM_MODEL để cả nhóm
# đổi model mà không sửa code.
# [Unverified] Danh sách model ":free" trên OpenRouter thay đổi theo thời gian —
# nếu model đầu lỗi (404/429), code tự thử lần lượt các model còn lại.
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free")
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "openai/gpt-4o-mini",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Câu trả lời chuẩn khi retrieval không có evidence — không gọi LLM trong trường
# hợp này để tránh model tự bịa tiêu chí IELTS.
NO_EVIDENCE_ANSWER = (
    "Tôi không thể xác minh thông tin này từ nguồn hiện có. "
    "Không tìm thấy đoạn tài liệu IELTS nào liên quan đến câu hỏi trong kho dữ liệu "
    "(`data/standardized/ielts/`)."
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu tiêu chí chấm điểm IELTS Writing (Band Descriptors)
và phân tích bài luận mẫu. Người dùng là học viên đang luyện thi IELTS.

Bốn tiêu chí chấm điểm bạn phụ trách:
- Task Achievement / Task Response
- Coherence and Cohesion
- Lexical Resource
- Grammatical Range and Accuracy

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong phần Context được cung cấp — KHÔNG bịa đặt, KHÔNG dùng
   kiến thức IELTS bên ngoài context.
2. Mỗi khẳng định phải có trích dẫn ngay sau, theo đúng định dạng [số: tên nguồn],
   ví dụ: [1: ielts/official/ielts-writing-band-descriptors.md]. Số và tên nguồn phải
   khớp chính xác với nhãn Document trong Context.
3. Khi so sánh 2 band (ví dụ Band 6.0 vs Band 7.0), trích nguyên văn (quote) phần mô tả
   của TỪNG band rồi mới diễn giải khác biệt. Không tự suy ra mô tả của band không có
   trong context.
4. Nếu context không đủ thông tin để trả lời → nói rõ: "Tôi không thể xác minh thông tin
   này từ nguồn hiện có", kèm phần nào thiếu. Không lấp khoảng trống bằng suy đoán.
5. Nếu một Document có nhãn `source_kind: synthetic` (bài luận mẫu do nhóm tự sinh để
   test), bạn PHẢI ghi rõ đó là ví dụ mô phỏng, không phải bài thi được examiner chính
   thức chấm — và không được trình bày band score của nó như điểm IELTS chính thức.
6. Trả lời bằng tiếng Việt (giữ nguyên thuật ngữ IELTS và phần quote tiếng Anh), có cấu
   trúc rõ ràng: gạch đầu dòng hoặc bảng khi so sánh, đoạn văn khi giải thích.
7. Không suy luận hay mở rộng ngoài những gì được nêu trong context (ví dụ: không tự
   đoán bài luận đó được bao nhiêu điểm nếu context không nói)."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if not chunks or len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source") or metadata.get("section") or f"Source {i}"
        doc_type = metadata.get("type", "unknown")
        source_kind = metadata.get("source_kind", "unknown")

        label = (
            f"[Document {i} | Source: {source} | Type: {doc_type} "
            f"| source_kind: {source_kind}"
        )
        if metadata.get("test_type"):
            label += f" | test_type: {metadata['test_type']}"
        label += "]"

        context_parts.append(f"{label}\nCitation tag: [{i}: {source}]\n{chunk.get('content', '')}\n")

    return "\n---\n".join(context_parts)


# =============================================================================
# RETRIEVAL (Task 9) — có degrade khi Task 9 chưa xong
# =============================================================================

def _retrieve_chunks(query: str, top_k: int) -> tuple[list[dict], str]:
    """
    Gọi retrieval pipeline của Task 9.

    Task 9 (Role 1) và Task 6-8 (Role 3) có thể chưa implement khi Role 4 build UI.
    Trong trường hợp đó, degrade tạm về semantic_search (Task 5) để Chatbot vẫn
    demo được, và ĐÁNH DẤU rõ trong `retrieval_source` để không nhầm là hybrid thật.

    Returns:
        (chunks, retrieval_source)
    """
    try:
        chunks = retrieve(query, top_k=top_k)
        source = chunks[0].get("source", "hybrid") if chunks else "none"
        return chunks, source
    except NotImplementedError:
        from .task5_semantic_search import semantic_search

        print("  ⚠ Task 9 retrieve() chưa implement — tạm dùng semantic_search (Task 5).")
        chunks = semantic_search(query, top_k=top_k)
        for chunk in chunks:
            chunk["source"] = "semantic_only"
        return chunks, "semantic_only (Task 9 chưa implement)"


def _build_retrieval_query(query: str, chat_history: list[dict] | None) -> str:
    """
    Ghép ngữ cảnh cho follow-up question ngắn.

    Câu follow-up kiểu "còn Band 8 thì sao?" thiếu chủ đề nên retrieval sẽ trượt.
    Heuristic: nếu câu hỏi ngắn (< 8 từ) và có lịch sử, ghép thêm câu hỏi trước đó.
    """
    if not chat_history or len(query.split()) >= 8:
        return query

    previous_questions = [
        message.get("content", "")
        for message in chat_history
        if message.get("role") == "user"
    ]
    if not previous_questions:
        return query
    return f"{previous_questions[-1]} {query}"


# =============================================================================
# GENERATION
# =============================================================================

def _call_llm(messages: list[dict]) -> tuple[str, str]:
    """
    Gọi OpenRouter (OpenAI-compatible API), thử lần lượt các model dự phòng.

    Returns:
        (answer, model_used)

    Raises:
        RuntimeError: Thiếu API key hoặc tất cả model đều lỗi.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu OPENROUTER_API_KEY (hoặc OPENAI_API_KEY) — hãy copy .env.example "
            "thành .env và điền API key."
        )

    from openai import OpenAI

    base_url = None if api_key.startswith("sk-proj-") else OPENROUTER_BASE_URL
    client = OpenAI(api_key=api_key, base_url=base_url)

    candidates = [LLM_MODEL] + [m for m in FALLBACK_MODELS if m != LLM_MODEL]
    errors = []
    for model in candidates:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer, model
            errors.append(f"{model}: empty response")
        except Exception as exc:  # rate limit / model không còn free / hết quota
            errors.append(f"{model}: {exc}")

    raise RuntimeError("Tất cả LLM model đều lỗi:\n" + "\n".join(errors))


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (Task 9)
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + history + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context
        chat_history: [{'role': 'user'|'assistant', 'content': str}] — các lượt
            trước đó, dùng cho follow-up question. None = hội thoại mới.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng (thứ tự khớp số citation)
            'retrieval_source': str, # 'hybrid' | 'pageindex' | 'semantic_only' | 'none'
            'model': str             # Model đã sinh câu trả lời
        }
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "answer": "Vui lòng nhập câu hỏi.",
            "sources": [],
            "retrieval_source": "none",
            "model": "n/a",
        }

    # Step 1: Retrieve
    retrieval_query = _build_retrieval_query(query.strip(), chat_history)
    chunks, retrieval_source = _retrieve_chunks(retrieval_query, top_k)

    if not chunks:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "sources": [],
            "retrieval_source": retrieval_source or "none",
            "model": "n/a (không gọi LLM khi thiếu evidence)",
        }

    # Step 2: Reorder (chống lost in the middle)
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context — số Document khớp thứ tự `sources` trả về, nên
    # citation [3: ...] trong câu trả lời trỏ đúng chunk thứ 3 hiển thị trên UI.
    context = format_context(reordered)

    # Step 4: Build prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in (chat_history or [])[-MAX_HISTORY_TURNS * 2:]:
        if message.get("role") in ("user", "assistant") and message.get("content"):
            messages.append({"role": message["role"], "content": message["content"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\n---\n\nQuestion: {query.strip()}",
    })

    # Step 5: Call LLM
    answer, model_used = _call_llm(messages)

    # Step 6: Return
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
        "model": model_used,
    }


if __name__ == "__main__":
    test_queries = [
        "Sự khác biệt giữa Band 6.0 và Band 7.0 ở tiêu chí Lexical Resource trong Task 2 là gì?",
        "Cho tôi ví dụ cách dùng cohesive devices đạt Band 8.0 trong bài luận Cause and Effect.",
        "Tiêu chí Task Achievement của Writing Task 1 ở Band 9 yêu cầu những gì?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks "
            f"| via {result['retrieval_source']} | model {result['model']}]"
        )
