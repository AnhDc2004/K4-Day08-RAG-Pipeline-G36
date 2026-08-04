"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(top_k, int) or top_k <= 0:
        return []

    from .task4_chunking_indexing import get_collection, get_embedding_model

    collection = get_collection()
    if collection is None or collection.count() == 0:
        return []

    model = get_embedding_model()
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for content, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        # Chroma returns cosine distance; convert it to a readable similarity.
        score = max(-1.0, min(1.0, 1.0 - float(distance)))
        output.append({
            "content": content,
            "score": round(score, 4),
            "metadata": metadata or {},
        })
    return sorted(output, key=lambda result: result["score"], reverse=True)


if __name__ == "__main__":
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
