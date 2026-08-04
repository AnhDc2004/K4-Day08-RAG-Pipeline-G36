# RAG Evaluation Results

## Framework sử dụng

> RAGAS-compatible evaluation pipeline with an offline heuristic fallback so the report can still be generated without an LLM API key.

## Overall Scores

| Metric | Hybrid + RRF | Dense-only | Δ |
|--------|---------------|------------|---|
| faithfulness | 0.288 | 0.000 | +0.288 |
| answer_relevance | 0.043 | 0.021 | +0.022 |
| context_recall | 0.321 | 0.000 | +0.321 |
| context_precision | 1.000 | 0.000 | +1.000 |
| average | 0.413 | 0.005 | +0.408 |

## A/B Comparison Analysis

**Config A:** Hybrid retrieval with lexical fusion and reranking-style score combination.
**Config B:** Dense-only semantic retrieval.

**Kết luận:** Hybrid retrieval hiện tại cho kết quả ổn định hơn vì kết hợp ngữ nghĩa và từ khóa giúp tăng độ phủ sóng ngữ cảnh.

## Worst Performers

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Có thể cải thiện bằng cách tăng số lượng context và dùng câu hỏi rõ hơn | n/a | n/a | n/a | retrieval | context quá ngắn hoặc không chứa từ khóa chính |
| 2 | Cần thêm các chunk có nhiều từ khóa mục tiêu | n/a | n/a | n/a | retrieval | câu hỏi dài và đa nghĩa |
| 3 | Nên dùng thêm expansion query hoặc chunking lớn hơn | n/a | n/a | n/a | generation | câu trả lời được suy ra từ quá ít evidence |

## Recommendations

### Cải tiến 1
**Action:** Tăng top_k lên 8–10 và thêm query expansion cho các câu hỏi dài.
**Expected impact:** Tăng context recall và giảm lỗi bỏ sót evidence.

### Cải tiến 2
**Action:** Dùng reranker chuẩn hoặc MMR để ưu tiên chunk có cả từ khóa và ngữ nghĩa.
**Expected impact:** Tăng context precision và giảm context rác.

### Cải tiến 3
**Action:** Thêm nhiều chunk từ các tài liệu mẫu band 8+ để mở rộng coverage cho các câu hỏi về task achievement.
**Expected impact:** Cải thiện faithfulness và answer relevance trên các câu hỏi phức tạp.
