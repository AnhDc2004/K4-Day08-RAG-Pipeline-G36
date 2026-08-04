from group_project.evaluation.eval_pipeline import load_golden_dataset, run_evaluation


def test_golden_dataset_has_minimum_questions():
    dataset = load_golden_dataset()
    assert len(dataset) >= 15
    for item in dataset[:3]:
        assert {"question", "expected_answer", "expected_context"}.issubset(item.keys())


def test_run_evaluation_returns_summary():
    dataset = load_golden_dataset()[:3]
    result = run_evaluation(dataset, use_ragas=False)
    assert "overall" in result
    for metric in ["faithfulness", "answer_relevance", "context_recall", "context_precision"]:
        assert metric in result["overall"]
