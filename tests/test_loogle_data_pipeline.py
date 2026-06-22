from __future__ import annotations

from hyde.loogle.chunking import chunk_documents_grouped_records, chunk_text
from hyde.loogle.dataset import parse_loogle_rows, select_frozen_subset
from hyde.loogle.labeling import build_retrieval_examples
from hyde.loogle.novelhopqa import parse_novelhop_rows
from hyde.loogle.qasper import parse_qasper_rows


def test_nested_and_flat_loogle_rows_have_stable_query_ids():
    rows = [
        {
            "doc_id": "nested",
            "context": "Nested document text.",
            "qa_pairs": '[{"Q": "Nested question?", "A": "answer", "S": "document text"}]',
        },
        {
            "document_id": "flat",
            "document": "Flat document text.",
            "question": "Flat question?",
            "answer": "answer",
            "evidence": "Flat document",
        },
    ]
    documents, qa_entries = parse_loogle_rows(rows)
    assert list(documents) == ["nested", "flat"]
    assert [row["id"] for row in qa_entries] == [0, 1]
    assert qa_entries[0]["retrieval_spans"] == ["document text"]
    assert qa_entries[1]["document_id"] == "flat"


def test_frozen_subset_uses_manifest_order_and_filters_questions():
    documents = {"d2": "two", "d1": "one", "extra": "ignored"}
    qa_entries = [
        {"id": 0, "document_id": "d1"},
        {"id": 1, "document_id": "extra"},
        {"id": 2, "document_id": "d2"},
    ]
    selected_docs, selected_qa, metadata = select_frozen_subset(
        documents, qa_entries, {"name": "fixture", "document_ids": ["d1", "d2"]}
    )
    assert list(selected_docs) == ["d1", "d2"]
    assert [row["id"] for row in selected_qa] == [0, 2]
    assert metadata["limited"] is False


def test_sentence_aware_500_word_chunking_and_stable_ids():
    first = " ".join(f"a{i}" for i in range(300)) + "."
    second = " ".join(f"b{i}" for i in range(250)) + "."
    chunks = chunk_text(f"{first} {second}", chunk_size=500, chunk_overlap=0)
    assert [len(text.split()) for text, _, _ in chunks] == [300, 250]
    grouped = chunk_documents_grouped_records([f"{first} {second}"], doc_ids=["doc"], chunk_size=500)
    assert [chunk.chunk_id for chunk in grouped[0]] == ["doc:0", "doc:1"]
    assert [(chunk.token_start, chunk.token_end) for chunk in grouped[0]] == [(0, 300), (300, 550)]


def test_labeling_builds_gold_and_cross_chunk_silver_group():
    text = "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa."
    chunks = chunk_documents_grouped_records([text], doc_ids=["d1"], chunk_size=5)[0]
    examples = build_retrieval_examples(
        [
            {"id": 0, "document_id": "d1", "question": "Gold?", "retrieval_spans": ["Alpha beta"]},
            {"id": 1, "document_id": "d1", "question": "Silver?", "retrieval_spans": ["epsilon Zeta"]},
        ],
        {"d1": chunks},
    )
    assert examples[0].gold_chunk_ids == ["d1:0"]
    assert examples[0].silver_chunk_ids == []
    assert examples[1].gold_chunk_ids == []
    assert examples[1].silver_chunk_ids == ["d1:0", "d1:1"]
    assert examples[1].silver_chunk_groups == [["d1:0", "d1:1"]]


def test_window_labeling_matches_all_chunks_overlapped_by_context():
    chunks = chunk_documents_grouped_records(
        ["alpha beta gamma delta. epsilon zeta eta theta."], doc_ids=["book"], chunk_size=4
    )[0]
    examples = build_retrieval_examples(
        [
            {
                "id": "hop_2:1",
                "document_id": "book",
                "question": "What crosses the boundary?",
                "retrieval_span_mode": "window",
                "retrieval_spans": ["gamma delta epsilon zeta"],
            }
        ],
        {"book": chunks},
    )
    assert examples[0].silver_chunk_groups == [["book:0", "book:1"]]


def test_qasper_parser_preserves_evidence_and_answer_variants():
    documents, qa_entries = parse_qasper_rows(
        [
            {
                "id": "paper-1",
                "full_text": {"paragraphs": [["First paragraph."], ["Second paragraph."]]},
                "qas": {
                    "question": ["What happened?"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": ["an answer"],
                                    "evidence": ["First paragraph."],
                                }
                            ]
                        }
                    ],
                },
            }
        ]
    )
    assert documents == {"paper-1": "First paragraph.\nSecond paragraph."}
    assert qa_entries[0]["answers"] == ["an answer"]
    assert qa_entries[0]["retrieval_spans"] == ["First paragraph."]


def test_novelhop_parser_uses_hop_ids_and_window_mode():
    qa_entries, missing = parse_novelhop_rows(
        {
            "hop_1": [
                {
                    "id": "question-1",
                    "book": "The Example Book",
                    "context": "A gold context window.",
                    "question": "Example question?",
                    "answer": "Example answer.",
                }
            ]
        },
        title_to_doc={"the example book": "B01"},
    )
    assert not missing
    assert qa_entries[0]["id"] == "hop_1:question-1"
    assert qa_entries[0]["document_id"] == "B01"
    assert qa_entries[0]["retrieval_span_mode"] == "window"
