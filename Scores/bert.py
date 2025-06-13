from bert_score import score as bert_scorer
from typing import Tuple

BERT_MODEL_TYPE = 'bert-base-uncased'


def calculate_bert(generated_report: str, optimal_report: str) -> Tuple[float, float, float]:
    candidates = [generated_report]
    references = [optimal_report]

    precision, recall, f1 = bert_scorer(
        candidates,
        references,
        lang="en",
        model_type=BERT_MODEL_TYPE,
        verbose=False
    )

    return precision.item(), recall.item(), f1.item()