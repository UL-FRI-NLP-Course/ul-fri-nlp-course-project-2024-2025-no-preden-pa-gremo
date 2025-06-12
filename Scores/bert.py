from bert_score import score as bert_scorer
import torch
from typing import Tuple

BERT_MODEL_TYPE = 'bert-base-uncased'
def calculate_bert(generated_report: str, optimal_report: str) -> Tuple[float, float, float]:
    candidates = [generated_report]
    references = [optimal_report]

    # By explicitly setting the model_type, we ensure that the same
    # model from Hugging Face is used every time.
    precision, recall, f1 = bert_scorer(
        candidates,
        references,
        lang="en",
        model_type=BERT_MODEL_TYPE,
        verbose=False # Set to True to see model loading progress
    )

    return precision.item(), recall.item(), f1.item()