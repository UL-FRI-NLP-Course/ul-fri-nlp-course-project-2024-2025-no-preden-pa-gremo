from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Union, List
import re
#import nltk
#nltk.download('punkt')
def calculate_bleu(generated_report: str, optimal_report: Union[str, List[str]]) -> float:
    # Simple tokenization: lowercase and split on spaces and punctuation
    tokenize = lambda text: re.findall(r'\b\w+\b', text.lower())

    candidate = tokenize(generated_report)

    if isinstance(optimal_report, str):
        references = [tokenize(optimal_report)]
    else:
        references = [tokenize(ref) for ref in optimal_report]

    # Using SmoothingFunction().method1 is a common choice
    smoothing = SmoothingFunction().method1

    bleu_score = sentence_bleu(references, candidate, smoothing_function=smoothing)
    return bleu_score