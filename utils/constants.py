from enum import Enum


LLM_RESPONSE = "llm_response"
PROMPT = "prompt"

SYNTHETIC_MISTAKE_TYPES = ['Entity_Error', 'Negation', 'Missing_Information', 'Out_of_Reference', 'Numerical_Error']

EVAL_COL_MAP = {
    "Correct": "Paraphrased",
    "Incorrect": "Incorrect",
    "gold": "generated_answer",
}

class RAGBENCH_COL_NAMES(Enum):
    GOLDEN_ANSWER = "response"
    QUESTION = "question"
    CONTEXT = "documents"
    GOLDEN_ANSWER_SENTENCES = "response_sentences"
    CONTEXT_SENTENCES = "documents_sentences"
    GENERATED_ANSWER = "generated_answer"
    KEY_POINTS = "key_points"

