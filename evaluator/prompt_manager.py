from enum import Enum, auto
from typing import Dict, Any

class BasePrompt(Enum):
    """Base class for prompt enums with template and output formatting"""
    @property
    def template(self) -> str:
        return self.value['template']
    
    @property
    def criteria(self) -> str:
        return self.value.get('criteria', '')
    
    @property
    def formatter(self) -> str:
        return self.value['formatter']
    
    @classmethod
    def get_prompt_type(cls, name: str) -> 'BasePrompt':
        return cls[name.upper()]

class EvaluationType(BasePrompt):
    """Enumeration of different evaluation prompt types with JSON formatting"""
    RELEVANCE = {
        'template': (
            "Evaluate the relevance of the answer to the question and context.\n"
            "Question: {question}\nContext: {context}\nAnswer: {answer}\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Does the answer directly address the question?\n"
            "2. Is the answer supported by the provided context?\n"
            "3. Does the answer stay focused on the key points?"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- relevance_score (float between 0-1)\n"
            "- reasons (array of 3 short strings)\n"
            "- confidence (float between 0-1)\n"
            "Example:\n"
            "```json\n"
            '{"relevance_score": 0.85, "reasons": ["Directly addresses question", '
            '"Uses context effectively", "Stays focused"], "confidence": 0.92}\n'
            "```"
        )
    }
    
    COHERENCE = {
        'template': (
            "Assess the coherence and clarity of the answer.\n"
            "Question: {question}\nAnswer: {answer}\n"
            "Consider these aspects: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Logical flow of ideas\n2. Grammatical correctness\n"
            "3. Readability and structure\n4. Consistency within the answer"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- coherence_score (float between 0-1)\n"
            "- strengths (array of 2 short strings)\n"
            "- weaknesses (array of 2 short strings)\n"
            "Example:\n"
            "```json\n"
            '{"coherence_score": 0.78, "strengths": ["Clear structure", "Good transitions"], '
            '"weaknesses": ["Some run-on sentences", "Abrupt conclusion"], "confidence": 0.88}\n'
            "```"
        )
    }
    
    FACTUAL_ACCURACY = {
        'template': (
            "Evaluate the factual accuracy based on the provided context.\n"
            "Context: {context}\nAnswer: {answer}\n"
            "Accuracy criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Alignment with contextual facts\n2. Absence of contradictions\n"
            "3. Support from authoritative sources (if applicable)"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- accuracy_score (float between 0-1)\n"
            "- supported_claims (array of strings)\n"
            "- unsupported_claims (array of strings)\n"
            "Example:\n"
            "```json\n"
            '{"accuracy_score": 0.92, "supported_claims": ["Climate change drivers", '
            '"CO2 impact"], "unsupported_claims": ["Mention of solar flares"], '
            '"confidence": 0.95}\n'
            "```"
        )
    }

    FACTUAL_CORRECTNESS = {
    'template': (
        "Evaluate the factual correctness of the generated answer compared to the golden (ground truth) answer.\n"
        "Golden Answer: {golden_answer}\n"
        "Generated Answer: {answer}\n"
        "Consider these criteria: {criteria}\n\n"
        "{formatter}"
    ),
    'criteria': (
        "1. Identify factual statements in both the golden answer and the generated answer.\n"
        "2. Classify statements as:\n"
        "   - True Positives (TP): Present in both answers.\n"
        "   - False Positives (FP): Present in the generated answer but not in the golden answer.\n"
        "   - False Negatives (FN): Present in the golden answer but missing in the generated answer.\n"
        "3. Ensure factual accuracy without adding or omitting key facts."
    ),
    'formatter': (
        "Respond ONLY with a JSON object containing:\n"
        "- extracted_statements (object with 'golden' and 'generated' arrays of key factual statements)\n"
        "- TP (integer): Number of True Positive statements\n"
        "- FP (integer): Number of False Positive statements\n"
        "- FN (integer): Number of False Negative statements\n"
        "- factual_correctness_score (float between 0-1, calculated as TP / (TP + FP + FN))\n"
        "- reasons (array of 3 short strings explaining the score)\n"
        "Example:\n"
        "```json\n"
        '{\n'
        '  "extracted_statements": {\n'
        '    "golden": ["The Eiffel Tower is in Paris", "It was built in 1889"],\n'
        '    "generated": ["The Eiffel Tower is in Paris", "It was built in 1890"]\n'
        '  },\n'
        '  "TP": 1,\n'
        '  "FP": 1,\n'
        '  "FN": 1,\n'
        '  "factual_correctness_score": 0.33,\n'
        '  "reasons": ["Correct location mentioned", "Incorrect construction year", "Missed one key fact"]\n'
        '}\n'
        "```"
    )
}


class PromptManager:
    """Manages prompt construction with JSON output formatting"""
    
    def __init__(self, default_type: EvaluationType = EvaluationType.RELEVANCE):
        self.default_type = default_type
    
    def build_prompt(
        self,
        answer: str,
        question: str = "",
        context: str = "",
        eval_type: EvaluationType = None,
        **kwargs: Any
    ) -> str:
        """
        Construct an evaluation prompt with JSON formatting instructions
        
        Args:
            question: User question/query
            context: Retrieved context used for generation
            answer: Generated answer to evaluate
            eval_type: Type of evaluation to perform
            kwargs: Additional template parameters
            
        Returns:
            Formatted evaluation prompt with JSON instructions
        """
        eval_type = eval_type or self.default_type
        
        return eval_type.template.format(
            question=question,
            context=context,
            answer=answer,
            criteria=eval_type.criteria,
            formatter=eval_type.formatter,
            **kwargs
        )
    

# Example usage
if __name__ == "__main__":
    # Create prompt manager with default evaluation type
    pm = PromptManager(default_type=EvaluationType.RELEVANCE)
    
    # Build a relevance evaluation prompt
    question = "What causes climate change?"
    context = "Scientific consensus attributes climate change to human activities..."
    answer = "Burning fossil fuels releases greenhouse gases that trap heat."
    
    prompt = pm.build_prompt(
        question=question,
        context=context,
        answer=answer,
        eval_type=EvaluationType.RELEVANCE
    )
    
    print("Relevance Evaluation Prompt:")
    print(prompt)
