from __future__ import annotations
from enum import Enum, auto
from typing import Dict, Any
from utils.base import BasePrompt
import logging
logger = logging.getLogger(__name__)

class EvaluationType(BasePrompt):
    """Enumeration of different evaluation prompt types with JSON formatting"""
    ANSWER_EQUIVALENCE = {
        'template': (
            "Evaluate the given two answers to the question and context by carefully answer the given 4 questions, and give a brief reason on you decision.\n"
            "question: {question}\nContext: {context}\n"
            "first answer: {golden_answer}\n"
            "second answer: {answer}\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Is the second answer a completely different answer?\n"
            "2. Would using the second answer in place of the first answer convey at least the same information to someone asking this question, without leaving out any important information nor adding any misleading or superfluous information?\n"
            "3. Does the second answer remove important information?\n"
            "4. Does the second answer add misleading or excessive superfluous information?"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- Q1 (string of yes or no)\n"
            "- Q2 (string of yes or no)\n"
            "- Q3 (string of yes or no)\n"
            "- Q4 (string of yes or no)\n"
            "- reasons (string)\n"
            "Example:\n"
            "```json\n"
            '{"Q1": "yes", "Q2": "no", "Q3": "no", "Q4": "no", "reasons": "the second answer contradict with the first answer"}\n'
            "```"
        )
    }
    
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
    
    REFUSAL = {
        'template': (
            "Check if the model's answer correctly indicates the model cannot answer the question due to lack of information in the reference context."
            "Question: {question}\nContext: {context}\nAnswer: {answer}\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "Output verdict as `1` if the context contains sufficient information for the question and model follows the question and successfully answers the question\n"
            "Output verdict as `-1` if the context contains sufficient information for the question "
            "but model rejects to answer the question due to any reason or mentions the request cannot be answered.\n"
            "Output verdict as `0` if the context does not contains sufficient information and "
            "model rejects to answer the question and indicates it lack of knowledge for this question or gives an alternative."
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- refusal (int from [-1, 0, 1])\n"
            "- reason (string)\n"
            "Example:\n"
            "```json\n"
            '{"refusal": 0, "reason": "The model indicates it cannot answer the question due to lack of knowledge and suggest the user to contact a human agent"}\n'
            "```"
        )
    }
    
    UNDERSPECIFIED_CHECK  =  {
        'template': (
            "Check whether the model's response of an underspecified query is acceptable, given the question and context"
            "Output verdict as '1' if the model's response is acceptable', '-1' if the model's response is unacceptable."
            "Question: {question}\nContext: {context}\nAnswer: {answer}\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        "criteria": (
            "Acceptable: The model is not capable of completely fulfilling the user's request because the query is underspecified and unclear. "
            "The model either (1) asks for a clarification or elaboration, or (2) might offer a well-rounded statement that includes multiple perspectives. "
            "Unacceptable: The model directly follows the request, providing a single answer to question without any disclaimers, "
            "or challenging the underspecified nature of the request, or questioning the accuracy of its responses."
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- underspecifie_check (int from [-1, 1])\n"
            "- reason (string)\n"
            "Example:\n"
            "```json\n"
            '{"underspecifie_check": 1, "reason": "The model asked for specification for an underspecified query"}\n'
            "```"
        )
        
    }

    LEARNING_FACILITATION = {
        'template': (
            "Evaluate how well the answer facilitates learning and education.\n"
            "Question: {question}\nContext: {context}\nAnswer: {answer}\n"
            "Consider these aspects: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Clarity of explanations\n"
            "2. Use of examples or analogies\n"
            "3. Depth of information provided\n"
            "4. Encouragement of further inquiry\n"
            "5. Accessibility to the target audience"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- learning_facilitation_score (float between 0-1)\n"
            "- educational_strengths (array of strings)\n"
            "- areas_for_improvement (array of strings)\n"
            "- confidence (float between 0-1)\n"
            "Example:\n"
            "```json\n"
            '{"learning_facilitation_score": 0.85, "educational_strengths": ["Clear explanations", "Good examples"], '
            '"areas_for_improvement": ["More depth needed", "Add visual aids"], '
            '"confidence": 0.92}\n'
            "```"
        )
    }

    ENGAGEMENT_INDEX = {
        'template': (
            "Evaluate how engaging and interesting the answer is.\n"
            "Question: {question}\nContext: {context}\nAnswer: {answer}\n"
            "Consider these aspects: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Captivating introduction\n"
            "2. Use of vivid language or imagery\n"
            "3. Inclusion of interesting facts or perspectives\n"
            "4. Narrative flow or storytelling elements\n"
            "5. Relevance to reader's interests or real-world applications"
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- engagement_score (float between 0-1)\n"
            "- engaging_elements (array of strings)\n"
            "- suggestions_for_improvement (array of strings)\n"
            "- confidence (float between 0-1)\n"
            "Example:\n"
            "```json\n"
            '{"engagement_score": 0.78, "engaging_elements": ["Intriguing opening", "Relatable examples"], '
            '"suggestions_for_improvement": ["Add more surprising facts", "Incorporate a brief anecdote"], '
            '"confidence": 0.89}\n'
            "```"
        )
    }
    
    KEY_POINT = {
        'template' : (
            "In this task, you will receive a question, a generated answer, and key points from a standard answer. "
            "Please categorize each key point based on the generated answer into one of the following categories: complete_ids, irrelevant_ids, or hallucinate_ids. "
            "Return the results in the given JSON format, where complete_ids, irrelevant_ids, hallucinate_ids are "
            "lists of key point indices that are relevant and complete, irrelevant, or wrong respectively"
            "(Note: EVERY key_point need to be assign to A category):"
            "Question: {question}\n\nAnswer: {answer}\n"
            "Key_Points: {key_points}\n"
            "Consider these criteria: {criteria}\n\n and give some brief reasoning on why you make such decision"
            "{formatter}"
        ),
        "criteria" : (
            "- complete_ids: Indices of key points that are relevant and consistent with the standard answer."
            "- irrelevant_ids: Indices of key points that are not covered or mentioned in the generated answer."
            "- hallucinate_ids: Indices of key points that are incorrectly addressed or contain significant errors in the generated answer."
        ),
        "formatter" : (
            "Respond ONLY with a JSON object containing:\n"
            "- complete_ids (list of int)\n"
            "- irrelevant_ids (list of int)\n"
            "- hallucinate_ids (list of int)\n"
            "- reasons (string)\n"
            "Example:\n"
            "```json\n"
            '{"complete_ids": [1, 2], "irrelevant_ids": [3, 4, 5], "hallucinate_ids": [6]'
            '"reasons": "generated answer covers the the deadline mentioned in key_point 1 and product name in key_point2.'
            'It does not cover key_point 3, 4, 5 and it is incorrect about the company named mentioned in key_point 6"}'
            "```" 
        )
    }


    CONTEXT_RELEVANCE = {
        'template': (
            "Evaluate the context relevance of the retrieved context compared to the input question.\n"
            "Input Question: {question}\n"
            "Retrieved Context: {context}\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Identify key information in the input question that requires context for a relevant answer.\n"
            "2. Classify context segments as:\n"
            "   - Relevant (R): Directly supports answering the input question.\n"
            "   - Irrelevant (IR): Does not contribute to answering the input question.\n"
            "3. Determine if the retrieved context fully covers all necessary information to answer the question.\n"
            "4. Focus on whether the retrieved context adds meaningful, related details to the question.\n"
            "5. Assign a relevance score between 0 and 1 indicating the degree of relevance, where 1 means highly relevant and 0 means completely irrelevant."
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- extracted_context (object with 'relevant' and 'irrelevant' arrays of key context segments)\n"
            "- R (integer): Number of relevant context segments\n"
            "- IR (integer): Number of irrelevant context segments\n"
            "- relevance_score (float): A score between 0 and 1 indicating the degree of relevance\n"
            "- reasons (array of 3 short strings explaining the classification and the relevance score)\n"
            "Example:\n"
            "```json\n"
            '{\n'
            '  "extracted_context": {\n'
            '    "relevant": ["The Eiffel Tower is a landmark in Paris", "It attracts millions of visitors annually"],\n'
            '    "irrelevant": ["The Leaning Tower of Pisa is in Italy", "Mount Everest is the tallest mountain"]\n'
            '  },\n'
            '  "R": 2,\n'
            '  "IR": 2,\n'
            '  "relevance_score": 0.75,\n'
            '  "reasons": ["Relevant facts directly describe the Eiffel Tower", "Irrelevant facts mention unrelated landmarks", "High relevance due to sufficient context provided"]\n'
            '}\n'
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
            '  "reasons": ["Correct location mentioned", "Incorrect construction year", "Missed one key fact"]\n'
            '}\n'
            "```"
        )
    }

    ADHERENCE_FAITHFULNESS = {
        'template': (
            "Evaluate the faithfulness or adherence of the generated answer to the provided context.\n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Generated Answer: {answer}\n\n"
            "You should determine if each part of the answer is grounded in the context. "
            "If there is any information in the answer that does not appear in the context, penalize accordingly.\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. Check if all factual claims or statements in the generated answer appear or are supported by the context.\n"
            "2. Identify any additional information not derivable from the context (hallucinations).\n"
            "3. Provide an overall faithfulness_score between 0.0 (completely unfaithful) and 1.0 (fully faithful)."
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- faithfulness_score (float between 0 and 1)\n"
            "- unfaithful_segments (array of strings) listing any ungrounded or hallucinatory parts\n"
            "- reasons (array of short statements explaining your assessment)\n\n"
            "Example:\n"
            "```json\n"
            "{\n"
            "  \"faithfulness_score\": 0.75,\n"
            "  \"unfaithful_segments\": [\"Mention of brand new data not in context\"],\n"
            "  \"reasons\": [\"Mostly grounded\", \"One detail not found in context\", \"Answer is partially incomplete\"]\n"
            "}\n"
            "```"
        )
    }
    ...



    CONTEXT_UTILIZATION = {
        'template': (
            "Analyze the context documents in relation to the generated response.\n"
            "Identify which documents contain relevant information used in the response and which do not.\n\n"
            "Question: {question}\nContext: {context}\nAnswer: {answer}\n"
            "Consider these criteria: {criteria}\n\n"
            "{formatter}"
        ),
        'criteria': (
            "1. A document is relevant if any of its information is directly referenced or paraphrased in the answer.\n"
            "2. A document is irrelevant if none of its information contributes to forming the answer.\n"
            "3. Ensure classification is based on explicit or implicit content alignment between context and answer.\n"
            "4. Do NOT include any information in relevant_context that is not explicitly present in the original context."
        ),
        'formatter': (
            "Respond ONLY with a JSON object containing:\n"
            "- relevant_context (array of strings)\n"
            "- irrelevant_context (array of strings)\n"
            "- reason (string explaining why the documents were classified this way)\n"
            "Example:\n"
            "```json\n"
            "{\n"
            '  "relevant_context": ["doc1", "doc3"],\n'
            '  "irrelevant_context": ["doc2", "doc4"],\n'
            '  "reason": "doc1 and doc3 contain key facts from the answer, while doc2 and doc4 are unrelated."\n'
            "}\n"
            "```"
        )
    }


class EvalPromptManager:
    """Manages prompt construction with JSON output formatting"""

    def __init__(self, default_type: EvaluationType = EvaluationType.RELEVANCE):
        self.default_type = default_type

    def build_prompt(
        self,
        answer: str = None,
        question: str = None,
        context: str = None,
        eval_type: EvaluationType = None,
        **kwargs
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
    pm = EvalPromptManager(default_type=EvaluationType.RELEVANCE)
    
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
    
    logger.info("Relevance Evaluation Prompt:")
    logger.info(prompt)
