from __future__ import annotations  # for pervious python version e.g. 3.9

import asyncio
import json
from typing import List, Dict, Union, Any
from evaluator.base_evaluator import RAGEvaluator
from evaluator.prompt_manager import EvaluationType, EvalPromptManager

try:
    from sentence_transformers import SentenceTransformer, util
    from bert_score import score as bert_score
except ImportError as e:
    print(f"Error: {e}")

from utils.llm import LLMClient
from utils.constants import RAGBENCH_COL_NAMES
from utils.constants import RAGBENCH_COL_NAMES, LLM_RESPONSE, PROMPT, EVAL_COL_MAP
import os
import logging

logger = logging.getLogger(__name__)


# TODO: add AnswerEquivalenceEvaluatorWithBert
class AnswerEquivalenceEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2202.07654, Used their definition of answer equivalence to build prompt.
    This method evaluates if the generated answer is equivalent to the reference answer.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["equivalence"]
        self.EVAL_SCORE_PREFIX = "answer_equivalence"
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls):
        return {
            "name": "evaluate_answer_equivalence",
            "description": "Evaluates if generated answer is equivalent to reference answer using LLM. Checks for "
                           "information parity without omissions/additions. Returns binary score (0/1) based on "
                           "structured criteria questions.",
            "parameters": {
                "question": "str",
                "context": "str",
                "generated_answer": "str",
                "reference_answer": "str",
            },
        }

    def pre_process_row(self, row: Dict) -> Dict:
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
                golden_answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(
        self,
        question: str | List[str],
        context: str | List[str],
        answer: str | List[str],
        **kwargs,
    ) -> str:
        assert "golden_answer" in kwargs, "Missing required input: golden_answer"
        golden_answer = kwargs.get("golden_answer")
        assert len(golden_answer) > 0, "golden_answer is empty"

        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ANSWER_EQUIVALENCE,
            golden_answer=golden_answer,
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)

            def get_score(result):
                if result["Q1"] == "no":
                    return 1
                elif result["Q2"] == "yes":
                    return 1
                return 0

            scores = {"equivalence": get_score(result), "raw_output": result}

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            return {"equivalence": -1, "raw_output": response_text, "error": str(e)}


# TODO: implement _process_split
class RefusalAccuracyEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["refusal_accuracy"]
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = "refusal_accuracy"
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls):
        return {
            "name": "evaluate_refusal_accuracy",
            "description": "Assesses model's ability to properly refuse answering unanswerable/ambiguous queries. "
                           "Combines refusal check and underspecification validation. Returns dual scores with "
                           "reasons.",
            "parameters": {
                "question": "str",
                "context": "str",
                "generated_answer": "str",
            },
        }

    def pre_process_row(self, row: Dict) -> Dict:
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(self, question, context, answer, **kwargs):
        pass

    def call_llm(self, processed_data):
        pass

    def post_process(self, llm_response, **kwargs):
        pass

    def evaluate(
        self,
        question: str | List[str],
        context: str | List[str],
        answer: str | List[str],
        **kwargs,
    ):
        prompt1 = EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.REFUSAL,
        )

        resp1 = self.llm.generate(prompt1)

        prompt2 = EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.UNDERSPECIFIED_CHECK,
        )

        resp2 = self.llm.generate(prompt2)

        try:
            response_text = resp1.strip().replace("```json", "").replace("```", "")
            result1 = json.loads(response_text)

            score1 = {"refusal": result1["refusal"], "reason": result1["reason"]}
        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {"refusal": 0xFFFFFFFF, "error": str(e)}

        try:
            response_text = resp2.strip().replace("```json", "").replace("```", "")
            result2 = json.loads(response_text)

            score2 = {
                "underspecifie_check": result2["underspecifie_check"],
                "reason": result2["reason"],
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {"underspecifie_check": 0, "error": str(e)}

        return {"refusal_result": score1, "underspecifie_check_score": score2}


class BERTScoreEvaluator(RAGEvaluator):
    """
    Computes BERTScore between the generated answer and the ground-truth answer.
    Paper: BERTScore: Evaluating Text Generation with BERT, https://arxiv.org/abs/1904.09675
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Args:
            model_name: The pretrained model name to use for BERTScore.
        """
        super().__init__()
        self.model_name = model_name
        self.EVAL_COLUMNS = ["precision", "recall", "f1"]
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = "bert_score"
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls) -> Dict:
        return {
            "name": "calculate_bert_score",
            "description": "Computes BERT-based precision, recall and F1 between generated and reference answers "
                           "using sentence embeddings. No LLM required.",
            "parameters": {"generated_answer": "str", "reference_answer": "str"},
        }

    def pre_process_row(self, row: Dict) -> Dict:
        pass

    async def a_call_llm(self, processed: Dict) -> Dict:
        pass

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        pass

    async def process_row(self, row: Dict, semaphore: asyncio.Semaphore) -> Dict:
        async with semaphore:
            question = row[RAGBENCH_COL_NAMES.QUESTION.value]
            context = row[RAGBENCH_COL_NAMES.CONTEXT.value]
            answer = row[EVAL_COL_MAP[self.answer_column]]
            if RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value not in row:
                raise KeyError("Missing golden_answer in row")
            golden_answer = row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value]

            if answer is None or golden_answer is None:
                raise ValueError(
                    "answer or golden_answer is None, cannot compute BERTScore"
                )

            evaluation_result = self.evaluate(
                question, context, answer, golden_answer=golden_answer
            )

            prefixed_result = {
                f"{self.EVAL_SCORE_PREFIX}_{key}": value
                for key, value in evaluation_result.items()
            }
            return prefixed_result

    def pre_process(self, question, context, answer, **kwargs):
        # No actual prompt needed.
        pass

    def call_llm(self, processed_data: Any) -> str:
        # Not calling an LLM.
        pass

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        # Not parsing any LLM JSON output.
        pass

    def evaluate(self, question, context, answer, **kwargs) -> Dict[str, float]:
        """
        Perform the main logic of computing BERTScore.
        """
        # 1. Validate that 'golden_answer' is provided
        if "golden_answer" not in kwargs:
            raise KeyError("BERTScoreEvaluator requires 'golden_answer' in kwargs.")
        golden_answer = kwargs["golden_answer"]

        # 2. Compute BERTScore
        P, R, F1 = bert_score([answer], [golden_answer], model_type=self.model_name)

        # 3. Return the final score dict
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }


class LearningFacilitationEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = [
            "learning_facilitation_score",
            "educational_strengths",
            "areas_for_improvement",
        ]
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = "learning_facilitation"
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls) -> Dict:
        return {
            "name": "calculate_bert_score",
            "description": "Computes BERT-based precision, recall and F1 between generated and reference answers "
                           "using sentence embeddings. No LLM required.",
            "parameters": {"generated_answer": "str", "reference_answer": "str"},
        }

    def pre_process_row(self, row: Dict) -> Dict:
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(
        self,
        question: str | List[str],
        context: str | List[str],
        answer: str | List[str],
        **kwargs,
    ) -> str:
        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.LEARNING_FACILITATION,
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            logger.info(f"Raw LLM response: {llm_response}")
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)

            scores = {
                "learning_facilitation_score": result["learning_facilitation_score"],
                "educational_strengths": result["educational_strengths"],
                "areas_for_improvement": result["areas_for_improvement"],
                "confidence": result["confidence"],
            }

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response: {response_text}")
            return {
                "learning_facilitation_score": -1,
                "educational_strengths": [],
                "areas_for_improvement": [],
                "confidence": -1,
                "error": str(e),
            }


class EngagementEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = [
            "engagement_score",
            "engaging_elements",
            "suggestions_for_improvement",
        ]
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = "engagement"
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls):
        return {
            "name": "evaluate_engagement",
            "description": "Measures answer engagement through language use, narrative flow and real-world relevance. "
                           "Provides scored analysis with enhancement recommendations.",
            "parameters": {
                "question": "str",
                "context": "str",
                "generated_answer": "str",
            },
        }

    def pre_process_row(self, row: Dict) -> Dict:
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(
        self,
        question: Union[str, List[str]],
        context: Union[str, List[str]],
        answer: Union[str, List[str]],
        **kwargs,
    ) -> str:
        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ENGAGEMENT_INDEX,
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(
        self, llm_response: str, **kwargs
    ) -> Dict[str, Union[float, List[str]]]:
        """Parse JSON response into scores dictionary"""
        try:
            logger.info(f"Raw LLM response: {llm_response}")
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)

            scores = {
                "engagement_score": result.get("engagement_score", -1),
                "engaging_elements": result.get("engaging_elements", []),
                "suggestions_for_improvement": result.get(
                    "suggestions_for_improvement", []
                ),
                "confidence": result.get("confidence", -1),
            }

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response: {llm_response}")
            return {
                "engagement_score": -1,
                "engaging_elements": [],
                "suggestions_for_improvement": [],
                "confidence": -1,
                "error": str(e),
            }


class ContextRelevanceEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2501.08208, Use their definition of context relevance to build prompt.
    This method evaluates the context relevance of the retrieved context compared to the input question.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["relevance_score"]
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = "Context_Relevance"
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls):
        return {
            "name": "evaluate_context_relevance",
            "description": "Evaluates relevance of retrieved context to question using LLM. Scores based on R/IR "
                           "segment classification and coverage analysis.",
            "parameters": {"question": "str", "context": "str"},
        }

    def pre_process_row(self, row: Dict) -> Dict:
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(
        self,
        question: str | List[str],
        context: str | List[str],
        answer: str | List[str],
        **kwargs,
    ) -> str:
        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            eval_type=EvaluationType.CONTEXT_RELEVANCE,
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)
            score = {"relevance_score": result["relevance_score"]}
            return score
        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response: {response_text}")
            return {"relevance_score": -1, "error": str(e)}


class FactualCorrectnessEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2407.12873, Use their definition of Factual Correctness to build prompt.
    This method evaluates factual correctness of the generated answer compared to the golden (ground truth) answer.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["TP", "FP", "FN", "F1_score"]
        self.EVAL_SCORE_PREFIX = "factual_correctness"
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = ""
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls) -> Dict:
        return {
            "name": "evaluate_factual_correctness",
            "description": "Compares generated vs reference answers using TP/FP/FN analysis. Calculates factual F1 "
                           "score through statement-level validation.",
            "parameters": {"generated_answer": "str", "reference_answer": "str"},
        }

    def pre_process_row(self, row: Dict) -> Dict:
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
                golden_answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],
                eval_type=EvaluationType.FACTUAL_CORRECTNESS,
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(
        self,
        question: str | List[str],
        context: str | List[str],
        answer: str | List[str],
        **kwargs,
    ) -> str:
        if "golden_answer" not in kwargs:
            raise KeyError("Missing required key: golden_answer")
        golden_answer = kwargs.get("golden_answer")
        return EvalPromptManager().build_prompt(
            answer=answer,
            eval_type=EvaluationType.FACTUAL_CORRECTNESS,
            golden_answer=golden_answer,
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)

            scores = {
                "TP": result["TP"],
                "FP": result["FP"],
                "FN": result["FN"],
                "F1_score": (
                    0
                    if (result["TP"] + result["FP"] + result["FN"]) == 0
                    else result["TP"] / (result["TP"] + result["FP"] + result["FN"])
                ),
            }

            return scores
        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response: {response_text}")
            return {"TP": -1, "FP": -1, "FN": -1, "F1_SCORE": -1, "error": str(e)}


# TODO: implement _process_split
class AnswerSimilarityEvaluator(RAGEvaluator):
    """
    Computes an embedding-based cosine similarity score between the generated answer and the ground-truth answer.
    Paper:Evaluation of RAG Metrics for Question Answering in the Telecom Domain,https://arxiv.org/abs/2407.12873
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            llm: Pass a dummy or None, we won't use it in this evaluator.
            prompt_manager: Not used here, but required by base class signature.
            model_name: The pretrained model name to use for sentence embedding.
        """
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.prompt_manager = "Answer_Similarity"
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = ""
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls):
        return {
            "name": "calculate_answer_similarity",
            "description": "Computes cosine similarity between answer embeddings using sentence transformers. "
                           "Measures semantic equivalence without LLMs.",
            "parameters": {"generated_answer": "str", "reference_answer": "str"},
        }

    def pre_process_row(self, row: Dict) -> Dict:
        pass

    async def a_call_llm(self, processed: Dict) -> Dict:
        pass

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        pass

    def pre_process(self, question, context, answer, **kwargs):
        # No actual prompt needed.
        pass

    def call_llm(self, processed_data: Any) -> str:
        # Not calling an LLM.
        pass

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        # Not parsing any LLM JSON output.
        pass

    def evaluate(self, question, context, answer, **kwargs) -> Dict[str, float]:
        """
        Perform the main logic of computing answer similarity using embeddings.
        """
        # 1. Validate that 'golden_answer' is provided
        if "golden_answer" not in kwargs:
            raise KeyError(
                "AnswerSimilarityEvaluator requires 'golden_answer' in kwargs."
            )
        golden_answer = kwargs["golden_answer"]

        # 2. Compute embeddings and cosine similarity
        gen_emb = self.model.encode(answer, convert_to_tensor=True)
        gold_emb = self.model.encode(golden_answer, convert_to_tensor=True)
        similarity = util.cos_sim(gen_emb, gold_emb).item()

        # 3. Return the final score dict
        return {"answer_similarity": float(similarity)}


class KeyPointEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2408.01262, using extracted key points generate from ground truth answer to check with generated answer,
    using the categorized key_points count to calculate generation scores.
    It can provide completeness, hallucination and irrelevance score.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.num_key_points = 0
        self.EVAL_COLUMNS = [
            "completeness_score",
            "irrelevant_score",
            "hallucination_score",
        ]
        self.EVAL_SCORE_PREFIX = "key_point"
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls):
        return {
            "name": "evaluate_keypoint_coverage",
            "description": "Assesses answer quality through key point alignment with reference. Scores completeness, "
                           "hallucination and irrelevance ratios.",
            "parameters": {
                "question": "str",
                "generated_answer": "str",
                "reference_keypoints": "List[str]",
            },
        }

    def pre_process_row(self, row: Dict) -> Dict:
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
                key_points=row[RAGBENCH_COL_NAMES.KEY_POINTS.value],
            ),
            "num_key_points": len(row[RAGBENCH_COL_NAMES.KEY_POINTS.value]),
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(
            llm_response=processed[LLM_RESPONSE],
            num_key_points=processed["num_key_points"],
        )
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(self, question, context, answer, **kwargs):
        if "key_points" not in kwargs:
            raise KeyError("Missing required input: key_points")
        key_points = kwargs.get("key_points")

        if not isinstance(key_points, list):
            raise ValueError("key_points is type of List[str]")

        if len(key_points) == 0:
            raise ValueError("key_points is an empty List, which is invalid")

        self.num_key_points = len(key_points)
        formatted_key_points = "\n".join(key_points)

        return EvalPromptManager().build_prompt(
            question=question,
            answer=answer,
            eval_type=EvaluationType.KEY_POINT,
            key_points=formatted_key_points,
        )

    def call_llm(self, processed_data):
        return self.llm.generate(processed_data)

    def post_process(self, llm_response, **kwargs):
        assert "num_key_points" in kwargs, "num_key_points is missing"
        try:
            # Clean response and parse JSON
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)

            scores = {
                "completeness_score": len(result["complete_ids"])
                / kwargs["num_key_points"],
                "irrelevant_score": len(result["irrelevant_ids"])
                / kwargs["num_key_points"],
                "hallucination_score": len(result["hallucinate_ids"])
                / kwargs["num_key_points"],
                "raw_output": result,
            }

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response: {llm_response}")
            return {
                "completeness_score": -1,
                "irrelevant_score": -1,
                "hallucination_score": -1,
                "raw_output": response_text,
                "error": str(e),
            }

    def evaluate(
        self,
        answer: str | List[str] = None,
        question: str | List[str] = None,
        context: str | List[str] = None,
        **kwargs,
    ) -> Dict:
        processed_data = self.pre_process(question, context, answer, **kwargs)
        llm_response = self.call_llm(processed_data)
        return self.post_process(llm_response, num_key_points=self.num_key_points)


class AdherenceFaithfulnessEvaluator(RAGEvaluator):
    """
    Uses an LLM to verify that all parts of the generated answer are grounded in the provided context.
    Returns a faithfulness_score between 0 and 1, plus any unfaithful (hallucinated) segments.
    Related paper:ASTRID - An Automated and Scalable TRIaD for the Evaluation of RAG-based Clinical Question Answering Systems,
    https://arxiv.org/abs/2501.08208
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["faithfulness_score", "unfaithful_segments"]
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = "Adherence_Faithfulness"
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls) -> Dict:
        return {
            "name": "evaluate_faithfulness",
            "description": "Validates answer grounding in context through hallucination detection. Scores "
                           "faithfulness and lists unsubstantiated claims.",
            "parameters": {
                "question": "str",
                "context": "str",
                "generated_answer": "str",
            },
        }

    def pre_process_row(self, row: Dict) -> Dict:
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(
        self,
        question: str | List[str],
        context: str | List[str],
        answer: str | List[str],
        **kwargs,
    ) -> str:

        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ADHERENCE_FAITHFULNESS,
        )

    def call_llm(self, processed_data: str) -> str:
        """
        Invoke the LLM with the processed prompt and return its raw text response.
        """
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """
        Parse the LLM's JSON output to extract the faithfulness score.
        """
        try:
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)
            return {
                "faithfulness_score": float(result.get("faithfulness_score", 0.0)),
                "unfaithful_segments": result.get("unfaithful_segments", []),
                "reasons": result.get("reasons", []),
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "faithfulness_score": -1.0,
                "unfaithful_segments": [],
                "reasons": [],
                "error": str(e),
            }


class ContextUtilizationEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.context = []
        self.EVAL_COLUMNS = ["faithfulness_score", "unfaithful_segments"]
        assert os.getenv(
            "ANSWER_TYPE", None
        ), "Environment variable ANSWER_TYPE must be defined for evaluation"
        self.answer_column = os.getenv("ANSWER_TYPE")
        self.EVAL_SCORE_PREFIX = "Context_Utilization"
        if self.EVAL_SCORE_PREFIX:
            self.EVAL_SCORE_PREFIX = f"{self.answer_column}_{self.EVAL_SCORE_PREFIX}"
        else:
            self.EVAL_SCORE_PREFIX = self.answer_column

    @classmethod
    def description(cls) -> Dict:
        return {
            "name": "evaluate_context_utilization",
            "description": "Measures effective use of provided context in answers through relevance classification. "
                           "Scores utilization ratio of context segments.",
            "parameters": {
                "question": "str",
                "context": "str",
                "generated_answer": "str",
            },
        }

    def pre_process_row(self, row: Dict) -> Dict:
        return {
            PROMPT: self.pre_process(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
                answer=row[EVAL_COL_MAP[self.answer_column]],
            ),
            "context": row[RAGBENCH_COL_NAMES.CONTEXT.value],
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert PROMPT in processed, f"Prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed[LLM_RESPONSE])
        try:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": result[key]
                for key in self.EVAL_COLUMNS
            }
        except KeyError:
            return {
                f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS
            }

    def pre_process(self, question, context, answer, **kwargs):
        self.context = context
        return EvalPromptManager().build_prompt(
            question=question,
            answer=answer,
            eval_type=EvaluationType.CONTEXT_UTILIZATION,
            context=context,
        )

    def call_llm(self, processed_data):
        return self.llm.generate(processed_data)

    def post_process(self, llm_response, **kwargs):
        assert "context" in kwargs, f"Missing context"
        try:
            logger.info(f"Raw LLM response: {llm_response}")
            response_text = (
                llm_response.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(response_text)

            context = kwargs["context"]

            logger.info(f"Context: {context}")
            relevant_context = result.get("relevant_context", [])
            # irrelevant_context = result.get("irrelevant_context", [])

            total_context = len(context)
            relevant_count = len(relevant_context)
            context_utilization_score = (
                relevant_count / total_context if total_context > 0 else 0
            )
            return context_utilization_score
        except (json.JSONDecodeError, KeyError) as e:
            logger.info(f"Error parsing LLM response: {llm_response}")
            return {"context_utilization_score": -1, "error": str(e)}

    def evaluate(
        self,
        answer: str | List[str] = None,
        question: str | List[str] = None,
        context: str | List[str] = None,
        **kwargs,
    ) -> Dict:
        processed_data = self.pre_process(question, context, answer, **kwargs)
        llm_response = self.call_llm(processed_data)
        return self.post_process(
            llm_response, context=self.context if hasattr(self, "context") else []
        )
