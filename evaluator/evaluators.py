from __future__ import annotations  # for pervious python version e.g. 3.9
import json
from typing import List, Dict, Union, Any
from evaluator.base_evaluator import RAGEvaluator
from evaluator.prompt_manager import EvaluationType, EvalPromptManager
from sentence_transformers import SentenceTransformer, util

from utils.llm import LLMClient
from utils.constants import RAGBENCH_COL_NAMES


# TODO: add AnswerEquivalenceEvaluatorWithBert
class AnswerEquivalenceEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2202.07654, Used their definition of answer equivalence to build prompt.
    This method evaluates if the generated answer is equivalent to the reference answer.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["equivalence"]

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
            gold_answer=row[RAGBENCH_COL_NAMES.GENERATED_ANSWER.value]
        )}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {key: result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {key: None for key in self.EVAL_COLUMNS}

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
            golden_answer=golden_answer
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)

            def get_score(result):
                if result['Q1'] == 'no':
                    return 1
                elif result['Q2'] == 'yes':
                    return 1
                return 0

            scores = {
                "equivalence": get_score(result),
                "raw_output": result
            }

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            return {
                "equivalence": -1,
                "raw_output": response_text,
                'error': str(e)
            }

# TODO: implement _process_split
class RefusalAccuracyEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["refusal_accuracy"]

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
        )}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {key: result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {key: None for key in self.EVAL_COLUMNS}

    def pre_process(self, question, context, answer, **kwargs):
        pass

    def call_llm(self, processed_data):
        pass

    def post_process(self, llm_response, **kwargs):
        pass

    def evaluate(self, question: str | List[str], context: str | List[str], answer: str | List[str], **kwargs):
        prompt1 = EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.REFUSAL
        )

        resp1 = self.llm.generate(prompt1)

        prompt2 = EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.UNDERSPECIFIED_CHECK
        )

        resp2 = self.llm.generate(prompt2)

        try:
            response_text = resp1.strip().replace('```json', '').replace('```', '')
            result1 = json.loads(response_text)

            score1 = {
                "refusal": result1['refusal'],
                "reason": result1['reason']
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {'refusal': 0xffffffff, "error": str(e)}

        try:
            response_text = resp2.strip().replace('```json', '').replace('```', '')
            result2 = json.loads(response_text)

            score2 = {
                "underspecifie_check": result2['underspecifie_check'],
                "reason": result2['reason']
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response on refusal: {response_text}")
            score1 = {'underspecifie_check': 0, "error": str(e)}

        return {'refusal_result': score1, "underspecifie_check_score": score2}


class LearningFacilitationEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["learning_facilitation_score", "educational_strengths", "areas_for_improvement"]

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
        )}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {key: result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {key: None for key in self.EVAL_COLUMNS}

    def pre_process(
            self,
            question: str | List[str],
            context: str | List[str],
            answer: str | List[str],
            **kwargs) -> str:
        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.LEARNING_FACILITATION
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            print(f"Raw LLM response: {llm_response}")
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)

            scores = {
                "learning_facilitation_score": result['learning_facilitation_score'],
                "educational_strengths": result['educational_strengths'],
                "areas_for_improvement": result['areas_for_improvement'],
                "confidence": result['confidence']
            }

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "learning_facilitation_score": -1,
                "educational_strengths": [],
                "areas_for_improvement": [],
                "confidence": -1,
                'error': str(e)
            }


class EngagementEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["engagement_score", "engaging_elements", "suggestions_for_improvement"]

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
        )}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {key: result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {key: None for key in self.EVAL_COLUMNS}

    def pre_process(
            self,
            question: Union[str, List[str]],
            context: Union[str, List[str]],
            answer: Union[str, List[str]],
            **kwargs) -> str:
        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ENGAGEMENT_INDEX
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, Union[float, List[str]]]:
        """Parse JSON response into scores dictionary"""
        try:
            print(f"Raw LLM response: {llm_response}")
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)

            scores = {
                "engagement_score": result.get('engagement_score', -1),
                "engaging_elements": result.get('engaging_elements', []),
                "suggestions_for_improvement": result.get('suggestions_for_improvement', []),
                "confidence": result.get('confidence', -1)
            }

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {llm_response}")
            return {
                "engagement_score": -1,
                "engaging_elements": [],
                "suggestions_for_improvement": [],
                "confidence": -1,
                'error': str(e)
            }


class ContextRelevanceEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2501.08208, Use their definition of context relevance to build prompt.
    This method evaluates the context relevance of the retrieved context compared to the input question.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["relevance_score"]

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
        )}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {key: result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {key: None for key in self.EVAL_COLUMNS}

    def pre_process(
            self, question: str | List[str], context: str | List[str], answer: str | List[str], **kwargs
    ) -> str:
        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            eval_type=EvaluationType.CONTEXT_RELEVANCE
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            score = {
                "relevance_score": result['relevance_score']
            }
            return score
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "relevance_score": -1,
                'error': str(e)
            }


class FactualCorrectnessEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2407.12873, Use their definition of Factual Correctness to build prompt.
    This method evaluates factual correctness of the generated answer compared to the golden (ground truth) answer.
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.EVAL_COLUMNS = ["TP", "FP", "FN", "F1_SCORE"]
        self.EVAL_SCORE_PREFIX = "factual_correctness"

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
        )}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {f"{self.EVAL_SCORE_PREFIX}_{key}": result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS}

    def pre_process(
            self, question: str | List[str], context: str | List[str], answer: str | List[str], **kwargs
    ) -> str:
        if "golden_answer" not in kwargs:
            raise KeyError("Missing required key: golden_answer")
        golden_answer = kwargs.get("golden_answer")
        return EvalPromptManager().build_prompt(
            answer=answer,
            eval_type=EvaluationType.FACTUAL_CORRECTNESS,
            golden_answer=golden_answer
        )

    def call_llm(self, processed_data: str) -> str:
        # Execute LLM call with constructed prompt
        return self.llm.generate(processed_data)

    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """Parse JSON response into scores dictionary"""
        try:
            # Clean response and parse JSON
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)

            scores = {
                "TP": result['TP'],
                "FP": result['FP'],
                "FN": result['FN'],
                "F1_score": 0 if (result['TP'] + result['FP'] + result['FN']) == 0 else result['TP'] / (
                        result['TP'] + result['FP'] + result['FN']),
            }

            return scores
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {response_text}")
            return {
                "TP": -1, "FP": -1, "FN": -1, "F1_SCORE": -1,
                'error': str(e)
            }

# TODO: implement _process_split
class AnswerSimilarityEvaluator(RAGEvaluator):
    """
    Computes an embedding-based cosine similarity score between the generated answer and the ground-truth answer.
    Paper:Evaluation of RAG Metrics for Question Answering in the Telecom Domain,https://arxiv.org/abs/2407.12873 
    """

    def __init__(self, llm, prompt_manager, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            llm: Pass a dummy or None, we won't use it in this evaluator.
            prompt_manager: Not used here, but required by base class signature.
            model_name: The pretrained model name to use for sentence embedding.
        """
        super().__init__(llm, prompt_manager)
        self.model = SentenceTransformer(model_name)

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
            raise KeyError("AnswerSimilarityEvaluator requires 'golden_answer' in kwargs.")
        golden_answer = kwargs["golden_answer"]

        # 2. Compute embeddings and cosine similarity
        gen_emb = self.model.encode(answer, convert_to_tensor=True)
        gold_emb = self.model.encode(golden_answer, convert_to_tensor=True)
        similarity = util.cos_sim(gen_emb, gold_emb).item()

        # 3. Return the final score dict
        return {
            "answer_similarity": float(similarity)
        }


class KeyPointEvaluator(RAGEvaluator):
    """
    From https://arxiv.org/abs/2408.01262, using extracted key points generate from ground truth answer to check with generated answer,
    using the categorized key_points count to calculate generation scores. 
    It can provide completeness, hallucination and irrelevance score. 
    """

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.num_key_points = 0
        self.EVAL_COLUMNS = ["completeness_score", "irrelevant_score", "hallucination_score"]
        self.EVAL_SCORE_PREFIX = "key_point"

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
            key_points=row[RAGBENCH_COL_NAMES.KEY_POINTS.value]
        ), "num_key_points": len(row[RAGBENCH_COL_NAMES.KEY_POINTS.value])}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(llm_response=processed['llm_response'], num_key_points=processed['num_key_points'])
        try:
            return {f"{self.EVAL_SCORE_PREFIX}_{key}": result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {f"{self.EVAL_SCORE_PREFIX}_{key}": None for key in self.EVAL_COLUMNS}

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
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)

            scores = {
                "completeness_score": len(result['complete_ids']) / kwargs["num_key_points"],
                "irrelevant_score": len(result['irrelevant_ids']) / kwargs["num_key_points"],
                "hallucination_score": len(result['hallucinate_ids']) / kwargs["num_key_points"],
                "raw_output": result
            }

            return scores

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {llm_response}")
            return {
                "completeness_score": -1,
                "irrelevant_score": -1,
                "hallucination_score": -1,
                "raw_output": response_text,
                "error": str(e),
            }

    def evaluate(self, answer: str | List[str] = None, question: str | List[str] = None,
                 context: str | List[str] = None, **kwargs) -> Dict:
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

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
        )}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {key: result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {key: None for key in self.EVAL_COLUMNS}

    def pre_process(
            self,
            question: str | List[str],
            context: str | List[str],
            answer: str | List[str],
            **kwargs
    ) -> str:

        return EvalPromptManager().build_prompt(
            question=question,
            context=context,
            answer=answer,
            eval_type=EvaluationType.ADHERENCE_FAITHFULNESS
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
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)
            return {
                "faithfulness_score": float(result.get("faithfulness_score", 0.0)),
                "unfaithful_segments": result.get("unfaithful_segments", []),
                "reasons": result.get("reasons", [])
            }
        except (json.JSONDecodeError, KeyError) as e:
            return {
                "faithfulness_score": -1.0,
                "unfaithful_segments": [],
                "reasons": [],
                "error": str(e)
            }


class ContextUtilizationEvaluator(RAGEvaluator):

    def __init__(self, llm_class: type[LLMClient] = None, **llm_kwargs):
        super().__init__(llm_class, **llm_kwargs)
        self.context = []
        self.EVAL_COLUMNS = ["faithfulness_score", "unfaithful_segments"]

    def pre_process_row(self, row: Dict) -> Dict:
        return {'prompt': self.pre_process(
            question=row[RAGBENCH_COL_NAMES.QUESTION.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],  # TODO: ragbench golden answer should be re-considered
        ), "context": row[RAGBENCH_COL_NAMES.CONTEXT.value]}

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert 'prompt' in processed, f'Prompt missing'
        processed['llm_response'] = await self.llm.a_generate(processed['prompt'])
        return processed

    def post_process_row(self, processed: Dict, row: Dict) -> Dict:
        result = self.post_process(processed['llm_response'])
        try:
            return {key: result[key] for key in self.EVAL_COLUMNS}
        except KeyError:
            return {key: None for key in self.EVAL_COLUMNS}

    def pre_process(self, question, context, answer, **kwargs):
        self.context = context
        return EvalPromptManager().build_prompt(
            question=question,
            answer=answer,
            eval_type=EvaluationType.CONTEXT_UTILIZATION,
            context=context
        )

    def call_llm(self, processed_data):
        return self.llm.generate(processed_data)

    def post_process(self, llm_response, **kwargs):
        assert "context" in kwargs, f"Missing context"
        try:
            print(f"Raw LLM response: {llm_response}")
            response_text = llm_response.strip().replace('```json', '').replace('```', '')
            result = json.loads(response_text)

            context = kwargs['context']

            print(f"Context: {context}")
            relevant_context = result.get("relevant_context", [])
            # irrelevant_context = result.get("irrelevant_context", [])

            total_context = len(context)
            relevant_count = len(relevant_context)
            context_utilization_score = relevant_count / total_context if total_context > 0 else 0
            return context_utilization_score
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {llm_response}")
            return {
                "context_utilization_score": -1,
                'error': str(e)
            }

    def evaluate(self, answer: str | List[str] = None, question: str | List[str] = None,
                 context: str | List[str] = None, **kwargs) -> Dict:
        processed_data = self.pre_process(question, context, answer, **kwargs)
        llm_response = self.call_llm(processed_data)
        return self.post_process(llm_response, context=self.context if hasattr(self, "context") else [])
