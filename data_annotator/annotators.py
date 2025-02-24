from __future__ import annotations
import json
from random import random
from typing import Dict, List

from data_annotator.base_annotator import DataAnnotator
from data_annotator.prompt_manager import AnnotationType, AnnotatePromptManager
from utils.constants import RAGBENCH_COL_NAMES, LLM_RESPONSE, PROMPT, SYNTHETIC_MISTAKE_TYPES
from utils.llm import LLMClient

import numpy as np


class KeyPointAnnotator(DataAnnotator):

    def __init__(
            self,
            llm_class: type[LLMClient] = None,
            **llm_kwargs,
    ):
        super().__init__(llm_class, **llm_kwargs)

    def pre_process(self, row: Dict) -> Dict:
        question = row[RAGBENCH_COL_NAMES.QUESTION.value]
        golden_answer = row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value]
        return {
            PROMPT: AnnotatePromptManager().build_prompt(
                question=question,
                golden_answer=golden_answer,
                eval_type=AnnotationType.KEY_POINT_EXTRACTION,
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        assert processed.get(PROMPT, None), "prompt missing"
        processed[LLM_RESPONSE] = await self.llm.a_generate(prompt=processed[PROMPT])
        return processed

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        try:
            # Clean response and parse JSON
            response_text = processed[LLM_RESPONSE].strip().replace("```json", "").replace("```", "")
            result = json.loads(response_text)
            return {"key_points": result["key_points"]}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response for row{row['id']}: {response_text}")
            return {"key_points": ["error"]}


class NumMistakesAnnotator(DataAnnotator):
    def __init__(
            self,
            llm_class: type[LLMClient] = None,
            **llm_kwargs,
    ):
        super().__init__(llm_class, **llm_kwargs)

    def pre_process(self, row: Dict) -> Dict:
        pass

    async def a_call_llm(self, processed: Dict) -> Dict:
        pass

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        return {'num_mistake': np.random.choice(3, p=[0.0, 0.7, 0.3])}


class MistakeDistributionAnnotator(DataAnnotator):
    def __init__(
            self,
            llm_class: type[LLMClient] = None,
            **llm_kwargs,
    ):
        super().__init__(llm_class, **llm_kwargs)
        self.mistake_type = SYNTHETIC_MISTAKE_TYPES

    def pre_process(self, row: Dict) -> Dict:
        return {
            PROMPT: AnnotatePromptManager().build_prompt(
                question=row[RAGBENCH_COL_NAMES.QUESTION.value],
                golden_answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],
                eval_type=AnnotationType.HAS_NUMERIC_INFO,
            )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        try:
            # Clean response and parse JSON
            response_text = processed[LLM_RESPONSE].strip().replace("```json", "").replace("```", "")
            result = json.loads(response_text)
            if result["has_numeric_info"].lower() == 'true':
                return {"mistake_distribution": self._distribute(True, row)}
            else:
                return {"mistake_distribution": self._distribute(False, row)}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response for row{row['id']}: {response_text}")
            return {"mistake_distribution": self._distribute(False, row)}

    def _distribute(self, has_numeric: bool, row: Dict) -> List:
        if has_numeric and np.random.choice([True, False], p=[0.75, 0.25]):
            counts = [0] * len(self.mistake_type)

            for _ in range(row["num_mistake"] - 1):
                idx = np.random.randint(0, len(self.mistake_type) - 2, 1)[0]
                counts[idx] += 1

            counts[-1] = 1
            return [json.dumps(inner) for inner in zip(self.mistake_type, counts)]
        else:
            counts = [0] * (len(self.mistake_type))

            for _ in range(row["num_mistake"]):
                idx = np.random.randint(0, len(self.mistake_type) - 2, 1)[0]
                counts[idx] += 1

            return [json.dumps(inner) for inner in zip(self.mistake_type, counts)]


class MistakeAnswerGenerator(DataAnnotator):
    def __init__(
            self,
            llm_class: type[LLMClient] = None,
            **llm_kwargs,
    ):
        super().__init__(llm_class, **llm_kwargs)

    def _pre_process_mistakes(self, mistake_distribution: list) -> str:
        """Convert mistake distribution list into instruction string"""
        mistake_distribution = [tuple(json.loads(item)) for item in mistake_distribution]
        errors = []
        for error_type, count in mistake_distribution:
            errors.extend([error_type] * count)

        return "\n".join([f"{i + 1}. Introduce a {error} error in only one place"
                          for i, error in enumerate(errors)])

    def pre_process(self, row: Dict) -> Dict:
        processed_mistakes = self._pre_process_mistakes(row['mistake_distribution'])

        return {PROMPT: AnnotatePromptManager().build_prompt(
            golden_answer=row[RAGBENCH_COL_NAMES.GOLDEN_ANSWER.value],
            context=row[RAGBENCH_COL_NAMES.CONTEXT.value],
            criteria_result=AnnotationType.MISTAKE_GENERATION.criteria(processed_mistakes),
            eval_type=AnnotationType.MISTAKE_GENERATION
        )
        }

    async def a_call_llm(self, processed: Dict) -> Dict:
        processed[LLM_RESPONSE] = await self.llm.a_generate(processed[PROMPT])
        return processed

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        try:
            # Clean response and parse JSON
            response_text = processed[LLM_RESPONSE].strip().replace("```json", "").replace("```", "")
            result = json.loads(response_text)
            return {"Paraphrased": result["Paraphrased"],
                    "Incorrect": result["Incorrect"],
                    "Error_Locations": result["Error_Locations"]}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response for row{row['id']}: {response_text}")
            return {"Paraphrased": None,
                    "Incorrect": None,
                    "Error_Locations": []}


# TODO discuss if we still need this, we can use a stronger model to direct get mistake answer standard score
class MistakeAnswerScoringAnnotator(DataAnnotator):
    def __init__(self, scores: List[str]):
        self.scores = scores
        super().__init__()

    def pre_process(self, row: Dict) -> Dict:
        pass

    def a_call_llm(self, processed: Dict) -> Dict:
        pass

    def post_process(self, processed: Dict, row: Dict) -> Dict:
        pass
