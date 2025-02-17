import json
from typing import Dict

from data_annotator.base_annotator import DataAnnotator
from data_annotator.prompt_manager import AnnotationType, AnnotatePromptManager
from utils.constants import RAGBENCH_COL_NAMES
from utils.llm import LLMClient


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
            "prompt": AnnotatePromptManager().build_prompt(
                question=question,
                golden_answer=golden_answer,
                eval_type=AnnotationType.KEY_POINT_EXTRACTION,
            )
        }

    async def call_llm(self, processed: Dict) -> str:
        assert processed.get("prompt", None), "prompt missing"
        return await self.llm.a_generate(prompt=processed["prompt"])

    def post_process(self, response: str, row: Dict) -> Dict:
        try:
            # Clean response and parse JSON
            response_text = response.strip().replace("```json", "").replace("```", "")
            result = json.loads(response_text)
            return {"key_points": result["key_points"]}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response for row{row['id']}: {response_text}")
            return {"key_points": ["error"]}
