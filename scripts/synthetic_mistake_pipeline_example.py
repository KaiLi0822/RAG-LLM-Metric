import asyncio
import sys

sys.path.append("..")

from execution_pipeline.execution_pipeline import ExecutionPipeline
from utils.llm import OpenAIClientLLM

from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "RAGEVALUATION-HJKMY/ragbench_10row_tester"

from data_annotator.annotators import NumMistakesAnnotator, MistakeDistributionAnnotator, MistakeAnswerGenerator
from evaluator.evaluators import FactualCorrectnessEvaluator


async def main():
    pipeline = ExecutionPipeline([NumMistakesAnnotator, MistakeDistributionAnnotator, MistakeAnswerGenerator])
    await pipeline.run_pipeline(dataset_name=DATASET_NAME, save_path="./tmp_data", upload_to_hub=True,
                                repo_id="RAGEVALUATION-HJKMY/ragbench_10row_tester_synthetic_mistake",
                                llm_class=OpenAIClientLLM)

    eval_pipeline = ExecutionPipeline([FactualCorrectnessEvaluator])
    await eval_pipeline.run_pipeline(dataset_name="RAGEVALUATION-HJKMY/ragbench_10row_tester_synthetic_mistake",
                                     save_path="./tmp_data", upload_to_hub=True,
                                     repo_id="RAGEVALUATION-HJKMY/ragbench_10row_tester_synthetic_mistake_evaluated",)


if __name__ == "__main__":
    asyncio.run(main())
