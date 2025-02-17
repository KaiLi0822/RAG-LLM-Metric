import asyncio
import sys
sys.path.append("..")

from execution_pipeline.execution_pipeline import ExecutionPipeline

from dotenv import load_dotenv
load_dotenv()

DATASET_NAME = "RAGEVALUATION-HJKMY/ragbench_10row_tester_annotated"

from evaluator.evaluators import KeyPointEvaluator

async def main():
    pipeline = ExecutionPipeline([KeyPointEvaluator])
    await pipeline.run_pipeline(dataset_name=DATASET_NAME, save_path="./tmp_data", upload_to_hub=True,
                                repo_id="RAGEVALUATION-HJKMY/ragbench_10row_tester_evaluated")
if __name__ == "__main__":
    asyncio.run(main())