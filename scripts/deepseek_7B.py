import asyncio
import sys
sys.path.append("..")

from execution_pipeline.execution_pipeline import ExecutionPipeline
from utils.llm import LocalDeepSeekR1
from dotenv import load_dotenv
load_dotenv()


import logging
import os
from datetime import datetime
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
# Generate log filename with current timestamp
log_filename = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log")
# Configure logging to write to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

DATASET_NAME = "RAGEVALUATION-HJKMY/deepseek_7B_evaluated_ragbench_delucionqa_400row_mistake_added_evaluated_Incorrect"

from evaluator.evaluators import KeyPointEvaluator, LearningFacilitationEvaluator, ContextRelevanceEvaluator

async def main():
    logger.info("Start processing pipeline")
    pipeline = ExecutionPipeline([ContextRelevanceEvaluator])
    await pipeline.run_pipeline(dataset_name=DATASET_NAME, save_path="./tmp_data", upload_to_hub=True, llm_class = LocalDeepSeekR1,
                                repo_id="RAGEVALUATION-HJKMY/deepseek_7B_evaluated_ragbench_delucionqa_400row_mistake_added_evaluated_context_relevance",model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", base_url="http://0.0.0.0:30000/v1")
if __name__ == "__main__":
    asyncio.run(main())