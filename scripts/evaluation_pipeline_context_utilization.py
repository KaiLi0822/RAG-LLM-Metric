import asyncio
import sys

sys.path.append("..")

from execution_pipeline.execution_pipeline import ExecutionPipeline

from dotenv import load_dotenv

load_dotenv()


import logging
import os
from datetime import datetime

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
# Generate log filename with current timestamp
log_filename = os.path.join(
    log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
)
# Configure logging to write to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)
logger = logging.getLogger(__name__)

DATASET_NAME = "RAGEVALUATION-HJKMY/ragbench_10row_tester_synthetic_mistake"

from evaluator.evaluators import (
    KeyPointIrrelevantEvaluator,
    KeyPointCompletenessEvaluator,
    KeyPointHallucinationEvaluator,
    ContextUtilizationEvaluator,
    BERTScoreEvaluator,
)


async def main():
    logger.info("Start processing pipeline")
    pipeline = ExecutionPipeline([ContextUtilizationEvaluator])
    await pipeline.run_pipeline(
        dataset_name=DATASET_NAME,
        save_path="./tmp_data",
        upload_to_hub=True,
        repo_id="RAGEVALUATION-HJKMY/cukai_ragbench_10row_tester_synthetic_mistake",
    )


if __name__ == "__main__":
    asyncio.run(main())
