import asyncio
import importlib
import inspect
import sys

sys.path.append("..")

from evaluator.base_evaluator import RAGEvaluator
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

DATASET_NAME = "RAGEVALUATION-HJKMY/TSBC_cleaned_demo"
OUTPUT_NAME = "RAGEVALUATION-HJKMY/TSBC_cleaned_demo_experiment"


def get_evaluator_classes():
    """Retrieve all implemented evaluators derived from RAGEvaluator."""
    module = importlib.import_module("evaluator.evaluators")
    evaluator_classes = []

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(cls, RAGEvaluator)
            and cls.__module__ == module.__name__
            and cls.__name__.endswith("Evaluator")
            and cls is not RAGEvaluator
        ):
            evaluator_classes.append(cls)

    return evaluator_classes


CLASSES = get_evaluator_classes()
logger.fatal(" ".join([i.__name__ for i in CLASSES]))


async def main():
    logger.info("Start processing pipeline")
    pipeline = ExecutionPipeline(CLASSES)
    await pipeline.run_pipeline(
        dataset_name=DATASET_NAME,
        save_path="./tmp_data",
        upload_to_hub=True,
        llm_class=LocalDeepSeekR1,
        repo_id=OUTPUT_NAME,
        model="deepseek-r1:7b",
        base_url="http://localhost:11434/v1",
    )


if __name__ == "__main__":
    asyncio.run(main())
