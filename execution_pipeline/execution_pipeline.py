import asyncio
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from datasets import Dataset, DatasetDict, load_dataset
from data_annotator.base_annotator import DataAnnotator
from evaluator.base_evaluator import RAGEvaluator
from data_annotator.annotators import KeyPointAnnotator
import os


def load_data(dataset_name: str, config: Optional[str] = None) -> DatasetDict:
    """Load dataset from Hugging Face hub"""
    dataset = load_dataset(dataset_name, config)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    return dataset


def detect_splits(dataset: DatasetDict) -> List[str]:
    """Detect available splits in the dataset"""
    return [split for split in ["train", "validation", "test"] if split in dataset]


class Executor:
    def __init__(self, processor_class: type[DataAnnotator] | type[RAGEvaluator], num_workers: int = os.cpu_count()):
        self.processor_class = processor_class
        self.num_workers = num_workers

    async def run(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        """Process entire DatasetDict across splits with parallel processing"""
        processed_splits = {}
        splits = detect_splits(dataset)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._process_split,
                    self.processor_class,
                    dataset[split],
                    kwargs,
                )
                for split in splits
            ]
            results = await asyncio.gather(*tasks)

        for split, result in zip(splits, results):
            processed_splits[split] = result

        return DatasetDict(processed_splits)

    @staticmethod
    def _process_split(processor_class: type[DataAnnotator] | type[RAGEvaluator], split_data: Dataset, kwargs):
        """Instantiate inside worker process"""
        annotator = processor_class(**kwargs)  # Create instance here
        processed = asyncio.run(annotator.process_split(split_data))
        for col_name, list_data in processed.items():
            split_data = split_data.add_column(col_name, list_data)
        return split_data


class ExecutionPipeline:
    def __init__(self, processor_classes: List[type[DataAnnotator] | type[RAGEvaluator]]):
        self.processor_classes = processor_classes
        self.executors = [Executor(cls) for cls in processor_classes]

    async def run_pipeline(self, dataset_name: str, save_path: str, upload_to_hub: bool = False,
                           repo_id: Optional[str] = None, **kwargs) -> DatasetDict:
        # Load initial dataset
        initial_dataset = load_data(dataset_name)
        current_dataset = initial_dataset

        # Create fresh instances in executor processes
        for _cls, executor in zip(self.processor_classes, self.executors):
            current_dataset = await executor.run(
                dataset=current_dataset,
                **kwargs
            )

        current_dataset.save_to_disk(save_path)

        # Upload to Hub if requested
        if upload_to_hub:
            if not repo_id:
                raise ValueError("repo_id is required for Hub upload")

            current_dataset.push_to_hub(repo_id=repo_id, token=os.getenv("HF_TOKEN"))

        return current_dataset
