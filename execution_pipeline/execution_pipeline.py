from __future__ import annotations
import asyncio
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from datasets import Dataset, DatasetDict, load_dataset
from data_annotator.base_annotator import DataAnnotator
from evaluator.base_evaluator import RAGEvaluator
import os
from tqdm import tqdm


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
    def __init__(
            self,
            processor_class: type[DataAnnotator] | type[RAGEvaluator],
            num_workers: int = os.cpu_count(),
    ):
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
    def _process_split(
            processor_class: type[DataAnnotator] | type[RAGEvaluator],
            split_data: Dataset,
            kwargs,
    ):
        """Instantiate inside worker process"""
        processor = processor_class(**kwargs)  # Create instance here
        processed = asyncio.run(processor.process_split(split_data))
        for col_name, list_data in processed.items():
            split_data = split_data.add_column(col_name, list_data)
        return split_data


class ExecutionPipeline:
    def __init__(
            self, processor_classes: List[type[DataAnnotator] | type[RAGEvaluator]]
    ):
        self.processor_classes = processor_classes
        self.executors = [Executor(cls) for cls in processor_classes]

    async def run_pipeline(
            self,
            dataset_name: Optional[str] = None,
            dataset_df: Optional[pd.DataFrame] = None,
            save_path: Optional[str] = None,
            upload_to_hub: bool = False,
            repo_id: Optional[str] = None,
            **kwargs
    ) -> Union[DatasetDict, pd.DataFrame]:
        """Handle both HF datasets and pandas DataFrames"""
        # Load initial dataset
        input_was_dataframe = False
        if dataset_name is not None:
            initial_dataset = load_data(dataset_name)
        elif dataset_df is not None:
            initial_dataset = DatasetDict({"train": Dataset.from_pandas(dataset_df)})
            input_was_dataframe = True
        else:
            raise ValueError("Must provide either dataset_name or dataset_df")

        current_dataset = initial_dataset

        # Process through all executors with progress bar
        for processor_class, executor in tqdm(  # Added tqdm
            zip(self.processor_classes, self.executors),
            total=len(self.processor_classes),
            desc="Processing pipeline stages"
        ):
            current_dataset = await executor.run(dataset=current_dataset, **kwargs)

        # Save processed dataset
        if save_path is not None:
            current_dataset.save_to_disk(save_path)

        # Upload to Hub if requested
        if upload_to_hub:
            if not repo_id:
                raise ValueError("repo_id is required for Hub upload")
            current_dataset.push_to_hub(repo_id=repo_id, token=os.getenv("HF_TOKEN"))

        # Return appropriate type
        return (
            current_dataset["train"].to_pandas()
            if input_was_dataframe
            else current_dataset
        )


class CompoundScoreExecutionPipeline(ExecutionPipeline):
    def __init__(self, evaluators_with_weights: List[Tuple[type[RAGEvaluator], float]]):
        processor_classes = [evaluator_cls for evaluator_cls, _ in evaluators_with_weights]
        super().__init__(processor_classes)
        self.evaluators_with_weights = evaluators_with_weights

    async def run_pipeline_with_weight(
            self,
            dataset_name: Optional[str] = None,
            dataset_df: Optional[pd.DataFrame] = None,
            save_path: Optional[str] = None,
            upload_to_hub: bool = False,
            repo_id: Optional[str] = None,
            **kwargs
    ) -> Union[DatasetDict, pd.DataFrame]:
        # Capture original columns
        if dataset_name is not None:
            original_dataset = load_data(dataset_name)
            original_columns = set(original_dataset['train'].column_names)
        elif dataset_df is not None:
            original_columns = set(dataset_df.columns)
        else:
            raise ValueError("Must provide either dataset_name or dataset_df")

        dataset = await super().run_pipeline(
            dataset_name=dataset_name,
            dataset_df=dataset_df,
            save_path=save_path,
            upload_to_hub=upload_to_hub,
            repo_id=repo_id,
            **kwargs
        )

        if isinstance(dataset, DatasetDict):
            split = detect_splits(dataset)[0]
            final_columns = set(dataset[split].column_names)
        else:
            final_columns = set(dataset.columns)
        new_columns = list(final_columns - original_columns)

        if len(new_columns) < len(self.evaluators_with_weights):
            raise ValueError("Mismatch between number of evaluators and new columns generated")

        evaluator_columns = new_columns[:len(self.evaluators_with_weights)]
        if isinstance(dataset, DatasetDict):
            for split_name in dataset:
                split_dataset = dataset[split_name]
                columns_with_weights = [
                    (col, weight)
                    for (_, weight), col in zip(self.evaluators_with_weights, evaluator_columns)
                ]

                def compute_weighted_score(ds):
                    total = 0.0
                    for col, weight in columns_with_weights:
                        try:
                            total += ds[col] * weight
                        except Exception as e:
                            pass
                    ds['Final_Score'] = total
                    return ds

                updated_split = split_dataset.map(compute_weighted_score)
                dataset[split_name] = updated_split
        else:
            df = dataset.copy()
            df['Final_Score'] = 0.0
            for col, (_, weight) in zip(evaluator_columns, self.evaluators_with_weights):
                df['Final_Score'] += df[col] * weight
            dataset = df

        # Re-save if save_path is provided
        if save_path is not None:
            if isinstance(dataset, DatasetDict):
                dataset.save_to_disk(save_path)
            else:
                dataset.to_csv(save_path, index=False)

        # Re-upload if requested
        if upload_to_hub:
            if not repo_id:
                raise ValueError("repo_id is required for Hub upload")
            if isinstance(dataset, DatasetDict):
                dataset.push_to_hub(repo_id=repo_id, token=os.getenv("HF_TOKEN"))
            else:
                hf_dataset = Dataset.from_pandas(dataset)
                hf_dataset.push_to_hub(repo_id=repo_id, token=os.getenv("HF_TOKEN"))

        return dataset
