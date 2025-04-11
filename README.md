[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/RAGEVALUATION-HJKMY)

# RAG-LLM-Metric

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using Large Language Models (LLMs). This project provides tools for evaluating RAG systems across multiple dimensions including key point evaluation, learning facilitation, and BERTScore metrics.

## Features

- **Multi-dimensional Evaluation**: Evaluate RAG systems using various metrics
- **Flexible Pipeline**: Modular execution pipeline for easy integration of new evaluators
- **GPU Acceleration**: Optional GPU support for faster processing
- **Dataset Integration**: Built-in support for Hugging Face datasets
- **Logging System**: Comprehensive logging for tracking evaluation progress

## Project Structure

```
RAG-LLM-Metric/
├── agent/                 # Agent-related components
├── data_annotator/        # Data annotation tools
├── evaluator/            # Evaluation metrics and tools
├── execution_pipeline/   # Pipeline execution framework
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Utility scripts and examples
├── utils/                # Utility functions
└── analysis/             # Analysis tools and results
```

## Installation

### Prerequisites
- Python 3.9+
- [Poetry](https://python-poetry.org/) (recommended)
- NVIDIA drivers (for GPU support only)

### Using Poetry (Recommended)

#### CPU Installation (Default)
For CPU-only environments:
```bash
poetry install -E cpu
```

#### GPU/CUDA Installation
```bash
poetry source add pytorch https://download.pytorch.org/whl/cu121
poetry install -E gpu
```

## Usage and How to Run

### Basic Evaluation Pipeline

The basic evaluation pipeline demonstrates how to use a single evaluator:

```python
import asyncio
from execution_pipeline.execution_pipeline import ExecutionPipeline
from evaluator.evaluators import BERTScoreEvaluator

async def main():
    pipeline = ExecutionPipeline([BERTScoreEvaluator])
    results = await pipeline.run_pipeline(
        dataset_df=your_dataset,
        save_path="./results",
        upload_to_hub=False
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

### Available Evaluators

The framework provides a comprehensive set of evaluators for different aspects of RAG system performance:

1. **Answer Equivalence** ([Paper](https://arxiv.org/abs/2202.07654))
   - Evaluates if generated answer is equivalent to reference answer
   - Checks for information parity without omissions/additions

2. **Refusal Accuracy** ([Paper](https://arxiv.org/html/2412.12300v1))
   - Assesses model's ability to properly refuse unanswerable/ambiguous queries
   - Combines refusal check and underspecification validation

3. **BERTScore** ([Paper](https://arxiv.org/abs/1904.09675))
   - Uses BERTScore to evaluate similarity between generated and reference answers

4. **Learning Facilitation**
   - Evaluates how well the answer facilitates learning and understanding

5. **Engagement**
   - Assesses the engagement level of the generated response

6. **Context Relevance** ([Paper](https://arxiv.org/abs/2501.08208))
   - Evaluates how relevant the generated answer is to the provided context

7. **Factual Correctness** ([Paper](https://arxiv.org/abs/2407.12873))
   - Checks the factual accuracy of the generated answer

8. **Answer Similarity** ([Paper](https://arxiv.org/abs/2407.12873))
   - Measures the similarity between generated and reference answers

9. **Key Point Analysis** ([Paper](https://arxiv.org/abs/2408.01262))
   - **Completeness**: Evaluates if all key points are covered
   - **Irrelevance**: Checks for irrelevant key points
   - **Hallucination**: Detects hallucinated key points

10. **Adherence Faithfulness** ([Paper](https://arxiv.org/abs/2501.08208))
    - Evaluates how faithful the answer is to the source context

11. **Context Utilization**
    - Assesses how well the context is utilized in the answer

12. **Coherence**
    - Evaluates the coherence and logical flow of the answer

13. **Factual Accuracy**
    - Checks the factual accuracy of the generated content

### Example Pipelines

#### 1. Agent-based Evaluation Pipeline

The agent-based pipeline uses a dynamic evaluation orchestrator to evaluate RAG systems with customizable metrics:

```bash
python scripts/agent_e2e.py
```

This script:
- Uses a dynamic evaluation orchestrator
- Supports custom metric weights
- Allows for multi-round discussions
- Requires OpenAI API key in `.env` file

#### 2. Synthetic Mistake Pipeline

The synthetic mistake pipeline helps in generating and analyzing synthetic mistakes in RAG systems:

```bash
python scripts/synthetic_mistake_pipeline_example.py
```

This pipeline:
- Annotates mistakes in the dataset
- Generates mistake distributions
- Creates synthetic answers with mistakes
- Requires OpenAI API key in `.env` file

#### 3. Mistral Experiment Pipeline (Example)

The Mistral experiment pipeline is an example that demonstrates how to run a comprehensive evaluation using all available evaluators with a local model. This example can be adapted to use different models and configurations:

```bash
python scripts/mistral_experiment_pipeline.py
```

This example pipeline:
- Automatically discovers and uses all available evaluators
- Can be configured to use different models by modifying the `model` and `base_url` parameters
- Supports both local and remote model serving
- Can upload results to Hugging Face Hub

To use a different model, modify the `run_pipeline` parameters in the script:
```python
res = await pipeline.run_pipeline(
    dataset_name=DATASET_NAME,
    save_path="./tmp_data",
    upload_to_hub=True,
    repo_id=OUTPUT_NAME,
    model="your-model-name",  # Change this to your desired model
    base_url="your-model-endpoint"  # Change this to your model's endpoint
)
```

Common model configurations:
- Local Mistral(with SGLang): `model="mistralai/Ministral-8B-Instruct-2410", base_url="http://127.0.0.1:30000/v1"` 
- OpenAI: `model="gpt-4", base_url="https://api.openai.com/v1/"`
- Anthropic: `model="claude-3-opus-20240229", base_url="https://api.anthropic.com/v1/"`
- Hugging Face: `model="your-model-id", base_url="https://api-inference.huggingface.co/models/"`

### Environment Setup

Before running any pipeline, ensure you have:

1. Created a `.env` file with necessary API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

2. For Mistral experiments, ensure your local model server is running:
```bash
python3 -m sglang.launch_server --model mistralai/Ministral-8B-Instruct-2410 --trust-remote-code
```

## Local LLM Setup (Optional)

### WSL2 Setup (for Windows users)
1. Configure WSL2:
```bash
wsl --set-default-version 2
wsl --list --online
wsl --install <Distribution Name>
```

2. Install VSCode WSL extension for better development experience

3. Install CUDA following the [NVIDIA installation guide](https://developer.nvidia.com/cuda-downloads)

4. Install SGLang following the [official documentation](https://docs.sglang.ai/start/install.html)

5. Start model server:
```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Contact

[Add your contact information here]

