[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/RAGEVALUATION-HJKMY)

# RAG-LLM-Metric

A retrieval-augmented generation LLM evaluation framework with optional GPU acceleration.

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
GPU/CUDA Installation
```bash
poetry source add pytorch https://download.pytorch.org/whl/cu121

poetry install -E gpu
```

### To run with local LLM
Manual for use GPU in XR LAB for model serving


#### install wsl2
 using the windows command prompt

 ```
 wsl --set-default-version 2
 ```

 Then run
 ```
 wsl --list --online
 ```

 choose a distribution to install
 ```
 wsl --install <Distribution Name>
 ```


#### Using VSCode
Install wsl extension in Vscode

Using the remote window option to connect to wsl, this way is easier for opening multiple terminals and IDE experience

#### install CUDA
In your terminal copy the command from [Nvidia install guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) and run

export the path, following the [Nvidia post install guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#mandatory-actions)

#### Install SGLang
Install SGLang follow the [official document](https://docs.sglang.ai/start/install.html)

#### Model serving
```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code
```


#### generate persona
```
python -m persona.persona --config-file config.yml
```
