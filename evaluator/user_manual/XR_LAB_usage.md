# Manual for use GPU in XR LAB for model serving


## install wsl2
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


## Using VSCode
Install wsl extension in Vscode

Using the remote window option to connect to wsl, this way is easier for opening multiple terminals and IDE experience

## install CUDA
In your terminal copy the command from [Nvidia install guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) and run

export the path, following the [Nvidia post install guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#mandatory-actions)

## Install SGLang
Install SGLang follow the [official document](https://docs.sglang.ai/start/install.html)

## Model serving
Follow the [quick guide](https://docs.sglang.ai/start/send_request.html) to serve your preferred model from HF

