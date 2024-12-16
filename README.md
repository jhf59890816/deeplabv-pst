下载解压deeplab3p-stcode文件 环境配置安装依赖：
$ pip install segmentation-models-pytorch 

For Aarch64: 

pip3 install torch==1.11.0 

For x86: 

pip3 install torch==1.11.0+cpu --index-url https://download.pytorch.org/whl/cpu

Install torch-npu dependencies Run the following command to install dependencies.

pip3 install pyyaml pip3 install setuptools

Install torch-npu Take Aarch64 architecture and Python 3.8 as an example.

wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc1-pytorch1.11.0/torch_npu-1.11.0.post11-cp38-cp38-linux_aarch64.whl

pip3 install torch_npu-1.11.0.post11-cp38-cp38-linux_aarch64.whl

Initialize CANN environment variable by running the command as shown below.
Default path, change it if needed.

source /usr/local/Ascend/ascend-toolkit/set_env.sh
From Source

In some special scenarios, users may need to compile torch-npu by themselves.Select a branch in table Ascend Auxiliary Software and a Python version in table PyTorch and Python Version Matching Table first. The docker image is recommended for compiling torch-npu through the following steps(It is recommended to mount the working path only and avoid the system path to reduce security risks), the generated .whl file path is ./dist/:

    Clone torch-npu

    git clone https://github.com/ascend/pytorch.git -b v1.11.0 --depth 1

    Build Docker Image

    cd pytorch/ci/docker/{arch} # {arch} for X86 or ARM
    docker build -t manylinux-builder:v1 .

    Enter Docker Container

    docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
    # {code_path} is the torch_npu source code path

    Compile torch-npu

    Take Python 3.8 as an example.

    cd /home/pytorch
    bash ci/build.sh --python=3.8

    Quick Verification

You can quickly experience Ascend NPU by the following simple examples.

import torch
import torch_npu

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)

print(z)

转换onnx
$python onnx_convert.py

onnx转换om
atc --model=./deeplabv3_plus.onnx --framework=5 --output=deeplabv3_plus --soc_version=Ascend310 -- --input_shape="input:1,3,512,512"

运行测试
python test.py
