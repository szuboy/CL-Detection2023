ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# Pull the docker image | 拉取镜像
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


# Set environment variables and compilation options | 设置环境变量和编译选项
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"


# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# (Optional, use Mirror to speed up downloads) | 可选方案，利用阿里云镜像和清华源镜像加快下载速度
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


# Install the required packages | 更新系统中的软件包列表，下载一些必要的包，并且清理本地存储库中获取的软件包文件和临时文件，释放磁盘空间
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# groupadd -r algorithm => 创建了一个名为 "algorithm" 的用户组
# useradd -m --no-log-init -r -g algorithm algorithm => 创建了一个名为 "algorithm" 的用户，并将其添加到 "algorithm" 用户组中
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm


# RUN mkdir .... => 创建了三个目录：/opt/algorithm、/input和/output。选项-p表示如果父级目录不存在，也会创建它们
# chown algorithm:algorithm ... => 将创建的三个目录的所有者和组设置为algorithm，以确保这些目录的访问权限是正确的，并且与容器内运行的进程用户匹配
RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output


# 将当前Docker容器内的用户切换为 "algorithm" 用户，并切换到 /opt/algorithm 目录
USER algorithm
WORKDIR /opt/algorithm


# 将环境变量 PATH 设置为 /home/algorithm/.local/bin 加上原先的 PATH 值。
# 这样做的目的是将新的路径添加到现有的 PATH 变量中，以便在容器中执行命令时可以搜索到该路径下的可执行文件。
ENV PATH="/home/algorithm/.local/bin:${PATH}"


# RUN python -m pip install --user -U pip =>  仅在当前用户下使用 Python 自带的 pip 工具来安装或升级 pip 包管理器本身
# python -m pip install --user pip-tools  =>  仅在当前用户下使用 pip 包管理器安装了 pip-tools 包
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


# 这个命令通过 COPY 指令将主机上的 requirements.txt 文件复制到容器中的 /opt/algorithm/ 目录下，并指定文件的所有者和所属组为 algorithm
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

# 依据 requirements.txt 列表，仅在当前用户下安装 Python 对应的依赖项
RUN python -m pip install --user -r requirements.txt

# 拷贝相关的依赖文件到 /opt/algorithm/ 目录下，包括处理的 python 代码，模型的权重和配置文件等
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm model.py /opt/algorithm/
COPY --chown=algorithm:algorithm best_model.pt /opt/algorithm/

# 在容器启动时以 python -m process $0 $@ 执行命令。
# python -m process 是要运行的脚本，$0 表示命令本身，$@ 表示传递给容器的所有命令行参数。
ENTRYPOINT python -m process $0 $@
