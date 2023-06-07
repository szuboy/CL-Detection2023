#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"


# 构建docker
./build.sh


# 生成一个随机的字符串，用作卷标（volume label）的后缀
VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)


# 设置算法镜像的内存限制为8GB。在Grand Challenge上，当前的内存限制是30GB，但可以在算法镜像设置中进行配置
MEM_LIMIT="8g"


# 创建了一个名为cldetection_alg_2023-output-$VOLUME_SUFFIX的Docker卷。
# 其中$VOLUME_SUFFIX是一个变量，它的值通过生成一个随机的32位哈希值来确定
docker volume create cldetection_alg_2023-output-$VOLUME_SUFFIX


# 这是固定的参数，还请不要改变，其主要功能为：在一系列限制的环境中运行名为cldetection_alg_2023的镜像，并将本地路径和卷挂载到容器中供其访问
# --network="none"：禁用容器的网络功能，即容器内部无法访问网络
# --cap-drop="ALL"：禁用容器中的所有特权功能
# --security-opt="no-new-privileges"：禁止在容器内启用新的特权
# --shm-size="128m"：设置共享内存的大小为128MB
# --pids-limit="256"：限制容器的进程数上限为256个
# -v $SCRIPTPATH/test/:/input/images/lateral-dental-x-rays/：
# 将$SCRIPTPATH/test/本地路径挂载到容器中的/input/images/lateral-dental-x-rays/目录，因为本地测试的图像是在放在当前目录的test文件夹
# -v cldetection_alg_2023-output-$VOLUME_SUFFIX:/output/：将名为cldetection_alg_2023-output-$VOLUME_SUFFIX的Docker卷挂载到容器中的/output/目录
# cldetection_alg_2023：指定要运行的镜像名称，如果之前的 ./build.sh 修改了镜象名词，这里还请同步修改，建议不修改
docker run --rm --gpus all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/images/lateral-dental-x-rays/ \
        -v cldetection_alg_2023-output-$VOLUME_SUFFIX:/output/ \
        cldetection_alg_2023

# 上面这个命令，已经进行了调用了模型进行了预测，得到了最后的预测结果

# 使用docker run命令运行一个容器，--rm 选项表示容器停止后自动删除容器。
# -v cldetection_alg_2023-output-$VOLUME_SUFFIX:/output/ 选项表示将之前创建的名为cldetection_alg_2023-output-$VOLUME_SUFFIX的Docker卷挂载到容器中的/output/目录。
# python:3.9-slim表示使用python:3.9-slim镜像作为容器的基础镜像
# cat /output/orthodontic-landmarks.json命令将容器内/output/目录下的orthodontic-landmarks.json文件的内容输出到标准输出
# python -m json.tool命令将标准输入的JSON格式数据进行格式化并输出
docker run --rm \
        -v cldetection_alg_2023-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/orthodontic-landmarks.json | python -m json.tool

# 上面的这个命令，是输出打印预测结果的json文件


docker run --rm \
        -v cldetection_alg_2023-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/images/lateral-dental-x-rays/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/orthodontic-landmarks.json')); f2 = json.load(open('/input/images/lateral-dental-x-rays/expected_output.json')); sys.exit(f1 != f2);"

# 上面的这个命令，比较docker模型预测 和 外部的测试结果，是否一致，保证结果是正确的

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm cldetection_alg_2023-output-$VOLUME_SUFFIX
