# 说明
介绍如何在linux下安装miniconda。

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 可以指定anaconda的安装路径
bash miniconda.sh -b -p $HOME/miniconda3

$HOME/miniconda3/bin/conda init

source ～/.bashrc

conda --version

# 创建Python虚拟环境
conda create -n py312 python=3.12.0
```
