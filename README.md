## [NTIRE 2024 Workshop and Challenge](https://cvlai.net/ntire/2024/) @ CVPR 2024
## Blind Compressed Image Enhancement Challenge


### UnifyFormer: Unifying Group Dynamics with Channel Attention for Blind Compressed Image Enhancement

#### Introduction

This repository is implemented for NTIRE2024 Blind Compressed Image Enhancement Challenge

By Titans
Members: [Yash Arora](https://github.com/yasharora102), [Aditya Arora](https://github.com/adityac8)


The code is tested under Ubuntu 18.04 environment with NVIDIA A100 GPU and PyTorch 1.12.1 Python 3.10 Cuda 11.3

#### Clone the repository
    
```bash
git clone https://github.com/yasharora102/UnifyFormer.git
cd UnifyFormer
```


#### Download the pretrained model

```bash
wget https://github.com/yasharora102/UnifyFormer/releases/download/Weights/model_unifyformer.pth 
```

#### Running the code

- For inference on DIV2K val set, run

```
python test.py --tile 400 --tile_overlap 32 --self_ensemble True
```
