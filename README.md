# Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation
This is the official implementation of the paper [Uncertainty-Guided Spatial Pruning Architecture for Efficient Frame Interpolation](https://arxiv.org/pdf/2307.16555), ACM MM 2023.

## Setup

* Clone this repository and navigate to Eigen-UGSP_Models folder

```
git clone https://github.com/Orange066/UGSP_Models.git
cd UGSP_Models
```

* We use Anaconda to create enviroment.

```
conda create -n ugsp python=3.9
conda activate ugsp
```

* Install Pytorch. 

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

* Install Python Packages: 

```
pip install -r requirements.txt
```

## Download the Datasets 

**UCF101**: Download [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow).

**Vimeo90K**: Download [Vimeo90K dataset](http://toflow.csail.mit.edu/).

**MiddleBury**: Download [MiddleBury OTHER dataset](https://vision.middlebury.edu/flow/data/).

The data path should be structured as follows:

```
UGSP_Models/
    datasets/
        middlebury/
        ucf101_interp/
        vimeo_triplet/ 
```

## Download the Pretrained Model 

Please download our pretrained model from the [Hugging Face](https://huggingface.co/Orange066/UGSP_Models) and extract it into the root folder. The data path should be structured as follows:

```
UGSP_Models/
	UGSP/
		train_log/
	UGSP_distill/
		train_log/
		train_log_32/
		train_log_35/
	UGSP_refine/
		train_log/
```

## Evaluation

Run the following codes:

```
# test UGSP
cd UGSP
CUDA_VISIBLE_DEVICES=0 python benchmark/Vimeo90K.py
CUDA_VISIBLE_DEVICES=0 python benchmark/UCF101.py
CUDA_VISIBLE_DEVICES=0 python benchmark/MiddleBury_Other.py

# test UGSP-distill
cd UGSP_distill
CUDA_VISIBLE_DEVICES=0 python benchmark/Vimeo90K.py
CUDA_VISIBLE_DEVICES=0 python benchmark/UCF101.py
CUDA_VISIBLE_DEVICES=0 python benchmark/MiddleBury_Other.py

# test UGSP-refine
cd UGSP_refine
CUDA_VISIBLE_DEVICES=0 python benchmark/Vimeo90K.py
CUDA_VISIBLE_DEVICES=0 python benchmark/UCF101.py
CUDA_VISIBLE_DEVICES=0 python benchmark/MiddleBury_Other.py
```

## Credit

Our code borrows from [RIFE](https://github.com/hzwer/ECCV2022-RIFE) and [IFRnet](https://github.com/ltkong218/IFRNet).