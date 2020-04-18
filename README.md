# ATSS-Pytorch


paper address: https://mmdetection.readthedocs.io/  

## Requirements

- Linux OS
- Python 3.6 or higher (Python 2 is not supported)
- PyTorch 1.2 or higher
- mmcv 0.4.3
- CUDA 10.0
- GCC(G++) 5.4.0 or higher


## Installation

a. Create a conda virtual environment and activate it (Optional but recommended).

```shell
conda create --name atss python=3.6
conda activate atss
```

b. Install pytorch and torchvision.  
pip is recommended,
```shell
pip install torch==1.2.0 torchvision==0.4.0  #  CUDA 10.0
```


c. Install mmdet (other dependencies wil be installed automatically).

```shell
pip install -r requirements.txt
python setup.py build develop
pip install -v -e .
```


d. Prepare dataset and checkpoint file.

Download [coco dataset](http://cocodataset.org/#download) and [checkpoint file](https://download.pytorch.org/models/resnet50-19c8e357.pth)

Fold structure should be as follows:

```
ATSS-Pytorch
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
├── backbone
│   ├── resnet50-19c8e357.pth
```


## Train
```shell
bash scripts/train.sh
```

## Test
```shell
bash scripts/test.sh
```

## Results

### RetinaNet

|    Backbone         | Lr schd | box AP |   AP50  |   AP75  |   AP_S  |   AP_M  |   AP_L  |                                                             Download                                                             |
| :-------------:     | :-----: | :----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------------------------------------------------------------------------------------------------------------------------------: |
|  R-50-FPN           |   1x    |        |         |         |         |         |         |                                                                -                                                                 |
|  R-50-FPN(ATSS)     |   1x    |        |         |         |         |         |         |                                                                -                                                                 |
|  R-50-FPN           |   2x    |        |         |         |         |         |         |                                                                -                                                                 |
|  R-50-FPN(ATSS)     |   2x    |        |         |         |         |         |         |                                                                -                                                                 |
|  R-50-FPN(A=1)      |   1x    |        |         |         |         |         |         |                                                                -                                                                 |
|  R-50-FPN(A=1,ATSS) |   1x    |        |         |         |         |         |         |                                                                -                                                                 |
|  R-50-FPN(A=1)      |   2x    |        |         |         |         |         |         |                                                                -                                                                 |
|  R-50-FPN(A=1,ATSS) |   2x    |        |         |         |         |         |         |                                                                -                                                                 |


[1] 1x and 2x mean the model is trained for 90K and 180K iterations, respectively.  
[2] (#A=1) mean retinanet only has one 8S square anchor in each level (S is stride)