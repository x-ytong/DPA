<h1 align="center">Enabling Country-Scale Land Cover Mapping with <br> Meter-Resolution Satellite Imagery</h1>

## Description
This is a [pytorch](https://pytorch.org/) implementation of [Dynamic Pseudo-label Assignment (DPA)](https://arxiv.org/abs/2209.00727) with U-Net as the backbone. DPA is an unsupervised domain adaptation (UDA) method applied to different satellite images for larg-scale land cover mapping. In our work, it has been validated for five megacities in China and six cities in other five Asian countries severally using unlabeled PlanetScope (PS), Gaofen-1 (GF-1), and Sentinel-2 (ST-2) satellite images.

## Usage
### Requisites
```
pip install tensorboardX tqdm
```
### Training
Input arguments (see full via python train.py --help):
```
usage: train.py [-h] [--source_dir /.../] [--target_dir /.../] [--target CITY_NAME]
                [--epochs N] [--lr LR] [--lr-scheduler {poly,step,cos}]
                [--batch-size BS_EACH_BRANCH)] [--momentum M] [--weight-decay M]
                [--workers N] [--gpu GPU_ID] [--multiprocessing-distributed]
                [--factor PSEUDO_LABEL_RATIO]
```
The data folder should be structured as follows (all files in folders other than the list folder are in the form of image patches; please refer to the data processing section of the article for more information on patch cropping):
```
├── source_dir/
│   ├── image/
│   ├── label/
│   ├── list/
|   |   ├── train.txt/
|   |   ├── val.txt/
```
```
├── target_dir/
│   ├── beijing/
│   ├── chengdu/
│   ├── guangzhou/
|   ├── shanghai/
|   ├── wuhan/
```
### Prediction
The models trained for the five Chinese megacities can be downloaded from [model_DPA](https://1drv.ms/f/c/f2b433c2113dc80b/EkemWwiHH31LnH_BigAXG-kBkAlyo0RtCy8AtjUAggXP1g?e=q9gAih). The city where the input images are located needs to correspond to the model.
```
usage: predict.py [-h] [--inputpath /.../] [--outputpath /.../]
                  [--modelname MODELNAME (e.g. unet_wuhan.pth.tar)]
```
### Evaluation
```
usage: evaluate.py [-h] [--labelpath /.../] [--resultpath /.../]
```
## Data
The data of the source and target domain can be downloaded from [Five-Billion-Pixels](https://x-ytong.github.io/project/Five-Billion-Pixels.html).
The city where the images in C-megacities are located:
```
Beijing:
    T50TMK_20210929T030539.tif
Guangzhou:
    T49QGF_20210218T025749.tif
Wuhan:
    GF1_PMS1_E114.0_N30.5_20160328_L1A0001492006-MSS1.tif
    GF1_PMS1_E114.6_N30.5_20160512_L1A0001577965-MSS1.tif
    GF1_PMS2_E113.7_N30.2_20160614_L1A0001642547-MSS2.tif
    GF1_PMS2_E114.4_N30.5_20160328_L1A0001492076-MSS2.tif
Chengdu:
    20190713_032217_1003_3B_AnalyticMS_SR.tif
    20190811_032055_1024_3B_AnalyticMS_SR.tif
    20190811_032058_1024_3B_AnalyticMS_SR.tif
    20190811_032450_1005_3B_AnalyticMS_SR.tif
    20190812_032440_1040_3B_AnalyticMS_SR.tif
    20190815_031457_0e20_3B_AnalyticMS_SR.tif
    20190815_032240_1024_3B_AnalyticMS_SR.tif
    20190815_032243_1024_3B_AnalyticMS_SR.tif
Shanghai:
    20190821_024714_84_1069_3B_AnalyticMS_SR.tif
    20190918_020334_0e26_3B_AnalyticMS_SR.tif
    20190918_021512_1035_3B_AnalyticMS_SR.tif
    20190918_021706_1014_3B_AnalyticMS_SR.tif
    20190924_024646_59_1069_3B_AnalyticMS_SR.tif
    20191030_024347_05_106e_3B_AnalyticMS_SR.tif
    20191101_021351_101b_3B_AnalyticMS_SR.tif
    20191108_024548_60_1065_3B_AnalyticMS_SR.tif
```
## Citation
If you find this code useful please consider citing:
```
@article{FBP2023,
  title={Enabling country-scale land cover mapping with meter-resolution satellite imagery},
  author={Tong, Xin-Yi and Xia, Gui-Song and Zhu, Xiao Xiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={196},
  pages={178-196},
  year={2023}
}
```
## Acknowledgement
Some codes are adapted from [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). We thank this excellent project.

