# TiO-Depth_pytorch
This is the official repo for our work 'Two-in-One Depth: Bridging the Gap Between Monocular and Binocular Self-supervised Depth Estimation' (ICCV 2023).
[Paper](https://arxiv.org/abs/2309.00933)  
Citation information:  
```
@article{zhou2023two,
  title={Two-in-One Depth: Bridging the Gap Between Monocular and Binocular Self-supervised Depth Estimation},
  author={Zhou, Zhengming and Dong, Qiulei},
  journal={arXiv preprint arXiv:2309.00933},
  year={2023}
}
```

## Setup
We built and ran the repo with CUDA 11.0, Python 3.7.11, and Pytorch 1.7.0. For using this repo, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/products/individual). Please open a terminal in the root of the repo folder for running the following commands and scripts.
```
conda env create -f environment.yml
conda activate pytorch170cu11
```

## Pre-trained models
|Model Name|Dataset(s)|Abs Rel.|Sq Rel.|RMSE|RMSElog|A1|
|----------|----------|--------|-------|----|-------|--|
|TiO-Depth_384_bs8[Baidu](https://pan.baidu.com/s/1rNZvLDcTSGq5XOBFHZRTjg)/[Google](https://drive.google.com/file/d/1XMFA9wQzikjKJ-dZFiPM0pKRr_yB-7E8/view?usp=sharing)|K|0.085|0.544|3.919|0.169|0.911|
|TiO-Depth_384_bs8 (Bino.)[Baidu](https://pan.baidu.com/s/1rNZvLDcTSGq5XOBFHZRTjg)/[Google](https://drive.google.com/file/d/1XMFA9wQzikjKJ-dZFiPM0pKRr_yB-7E8/view?usp=sharing)|K|0.063|0.523|3.611|0.153|0.943|
|TiO-Depth_384_bs8_kittifull [Baidu](https://pan.baidu.com/s/1NJUA10rLDFJcS2StjbqRrw)/[Google](https://drive.google.com/file/d/1ylElx3LMm70Dmq0t-InwqU60_QcRFAeu/view?usp=sharing)|K||0.075|0.458|3.717|0.130|0.925|
|TiO-Depth_384_bs8_kittifull (Bino.) [Baidu](https://pan.baidu.com/s/1NJUA10rLDFJcS2StjbqRrw)/[Google](https://drive.google.com/file/d/1ylElx3LMm70Dmq0t-InwqU60_QcRFAeu/view?usp=sharing)|K||0.050|0.434|3.239|0.104|0.967|

* **code for all the download links of Baidu is `smde`**

## Prediction
To predict depth maps for your images, please firstly download the pretrained model from the column named `Model Name` in the above table. After unzipping the downloaded model, you could predict the depth maps for your images by
```
python predict.py\
 --image_path <path to your image or folder name for your images>\
 --exp_opts <path to the method training option>\
 --model_path <path to the downloaded or trained model>
```
You also could set `--input_size` to decide the size that the images are reshaped before they are input to the model. If you want to predict on CPU, please set `--cpu`. The depth results `<image name>_pred.npy` and the visualization results `<image name>_visual.png` will be saved in the same folder as the input images.  

## Data preparation
##### Set data path
We give an example `path_example.py` for setting the path in the repository.
Please create a python file named `path_my.py` and copy the code in `path_example.py` to the `path_my.py`. Then you can replace the used paths to your folder in the `path_my.py`.
the folder for each dataset should be organized like:
```
<root of kitti>
|---2011_09_26
|   |---2011_09_26_drive_0001_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   |---2011_09_26_drive_0002_sync
|   |   |---image_02
|   |   |---image_03
|   |   |---velodyne_points
|   |   |---...
|   '''
|---2011_09_28
|   |--- ...
|---gt_depths_raw.npz (for raw Eigen test set)
|---gt_depths_improved.npz (for improved Eigen test set)
```
```
<root of kitti 2015>
|---training
|   |---image_2
|   |   |---000000_10.png
|   |   |---000000_11.png
|   |   |---000001_10.png
|   |   |---...
|   |---image_3
|   |   |---000000_10.png
|   |   |---000000_11.png
|   |   |---000001_10.png
|   |   |---...
|   |---disp_occ_0
|   |   |---000000_10.png
|   |   |---000000_11.png
|   |   |---000001_10.png
|   '''
|---testing
|   |--- ...
```
##### KITTI
For training the methods on the KITTI dataset (the Eigen split), you should download the entire KITTI dataset (about 175GB) by:
```
wget -i ./datasets/kitti_archives_to_download.txt -P <save path>
```
And you could unzip them with:
```
cd <save path>
unzip "*.zip"
```

For evaluating the methods on the KITTI (Eigen raw test set), you should further generate the ground-truth depth file by (as done in the [Monodepth2](https://github.com/nianticlabs/monodepth2)):

```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split raw
```
If you want to evaluate the method on the KITTI improved test set, you should download the `annotated depth maps` (about 15GB) at [Here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and unzip it. Then you could generate the imporved ground-truth depth file by:
```
python datasets/utils/export_kitti_gt_depth.py --data_path <root of KITTI> --split improved
```
As an alternative, we provide the Eigen test subset (with `.png` images [Here](https://pan.baidu.com/s/1NejtxajjJt6pQ-VIRJDcUg) or with `.jpg` images [Here](https://pan.baidu.com/s/1AMkcaxh1Ua4cL1VsTXt4Ww), about 2GB) and the generated `gt_depth` files for the people who just want to do the evaluation.

##### KITTI Stereo 2015
For evaluating the model on the KITTI Stereo 2015 training set as many stereo matching methods, you should download the corresponding dataset [Here](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and unzip it.
It is noted that the training of the model also requires the entire KITTI dataset.

### Evaluate the methods
To evaluate the methods on the prepared dataset, you could simply use 
```
python evaluate.py\
 --exp_opts <path to the method EVALUATION option>\
 --model_path <path to the downloaded or trained model>
```
We provide the EVALUATION option files in `options/<Method Name>/eval/*`. Here we introduce some important arguments.
|Argument|Information|
|--------|-----------|
|`--metric_name depth_kitti_mono`|Enable the median scaling for the methods traind with monocular sequences (Sup = Mono)|
|`--visual_list`|The samples which you want to save the output (path to a `.txt` file)|
|`--save_pred`|Save the predicted depths of the samples which are in `--visual_list`|
|`--save_visual`|Save the visualization results of the samples which are in `--visual_list`|
|`-fpp`,`-gpp`, `-mspp`|Adopt different post-processing steps. (Please choose one in each time)|

The output files are saved in `eval_res\` by default. Please check `evaluate.py` for more information about arguments.

# todo
You can reproduce our results on the KITTI Eigen test set by:
```
python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti.yaml\
 --model_path pretrained_models/TiO-Depth_384_bs8/model/last_model.pth

python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti.yaml\
 --model_path pretrained_models/TiO-Depth_384_bs8/model/last_model.pth\
 -fpp

python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kittistereo.yaml\
 --model_path pretrained_models/TiO-Depth_384_bs8/model/last_model.pth
```
You can reproduce our results on the KITTI 2015 training set by:

```
python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti2015stereo.yaml\
 --model_path pretrained_models/TiO-Depth_384_bs8_kittifull/model/last_model.pth\
 --metric_name depth_kitti_stereo2015

python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti2015.yaml\
 --model_path pretrained_models/TiO-Depth_384_bs8_kittifull/model/last_model.pth\
 --metric_name depth_kitti_stereo2015

python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti2015.yaml\
 --model_path pretrained_models/TiO-Depth_384_bs8_kittifull/model/last_model.pth\
 --metric_name depth_kitti_stereo2015\
 -fpp
```

## Train the methods
Plese firstly download the pretrained Swin-trainsformer (Tiny Size) in their [official repo](https://github.com/microsoft/Swin-Transformer) and don't forget to set the path in `path_my.py` to the downloaded model. Then, you could train TiO-Depth simply use the commands provided in `options/TiO-Depth/train/train_scripts.sh`.

## Acknowledgment
[Mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
[Mmcv](https://github.com/open-mmlab/mmcv)  
[Mmengine](https://github.com/open-mmlab/mmengine)  
[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)  
[Monodepth2](https://github.com/nianticlabs/monodepth2)  
[FAL-Net](https://github.com/JuanLuisGonzalez/FAL_net)  
[DepthHints](https://github.com/nianticlabs/depth-hints)  
[DIFFNet](https://github.com/brandleyzhou/DIFFNet)  
[EPCDepth](https://github.com/prstrive/EPCDepth)  
[EdgeOfDepth](https://github.com/TWJianNuo/EdgeDepth-Release)  
[PackNet](https://github.com/TRI-ML/packnet-sfm)  
[P2Net](https://github.com/svip-lab/Indoor-SfMLearner)  
[HRDepth](https://github.com/shawLyu/HR-Depth)  
[FSRE-Depth](https://github.com/hyBlue/FSRE-Depth)  
[ManyDepth](https://github.com/nianticlabs/manydepth)  
[R-MSFM](https://github.com/jsczzzk/R-MSFM)
[ApolloScape Dataset](http://apolloscape.auto/index.html)  
[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/index.php)  
[NYUv2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)  
[Make3D Dataset](http://make3d.cs.cornell.edu/data.html#make3d)  
[Cityscapes Dataset](https://www.cityscapes-dataset.com)
