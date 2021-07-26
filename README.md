* 2021.3.21: 对模型结构进行细微调整优化，更新Yolo-Fastest-1.1模型
* 2021.3.19: NCNN Camera Demo https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/sample/ncnn
* 2021.3.16: 修复分组卷积在某些旧架构GPU推理耗时异常的问题

# :zap:Yolo-Fastest:zap:[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5131532.svg)](https://doi.org/10.5281/zenodo.5131532)
* Simple, fast, compact, easy to transplant
* A real-time target detection algorithm for all platforms
* The fastest and smallest known universal target detection algorithm based on yolo
* ***Optimized design for ARM mobile terminal, optimized to support [NCNN](https://github.com/Tencent/ncnn) reasoning framework***
* Based on NCNN deployed on RK3399 ,Raspberry Pi 4b... and other embedded devices to achieve full real-time 30fps+

![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/fast.jpg)
* ***中文介绍https://zhuanlan.zhihu.com/p/234506503*** 
* ***相比AlexeyAB/darknet，此版本的darknet修复分组卷积在某些旧架构GPU推理耗时异常的问题(例如1050ti:40ms->4ms速度提升10倍)，强烈建议用此仓库框架训练模型***
* ***Compared with AlexeyAB/darknet, this version of darknet fixes the problem of abnormal time-consuming inference of grouped convolution in some old architecture GPUs (for example, 1050ti:40ms->4ms speed up 10 times), it is strongly recommended to use this warehouse framework for training model***
* ***Darknet CPU推理效率优化不好，不建议使用Darknet作为CPU端的推理框架，建议使用NCNN***
* ***Darknet CPU reasoning efficiency optimization is not good, it is not recommended to use Darknet as the CPU side reasoning framework, it is recommended to use ncnn***
* ***Based on pytorch training framework: https://github.com/dog-qiuqiu/yolov3***

# Evaluating indicator/Benchmark
Network|COCO mAP(0.5)|Resolution|Run Time(Ncnn 4xCore)|Run Time(Ncnn 1xCore)|FLOPS|Params|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
[Yolo-Fastest-1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40 %|320X320|5.59 ms|7.52 ms|0.252BFlops|0.35M|1.4M
[Yolo-Fastest-1.1-xl](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|34.33 %|320X320|9.27ms|15.72ms|0.725BFlops|0.925M|3.7M
[Yolov3-Tiny-Prn](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny-prn.cfg)|33.1%|416X416|%ms|%ms|3.5BFlops|4.7M|18.8M
[Yolov4-Tiny](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)|40.2%|416X416|23.67ms|40.14ms|6.9 BFlops|5.77M|23.1M

* ***Test platform Mi 11 Snapdragon 888 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
* COCO 2017 Val mAP（no group label）
* Suitable for hardware with extremely tight computing resources
* This model is recommended to do some simple single object detection suitable for simple application scenarios

# Yolo-Fastest-1.1 Multi-platform benchmark
Equipment|Computing backend|System|Framework|Run time
:---:|:---:|:---:|:---:|:---:
Mi 11|Snapdragon 888|Android(arm64)|ncnn|5.59ms
Mate 30|Kirin 990|Android(arm64)|ncnn|6.12ms
Meizu 16|Snapdragon 845|Android(arm64)|ncnn|7.72ms
Development board|Snapdragon 835(Monkey version)|Android(arm64)|ncnn|20.52ms
Development board|RK3399|Linux(arm64)|ncnn|35.04ms
Raspberrypi 3B|4xCortex-A53|Linux(arm64)|ncnn|62.31ms
Orangepi Zero Lts|H2+ 4xCortex-A7|Linux(armv7)|ncnn|550ms
Nvidia|Gtx 1050ti|Ubuntu(x64)|darknet|4.73ms
Intel|i7-8700|Ubuntu(x64)|ncnn|5.78ms
* The above is a multi-core test benchmark
* The above speed benchmark is tested by ***big core*** in big.little CPU
* Raspberrypi 3B enable bf16s optimization，[Raspberrypi 64 Bit OS](http://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2020-08-24/)
* [Rk3399 needs to lock the cpu to the highest frequency](http://blog.sina.com.cn/s/blog_15d5280590102yarw.html), ncnn and enable bf16s optimization

# Pascal VOC performance index comparison
Network|Model Size|mAP(VOC 2007)|FLOPS
:---:|:---:|:---:|:---:
Tiny YOLOv2|60.5MB|57.1%|6.97BFlops
Tiny YOLOv3|33.4MB|58.4%|5.52BFlops
YOLO Nano|4.0MB|69.1%|4.51Bflops
MobileNetv2-SSD-Lite|13.8MB|68.6%|&Bflops
MobileNetV2-YOLOv3|11.52MB|70.20%|2.02Bflos
Pelee-SSD|21.68MB|70.09%|2.40Bflos
***Yolo Fastest***|1.3MB|61.02%|0.23Bflops
***Yolo Fastest-XL***|3.5MB|69.43%|0.70Bflops
***MobileNetv2-Yolo-Lite***|8.0MB|73.26%|1.80Bflops
* Performance indicators reference from the papers and public indicators in the github project
* MobileNetv2-Yolo-Lite: https://github.com/dog-qiuqiu/MobileNet-Yolo#mobilenetv2-yolov3-litenano-darknet

# Yolo-Fastest-1.1 Pedestrian detection
Equipment|System|Framework|Run time
:---:|:---:|:---:|:---:
Raspberrypi 3B|Linux(arm64)|ncnn|62ms
* Simple real-time pedestrian detection model based on yolo-fastest-1.1
* Enable bf16s optimization，[Raspberrypi 64 Bit OS](http://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2020-08-24/)

## Demo
![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/p1.jpg)
![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/p2.jpg)

# Compile 
## How to compile on Linux
* This repo is based on Darknet project so the instructions for compiling the project are same
(https://github.com/MuhammadAsadJaved/darknet#how-to-compile-on-windows-legacy-way)


Just do `make` in the Yolo-Fastest-master directory. Before make, you can set such options in the `Makefile`: [link](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/Makefile#L1)

* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* Set the other options in the `Makefile` according to your need.

# Test/Demo
*Run Yolo-Fastest , Yolo-Fastest-xl  , Yolov3 or Yolov4 on image or video inputs
## Demo on image input
*Note: change  .data , .cfg , .weights and input image file in `image_yolov3.sh` for Yolo-Fastest-x1, Yolov3 and Yolov4

```
  sh image_yolov3.sh
```
## Demo on video input
*Note: Use any input video and place in the `data` folder or use `0` in the `video_yolov3.sh` for webcam

*Note: change  .data , .cfg , .weights and input video file in `video_yolov3.sh` for Yolo-Fastest-x1, Yolov3 and Yolov4

```
  sh video_yolov3.sh
```
## Yolo-Fastest Test
![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/predictions_2.png)

## Yolo-Fastest-xl Test
![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/projections.jpg)

# How to Train
## Generate a pre-trained model for the initialization of the model backbone
```
  ./darknet partial yolo-fastest.cfg yolo-fastest.weights yolo-fastest.conv.109 109
```
## Train
* 交流qq群:1062122604
* https://github.com/AlexeyAB/darknet
```
  ./darknet detector train voc.data yolo-fastest.cfg yolo-fastest.conv.109 
```
# Deploy
## NCNN
### NCNN Conversion Tutorial
* Benchmark:https://github.com/Tencent/ncnn/tree/master/benchmark
* NCNN supports direct conversion of darknet models
* darknet2ncnn: https://github.com/Tencent/ncnn/tree/master/tools/darknet
### NCNN Sample
* CamSample:https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/sample/ncnn
* AndroidSample: https://github.com/WZTENG/YOLOv5_NCNN
## MNN&TNN&MNN
* https://github.com/dog-qiuqiu/MobileNet-Yolo#darknet2caffe-tutorial
* ***Based on MNN: https://github.com/geekzhu001/Yolo-Fastest-MNN Run on : raspberry pi 4B 2G Input size : 320*320 Average inference time : 0.035s*** 
## ONNX&TensorRT
* https://github.com/CaoWGG/TensorRT-YOLOv4
* It is not efficient to run on Psacal and earlier GPU architectures. It is not recommended to deploy on such devices such as jeston nano(17ms/img), Tx1, Tx2, but there is no such problem in Turing GPU, such as jetson-Xavier-NX Can run efficiently
## OpenCV DNN
* https://blog.csdn.net/nihate/article/details/108670542
# Thanks
* https://github.com/AlexeyAB/darknet
* https://github.com/Tencent/ncnn

## Cite as
dog-qiuqiu. (2021, July 24). dog-qiuqiu/Yolo-Fastest: 
yolo-fastest-v1.1.0 (Version v.1.1.0). Zenodo. 
http://doi.org/10.5281/zenodo.5131532
