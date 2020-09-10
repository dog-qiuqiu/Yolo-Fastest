

![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/fast.jpg)

# :zap:Yolo-Fastest:zap:
* Simple, fast, compact, easy to transplant
* A real-time target detection algorithm for all platforms
* The fastest and smallest known universal target detection algorithm based on yolo
* Optimized design for ARM mobile terminal, optimized to support [NCNN](https://github.com/Tencent/ncnn) reasoning framework
* The speed is 45% faster than [mobilenetv2-yolov3-nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Nano), and the parameter amount is reduced by 56%

# Evaluating indicator/Benchmark
Network|VOC mAP(0.5)|COCO mAP(0.5)|Resolution|Run Time(Ncnn 1xCore)|Run Time(Ncnn 4xCore)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
[MobileNetV2-YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Nano)|65.27|30.13|320|11.36ms|5.48ms|0.55BFlops|3.0MB
[Yolo-Fastest(our)](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/Yolo-Fastest)|61.02|&|320|6.74ms|4.42ms|0.23BFlops|1.3MB
[Yolo-Fastest-XL(our)](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/Yolo-Fastest)|69.43|32.45|320|15.15ms|7.09ms|0.70BFlops|3.5MB
* ***Test platform Kirin 990 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
* Suitable for hardware with extremely tight computing resources
* This model is recommended to do some simple single object detection suitable for simple application scenarios

# Pascal VOC performance index comparison
Network|Model Size|mAP(VOC 2017)|FLOPS
:---:|:---:|:---:|:---:
Tiny YOLOv2|60.5MB|57.1%|6.97BFlops
Tiny YOLOv3|33.4MB|58.4%|5.52BFlops
YOLO Nano|4.0MB|69.1%|4.51Bflops
MobileNetv2-SSD-Lite|13.8MB|68.6%|&Bflops
MobileNetV2-YOLOv3|11.52MB|70.20%|2.02Bflos
Pelee-SSD|21.68MB|70.09%|2.40Bflos
***Yolo Fastest***|1.3MB|61.02%|0.23Bflops
***Yolo Fastest-XL***|3.5MB|69.43%|0.70Bflops
* Performance indicators reference from the papers and public indicators in the github project
# Raspberrypi 3b Ncnn bf16s benchmark(4xA53 1.2Ghz)
```
loop_count = 4
num_threads = 4
powersave = 0
gpu_device = -1
cooling_down = 1
        yolo-fastest  min =   62.58  max =   62.76  avg =   62.70
      squeezenet_ssd  min =  380.98  max =  391.39  avg =  387.53
 squeezenet_ssd_int8  min =  458.05  max =  467.54  avg =  463.12
       mobilenet_ssd  min =  212.31  max =  223.34  avg =  218.93
  mobilenet_ssd_int8  min =  359.98  max =  374.03  avg =  365.17
      mobilenet_yolo  min =  619.65  max =  635.44  avg =  628.29
  mobilenetv2_yolov3  min =  294.92  max =  304.95  avg =  298.43
         yolov4-tiny  min =  855.50  max = 1074.92  avg =  962.78


```
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
![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/predictions_2.jpg)

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
# Thanks
* https://github.com/AlexeyAB/darknet
* https://github.com/Tencent/ncnn
