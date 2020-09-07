

![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/fast.jpg)

# :zap:Yolo-Fastest:zap:
* Simple, fast, compact, easy to transplant
* A real-time target detection algorithm for all platforms
* The fastest and smallest known universal target detection algorithm based on yolo
* Optimized design for ARM mobile terminal, optimized to support [NCNN](https://github.com/Tencent/ncnn) reasoning framework
* The speed is 45% faster than [mobilenetv2-yolov3-nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Nano), and the parameter amount is reduced by 56%

# Evaluating indicator
Network|VOC mAP(0.5)|Resolution|Run Time(Ncnn 1xCore)|Run Time(Ncnn 4xCore)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[MobileNetV2-YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Nano)|65.27|320|11.36ms|5.48ms|0.55BFlops|3.0MB
[Yolo-Fastest(our)](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/Yolo-Fastest)|60.8|320|6.74ms|4.42ms|0.23BFlops|1.3MB
[MobileNetV2 SSD-Lite](https://github.com/qfgaohao/pytorch-ssd#mobilenetv2-ssd-lite)|68.6|300|&ms|&ms|&BFlops|13.8MB
[Yolo-Fastest-XL(our)](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/Yolo-Fastest)|68.8|320|15.15ms|7.09ms|0.70BFlops|3.5MB
* Test platform Kirin 990，Based on [NCNN](https://github.com/Tencent/ncnn)
* Suitable for hardware with extremely tight computing resources
* This model is recommended to do some simple single object detection suitable for simple application scenarios

# Test 
![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/predictions.jpg)

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
