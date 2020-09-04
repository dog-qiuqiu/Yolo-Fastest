

![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/fast.jpg)

# :zap:Yolo-Fastest:zap:
* Simple, fast, compact, easy to transplant
* A real-time target detection algorithm for all platforms
* The fastest and smallest known universal target detection algorithm based on yolo
* The speed is 45% faster than mobilenetv2-yolov3-nano, and the parameter amount is reduced by 56%

# Evaluating indicator
Network|VOC mAP(0.5)|Resolution|Run Time(Ncnn 1xCore)|Run Time(Ncnn 4xCore)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[MobileNetV2-YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Nano)|65.27|320|11.36ms|5.48ms|0.55BFlops|3.0MB
Yolo-Fastest|&|320|6.74ms|4.42ms|0.23BFlops|1.3MB
* Test platform Kirin 990

# Test 
![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/predictions.jpg)
