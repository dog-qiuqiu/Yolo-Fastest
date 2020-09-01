

![image](https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/data/flash.gif)

# :zap:Yolo-Fastest:zap:
* A real-time target detection algorithm for all platforms
* Combine EfficientNet,FPN,Attention mechanism...
* The fastest and smallest known universal target detection algorithm based on yolo
* The speed is 25% faster than mobilenetv2-yolov3-nano, and the parameter amount is reduced by 40%

# Evaluating indicator
Network|VOC mAP(0.5)|Resolution|Inference time (NCNN/Kirin 990)|Inference time (Darknet/i7)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Tiny-YoloV3|58.4%|416|& ms|697 ms|5.52BFlops|33.4MB
[MobileNetV2-YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Nano)|65.27|320|10.15ms|77 ms|0.5BFlops|3.0MB
Yolo-Fastest|&%|320|& ms|57 ms|0.25 BFlops|1.8MB

