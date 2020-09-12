## NCNN 
* https://github.com/Tencent/ncnn
## Compile
* g++ -o yolo-fastest yolo-fastest.cpp -I include/ncnn/ lib/libncnn.a `pkg-config --libs --cflags opencv` -fopenmp
* Usage: ./yolo-fastest
* AMD R3-3100: 66FPS
