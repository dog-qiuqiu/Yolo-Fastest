#ifndef NCNN_H_
#define NCNN_H_

#include "net.h"
#include "benchmark.h"

#include <opencv2/opencv.hpp>

//Model input image size
#define IW 320  //width
#define IH 320  //height

//cpu num threads
#define NUMTHREADS 8

typedef struct _TargetBox {
    int x1;        //left
    int y1;        //top
    int x2;        //right
    int y2;        //bottom
 
    int cate;      //category
    float score;   //Confidence level
}TargetBox;

class YoloDet {
public:

    //model init
    int init(const char* paramPath, const char* binPath);
    //body detect
    int detect(cv::Mat& srcImg, std::vector<TargetBox>& dstBoxes);

private:

};


#endif  //NCNN_H_