#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>

int demo(cv::Mat& image, ncnn::Net &detector, int detector_size_width, int detector_size_height)
{

    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };

    cv::Mat bgr = image.clone();
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 bgr.cols, bgr.rows, detector_size_width, detector_size_height);

    //数据预处理
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = detector.create_extractor();
    ex.set_num_threads(8);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);

    for (int i = 0; i < out.h; i++)
    {
        int label;
        float x1, y1, x2, y2, score;
        float pw,ph,cx,cy;
        const float* values = out.row(i);
        
        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        score = values[1];
        label = values[0];

        //处理坐标越界问题
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if(x2<0) x2=0;
        if(y2<0) y2=0;

        if(x1>img_w) x1=img_w;
        if(y1>img_h) y1=img_h;
        if(x2>img_w) x2=img_w;
        if(y2>img_h) y2=img_h;
        cv::rectangle (image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 1, 1, 0);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[label], score * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(image, text, cv::Point(x1, y1 + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    return 0;
}

//摄像头测试
int test_cam()
{
    //定义yolo-fastest VOC检测器
    ncnn::Net detector;  
    detector.load_param("model/yolo-fastest.param");
    detector.load_model("model/yolo-fastest.bin");
    int detector_size_width  = 320;
    int detector_size_height = 320;

    cv::Mat frame;
    cv::VideoCapture cap(0);

    while (true)
    {
        cap >> frame;
        double start = ncnn::get_current_time();
        demo(frame, detector, detector_size_width, detector_size_height);
        double end = ncnn::get_current_time();
        double time = end - start;
        printf("Time:%7.2f \n",time);
        cv::imshow("demo", frame);
        cv::waitKey(1);
    }
    return 0;
}
int main()
{
    test_cam();
    return 0;
}
