#include "YoloDet.h"

int drawBoxes(cv::Mat srcImg, std::vector<TargetBox> boxes)
{   
    printf("Detect box num: %d\n", boxes.size());
    for (int i = 0; i < boxes.size(); i++)
    {
       cv::rectangle (srcImg, cv::Point(boxes[i].x1, boxes[i].y1), 
                              cv::Point(boxes[i].x2, boxes[i].y2), 
                                cv::Scalar(255, 255, 0), 2, 2, 0);

        std::string cate =std::to_string(boxes[i].cate);
        std::string Ttext = "Category:" + cate;
        cv::Point Tp = cv::Point(boxes[i].x1, boxes[i].y1-20);
        cv::putText(srcImg, Ttext, Tp, cv::FONT_HERSHEY_TRIPLEX, 0.5, 
                                   cv::Scalar(0, 255, 0), 1, CV_AA);

        std::string score =std::to_string(boxes[i].score);
        std::string Stext = "Score:" + score;
        cv::Point Sp = cv::Point(boxes[i].x1, boxes[i].y1-5);
        cv::putText(srcImg, Stext, Sp, cv::FONT_HERSHEY_TRIPLEX, 0.5, 
                                   cv::Scalar(0, 0, 255), 1, CV_AA);

    }
    return 0;
}


int testCam() {
    YoloDet api;
    //Init model
    api.init("model/yolo-fastest-1.1_body.param", 
              "model/yolo-fastest-1.1_body.bin");

    cv::Mat frame;    
    std::vector<TargetBox> output;

    cv::VideoCapture cap(0);

    while (true) {
        printf("=========================\n");
        cap >> frame;
        if (frame.empty()) break; //如果某帧为空则退出循环

        double start = ncnn::get_current_time();
        api.detect(frame, output);
        double end = ncnn::get_current_time();
        double time = end - start;
        printf("Detect Time:%7.2f \n",time);

        drawBoxes(frame, output);
        output.clear();
        
        cv::imshow("demo", frame);
        cv::waitKey(20);
    }

    cap.release();//释放资源
    return 0;
}

int main() {
    testCam();
    return 0;
}
