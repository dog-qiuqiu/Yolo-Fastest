#include "YoloDet.h"

ncnn::Net DNet;

static ncnn::PoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

int YoloDet::init(const char* paramPath, const char* binPath)
{
    printf("Ncnn mode init:\n %s \n %s\n", paramPath, binPath);
    
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    g_blob_pool_allocator.set_size_compare_ratio(0.0f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.0f);

    DNet.load_param(paramPath);
    DNet.load_model(binPath);
    return 0;
}

int YoloDet::detect(cv::Mat& srcImg, std::vector<TargetBox>& dstBoxes)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(srcImg.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 srcImg.cols, srcImg.rows, IW, IH);
    
    //Normalization of input image data
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
     
    //Forward
    ncnn::Mat out;
    ncnn::Extractor ex = DNet.create_extractor();
    ex.set_num_threads(NUMTHREADS);
    ex.input("data", in);
    ex.extract("output", out);
    
    //doresult
    dstBoxes.resize(out.h);
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        dstBoxes[i].cate = values[0];
        dstBoxes[i].score = values[1];

        dstBoxes[i].x1 = values[2] * srcImg.cols;
        dstBoxes[i].y1 = values[3] * srcImg.rows;
        dstBoxes[i].x2 = values[4] * srcImg.cols;
        dstBoxes[i].y2 = values[5] * srcImg.rows;
    }
    return 0;
}