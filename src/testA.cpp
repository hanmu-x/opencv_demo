

#include <filesystem>
#include "opencv_tool/opencv_IO.h"
#include "opencv_tool/opencv_drawer.h"
#include "opencv_tool/opencv_transform.h"
#include "opencv_tool/opencv_filter.h"
#include "opencv_tool/opencv_algo.h"
#include "file_tool/file_tool.h"


int main(int argc, char** argv)
{
    std::filesystem::path imageGraph(DEFAULT_DATA_DIR);
    imageGraph += "/graph.jpg";

    // 双目相机生成深度图
    std::filesystem::path leftImagePath(DEFAULT_DATA_DIR);
    leftImagePath += "/two_eyes/left.png";
    std::filesystem::path rightImagePath(DEFAULT_DATA_DIR);
    rightImagePath += "/two_eyes/right.png";

    cv::Mat leftImage = opencvIO::openImage(leftImagePath.string());
    opencvIO::showImage(leftImage);
    cv::Mat rightImage = opencvIO::openImage(rightImagePath.string());
    opencvIO::showImage(rightImage);
    cv::Mat depth = opencvAlgo::depthMap(leftImage, rightImage);
    opencvIO::showImage(depth);

    return 0;




    // 霍夫变换
    cv::Mat graphImage = opencvIO::openImage(imageGraph.string());
    //// 霍夫直线变换
    //cv::Mat linesResult = opencvFilter::houghDetectLines(graphImage);
    //opencvIO::showImage(linesResult);
    //// 霍夫圆变换
    cv::Mat circlesResult = opencvFilter::houghDetectCircles(graphImage);
    opencvIO::showImage(circlesResult);


    return 0;
}