
#ifndef OPENCV_ALGO_H
#define OPENCV_ALGO_H

#include "opencv_global.h"
#include <opencv2/opencv.hpp>

class opencvAlgo
{
  public:
    /// <summary>
    /// 棋盘格内参标定标定
    /// </summary>
    /// <param name="imageFolderPath"></param>
    /// <param name="cameraMatrix">相机内参</param>
    /// <param name="distCoeffs">畸变系数</param>
    static void checkerBoardCalibration(const std::string& imageFolderPath, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);
};

#endif  // OPENCV_ALGO_H
