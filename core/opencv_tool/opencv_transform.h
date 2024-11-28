
#ifndef OPENCV_TRANSFORM_H
#define OPENCV_TRANSFORM_H

#include "opencv_global.h"
#include <opencv2/opencv.hpp>

class opencvTF
{
  public:
    /// <summary>
    /// 图片缩放,会改变原图片的分辨率
    /// </summary>
    /// <param name="img"></param>
    /// <param name="scale_factor">放大缩小的倍数,2.0:放大两倍,0.5缩小一倍</param>
    /// <returns></returns>
    static cv::Mat resizeImage(const cv::Mat& img, double scale_factor);

    /// <summary>
    /// 图片平移
    /// </summary>
    /// <param name="img"></param>
    /// <param name="dx">x方向平移多少</param>
    /// <param name="dy">y方向平移多少</param>
    /// <returns></returns>
    static cv::Mat translateImage(const cv::Mat& img, int dx, int dy);

    /// <summary>
    /// 图片的旋转
    /// </summary>
    /// <param name="img"></param>
    /// <param name="angle">旋转角度:正数，则表示逆时针旋转;负数，则表示顺时针旋转</param>
    /// <returns></returns>
    static cv::Mat rotateImage(const cv::Mat& img, double angle);

    /// <summary>
    /// 图片裁剪
    /// </summary>
    /// <param name="img"></param>
    /// <param name="x">裁剪起始x</param>
    /// <param name="y">裁剪起始y</param>
    /// <param name="width">裁剪图片的宽度</param>
    /// <param name="height">裁剪图片的宽度</param>
    /// <returns>注意,x + width > img.cols,会自动只裁剪到图片结尾的宽度,高度同理</returns>
    static cv::Mat cutImage(const cv::Mat& img, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height);
    static cv::Mat cutImage(const std::string& imgPath, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height);
};

#endif  // OPENCV_TRANSFORM_H
