
#ifndef OPENCV_FILTER_H
#define OPENCV_FILTER_H

#include "opencv_global.h"
#include <opencv2/opencv.hpp>

class opencvFilter
{
  public:
    /// <summary>
    /// 提取边界
    /// </summary>
    /// <param name="img"></param>
    /// <param name="low_threshold"></param>
    /// <param name="height_threshold"></param>
    /// <returns></returns>
    static cv::Mat edgeDetection(const cv::Mat img, int low_threshold, int height_threshold);

    //////////////// 霍夫变换 ////////////////
    // 霍夫直线变换
    static cv::Mat houghDetectLines(const cv::Mat& inputImage);

    // 霍夫圆变换(参数有问题)
    static cv::Mat houghDetectCircles(const cv::Mat& inputImage);

    /// <summary>
    /// 绘制轮廓
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    static cv::Mat drawOutline(const cv::Mat& image);

    /// <summary>
    /// 绘制矩形框
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    static cv::Mat drawRectangleOutline(const cv::Mat& image);

    /// <summary>
    /// 灰度直方图
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    static cv::Mat calculateHistogram(const cv::Mat& image);

    /// <summary>
    /// 角点检测
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    static cv::Mat detectAndMarkCorners(const cv::Mat& image);

    /////////////////////// 图像滤波 ///////////////////////

    //////////////// 线性滤波 ////////////////

    /// <summary>
    /// 均值滤波
    /// </summary>
    /// <param name="inputImage">需处理的图片</param>
    /// <param name="kernelSize">均值滤波核(越小:保留细节多,去噪效果欠佳, 越大:更加平滑,去噪效果好,图片细节模糊)</param>
    /// <returns></returns>
    static cv::Mat meanFilter(const cv::Mat& inputImage, int kernelSize);

    /// <summary>
    /// 高斯滤波
    /// </summary>
    /// <param name="inputImage"></param>
    /// <param name="kernelSize">滤波核</param>
    /// <returns></returns>
    static cv::Mat gaussianBlurFilter(const cv::Mat& inputImage, int kernelSize);

    /// <summary>
    /// 方框滤波
    /// </summary>
    /// <param name="inputImage"></param>
    /// <param name="kernelSize">滤波核</param>
    /// <returns></returns>
    static cv::Mat applyBoxFilter(const cv::Mat& inputImage, int kernelSize);

    //////////////// 非线性滤波 ////////////////

    /// <summary>
    /// 中值滤波
    /// </summary>
    /// <param name="inputImage"></param>
    /// <param name="kernelSize"></param>
    /// <returns></returns>
    static cv::Mat medianFilter(const cv::Mat& inputImage, int kernelSize);

    /// <summary>
    /// 双边滤波
    /// </summary>
    /// <param name="inputImage"></param>
    /// <param name="diameter">滤波器的直径，控制在像素范围内运行的滤波器直径</param>
    /// <param name="sigmaColor">色彩空间的标准偏差，用于根据颜色相似性衡量两个像素之间的距离</param>
    /// <param name="sigmaSpace">空间坐标的标准偏差，用于根据空间距离衡量两个像素之间的距离</param>
    /// <returns></returns>
    static cv::Mat applyBilateralFilter(const cv::Mat& inputImage, int diameter, double sigmaColor, double sigmaSpace);

    /// <summary>
    /// 非局部均值滤波
    /// </summary>
    /// <param name="inputImage"></param>
    /// <param name="h">控制滤波器强度，一般取一个较小的值，例如 3.0</param>
    /// <param name="templateWindowSize">均值估计过程中考虑的像素邻域窗口大小</param>
    /// <param name="searchWindowSize">搜索区域窗口大小，控制在哪个区域搜索相似像素</param>
    /// <returns></returns>
    static cv::Mat nonLocalMeansFilter(const cv::Mat& inputImage, double h, int templateWindowSize, int searchWindowSize);

    //////////////// 形态学滤波 ////////////////

    /// <summary>
    /// 腐蚀操作
    /// 将核覆盖区域内的像素设置为核内的最小像素值。有助于消除较小的白色噪声，缩小物体的大小，并分离相互连接的物体
    /// </summary>
    /// <param name="src"></param>
    /// <param name="kernel"></param>
    /// <returns></returns>
    static cv::Mat erosionFilter(const cv::Mat& src, const int kernelSize);

    /// <summary>
    /// 膨胀操作
    /// 将核内的最大像素值赋予覆盖区域内的像素。膨胀操作可以扩大物体的尺寸，并填充图像中的小洞
    /// </summary>
    /// <param name="src"></param>
    /// <returns></returns>
    static cv::Mat dilationFilter(const cv::Mat& src, const int kernelSize);

    /// <summary>
    /// 开运算
    /// 先腐蚀操作，再膨胀操作的组合。有助于消除小物体、平滑边界、分离物体,用于去除噪声，并保持物体的形状
    /// </summary>
    /// <param name="src"></param>
    /// <param name="kernel"></param>
    /// <returns></returns>
    static cv::Mat openingFilter(const cv::Mat& src, const int kernelSize);

    /// <summary>
    /// 闭运算
    /// 先膨胀操作，再腐蚀操作的组合。有助于填充物体内的小洞、连接相邻物体等，常用于处理前景对象的内部区域
    /// </summary>
    /// <param name="src"></param>
    /// <param name="kernel"></param>
    /// <returns></returns>
    static cv::Mat closingFilter(const cv::Mat& src, const int kernelSize);

    /// <summary>
    /// 计算并显示直方图
    /// </summary>
    /// <param name="filename"></param>
    static void computeAndShowHistogram(const std::string& filename);
};

#endif  // OPENCV_FILTER_H
