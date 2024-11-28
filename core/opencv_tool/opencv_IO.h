
#ifndef OPENCVIO_H
#define OPENCVIO_H

#include "opencv_global.h"
#include <opencv2/opencv.hpp>

class opencvIO
{
  public:
    /// <summary>
    /// 1. OpenCV 的坐标系在左上角,横轴为x轴(向右为正方向),纵轴为y轴(向下为正方向)
    /// 2. OpenCV 颜色通道顺序为 BGR
    /// </summary>

    /// <summary>
    /// 将数据类型转换为字符串
    /// </summary>
    /// <param name="type"></param>
    /// <returns> 如"8UC3"，它表示图像的深度为 8 位无符号整数（8U）且具有 3 个颜色通道（C3）</returns>
    static std::string type2str(int type);

    static cv::Mat openImage(const std::string& image_path);

    /// <summary>
    /// 可视化展示图片
    /// </summary>
    /// <param name="image_p"></param>
    /// <returns></returns>
    static bool showImage(std::string image_p);
    static bool showImage(cv::Mat image);

    /// <summary>
    /// 创建一个空白图片
    /// </summary>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="image_p"></param>
    /// <returns></returns>
    static bool creatEmptyImage(unsigned int width, unsigned int height, std::string image_p);
    static cv::Mat creatEmptyMat(unsigned int width, unsigned int height, int imageType = CV_8UC3);

    /// <summary>
    /// 创建一个渐变彩色图片
    /// </summary>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="imageType"></param>
    /// <returns></returns>
    static cv::Mat creatColorMat(unsigned int width, unsigned int height, int imageType = CV_8UC3);

    /// <summary>
    /// 创建一个渐变彩色图片
    /// </summary>
    /// <param name="image_p"></param>
    /// <returns></returns>
    static bool creatColor(std::string image_p);

    /// <summary>
    /// 保存图片
    /// </summary>
    /// <param name="path"></param>
    /// <param name="image"></param>
    /// <returns></returns>
    static bool saveImage(const std::string path, const cv::Mat image);

    /// <summary>
    /// 将文件字符串转换为Mat
    /// </summary>
    /// <param name="base64Str">一个图片文件的全部字符串</param>
    /// <returns></returns>
    static cv::Mat memoryEncode(const std::string& base64Str);

    /// <summary>
    /// 将Mat转换为文件字符串
    /// </summary>
    /// <param name="img"></param>
    /// <returns></returns>
    static std::string memoryDecode(const cv::Mat& img, const std::string extension = "png");

    /// <summary>
    /// BGR图片转HSV图片
    /// </summary>
    /// <param name="bgr_image"></param>
    /// <returns></returns>
    static cv::Mat BGRToHSV(cv::Mat bgr_image);
};

#endif  // OPENCVIO_H