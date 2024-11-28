
#ifndef OPENCV_DRAWER_H
#define OPENCV_DRAWER_H

#include "opencv_global.h"
#include <opencv2/opencv.hpp>

typedef std::vector<cv::Point> Polygon;

class opencvDrawer
{
  public:
    /// <summary>
    /// 绘制多边形
    /// </summary>
    /// <param name="image_p"></param>
    /// <param name="line"></param>
    /// <returns></returns>
    static bool drawPolygon(std::string image_p, std::vector<cv::Point> points);
    static bool drawPolygon(cv::Mat& image, std::vector<cv::Point> points, int lineWidth = 1);

    /// <summary>
    /// 绘制线段
    /// </summary>
    /// <param name="image_p"></param>
    /// <param name="points"></param>
    /// <returns></returns>
    static bool drawLines(std::string image_p, std::vector<cv::Point> points);
    static bool drawLines(cv::Mat& image, std::vector<cv::Point> points, int lineWidth = 1);

    /// <summary>
    /// 绘制矩形
    /// </summary>
    /// <param name="image_p"></param>
    /// <param name="upperLeft"></param>
    /// <param name="lowerRight"></param>
    /// <param name="lineWidth"></param>
    /// <returns></returns>
    static bool drawRectangle(const std::string image_p, const cv::Point& upperLeft, const cv::Point& lowerRight, int lineWidth = 1);
    static bool drawRectangle(cv::Mat& image, const cv::Point& upperLeft, const cv::Point& lowerRight, int lineWidth = 1);

    /// <summary>
    /// 改变指定像素点的颜色
    /// </summary>
    /// <param name="image_p"></param>
    /// <param name="x_coor">x轴坐标</param>
    /// <param name="y_coor">y轴坐标</param>
    /// <param name="color">颜色</param>
    /// <returns></returns>
    static bool changeColor(const std::string image_p, int x_coor, int y_coor, const cv::Scalar color);
    static bool changeColor(cv::Mat& image, int x_coor, int y_coor, const cv::Scalar color);

    /// <summary>
    /// 给图片添加文字
    /// </summary>
    /// <param name="image"></param>
    /// <param name="text"></param>
    /// <param name="position">文字的位置</param>
    /// <param name="fontScale">字体大小</param>
    /// <param name="color">颜色</param>
    /// <param name="thickness">粗细</param>
    /// <param name="fontFace">字体</param>
    /// <returns></returns>
    static bool addText(cv::Mat& image, const std::string text, const cv::Point& position, double fontScale = (1.0), cv::Scalar color = cv::Scalar(0, 0, 255), int thickness = 1, int fontFace = cv::FONT_HERSHEY_COMPLEX);

    /// <summary>
    /// 添加水印
    /// </summary>
    /// <param name="image"></param>
    /// <param name="watermark"></param>
    /// <param name="posX"></param>
    /// <param name="posY"></param>
    /// <returns></returns>
    static bool addWatermark(cv::Mat& image, cv::Mat& watermark, int posX, int posY);
    static bool addWatermark(cv::Mat& image, const std::string text, const cv::Point& position, double fontScale = (1.0), cv::Scalar color = cv::Scalar(0, 0, 255), int thickness = 1, int fontFace = cv::FONT_HERSHEY_COMPLEX);

    /// <summary>
    /// 鼠标绘制线段
    /// </summary>
    static void drawingByMouse();

    /// <summary>
    /// BGR图片转HSV图片
    /// </summary>
    /// <param name="bgr_image"></param>
    /// <returns></returns>
    static cv::Mat BGRToHSV(cv::Mat bgr_image);
};

#endif  // OPENCV_DRAWER_H
