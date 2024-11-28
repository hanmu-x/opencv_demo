
#include "opencv_transform.h"
#include "opencv_IO.h"


cv::Mat opencvTF::resizeImage(const cv::Mat& img, double scale_factor)
{
    cv::Mat resized_img;
    // 使用缩放因子进行放大
    cv::resize(img, resized_img, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);

    //// 使用目标大小进行放大
    // cv::Mat resized_img;
    // cv::resize(img, resized_img, cv::Size(2 * width, 2 * height), 0, 0, cv::INTER_CUBIC);

    return resized_img;
}

// 定义函数，实现图像的平移变换
cv::Mat opencvTF::translateImage(const cv::Mat& img, int dx, int dy)
{
    // 获取图像尺寸
    int rows = img.rows;
    int cols = img.cols;

    // 定义仿射变换矩阵
    cv::Mat M = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);

    // 进行仿射变换
    cv::Mat dst;
    cv::warpAffine(img, dst, M, cv::Size(cols, rows));

    return dst;
}

cv::Mat opencvTF::rotateImage(const cv::Mat& img, double angle)
{
    // 获取图像尺寸
    int rows = img.rows;
    int cols = img.cols;

    // 计算旋转中心
    cv::Point2f center((cols - 1) / 2.0, (rows - 1) / 2.0);

    // 获取旋转矩阵, 缩放因子（这里是1，表示不进行缩放）。
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1);

    // 进行仿射变换
    cv::Mat dst;
    cv::warpAffine(img, dst, M, cv::Size(cols, rows));

    return dst;
}

cv::Mat opencvTF::cutImage(const cv::Mat& img, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height)
{
    unsigned int cols = width;
    unsigned int rows = height;
    // 检查裁剪区域是否在图像范围内
    if (x + width > img.cols)
    {
        cols = img.cols - x;
    }
    if (y + height > img.rows)
    {
        rows = img.rows - y;
    }

    // 裁剪图像，使用矩形区域的坐标（x, y）和尺寸（width, height）
    cv::Rect roi(x, y, width, rows);
    // cv::Rect roi(x, y, cols, height);
    cv::Mat croppedImage = img(roi);

    return croppedImage;
}

cv::Mat opencvTF::cutImage(const std::string& imgPath, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height)
{
    cv::Mat img = opencvIO::openImage(imgPath);
    if (img.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return img;
    }

    cv::Mat croppedImage = opencvTF::cutImage(img, x, y, width, height);

    return croppedImage;
}