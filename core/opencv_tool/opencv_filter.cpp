
#include "opencv_filter.h"
#include "opencv_IO.h"
#include <opencv2/imgproc.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/calib3d.hpp>

cv::Mat opencvFilter::edgeDetection(const cv::Mat img, int low_threshold, int height_threshold)
{
    // 读取图像
    // cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    // if (img.empty()) {
    //	std::cout << "Error: Unable to load image " << filename << std::endl;
    //	return;
    //}

    // 边缘检测
    cv::Mat edges;
    cv::Canny(img, edges, low_threshold, height_threshold);

    return edges;
}

cv::Mat opencvFilter::houghDetectLines(const cv::Mat& inputImage)
{
    cv::Mat gray, edges;
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150, 3);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 150);

    cv::Mat result(inputImage.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    inputImage.copyTo(result);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        cv::line(result, pt1, pt2, cv::Scalar(0, 0, 255), 2, 8);
    }
    return result;
}

cv::Mat opencvFilter::houghDetectCircles(const cv::Mat& inputImage)
{
    cv::Mat gray, edges;
    cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    opencvIO::showImage(gray);
    // cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);  // 用于降噪的GaussianBlur
    cv::Canny(gray, edges, 0, 200, 3);
    opencvIO::showImage(edges);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(edges, circles, 3, 1, edges.rows / 8, 200, 100, 0, 0);

    cv::Mat result(inputImage.size(), CV_8UC3);
    // inputImage.copyTo(result);
    for (size_t i = 0; i < circles.size(); ++i)
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(result, center, radius, cv::Scalar(0, 0, 255), 20, 16);
    }
    return result;
};

cv::Mat opencvFilter::drawOutline(const cv::Mat& image)
{
    cv::Mat imgray;
    cv::cvtColor(image, imgray, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    cv::threshold(imgray, thresh, 127, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat contourImg = cv::Mat::zeros(imgray.size(), CV_8UC3);  // 创建一个空白图像，与原图大小相同

    // 绘制轮廓
    for (size_t i = 0; i < contours.size(); ++i)
    {
        cv::Scalar color = cv::Scalar(0, 255, 0);  // 轮廓颜色为绿色
        cv::drawContours(contourImg, contours, static_cast<int>(i), color, 2, 8, hierarchy, 0);
    }

    return contourImg;
}

cv::Mat opencvFilter::drawRectangleOutline(const cv::Mat& image)
{
    cv::Mat outputImage = image.clone();  // 创建输出图像并复制输入图像

    // 将输入图像转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 对灰度图像进行阈值处理
    cv::Mat thresholdedImage;
    cv::threshold(grayImage, thresholdedImage, 127, 255, cv::THRESH_BINARY);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thresholdedImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓
    for (size_t i = 0; i < contours.size(); ++i)
    {
        cv::Scalar color = cv::Scalar(0, 255, 0);  // 绿色
        drawContours(outputImage, contours, static_cast<int>(i), color, 2, 8, hierarchy, 0);
    }

    return outputImage;
}

cv::Mat opencvFilter::calculateHistogram(const cv::Mat& image)
{
    // 如果输入图像尚未处于灰度级，请将其转换为灰度级
    cv::Mat grayscale_image;
    if (image.channels() > 1)
    {
        cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);
    }
    else
    {
        grayscale_image = image.clone();
    }

    // 计算直方图
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    cv::calcHist(&grayscale_image, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // 绘制直方图
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))), cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), cv::Scalar(0, 0, 0), 2, 8, 0);
    }

    return histImage;
}

cv::Mat opencvFilter::meanFilter(const cv::Mat& inputImage, int kernelSize)
{
    cv::Mat blurredImage;
    cv::blur(inputImage, blurredImage, cv::Size(kernelSize, kernelSize));
    return blurredImage;
}

cv::Mat opencvFilter::gaussianBlurFilter(const cv::Mat& inputImage, int kernelSize)
{
    cv::Mat blurredImage;
    double sigmaX = 0;  // 设置标准差为0（自动计算）
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(kernelSize, kernelSize), sigmaX, sigmaX);
    return blurredImage;
}

cv::Mat opencvFilter::applyBoxFilter(const cv::Mat& inputImage, int kernelSize)
{
    cv::Mat filteredImage;
    // -1 表示输出图像与输入图像具有相同的深度
    cv::boxFilter(inputImage, filteredImage, -1, cv::Size(kernelSize, kernelSize));
    return filteredImage;
}

cv::Mat opencvFilter::medianFilter(const cv::Mat& inputImage, int kernelSize)
{
    cv::Mat filteredImage;
    cv::medianBlur(inputImage, filteredImage, kernelSize);
    return filteredImage;
}

cv::Mat opencvFilter::applyBilateralFilter(const cv::Mat& inputImage, int diameter, double sigmaColor, double sigmaSpace)
{
    cv::Mat filteredImage;
    cv::bilateralFilter(inputImage, filteredImage, diameter, sigmaColor, sigmaSpace);
    return filteredImage;
}

// 定义非局部均值滤波函数
cv::Mat opencvFilter::nonLocalMeansFilter(const cv::Mat& inputImage, double h, int templateWindowSize, int searchWindowSize)
{
    cv::Mat denoisedImage;
    cv::fastNlMeansDenoising(inputImage, denoisedImage, h, templateWindowSize, searchWindowSize);
    return denoisedImage;
}

// 函数：腐蚀操作
cv::Mat opencvFilter::erosionFilter(const cv::Mat& src, const int kernelSize)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::Mat result;
    cv::erode(src, result, kernel);
    return result;
}

// 函数：膨胀操作
cv::Mat opencvFilter::dilationFilter(const cv::Mat& src, const int kernelSize)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::Mat result;
    cv::dilate(src, result, kernel);
    return result;
}

// 函数：开运算
cv::Mat opencvFilter::openingFilter(const cv::Mat& src, const int kernelSize)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::Mat result;
    cv::morphologyEx(src, result, cv::MORPH_OPEN, kernel);
    return result;
}

// 函数：闭运算
cv::Mat opencvFilter::closingFilter(const cv::Mat& src, const int kernelSize)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::Mat result;
    cv::morphologyEx(src, result, cv::MORPH_CLOSE, kernel);
    return result;
}

// int opencvFilter::filtering_comparison(cv::Mat src)
//{
//
//	// 定义结构元素
//	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//
//	// 进行腐蚀操作
//	cv::Mat eroded;
//	cv::erode(src, eroded, kernel);
//
//	// 进行膨胀操作
//	cv::Mat dilated;
//	cv::dilate(src, dilated, kernel);
//
//	// 进行开运算
//	cv::Mat opened;
//	cv::morphologyEx(src, opened, cv::MORPH_OPEN, kernel);
//
//	// 进行闭运算
//	cv::Mat closed;
//	cv::morphologyEx(src, closed, cv::MORPH_CLOSE, kernel);
//
//	// 显示结果图像
//	cv::imshow("原图", src);
//	cv::imshow("腐蚀", eroded);
//	cv::imshow("膨胀", dilated);
//	cv::imshow("开运算", opened);
//	cv::imshow("闭运算", closed);
//
//	// 等待按键
//	cv::waitKey(0);
//	return 0;
// }

cv::Mat opencvFilter::detectAndMarkCorners(const cv::Mat& image)
{
    cv::Mat marked_image = image.clone();

    // 将输入图像转换为灰度图像
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Harris角点检测参数
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    // 使用Harris角点检测检测角点
    cv::Mat dst, dst_norm;
    dst = cv::Mat::zeros(image.size(), CV_32FC1);
    // Harris角点检测需要灰度图
    cv::cornerHarris(gray_image, dst, blockSize, apertureSize, k);

    // 对Harris角点检测的输出进行归一化处理
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // 在输入图像上标记角点
    for (int i = 0; i < dst_norm.rows; ++i)
    {
        for (int j = 0; j < dst_norm.cols; ++j)
        {
            if ((int)dst_norm.at<float>(i, j) > 100)
            {  // 阈值可调
                cv::circle(marked_image, cv::Point(j, i), 20, cv::Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }

    return marked_image;
}

void opencvFilter::computeAndShowHistogram(const std::string& filename)
{
    // Load image
    cv::Mat img = cv::imread(filename);
    if (img.empty())
    {
        std::cout << "Error: Could not open or find the image: " << filename << std::endl;
        return;
    }

    // 转换为灰度
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    // 显示原始图像
    cv::imshow("[img1]", img);
    cv::imshow("[gray_img]", gray_img);

    // 计算直方图
    cv::MatND hist;
    int dims = 1;
    float hranges[] = {0, 255};
    const float* ranges[] = {hranges};
    int size = 256;
    int channels = 0;
    cv::calcHist(&gray_img, 1, &channels, cv::Mat(), hist, dims, &size, ranges);

    // 创建直方图可视化
    int scale = 1;
    int hist_height = 256;
    cv::Mat hist_img(hist_height, size * scale, CV_8U, cv::Scalar(0));

    double minValue = 0;
    double maxValue = 0;
    cv::minMaxLoc(hist, &minValue, &maxValue);

    int hpt = static_cast<int>(0.9 * size);
    for (int i = 0; i < size; ++i)
    {
        float binValue = hist.at<float>(i);
        int realValue = cv::saturate_cast<int>(binValue * hpt / maxValue);
        cv::rectangle(hist_img, cv::Point(i * scale, hist_height - 1), cv::Point((i + 1) * scale - 1, hist_height - realValue), cv::Scalar(255));
    }

    // 显示柱状图
    cv::imshow("[hist]", hist_img);

    // 分解直方图
    cv::Mat equal_img;
    cv::equalizeHist(gray_img, equal_img);
    cv::imshow("[equal_img]", equal_img);

    // 计算均衡直方图
    cv::calcHist(&equal_img, 1, &channels, cv::Mat(), hist, dims, &size, ranges);
    cv::minMaxLoc(hist, &minValue, &maxValue);

    // 创建均衡直方图可视化
    cv::Mat equal_hist_img(hist_height, size * scale, CV_8U, cv::Scalar(0));
    for (int i = 0; i < size; ++i)
    {
        float binValue = hist.at<float>(i);
        int realValue = cv::saturate_cast<int>(binValue * hpt / maxValue);
        cv::rectangle(equal_hist_img, cv::Point(i * scale, hist_height - 1), cv::Point((i + 1) * scale - 1, hist_height - realValue), cv::Scalar(255));
    }

    // 显示均衡直方图
    cv::imshow("[equal_hist]", equal_hist_img);

    cv::waitKey(0);
}