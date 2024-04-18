
#include "opencvTool.h"


//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"

#include <opencv2/core.hpp>
#include <filesystem>

using namespace std;
//using namespace cv;

int tool_class::opencvReadImage(std::string image_p)
{
    cv::Mat image = cv::imread(image_p.c_str());
    if (image.empty())
    {
        return -1;
    }
    //namedWindow("test", WINDOW_AUTOSIZE);
    cv::imshow("test", image);
    cv::waitKey(0);  //持续显示窗口
    cv::destroyAllWindows();  //用于关闭所有由 OpenCV 创建的窗口
    return 0;



}

bool tool_class::creatEmpty(int width, int height, std::string image_p)
{

    // 创建一个空白图像
    cv::Mat blankImage(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 保存图像为文件（可选）
    cv::imwrite(image_p.c_str(), blankImage);

    // 显示空白图像
    cv::imshow("Blank Image", blankImage);

    // 等待用户按下任意键后关闭窗口
    cv::waitKey(0);

    // 关闭窗口
    cv::destroyAllWindows();
    return true;
}




bool tool_class::creatColor(std::string image_p)
{
    // 指定图像的宽度和高度
    int width = 640;
    int height = 480;

    // 创建一个空白图像
    cv::Mat gradientImage(height, width, CV_8UC3);

    // 生成渐变色
    for (int y = 0; y < height; y++) 
    {
        for (int x = 0; x < width; x++) 
        {
            // 计算RGB颜色值，根据x和y的位置生成渐变色
            uchar blue = static_cast<uchar>(x * 255 / width);
            uchar green = static_cast<uchar>((x + y) * 255 / (width + height));
            uchar red = static_cast<uchar>(y * 255 / height);
            // 设置像素颜色
            // (三通道:用at<Vec3b>(row, col)
            // (单通道:at<uchar>(row, col))
            gradientImage.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
        }
    }

    // 保存图像为文件（可选）
    cv::imwrite(image_p.c_str(), gradientImage);

    // 显示渐变色图像
    cv::imshow("Gradient Image", gradientImage);

    // 等待用户按下任意键后关闭窗口
    cv::waitKey(0);

    // 关闭窗口
    cv::destroyAllWindows();

    return true;
}


bool tool_class::drawPolygon(std::string image_p, std::vector<cv::Point> points)
{
    cv::Mat ima = cv::imread(image_p.c_str()); // 读取图像，替换为你的图片路径  

    cv::Scalar red = cv::Scalar(0, 0, 255);  // Red color  
    cv::Scalar blue = cv::Scalar(255, 0, 0);  // Red color  
    int thickness = 2;

    // 使用polylines函数给图片绘制多边形
    cv::polylines(ima, points, true, red, thickness, 8, 0);
    // 填充颜色
    cv::fillPoly(ima, std::vector<std::vector<cv::Point>>{points}, blue, 8, 0);
    
    cv::imwrite(image_p.c_str(), ima);

    // // 显示图像  
    //cv::imshow("Image with line", ima);
    // // 等待用户按下任意键后关闭窗口
    //cv::waitKey(0);
    // // 关闭窗口
    //cv::destroyAllWindows();

    return true;
}


bool tool_class::drawLines(std::string image_p, std::vector<cv::Point> points)
{
    cv::Mat ima = cv::imread(image_p.c_str()); // 读取图像，替换为你的图片路径  
    cv::Scalar red = cv::Scalar(0, 0, 255);  // Red color  
    int thickness = 2;

    // 遍历点列表，绘制线段
    for (size_t i = 0; i < points.size() - 1; i++)
    {
        cv::Point2f start = points[i];
        cv::Point2f end = points[i + 1];

        cv::line(ima, start, end, red, thickness);
    }

    cv::imwrite(image_p.c_str(), ima);

    return true;
}


bool tool_class::changeColor(std::string image_p, int width, int height)
{
    cv::Mat ima = cv::imread(image_p.c_str()); // 读取图像，替换为你的图片路径  
    cv::Scalar Red = cv::Scalar(0, 0, 255);  // Red color  

    // 改变像素点的颜色
    ima.at<cv::Vec3b>(height, width)[0] = 0;
    ima.at<cv::Vec3b>(height, width)[1] = 0;
    ima.at<cv::Vec3b>(height, width)[2] = 255;

    // 或者
    //uchar blue = 0;
    //uchar green = 0;
    //uchar red = 255;
    //ima.at<cv::Vec3b>(height, width) = cv::Vec3b(blue, green, red);

    cv::imwrite(image_p.c_str(), ima);
    return true;

}

















tool_class::tool_class()
{
	std::cout << "Start tool_class" << std::endl;

}
tool_class::~tool_class()
{
	std::cout << "End tool_class" << std::endl;

}













