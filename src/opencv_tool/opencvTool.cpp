
#include "opencvTool.h"


//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"

//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

using namespace std;
//using namespace cv;

// 将数据类型转换为字符串
std::string opencvTool::type2str(int type) 
{
	std::string r;
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) 
	{
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}
	r += "C";
	r += (chans + '0'); //3 个颜色通道（C3）
	return r;
}

bool opencvTool::showImage(std::string image_p)
{
	cv::Mat image = cv::imread(image_p.c_str());
	if (image.empty())
	{
		std::cout << "Error: empyt mat " << std::endl;
		return false;
	}

	// 打印图像信息
	std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;
	std::cout << "Number of channels: " << image.channels() << std::endl;  // 通道数
	std::cout << "Data type: " << type2str(image.type()) << std::endl; // 自定义函数，用于将数据类型转换为字符串

	cv::imshow("test", image);
	cv::waitKey(0);  //持续显示窗口
	cv::destroyAllWindows();  //用于关闭所有由 OpenCV 创建的窗口
	return true;
}

bool opencvTool::showImage(cv::Mat image)
{
	if (image.empty())
	{
		std::cout << "Error: empty mat " << std::endl;
		return false;
	}

	// 打印图像信息
	std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
	std::cout << "Number of channels: " << image.channels() << std::endl;  // 通道数
	std::cout << "Data type: " << type2str(image.type()) << std::endl; // 自定义函数，用于将数据类型转换为字符串

	// 显示图像
	cv::imshow("test", image);
	cv::waitKey(0);  // 持续显示窗口
	cv::destroyAllWindows();  // 用于关闭所有由 OpenCV 创建的窗口
	return true;
}


bool opencvTool::creatEmptyImage(unsigned int width, unsigned int height, std::string image_p)
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

cv::Mat opencvTool::creatEmptyMat(unsigned int width, unsigned int height)
{
	// 创建一个空白图像
	cv::Mat blankImage(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
	return blankImage;
}


bool opencvTool::creatColor(std::string image_p)
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


bool opencvTool::saveImage(const std::string path, const cv::Mat image)
{
	if (image.empty())
	{
		std::cout << "Error: empty mat " << std::endl;
		return false;
	}

	// 打印图像信息
	std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
	std::cout << "Number of channels: " << image.channels() << std::endl;  // 通道数
	std::cout << "Data type: " << type2str(image.type()) << std::endl; // 自定义函数，用于将数据类型转换为字符串

	// 保存图像文件
	try 
	{
		cv::imwrite(path, image);
		std::cout << "Image saved successfully!" << std::endl;
		return true;
	}
	catch (cv::Exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
		return false;
	}
}




bool opencvTool::drawPolygon(std::string image_p, std::vector<cv::Point> points)
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


bool opencvTool::drawLines(std::string image_p, std::vector<cv::Point> points)
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


bool opencvTool::changeColor(const std::string image_p, int x_coor, int y_coor, const cv::Scalar color)
{
	std::filesystem::path file(image_p);
	if (!std::filesystem::exists(file))
	{
		std::cout << image_p << " is not exist" << std::endl;
		return false;
	}
	cv::Mat ima = cv::imread(image_p.c_str()); // 读取图像，替换为你的图片路径  
	cv::Scalar Red = cv::Scalar(0, 0, 255);  // Red color  

	if (x_coor> ima.cols || x_coor < 0)
	{
		printf("Input x_coor: %d which exceeds width range of the image: %d \n", x_coor, ima.cols);
		return false;
	}
	if (y_coor > ima.rows || y_coor< 0)
	{
		printf("Input y_coor: %d which exceeds height range of the image: %d \n", y_coor, ima.rows);
		return false;
	}

	// 改变像素点的颜色
	ima.at<cv::Vec3b>(y_coor, x_coor)[0] = 0;
	ima.at<cv::Vec3b>(y_coor, x_coor)[1] = 0;
	ima.at<cv::Vec3b>(y_coor, x_coor)[2] = 255;

	// 或者
	//uchar blue = 0;
	//uchar green = 0;
	//uchar red = 255;
	//ima.at<cv::Vec3b>(y_coor, x_coor) = cv::Vec3b(blue, green, red);

	cv::imwrite(image_p.c_str(), ima);
	return true;

}

bool opencvTool::changeColor(cv::Mat& image, int x_coor, int y_coor, const cv::Scalar color)
{
	if (image.empty())
	{
		std::cout << "Error: empty mat" << std::endl;
		return false;
	}

	if (x_coor > image.cols || x_coor < 0)
	{
		printf("Input x_coor: %d which exceeds width range of the image: %d \n", x_coor, image.cols);
		return false;
	}
	if (y_coor > image.rows || y_coor < 0)
	{
		printf("Input y_coor: %d which exceeds height range of the image: %d \n", y_coor, image.rows);
		return false;
	}

	// 更改指定坐标点的颜色
	image.at<cv::Vec3b>(y_coor, x_coor) = cv::Vec3b(color[0], color[1], color[2]);

	return true;
}














opencvTool::opencvTool()
{
	std::cout << "Start opencvTool" << std::endl;

}
opencvTool::~opencvTool()
{
	std::cout << "End opencvTool" << std::endl;

}













