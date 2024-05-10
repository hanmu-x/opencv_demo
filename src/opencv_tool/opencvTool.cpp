
#include "opencvTool.h"

//#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <opencv2/imgproc.hpp>

using namespace std;
//using namespace cv;

// 将数据类型转换为字符串
std::string opencvTool::type2str(int type) 
{
	std::string typeStr;
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) 
	{
	case CV_8U:  typeStr = "8U"; break;
	case CV_8S:  typeStr = "8S"; break;
	case CV_16U: typeStr = "16U"; break;
	case CV_16S: typeStr = "16S"; break;
	case CV_32S: typeStr = "32S"; break;
	case CV_32F: typeStr = "32F"; break;
	case CV_64F: typeStr = "64F"; break;
	default:     typeStr = "User"; break;
	}
	typeStr += "C";
	typeStr += (chans + '0'); // 3 个颜色通道（C3）
	return typeStr;
}

cv::Mat opencvTool::openImage(const std::string& image_path)
{
	// 使用imread函数加载图像
	cv::Mat image = cv::imread(image_path);

	// 检查图像是否成功加载
	if (image.empty()) 
	{
		std::cout << "Could not open or find the image: " << image_path << std::endl;
	}

	return image;
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

cv::Mat opencvTool::creatEmptyMat(unsigned int width, unsigned int height, int imageType)
{
	// 创建一个空白图像
	cv::Mat blankImage(height, width, imageType, cv::Scalar(255, 255, 255));
	return blankImage;
}


cv::Mat opencvTool::creatColorMat(unsigned int width, unsigned int height, int imageType)
{
	// 创建一个空白图像
	cv::Mat gradientImage(height, width, imageType);

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

	return gradientImage;
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
	//cv::fillPoly(ima, std::vector<std::vector<cv::Point>>{points}, blue, 8, 0);
	cv::imwrite(image_p.c_str(), ima);
	return true;
}

bool opencvTool::drawPolygon(cv::Mat& image, std::vector<cv::Point> points, int lineWidth)
{
	if (image.empty())
	{
		std::cout << "Error: empty mat" << std::endl;
		return false;
	}

	// 确保多边形点的数量大于等于3
	if (points.size() < 3)
	{
		std::cout << "Error: need at least 3 points to draw a polygon" << std::endl;
		return false;
	}

	// 绘制多边形
	cv::polylines(image, points, true, cv::Scalar(0, 0, 255), lineWidth);

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

bool opencvTool::drawLines(cv::Mat& image, std::vector<cv::Point> points, int lineWidth)
{
	cv::Scalar red = cv::Scalar(0, 0, 255);  // Red color  

	// 遍历点列表，绘制线段
	for (size_t i = 0; i < points.size() - 1; i++)
	{
		cv::Point2f start = points[i];
		cv::Point2f end = points[i + 1];
		cv::line(image, start, end, red, lineWidth);
	}
	return true;
}


bool opencvTool::drawRectangle(const std::string image_p, const cv::Point& upperLeft, const cv::Point& lowerRight, int lineWidth)
{
	// 读取图像
	cv::Mat image = cv::imread(image_p);
	if (image.empty()) 
	{
		std::cerr << "Error: Unable to read image " << image_p << std::endl;
		return false;
	}

	// 绘制矩形
	cv::rectangle(image, upperLeft, lowerRight, cv::Scalar(255, 255, 255), lineWidth);

	cv::imwrite(image_p.c_str(), image);
	return true;
}


bool opencvTool::drawRectangle(cv::Mat& image, const cv::Point& upperLeft, const cv::Point& lowerRight, int lineWidth)
{
	// 访问左上角和右下角点的坐标
	int upperLeft_x = upperLeft.x;
	int upperLeft_y = upperLeft.y;
	int lowerRight_x = lowerRight.x;
	int lowerRight_y = lowerRight.y;

	std::cout << "Upper Left Point: (" << upperLeft_x << ", " << upperLeft_y << ")" << std::endl;
	std::cout << "Lower Right Point: (" << lowerRight_x << ", " << lowerRight_y << ")" << std::endl;
	// 绘制矩形
	cv::rectangle(image, upperLeft, lowerRight, cv::Scalar(0, 0, 255), lineWidth);

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

	//// 改变像素点的颜色
	//ima.at<cv::Vec3b>(y_coor, x_coor)[0] = 0;
	//ima.at<cv::Vec3b>(y_coor, x_coor)[1] = 0;
	//ima.at<cv::Vec3b>(y_coor, x_coor)[2] = 255;

	// 或者
	//uchar blue = 0;
	//uchar green = 0;
	//uchar red = 255;
	//ima.at<cv::Vec3b>(y_coor, x_coor) = cv::Vec3b(blue, green, red);

	// 更改指定坐标点的颜色
	ima.at<cv::Vec3b>(y_coor, x_coor) = cv::Vec3b(color[0], color[1], color[2]);
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


bool opencvTool::addText(cv::Mat& image, const std::string text, const cv::Point& position, double fontScale, cv::Scalar color, int thickness, int fontFace)
{
	cv::putText(image, text, position, fontFace, fontScale, color, thickness);
	return true;
}

bool opencvTool::addWatermark(cv::Mat& image, cv::Mat& watermark, int posX, int posY)
{
	// 确定水印在原始图像上的位置
	cv::Rect roi(posX, posY, watermark.cols, watermark.rows);

	// 为水印图像创建蒙版
	cv::Mat mask;
	cv::cvtColor(watermark, mask, cv::COLOR_BGR2GRAY);
	cv::threshold(mask, mask, 128, 255, cv::THRESH_BINARY);

	// 将水印叠加到原始图像上
	watermark.copyTo(image(roi), mask);

	return true;
}


bool opencvTool::addWatermark(cv::Mat& image, const std::string text, const cv::Point& position, double fontScale, cv::Scalar color, int thickness, int fontFace)
{
	// 创建一个透明的图像，大小与原始图像相同
	cv::Mat watermark = cv::Mat::zeros(image.size(), image.type());

	// 在透明图像上添加文字
	cv::putText(watermark, text, position, fontFace, fontScale, color, thickness);

	// 将透明图像作为水印叠加到原始图像上
	cv::addWeighted(image, 1.0, watermark, 0.5, 0.0, image);
	return true;
}

// 鼠标回调函数
void draw_circle(int event, int x, int y, int flags, void* param)
{
	cv::Mat* img = (cv::Mat*)param;
	if (event == cv::EVENT_LBUTTONDBLCLK)
	{
		cv::circle(*img, cv::Point(x, y), 100, cv::Scalar(0, 0, 255), -1);
	}
}

// 鼠标回调函数
void draw_line(int event, int x, int y, int flags, void* param)
{
	static cv::Point draw_line_startp;  // 一定要是static
	cv::Mat* img = (cv::Mat*)param;

	if (event == cv::EVENT_LBUTTONDOWN) // 鼠标左键按下时执行以下代码块。
	{
		draw_line_startp = cv::Point(x, y); // 记录鼠标按下时的坐标作为起始点
	}
	else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) //当鼠标左键按下并移动时执行以下代码块。
	{
		cv::Point end_point(x, y); // 获取当前鼠标移动位置作为终点
		cv::line(*img, draw_line_startp, end_point, cv::Scalar(0, 0, 255), 2); // 在图像上绘制线段
		draw_line_startp = end_point; // 更新起始点为当前终点，以便下一次绘制
	}

}

void opencvTool::drawingByMouse()
{
	// 创建一个黑色的图像
	//cv::Mat img = cv::Mat::zeros(512, 512, CV_8UC3);  
	// 创建一个白色的图像
	cv::Mat img(512, 512, CV_8UC3, cv::Scalar(255, 255, 255));

	// 创建一个窗口并绑定回调函数
	cv::namedWindow("image");
	cv::setMouseCallback("image", draw_line, &img); // 该函数将在鼠标事件发生时被调用

	// 进入主循环，显示图像
	while (true)
	{
		imshow("image", img);
		if (cv::waitKey(20) == 27) // 按下Esc键（对应的ASCII码是27）。
		{
			break;
		}
	}
	// 关闭窗口
	cv::destroyAllWindows();
	return;
}

cv::Mat opencvTool::BGRToHSV(cv::Mat bgr_image)
{
	// 创建一个用于存储HSV图像的Mat对象
	cv::Mat hsv_image;

	// 将BGR图像转换为HSV图像
	cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
	return hsv_image;
}

// 调整图像大小的函数
cv::Mat opencvTool::resizeImage(const cv::Mat& img, double scale_factor)
{
	cv::Mat resized_img;
	// 使用缩放因子进行放大
	cv::resize(img, resized_img, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);

	//// 使用目标大小进行放大
	//cv::Mat resized_img;
	//cv::resize(img, resized_img, cv::Size(2 * width, 2 * height), 0, 0, cv::INTER_CUBIC);

	return resized_img;
}

// 定义函数，实现图像的平移变换
cv::Mat opencvTool::translateImage(const cv::Mat& img, int dx, int dy)
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

cv::Mat opencvTool::rotateImage(const cv::Mat& img, double angle)
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


cv::Mat opencvTool::edgeDetection(const cv::Mat img, int low_threshold, int height_threshold)
{
	// 读取图像
	//cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
	//if (img.empty()) {
	//	std::cerr << "Error: Unable to load image " << filename << std::endl;
	//	return;
	//}

	// 边缘检测
	cv::Mat edges;
	cv::Canny(img, edges, low_threshold, height_threshold);

	return edges;

}


cv::Mat opencvTool::drawOutline(const cv::Mat& image)
{
	cv::Mat imgray;
	cv::cvtColor(image, imgray, cv::COLOR_BGR2GRAY);

	cv::Mat thresh;
	cv::threshold(imgray, thresh, 127, 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat contourImg = cv::Mat::zeros(imgray.size(), CV_8UC3); // 创建一个空白图像，与原图大小相同

	// 绘制轮廓
	for (size_t i = 0; i < contours.size(); ++i) {
		cv::Scalar color = cv::Scalar(0, 255, 0); // 轮廓颜色为绿色
		cv::drawContours(contourImg, contours, static_cast<int>(i), color, 2, 8, hierarchy, 0);
	}

	return contourImg;

}



cv::Mat opencvTool::drawRectangleOutline(const cv::Mat& image)
{
	cv::Mat outputImage = image.clone(); // 创建输出图像并复制输入图像

	// 将输入图像转换为灰度图像
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	// 对灰度图像进行阈值处理
	cv::Mat thresholdedImage;
	cv::threshold(grayImage, thresholdedImage, 127, 255, cv::THRESH_BINARY);

	// 查找轮廓
	std::vector<vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(thresholdedImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	// 绘制轮廓
	for (size_t i = 0; i < contours.size(); ++i) 
	{
		cv::Scalar color = cv::Scalar(0, 255, 0); // 绿色
		drawContours(outputImage, contours, static_cast<int>(i), color, 2, 8, hierarchy, 0);
	}

	return outputImage;
}


cv::Mat opencvTool::calculateHistogram(const cv::Mat& image)
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
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	cv::calcHist(&grayscale_image, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	// 绘制直方图
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for (int i = 1; i < histSize; i++)
	{
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

	return histImage;
}









opencvTool::opencvTool()
{
	std::cout << "Start opencvTool" << std::endl;

}
opencvTool::~opencvTool()
{
	std::cout << "End opencvTool" << std::endl;

}













