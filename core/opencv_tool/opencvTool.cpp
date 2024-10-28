
#include "opencvTool.h"

#include <opencv2/core.hpp>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <filesystem>

//using namespace std;
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
		std::cout << "Error: Unable to read image " << image_p << std::endl;
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

	if (x_coor > ima.cols || x_coor < 0)
	{
		printf("Input x_coor: %d which exceeds width range of the image: %d \n", x_coor, ima.cols);
		return false;
	}
	if (y_coor > ima.rows || y_coor < 0)
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
	//	std::cout << "Error: Unable to load image " << filename << std::endl;
	//	return;
	//}

	// 边缘检测
	cv::Mat edges;
	cv::Canny(img, edges, low_threshold, height_threshold);

	return edges;

}




// 霍夫直线变换
cv::Mat opencvTool::houghDetectLines(const cv::Mat& inputImage)
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



// 霍夫圆变换
cv::Mat opencvTool::houghDetectCircles(const cv::Mat& inputImage)
{
	cv::Mat gray, edges;
	cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
	showImage(gray);
	//cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);  // 用于降噪的GaussianBlur
	cv::Canny(gray, edges, 0, 200, 3);
	showImage(edges);


	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(edges, circles, 3, 1, edges.rows / 8, 200, 100, 0, 0);

	cv::Mat result(inputImage.size(), CV_8UC3);
	//inputImage.copyTo(result);
	for (size_t i = 0; i < circles.size(); ++i)
	{
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		cv::circle(result, center, radius, cv::Scalar(0, 0, 255), 20, 16);
	}
	return result;
};





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



cv::Mat opencvTool::meanFilter(const cv::Mat& inputImage, int kernelSize)
{
	cv::Mat blurredImage;
	cv::blur(inputImage, blurredImage, cv::Size(kernelSize, kernelSize));
	return blurredImage;
}

cv::Mat opencvTool::gaussianBlurFilter(const cv::Mat& inputImage, int kernelSize)
{
	cv::Mat blurredImage;
	double sigmaX = 0;  // 设置标准差为0（自动计算）
	cv::GaussianBlur(inputImage, blurredImage, cv::Size(kernelSize, kernelSize), sigmaX, sigmaX);
	return blurredImage;
}


cv::Mat opencvTool::applyBoxFilter(const cv::Mat& inputImage, int kernelSize)
{
	cv::Mat filteredImage;
	// -1 表示输出图像与输入图像具有相同的深度
	cv::boxFilter(inputImage, filteredImage, -1, cv::Size(kernelSize, kernelSize));
	return filteredImage;
}



cv::Mat opencvTool::medianFilter(const cv::Mat& inputImage, int kernelSize)
{
	cv::Mat filteredImage;
	cv::medianBlur(inputImage, filteredImage, kernelSize);
	return filteredImage;
}

cv::Mat opencvTool::applyBilateralFilter(const cv::Mat& inputImage, int diameter, double sigmaColor, double sigmaSpace)
{
	cv::Mat filteredImage;
	cv::bilateralFilter(inputImage, filteredImage, diameter, sigmaColor, sigmaSpace);
	return filteredImage;
}


// 定义非局部均值滤波函数
cv::Mat opencvTool::nonLocalMeansFilter(const cv::Mat& inputImage, double h, int templateWindowSize, int searchWindowSize)
{
	cv::Mat denoisedImage;
	cv::fastNlMeansDenoising(inputImage, denoisedImage, h, templateWindowSize, searchWindowSize);
	return denoisedImage;
}



// 函数：腐蚀操作
cv::Mat opencvTool::erosionFilter(const cv::Mat& src, const int kernelSize)
{
	cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
	cv::Mat result;
	cv::erode(src, result, kernel);
	return result;
}

// 函数：膨胀操作
cv::Mat opencvTool::dilationFilter(const cv::Mat& src, const int kernelSize)
{
	cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
	cv::Mat result;
	cv::dilate(src, result, kernel);
	return result;
}

// 函数：开运算
cv::Mat opencvTool::openingFilter(const cv::Mat& src, const int kernelSize)
{
	cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
	cv::Mat result;
	cv::morphologyEx(src, result, MORPH_OPEN, kernel);
	return result;
}

// 函数：闭运算
cv::Mat opencvTool::closingFilter(const cv::Mat& src, const int kernelSize)
{
	cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
	cv::Mat result;
	cv::morphologyEx(src, result, MORPH_CLOSE, kernel);
	return result;
}



//int opencvTool::filtering_comparison(cv::Mat src)
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
//}




cv::Mat opencvTool::detectAndMarkCorners(const cv::Mat& image)
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
			{ // 阈值可调
				cv::circle(marked_image, cv::Point(j, i), 20, cv::Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}

	return marked_image;
}


void opencvTool::checkerBoardCalibration(const std::string& imageFolderPath, cv::Mat& cameraMatrix, cv::Mat& distCoeffs)
{
	// 定义棋盘格尺寸
	const int BOARDSIZE[2]{ 9,6 }; // 第一个参数是几行,第二个是几列

	std::vector<std::vector<cv::Point3f>> objpoints_img;	// 存储棋盘格角点的三维坐标
	std::vector<std::vector<cv::Point2f>> images_points;	// 存储每幅图像检测到的棋盘格二维角点坐标
	std::vector<cv::String> images_path;	// 存储输入图像文件夹中的图像路径
	std::vector<cv::Point3f> obj_world_pts;	// 存储棋盘格在世界坐标系中的三维点坐标

	//  函数获取指定文件夹中所有图像文件的路径
	cv::glob(imageFolderPath, images_path);

	// 转换世界坐标系
	for (int i = 0; i < BOARDSIZE[1]; i++)
	{
		for (int j = 0; j < BOARDSIZE[0]; j++)
		{
			obj_world_pts.push_back(cv::Point3f(j, i, 0));
		}
	}
	// image 和 img_gray 分别用于存储读取的原始图像和转换为灰度的图像
	cv::Mat image, img_gray;

	// 遍历每张图像进行角点检测和存储
	for (const auto& imagePath : images_path)
	{
		image = cv::imread(imagePath);
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

		// 检测角点 findChessboardCorners 检测当前灰度图像中的棋盘格角点，并存储在 img_corner_points 中
		std::vector<cv::Point2f> img_corner_points;
		bool found_success = cv::findChessboardCorners(img_gray, cv::Size(BOARDSIZE[0], BOARDSIZE[1]),
			img_corner_points,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		// 如果成功检测到角点
		if (found_success)
		{
			// 进行亚像素级别的角点定位
			// 使用 cv::cornerSubPix 对角点进行亚像素级的精确化处理，提高检测精度
			// 并用 cv::drawChessboardCorners 在原始图像上绘制检测到的角点
			cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
			cv::cornerSubPix(img_gray, img_corner_points, cv::Size(11, 11), cv::Size(-1, -1), criteria);
			// 绘制角点
			cv::drawChessboardCorners(image, cv::Size(BOARDSIZE[0], BOARDSIZE[1]), img_corner_points, found_success);

			// 存储世界坐标系下的角点和图像坐标系下的角点
			objpoints_img.push_back(obj_world_pts); // 棋盘格三维点坐标 
			images_points.push_back(img_corner_points); // 二维角点坐标
		}

	}

	// 标定相机并获得相机矩阵、畸变系数、旋转向量和平移向量
	//cv::Mat cameraMatrix, distCoeffs;
	std::vector<cv::Mat> rvecs, tvecs;
	cv::calibrateCamera(objpoints_img, images_points, img_gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs);
	// 输出标定结果
	std::cout << "相机内参：" << std::endl;
	std::cout << cameraMatrix << std::endl;
	std::cout << "*****************************" << std::endl;
	std::cout << "畸变系数：" << std::endl;
	std::cout << distCoeffs << std::endl;
	std::cout << "*****************************" << std::endl;

	std::vector<cv::Mat> rotations;
	// 打印每张图像的旋转矩阵和平移向量
	for (size_t i = 0; i < rvecs.size(); ++i)
	{
		// 从旋转向量转换为旋转矩阵
		cv::Mat R;
		// Rodrigues 函数被用来将旋转向量转换成旋转矩阵
		cv::Rodrigues(rvecs[i], R); // rvecs[i] 是旋转向量, R 是旋转矩阵
		rotations.push_back(R);
	}

	for (size_t i = 0; i < rotations.size(); ++i)
	{
		// 旋转矩阵
		std::cout << "Image " << i + 1 << " Rotation Matrix:" << std::endl;
		std::cout << rotations[i] << std::endl;
		// 平移向量
		std::cout << "Image " << i + 1 << " Translation Vector:" << std::endl;
		std::cout << tvecs[i] << std::endl;
		std::cout << "====================================" << std::endl;

	}


	for (const auto& once : images_path)
	{
		// 读取一张测试图像进行畸变校正
		cv::Mat src = cv::imread(once);
		// 畸变校正
		cv::Mat dstImage;
		cv::undistort(src, dstImage, cameraMatrix, distCoeffs);
		// 显示校正结果并保存
		std::filesystem::path file(once);
		std::filesystem::path out;
		out = file.parent_path();
		out.append("undistort");
		std::filesystem::create_directories(out);
		std::string name = "undistort" + file.filename().string();
		out.append(name);
		saveImage(out.string(), dstImage);
		src.release();
		dstImage.release();
		//break;
	}
}



void opencvTool::computeAndShowHistogram(const std::string& filename)
{
	// Load image
	cv::Mat img = cv::imread(filename);
	if (img.empty()) {
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
	float hranges[] = { 0, 255 };
	const float* ranges[] = { hranges };
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
	for (int i = 0; i < size; ++i) {
		float binValue = hist.at<float>(i);
		int realValue = cv::saturate_cast<int>(binValue * hpt / maxValue);
		cv::rectangle(hist_img, cv::Point(i * scale, hist_height - 1),
			cv::Point((i + 1) * scale - 1, hist_height - realValue),
			cv::Scalar(255));
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
	for (int i = 0; i < size; ++i) {
		float binValue = hist.at<float>(i);
		int realValue = cv::saturate_cast<int>(binValue * hpt / maxValue);
		cv::rectangle(equal_hist_img, cv::Point(i * scale, hist_height - 1),
			cv::Point((i + 1) * scale - 1, hist_height - realValue),
			cv::Scalar(255));
	}

	// 显示均衡直方图
	cv::imshow("[equal_hist]", equal_hist_img);

	cv::waitKey(0);
}

































//void opencvTool::imageRegistration(const std::string& image1_path, const std::string& image2_path)
//{
//	cv::Mat image1, image2;
//	cv::Mat H;
//
//	// 读取图像
//	image1 = cv::imread(image1_path, cv::IMREAD_ANYCOLOR);
//	image2 = cv::imread(image2_path, cv::IMREAD_ANYCOLOR);
//	assert(!image1.empty() && !image2.empty());
//
//	// 创建特征检测器、描述符提取器和特征匹配器
//	std::vector<cv::KeyPoint> kp1, kp2;
//	std::vector<cv::DMatch> matches;
//	cv::Mat descriptor1, descriptor2;
//	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create()
//	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20)
//	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
//
//	// 检测特征点
//	detector->detect(image1, kp1);
//	detector->detect(image2, kp2);
//
//	// 提取描述符
//	descriptor->compute(image1, kp1, descriptor1);
//	descriptor->compute(image2, kp2, descriptor2);
//
//	// 匹配描述符
//	matcher->match(descriptor1, descriptor2, matches);
//
//	// 选取好的匹配点
//	auto min_max = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; });
//	double min_dist = min_max.first->distance;
//	std::vector<cv::DMatch> good_matches;
//	for (size_t i = 0; i < matches.size(); i++) {
//		if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
//			good_matches.push_back(matches[i]);
//		}
//	}
//
//	// 提取好的匹配点对应的特征点
//	std::vector<cv::Point2f> good_kp1, good_kp2;
//	for (size_t i = 0; i < good_matches.size(); i++) {
//		good_kp1.push_back(kp1.at(good_matches[i].queryIdx).pt);
//		good_kp2.push_back(kp2.at(good_matches[i].trainIdx).pt);
//	}
//
//	// 估计单应性矩阵
//	H = findHomography(good_kp2, good_kp1);
//
//	// 图像配准
//	cv::Mat corner00 = (cv::Mat_<double>(3, 1) << 0, 0, 1);
//	cv::Mat corner01 = (cv::Mat_<double>(3, 1) << image2.cols, 0, 1);
//	cv::Mat corner10 = (cv::Mat_<double>(3, 1) << 0, image2.rows, 1);
//	cv::Mat corner11 = (cv::Mat_<double>(3, 1) << image2.cols, image2.rows, 1);
//	cv::Mat new_corner00 = H * corner00;
//	cv::Mat new_corner01 = H * corner01;
//	cv::Mat new_corner10 = H * corner10;
//	cv::Mat new_corner11 = H * corner11;
//	new_corner00 = new_corner00 / new_corner00.at<double>(2, 0);
//	new_corner01 = new_corner01 / new_corner01.at<double>(2, 0);
//	new_corner10 = new_corner10 / new_corner10.at<double>(2, 0);
//	new_corner11 = new_corner11 / new_corner11.at<double>(2, 0);
//
//	int maxx = ceil(std::max(new_corner01.at<double>(0, 0), new_corner11.at<double>(0, 0)));
//	int maxy = ceil(std::max(new_corner01.at<double>(1, 0), new_corner11.at<double>(1, 0)));
//	cv::Mat dst;
//	cv::warpPerspective(image2, dst, H, cv::Size(maxx, maxy));
//	cv::Mat pimage2 = dst.clone();
//	image1.copyTo(dst(cv::Rect(0, 0, image1.cols, image1.rows)));
//
//	int start = std::min(new_corner00.at<double>(0, 0), new_corner10.at<double>(0, 0));
//	double processWidth = image1.cols - start;  // 重叠区域的列宽度
//	double alpha = 1;
//
//	for (int i = 0; i < image1.rows; i++) {
//		uchar* p = image1.ptr<uchar>(i);
//		uchar* p2 = pimage2.ptr<uchar>(i);
//		uchar* r = dst.ptr<uchar>(i);
//
//		for (int j = start; j < image1.cols; j++) {
//			if (p2[j * 3] == 0 && p2[j * 3 + 1] == 0 && p2[j * 3 + 2] == 0) {
//				alpha = 1;
//			}
//			else {
//				alpha = (processWidth - (j - start)) / processWidth;
//			}
//			r[j * 3] = p[j * 3] * (alpha)+p2[j * 3] * (1 - alpha);
//			r[j * 3 + 1] = p[j * 3 + 1] * alpha + p2[j * 3 + 1] * (1 - alpha);
//			r[j * 3 + 2] = p[j * 3 + 2] * alpha + p2[j * 3 + 2] * (1 - alpha);
//		}
//	}
//
//	cv::imshow("optimizer", dst);
//	cv::waitKey();
//}





opencvTool::opencvTool()
{
	std::cout << "Start opencvTool" << std::endl;

}
opencvTool::~opencvTool()
{
	std::cout << "End opencvTool" << std::endl;

}













