
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

struct point
{
	int row;
	int col;
};
typedef std::vector<cv::Point> Polygon;

class opencvTool
{
public:
	opencvTool();
	~opencvTool();

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


	/// <summary>
	/// 改变图片的大小,会改变原图片的分辨率
	/// </summary>
	/// <param name="img"></param>
	/// <param name="scale_factor">放大缩小的倍数,2.0:放大两倍,0.5缩小一倍</param>
	/// <returns></returns>
	static cv::Mat resizeImage(const cv::Mat& img, double scale_factor);


	/////////////////////// 图像变换 ///////////////////////

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
	/// 提取边界
	/// </summary>
	/// <param name="img"></param>
	/// <param name="low_threshold"></param>
	/// <param name="height_threshold"></param>
	/// <returns></returns>
	static cv::Mat edgeDetection(const cv::Mat img, int low_threshold, int height_threshold);

	//////////////// 射影变换 ////////////////
	//////////////// 霍夫变换 ////////////////
	// 霍夫直线变换
	static cv::Mat houghDetectLines(const cv::Mat& inputImage);

	// 霍夫圆变换(参数有问题)
	static cv::Mat houghDetectCircles(const cv::Mat& inputImage);


	//////////////// 边缘检测 ////////////////
	//////////////// 直方图均衡 ////////////////


	/// <summary>
	/// 绘制轮廓
	/// </summary>
	/// <param name="image"></param>
	/// <returns></returns>
	static cv::Mat drawOutline(const cv::Mat& image);


	/// <summary>
	/// 
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
	/// 棋盘格内参标定标定
	/// </summary>
	/// <param name="imageFolderPath"></param>
	/// <param name="cameraMatrix">相机内参</param>
	/// <param name="distCoeffs">畸变系数</param>
	static void checkerBoardCalibration(const std::string& imageFolderPath, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);


	// 形体对比
	//static int filtering_comparison(cv::Mat src);




	//图像特征点提取
	//	SIFT特征点
	//	SURF特征点
	//	ORB特征点



	static void computeAndShowHistogram(const std::string& filename);

	//static void imageRegistration(const std::string& image1_path, const std::string& image2_path);


private:

};





