
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

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



private:

};





