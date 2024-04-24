
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
	/// 可视化展示图片
	/// </summary>
	/// <param name="image_p"></param>
	/// <returns></returns>
	static bool showImage(std::string image_p);
	static bool showImage(cv::Mat image);

	/// <summary>
	/// 将数据类型转换为字符串
	/// </summary>
	/// <param name="type"></param>
	/// <returns> 如"8UC3"，它表示图像的深度为 8 位无符号整数（8U）且具有 3 个颜色通道（C3）</returns>
	static std::string type2str(int type);

	/// <summary>
	/// 创建一个空白图片
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="image_p"></param>
	/// <returns></returns>
	static bool creatEmptyImage(unsigned int width, unsigned int height, std::string image_p);

	static cv::Mat creatEmptyMat(unsigned int width, unsigned int height);


	/// <summary>
	/// 创建一个渐变彩色图片
	/// </summary>
	/// <param name="image_p"></param>
	/// <returns></returns>
	static bool creatColor(std::string image_p);

	/// <summary>
	/// 绘制多边形
	/// </summary>
	/// <param name="image_p"></param>
	/// <param name="line"></param>
	/// <returns></returns>
	static bool drawPolygon(std::string image_p, std::vector<cv::Point> points);
	
	/// <summary>
	/// 
	/// </summary>
	/// <param name="image_p"></param>
	/// <param name="points"></param>
	/// <returns></returns>
	static bool drawLines(std::string image_p, std::vector<cv::Point> points);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="image_p"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <returns></returns>
	static bool changeColor(std::string image_p, int width, int height);


private:

};







