
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

class tool_class
{
public:
	tool_class();
	~tool_class();

	/// <summary>
	/// 打开一个图片并用图框展示
	/// </summary>
	/// <param name="image_p"></param>
	/// <returns></returns>
	int opencvReadImage(std::string image_p);

	/// <summary>
	/// 创建一个空白图片
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="image_p"></param>
	/// <returns></returns>
	bool creatEmpty(int width, int height, std::string image_p);

	/// <summary>
	/// 创建一个渐变彩色图片
	/// </summary>
	/// <param name="image_p"></param>
	/// <returns></returns>
	bool creatColor(std::string image_p);

	/// <summary>
	/// 绘制多边形
	/// </summary>
	/// <param name="image_p"></param>
	/// <param name="line"></param>
	/// <returns></returns>
	bool drawPolygon(std::string image_p, std::vector<cv::Point> points);

	bool drawLines(std::string image_p, std::vector<cv::Point> points);


	bool changeColor(std::string image_p, int width, int height);


private:

};







