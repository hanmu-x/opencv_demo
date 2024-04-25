
#include "opencv_tool/opencvTool.h"
#include "opencv_tool/config.hpp"
#include <filesystem>


int main()
{
	opencvTool tc;
    std::filesystem::path imageEmpty(DEFAULT_DATA_DIR);
    imageEmpty += "/empty_1.jpg";

    std::filesystem::path imageEmpty_2(DEFAULT_DATA_DIR);
    imageEmpty_2 += "/empty500.jpg";

    Polygon poly;
    poly.push_back(cv::Point(100, 20));
    poly.push_back(cv::Point(320, 20));
    poly.push_back(cv::Point(320, 240));
    poly.push_back(cv::Point(100, 240));

    opencvTool::drawingByMouse();

    return 0;

    cv::Mat color = opencvTool::creatColorMat(500, 500);
    std::string text = "Hello, OpenCV!";
    cv::Point position(50, 200); // 文字的位置
    opencvTool::addWatermark(color, text, position);
    opencvTool::showImage(color);

    return 0;

    cv::Mat empyt = opencvTool::creatEmptyMat(500, 500);
    opencvTool::addText(empyt,  text, position);
    opencvTool::showImage(empyt);

    return 0;

    opencvTool::drawRectangle(empyt, cv::Point(100, 20), cv::Point(320, 240), 3);
    opencvTool::showImage(empyt);

    return 0;
    opencvTool::drawPolygon(empyt, poly);
    //opencvTool::drawPolygon(empyt, poly);
    //opencvTool::drawLines(empyt, poly);
    //opencvTool::showImage(empyt);



    return 0;



    cv::Scalar Red = cv::Scalar(0, 0, 255);  // Red color  
    opencvTool::changeColor(empyt, 100, 200, Red);

    //opencvTool::showImage(empyt);
    //opencvTool::showImage(imageEmpty.string());
    opencvTool::saveImage(imageEmpty_2.string(), empyt);
    opencvTool::changeColor(imageEmpty_2.string(), 100, 600, Red);


    return 0;

    //int w = 640, h = 480;
    //bool a = tc.creatEmptyImage(w, h, imageEmpty.string());

    std::filesystem::path imageColor(DEFAULT_DATA_DIR);
    imageColor += "/color_1.jpg";

    
    //tc.drawPolygon(imageColor.string(), poly);
    //tc.drawLines(imageEmpty.string(), poly);
    int w1 = 64, h1 = 48;
    //tc.changeColor(imageEmpty.string(), w1, h1);


	return 0;




//	Config config;
//#ifndef NDEBUG
//	std::string configPath = "../../../../Config/my_config.json";
//#else
//	std::string configPath = "./my_config.json";
//#endif
//    if (config.read_config(configPath))
//    {
//        std::cout << "Read config file succession " << std::endl;
//    }
//    else
//    {
//        std::cout << "ERROR : Failed to read config file " << std::endl;
//        return 1;
//    }

}