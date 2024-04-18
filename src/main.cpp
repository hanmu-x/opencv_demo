
#include "opencv_tool/opencvTool.h"
#include "opencv_tool/config.hpp"
#include <filesystem>


int main()
{

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

	tool_class tc;

    std::filesystem::path imageEmpty(DEFAULT_DATA_DIR);
    imageEmpty += "/empty_1.jpg";

    int w = 640, h = 480;
    bool a = tc.creatEmpty(w, h, imageEmpty.string());

    std::filesystem::path imageColor(DEFAULT_DATA_DIR);
    imageColor += "/color_1.jpg";

    Polygon poly;
    poly.push_back(cv::Point(100, 20));
    poly.push_back(cv::Point(320, 20));
    poly.push_back(cv::Point(320, 240));
    //poly.push_back(cv::Point(100, 240));
    
    //tc.drawPolygon(imageColor.string(), poly);
    //tc.drawLines(imageEmpty.string(), poly);
    int w1 = 64, h1 = 48;
    tc.changeColor(imageEmpty.string(), w1, h1);


	return 0;
}