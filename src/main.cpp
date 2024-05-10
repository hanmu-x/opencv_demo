
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

    std::filesystem::path imageColor(DEFAULT_DATA_DIR);
    imageColor += "/color.jpg";

    std::filesystem::path imageColorresize(DEFAULT_DATA_DIR);
    imageColorresize += "/color_resize.jpg";

    std::filesystem::path radar(DEFAULT_DATA_DIR);
    radar += "/radar.jpg";

    std::filesystem::path triangle(DEFAULT_DATA_DIR);
    triangle += "/triangle.jpg";

    // 绘制直方图
    cv::Mat radar_mat = opencvTool::openImage(radar.string());
    cv::Mat histogram_image = opencvTool::calculateHistogram(radar_mat);
    opencvTool::showImage(histogram_image);


    return 0;

    cv::Mat triangle_mat = opencvTool::openImage(triangle.string());

    // 绘制矩形边界
    cv::Mat RectangleoutLine = opencvTool::drawRectangleOutline(triangle_mat);

    opencvTool::showImage(RectangleoutLine);

    return 0;

    // 绘制边界
    cv::Mat outLine = opencvTool::drawOutline(triangle_mat);

    opencvTool::showImage(outLine);


    return 0;


    cv::Mat dege_radar = opencvTool::edgeDetection(radar_mat,100,200);
    opencvTool::showImage(dege_radar);

    return 0;
    cv::Mat color_mat_bgr = opencvTool::openImage(imageColor.string());

    // 图片的旋转

    cv::Mat rotate_image = opencvTool::rotateImage(color_mat_bgr, 90.0);
    opencvTool::showImage(rotate_image);


    return 0;

    // 图片的平移
    cv::Mat trans_image = opencvTool::translateImage(color_mat_bgr, 50, 100);
    opencvTool::showImage(trans_image);


    return 0;
    cv::Mat resize_mat = opencvTool::resizeImage(color_mat_bgr, 0.5);
    opencvTool::saveImage(imageColorresize.string(), resize_mat);

    return 0;

    // BGR图片转HSV
    cv::Mat color_mat_hsv = opencvTool::BGRToHSV(color_mat_bgr);
    opencvTool::showImage(color_mat_hsv);
    return 0;

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