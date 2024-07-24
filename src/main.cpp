
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

    std::filesystem::path chessboard_grid(DEFAULT_DATA_DIR);
    chessboard_grid += "/chessboard_grid.jpg";

    std::filesystem::path Corner_grid(DEFAULT_DATA_DIR);
    Corner_grid += "/Corner_grid.jpg";

    std::filesystem::path imagePath1(DEFAULT_DATA_DIR);
    imagePath1 += "/image1.png";
    std::filesystem::path imagePath2(DEFAULT_DATA_DIR);
    imagePath2 += "/image2.png";

    std::filesystem::path imagePatht(DEFAULT_DATA_DIR);
    imagePatht += "/1.jpg";

    std::filesystem::path graphPath(DEFAULT_DATA_DIR);
    graphPath += "/cricle.jpg";


    // 棋盘格标定
    cv::Mat cameraMatrix, distCoeffs;
    opencvTool::checkerBoardCalibration("D:/1_wangyingjie/learn/github_project/5_OpenCV/opencv_demo/data/calibration/*.jpg", cameraMatrix, distCoeffs);

    return 0;





    // 霍夫变换
    cv::Mat graphImage = opencvTool::openImage(graphPath.string());

    cv::Mat linesResult = opencvTool::houghDetectCircles(graphImage);
    opencvTool::showImage(linesResult);

    return 0;


    cv::Mat bifilterbefor = opencvTool::openImage(imagePatht.string());

    //int b = opencvTool::filtering_comparison(bifilterbefor);
    return 0;


    int kernel = 5;
    // 应用各个形态学操作
    cv::Mat closed = opencvTool::closingFilter(bifilterbefor, kernel);
    opencvTool::showImage(closed);
    return 0;

    cv::Mat eroded = opencvTool::erosionFilter(bifilterbefor, 5);
    cv::Mat dilated = opencvTool::dilationFilter(bifilterbefor, kernel);
    cv::Mat opened = opencvTool::openingFilter(bifilterbefor, kernel);

    // 非局部均值滤波
    double h = 3.0;                // 控制滤波器强度，一般取3
    int templateWindowSize = 7;    // 均值估计过程中考虑的像素邻域窗口大小
    int searchWindowSize = 21;     // 搜索区域窗口大小

    cv::Mat denoisedImage = opencvTool::nonLocalMeansFilter(bifilterbefor, h, templateWindowSize, searchWindowSize);
    opencvTool::showImage(denoisedImage);

    return 0;

    // 应用双边滤波
    int diameter = 9;       
    double sigmaColor = 75; 
    double sigmaSpace = 75; 
    cv::Mat filteredImage = opencvTool::applyBilateralFilter(bifilterbefor, diameter, sigmaColor, sigmaSpace);
    opencvTool::showImage(filteredImage);

    return 0;

    // 中值滤波
    cv::Mat mefilterbefor = opencvTool::openImage(imagePatht.string());
    cv::Mat mefilterafter = opencvTool::medianFilter(mefilterbefor, 5);
    opencvTool::showImage(mefilterafter);

    return 0;


    // 方框滤波
    cv::Mat bfilterbefor = opencvTool::openImage(imagePatht.string());
    cv::Mat bfilterafter = opencvTool::applyBoxFilter(bfilterbefor, 5);
    opencvTool::showImage(bfilterafter);

    return 0;


    // 高斯滤波
    cv::Mat gfilterbefor = opencvTool::openImage(imagePatht.string());
    cv::Mat gfilterafter = opencvTool::gaussianBlurFilter(gfilterbefor, 5);
    opencvTool::showImage(gfilterafter);

    return 0;

    //均值滤波
    cv::Mat mfilterbefor = opencvTool::openImage(imagePatht.string());
    cv::Mat mfilterafter = opencvTool::meanFilter(mfilterbefor,5);
    opencvTool::showImage(mfilterafter);

    return 0;
    //opencvTool::imageRegistration(imagePath1.string(), imagePath2.string());
    opencvTool::computeAndShowHistogram(imagePatht.string());

    return 0;


    // 检测和标记拐角
    cv::Mat mat_grid = opencvTool::openImage(chessboard_grid.string());
    cv::Mat marked_image = opencvTool::detectAndMarkCorners(mat_grid);
    //opencvTool::showImage(marked_image);
    opencvTool::saveImage(Corner_grid.string(), marked_image);

    return 0;

    // 绘制直方图
    cv::Mat radar_mat = opencvTool::openImage(radar.string());
    cv::Mat histogram_image = opencvTool::calculateHistogram(radar_mat);
    opencvTool::showImage(histogram_image);
    // 检测并标记角点

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