
#include "opencv_tool/config.hpp"
#include "file_tool/file_tool.h"

#include <filesystem>
#include "opencv_tool/opencv_IO.h"
#include "opencv_tool/opencv_drawer.h"
#include "opencv_tool/opencv_transform.h"
#include "opencv_tool/opencv_filter.h"
#include "opencv_tool/opencv_algo.h"


int main()
{
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

    std::filesystem::path testPng(DEFAULT_DATA_DIR);
    testPng += "/cricle.png";

    std::filesystem::path testPngCopy(DEFAULT_DATA_DIR);
    testPngCopy += "/cricle_copy.png";

    // 剪切
    cv::Mat img = opencvTF::cutImage(imageColor.string(), 80, 78, 500, 500);
    opencvIO::showImage(img);

    return 0;


    std::string pngStr = fileToString(testPng.string());
    // 将字符串转换为 cv::Mat 对象
    cv::Mat imgPng = opencvIO::memoryEncode(pngStr);
    if (imgPng.empty())
    {
        std::cout << "Failed to convert string to Mat object." << std::endl;
        return 1;
    }
    //opencvIO::showImage(imgPng);
    std::string decode_copy = opencvIO::memoryDecode(imgPng);
    stringToFile(testPngCopy.string(), decode_copy);


    return 0;

    // 棋盘格内参标定
    std::filesystem::path calibPath(DEFAULT_DATA_DIR);
    calibPath += "/calibration/*.jpg";
    cv::Mat cameraMatrix, distCoeffs;
    opencvAlgo::checkerBoardCalibration(calibPath.string(), cameraMatrix, distCoeffs);

    return 0;





    // 霍夫变换
    cv::Mat graphImage = opencvIO::openImage(graphPath.string());

    cv::Mat linesResult = opencvFilter::houghDetectCircles(graphImage);
    opencvIO::showImage(linesResult);

    return 0;


    cv::Mat bifilterbefor = opencvIO::openImage(imagePatht.string());

    //int b = opencvFilter::filtering_comparison(bifilterbefor);
    return 0;


    int kernel = 5;
    // 应用各个形态学操作
    cv::Mat closed = opencvFilter::closingFilter(bifilterbefor, kernel);
    opencvIO::showImage(closed);
    return 0;

    cv::Mat eroded = opencvFilter::erosionFilter(bifilterbefor, 5);
    cv::Mat dilated = opencvFilter::dilationFilter(bifilterbefor, kernel);
    cv::Mat opened = opencvFilter::openingFilter(bifilterbefor, kernel);

    // 非局部均值滤波
    double h = 3.0;                // 控制滤波器强度，一般取3
    int templateWindowSize = 7;    // 均值估计过程中考虑的像素邻域窗口大小
    int searchWindowSize = 21;     // 搜索区域窗口大小

    cv::Mat denoisedImage = opencvFilter::nonLocalMeansFilter(bifilterbefor, h, templateWindowSize, searchWindowSize);
    opencvIO::showImage(denoisedImage);

    return 0;

    // 应用双边滤波
    int diameter = 9;       
    double sigmaColor = 75; 
    double sigmaSpace = 75; 
    cv::Mat filteredImage = opencvFilter::applyBilateralFilter(bifilterbefor, diameter, sigmaColor, sigmaSpace);
    opencvIO::showImage(filteredImage);

    return 0;

    // 中值滤波
    cv::Mat mefilterbefor = opencvIO::openImage(imagePatht.string());
    cv::Mat mefilterafter = opencvFilter::medianFilter(mefilterbefor, 5);
    opencvIO::showImage(mefilterafter);

    return 0;


    // 方框滤波
    cv::Mat bfilterbefor = opencvIO::openImage(imagePatht.string());
    cv::Mat bfilterafter = opencvFilter::applyBoxFilter(bfilterbefor, 5);
    opencvIO::showImage(bfilterafter);

    return 0;


    // 高斯滤波
    cv::Mat gfilterbefor = opencvIO::openImage(imagePatht.string());
    cv::Mat gfilterafter = opencvFilter::gaussianBlurFilter(gfilterbefor, 5);
    opencvIO::showImage(gfilterafter);

    return 0;

    //均值滤波
    cv::Mat mfilterbefor = opencvIO::openImage(imagePatht.string());
    cv::Mat mfilterafter = opencvFilter::meanFilter(mfilterbefor,5);
    opencvIO::showImage(mfilterafter);

    return 0;
    //opencvFilter::imageRegistration(imagePath1.string(), imagePath2.string());
    opencvFilter::computeAndShowHistogram(imagePatht.string());

    return 0;


    // 检测和标记拐角
    cv::Mat mat_grid = opencvIO::openImage(chessboard_grid.string());
    cv::Mat marked_image = opencvFilter::detectAndMarkCorners(mat_grid);
    //opencvIO::showImage(marked_image);
    opencvIO::saveImage(Corner_grid.string(), marked_image);

    return 0;

    // 绘制直方图
    cv::Mat radar_mat = opencvIO::openImage(radar.string());
    cv::Mat histogram_image = opencvFilter::calculateHistogram(radar_mat);
    opencvIO::showImage(histogram_image);
    // 检测并标记角点

    return 0;

    cv::Mat triangle_mat = opencvIO::openImage(triangle.string());

    // 绘制矩形边界
    cv::Mat RectangleoutLine = opencvFilter::drawRectangleOutline(triangle_mat);

    opencvIO::showImage(RectangleoutLine);

    return 0;

    // 绘制边界
    cv::Mat outLine = opencvFilter::drawOutline(triangle_mat);

    opencvIO::showImage(outLine);


    return 0;


    cv::Mat dege_radar = opencvFilter::edgeDetection(radar_mat,100,200);
    opencvIO::showImage(dege_radar);

    return 0;
    cv::Mat color_mat_bgr = opencvIO::openImage(imageColor.string());

    // 图片的旋转

    cv::Mat rotate_image = opencvTF::rotateImage(color_mat_bgr, 90.0);
    opencvIO::showImage(rotate_image);


    return 0;

    // 图片的平移
    cv::Mat trans_image = opencvTF::translateImage(color_mat_bgr, 50, 100);
    opencvIO::showImage(trans_image);


    return 0;
    cv::Mat resize_mat = opencvTF::resizeImage(color_mat_bgr, 0.5);
    opencvIO::saveImage(imageColorresize.string(), resize_mat);

    return 0;

    // BGR图片转HSV
    cv::Mat color_mat_hsv = opencvIO::BGRToHSV(color_mat_bgr);
    opencvIO::showImage(color_mat_hsv);
    return 0;

    Polygon poly;
    poly.push_back(cv::Point(100, 20));
    poly.push_back(cv::Point(320, 20));
    poly.push_back(cv::Point(320, 240));
    poly.push_back(cv::Point(100, 240));

    opencvDrawer::drawingByMouse();

    return 0;

    cv::Mat color = opencvIO::creatColorMat(500, 500);
    std::string text = "Hello, OpenCV!";
    cv::Point position(50, 200); // 文字的位置
    opencvDrawer::addWatermark(color, text, position);
    opencvIO::showImage(color);

    return 0;

    cv::Mat empyt = opencvIO::creatEmptyMat(500, 500);
    opencvDrawer::addText(empyt, text, position);
    opencvIO::showImage(empyt);

    return 0;

    opencvDrawer::drawRectangle(empyt, cv::Point(100, 20), cv::Point(320, 240), 3);
    opencvIO::showImage(empyt);

    return 0;
    opencvDrawer::drawPolygon(empyt, poly);
    //opencvFilter::drawPolygon(empyt, poly);
    //opencvFilter::drawLines(empyt, poly);
    //opencvIO::showImage(empyt);



    return 0;



    cv::Scalar Red = cv::Scalar(0, 0, 255);  // Red color  
    opencvDrawer::changeColor(empyt, 100, 200, Red);

    //opencvIO::showImage(empyt);
    //opencvIO::showImage(imageEmpty.string());
    opencvIO::saveImage(imageEmpty_2.string(), empyt);
    opencvDrawer::changeColor(imageEmpty_2.string(), 100, 600, Red);


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