
#include "opencv_algo.h"
#include "opencv_IO.h"
#include <filesystem>

void opencvAlgo::checkerBoardCalibration(const std::string& imageFolderPath, cv::Mat& cameraMatrix, cv::Mat& distCoeffs)
{
    // 定义棋盘格尺寸
    const int BOARDSIZE[2]{9, 6};  // 第一个参数是几行,第二个是几列

    std::vector<std::vector<cv::Point3f>> objpoints_img;  // 存储棋盘格角点的三维坐标
    std::vector<std::vector<cv::Point2f>> images_points;  // 存储每幅图像检测到的棋盘格二维角点坐标
    std::vector<cv::String> images_path;                  // 存储输入图像文件夹中的图像路径
    std::vector<cv::Point3f> obj_world_pts;               // 存储棋盘格在世界坐标系中的三维点坐标

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
        bool found_success = cv::findChessboardCorners(img_gray, cv::Size(BOARDSIZE[0], BOARDSIZE[1]), img_corner_points, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

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
            objpoints_img.push_back(obj_world_pts);      // 棋盘格三维点坐标
            images_points.push_back(img_corner_points);  // 二维角点坐标
        }
    }

    // 标定相机并获得相机矩阵、畸变系数、旋转向量和平移向量
    // cv::Mat cameraMatrix, distCoeffs;
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
        cv::Rodrigues(rvecs[i], R);  // rvecs[i] 是旋转向量, R 是旋转矩阵
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
        opencvIO::saveImage(out.string(), dstImage);
        src.release();
        dstImage.release();
        // break;
    }
}
