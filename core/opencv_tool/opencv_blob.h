
#ifndef OPENCV_BLOB_H
#define OPENCV_BLOB_H

#include "opencv_global.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

class opencvBlob
{
  public:
    static cv::Mat check(const cv::Mat img)
    {
        // 设置BLOB检测器参数
        cv::SimpleBlobDetector::Params params;
        params.minThreshold = 10;
        params.maxThreshold = 200;
        params.filterByArea = true;
        params.minArea = 100;   // 设置最小面积
        params.maxArea = 5000;  // 设置最大面积
        params.filterByCircularity = true;
        params.minCircularity = 0.7;
        params.filterByConvexity = true;
        params.minConvexity = 0.8;
        params.filterByInertia = true;
        params.minInertiaRatio = 0.01;

        // 创建BLOB检测器
        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

        // 检测BLOB
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(img, keypoints);

        // 绘制检测到的BLOB
        cv::Mat output_img;
        cv::drawKeypoints(img, keypoints, output_img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        return output_img;
    }
};

#endif  // OPENCV_BLOB_H
