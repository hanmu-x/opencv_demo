
#include "opencv_IO.h"

// 将数据类型转换为字符串
std::string opencvIO::type2str(int type)
{
    std::string typeStr;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
        case CV_8U:
            typeStr = "8U";
            break;
        case CV_8S:
            typeStr = "8S";
            break;
        case CV_16U:
            typeStr = "16U";
            break;
        case CV_16S:
            typeStr = "16S";
            break;
        case CV_32S:
            typeStr = "32S";
            break;
        case CV_32F:
            typeStr = "32F";
            break;
        case CV_64F:
            typeStr = "64F";
            break;
        default:
            typeStr = "User";
            break;
    }
    typeStr += "C";
    typeStr += (chans + '0');  // 3 个颜色通道（C3）
    return typeStr;
}

cv::Mat opencvIO::openImage(const std::string& image_path)
{
    // 使用imread函数加载图像
    cv::Mat image = cv::imread(image_path);

    // 检查图像是否成功加载
    if (image.empty())
    {
        std::cout << "Could not open or find the image: " << image_path << std::endl;
    }

    return image;
}

bool opencvIO::showImage(std::string image_p)
{
    cv::Mat image = cv::imread(image_p.c_str());
    if (image.empty())
    {
        std::cout << "Error: empyt mat " << std::endl;
        return false;
    }

    // 打印图像信息
    std::cout << "Image size: " << image.cols << " x " << image.rows << std::endl;
    std::cout << "Number of channels: " << image.channels() << std::endl;  // 通道数
    std::cout << "Data type: " << type2str(image.type()) << std::endl;     // 自定义函数，用于将数据类型转换为字符串

    cv::imshow("show", image);
    cv::waitKey(0);           // 持续显示窗口
    cv::destroyAllWindows();  // 用于关闭所有由 OpenCV 创建的窗口
    return true;
}

bool opencvIO::showImage(cv::Mat image)
{
    if (image.empty())
    {
        std::cout << "Error: empty mat " << std::endl;
        return false;
    }

    // 打印图像信息
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Number of channels: " << image.channels() << std::endl;  // 通道数
    std::cout << "Data type: " << type2str(image.type()) << std::endl;     // 自定义函数，用于将数据类型转换为字符串

    // 显示图像
    cv::imshow("show", image);
    cv::waitKey(0);           // 持续显示窗口
    cv::destroyAllWindows();  // 用于关闭所有由 OpenCV 创建的窗口
    return true;
}

bool opencvIO::creatEmptyImage(unsigned int width, unsigned int height, std::string image_p)
{
    // 创建一个空白图像
    cv::Mat blankImage(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 保存图像为文件（可选）
    cv::imwrite(image_p.c_str(), blankImage);

    // 显示空白图像
    cv::imshow("Blank Image", blankImage);

    // 等待用户按下任意键后关闭窗口
    cv::waitKey(0);

    // 关闭窗口
    cv::destroyAllWindows();
    return true;
}

cv::Mat opencvIO::creatEmptyMat(unsigned int width, unsigned int height, int imageType)
{
    // 创建一个空白图像
    cv::Mat blankImage(height, width, imageType, cv::Scalar(255, 255, 255));
    return blankImage;
}

cv::Mat opencvIO::creatColorMat(unsigned int width, unsigned int height, int imageType)
{
    // 创建一个空白图像
    cv::Mat gradientImage(height, width, imageType);

    // 生成渐变色
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // 计算RGB颜色值，根据x和y的位置生成渐变色
            uchar blue = static_cast<uchar>(x * 255 / width);
            uchar green = static_cast<uchar>((x + y) * 255 / (width + height));
            uchar red = static_cast<uchar>(y * 255 / height);
            // 设置像素颜色
            // (三通道:用at<Vec3b>(row, col)
            // (单通道:at<uchar>(row, col))
            gradientImage.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
        }
    }

    return gradientImage;
}

bool opencvIO::creatColor(std::string image_p)
{
    // 指定图像的宽度和高度
    int width = 640;
    int height = 480;

    // 创建一个空白图像
    cv::Mat gradientImage(height, width, CV_8UC3);

    // 生成渐变色
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // 计算RGB颜色值，根据x和y的位置生成渐变色
            uchar blue = static_cast<uchar>(x * 255 / width);
            uchar green = static_cast<uchar>((x + y) * 255 / (width + height));
            uchar red = static_cast<uchar>(y * 255 / height);
            // 设置像素颜色
            // (三通道:用at<Vec3b>(row, col)
            // (单通道:at<uchar>(row, col))
            gradientImage.at<cv::Vec3b>(y, x) = cv::Vec3b(blue, green, red);
        }
    }

    // 保存图像为文件（可选）
    cv::imwrite(image_p.c_str(), gradientImage);

    // 显示渐变色图像
    cv::imshow("Gradient Image", gradientImage);

    // 等待用户按下任意键后关闭窗口
    cv::waitKey(0);

    // 关闭窗口
    cv::destroyAllWindows();

    return true;
}

bool opencvIO::saveImage(const std::string path, const cv::Mat image)
{
    if (image.empty())
    {
        std::cout << "Error: empty mat " << std::endl;
        return false;
    }

    // 打印图像信息
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Number of channels: " << image.channels() << std::endl;  // 通道数
    std::cout << "Data type: " << type2str(image.type()) << std::endl;     // 自定义函数，用于将数据类型转换为字符串

    // 保存图像文件
    try
    {
        cv::imwrite(path, image);
        std::cout << "Image saved successfully!" << std::endl;
        return true;
    }
    catch (cv::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat opencvIO::memoryEncode(const std::string& base64Str)
{
    // 将Base64字符串解码为二进制数据
    std::vector<uint8_t> imageData;
    // 将二进制数据解码为图像
    cv::Mat img = cv::imdecode(std::vector<uint8_t>(base64Str.begin(), base64Str.end()), cv::IMREAD_UNCHANGED);

    if (img.empty())
    {
        std::cout << "Failed to decode image data." << std::endl;
    }
    return img;
}

std::string opencvIO::memoryDecode(const cv::Mat& img, const std::string extension)
{
    std::string encodedBuffer;

    if (img.empty())
    {
        std::cout << "Error: empty image" << std::endl;
        return "";
    }

    if (extension == "jpeg" || extension == "jpg" || extension == "JPG" || extension == "JPEG")
    {
        // 将图像编码为 JPEG 或 PNG 格式
        std::vector<int> compressionParams;
        compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
        compressionParams.push_back(95);  // JPEG 质量
        std::vector<uint8_t> buffer;

        // 编码图像
        bool success = cv::imencode(".jpg", img, buffer, compressionParams);
        if (!success)
        {
            std::cerr << "Failed to encode image data." << std::endl;
            return "";
        }

        // 将编码后的图像数据转换为 Base64 编码的字符串
        encodedBuffer.clear();
        // 将编码后的图像数据转换为字符串
        encodedBuffer.assign(buffer.begin(), buffer.end());
    }
    else if (extension == "png" || extension == "PNG")
    {
        // 将图像编码为 PNG 格式
        std::vector<int> compressionParams;
        compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compressionParams.push_back(3);  // PNG 压缩级别
        std::vector<uint8_t> buffer;

        // 编码图像
        bool success = cv::imencode(".png", img, buffer, compressionParams);
        if (!success)
        {
            std::cerr << "Failed to encode image data." << std::endl;
            return "";
        }

        // 将编码后的图像数据转换为字符串
        encodedBuffer.assign(buffer.begin(), buffer.end());
    }
    else
    {
        std::cerr << "Unsupported format: " << extension << std::endl;
        return "";
    }

    return encodedBuffer;
}

cv::Mat opencvIO::BGRToHSV(cv::Mat bgr_image)
{
    // 创建一个用于存储HSV图像的Mat对象
    cv::Mat hsv_image;

    // 将BGR图像转换为HSV图像
    cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
    return hsv_image;
}