
#include <iostream>
#include <fstream>
#include <string>

std::string fileToString(const std::string& filePath)
{
    // 打开文件
    std::ifstream file(filePath, std::ios::binary);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return "";
    }

    // 使用 std::string 的构造函数从文件流中读取内容
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // 关闭文件
    file.close();

    return content;
}



bool stringToFile(const std::string& filePath, const std::string& content)
{
    // 打开文件
    std::ofstream file(filePath, std::ios::binary);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        std::cerr << "Failed to open file for writing: " << filePath << std::endl;
        return false;
    }

    // 将字符串写入文件
    file.write(content.c_str(), content.size());

    // 关闭文件
    file.close();

    return true;  // 返回成功
}
