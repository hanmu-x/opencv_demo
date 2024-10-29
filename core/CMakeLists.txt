cmake_minimum_required (VERSION 3.15)


project(video_tool CXX)
foreach(dirname ${CMAKE_CURRENT_SOURCE_DIR})
    file(GLOB_RECURSE SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "${dirname}/*.h"
            "${dirname}/*.hpp"
            "${dirname}/*.c"
            "${dirname}/*.cpp"
            )
endforeach()




add_library(${PROJECT_NAME} STATIC ${SRC})

find_package(jsoncpp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC JsonCpp::JsonCpp)


set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv2")
find_package(OpenCV CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})





