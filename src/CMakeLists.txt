cmake_minimum_required (VERSION 3.15)

project(video_main CXX)


add_executable(${PROJECT_NAME}  "main.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC video_tool)


add_executable(testA  "testA.cpp" )
target_link_libraries(testA PUBLIC video_tool)




