cmake_minimum_required(VERSION 3.10)

project(cam)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)
#find_package(openni REQUIRED)
#find_package(depthai REQUIRED)

include_directories(../librealsense/include)
include_directories(/usr/local/include/opencv4)
#include_directories(/usr/local/zed/include)
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/include/depthai)
include_directories(/usr/local/include/openni)
include_directories(/home/arek/libs/depthai-core/build/install/lib/cmake/depthai/dependencies/include)

link_directories(/usr/local/lib)

add_executable(cam main.cpp cameras.cpp)
target_link_libraries(cam realsense2 ${OpenCV_LIBS} ${ZED_LIBRARIES})
#set_target_properties (pomiary PROPERTIES FOLDER "Examples")

