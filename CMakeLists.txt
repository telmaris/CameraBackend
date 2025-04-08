cmake_minimum_required(VERSION 3.10)

project(cam)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(ZED REQUIRED)
find_package(CUDA 11.4 REQUIRED)
find_package(depthai CONFIG REQUIRED)
find_package(PCL REQUIRED)

include_directories(../librealsense/include)
include_directories(/usr/local/include/opencv4)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/depthai/include)
include_directories(/home/mobilne/libs/depthai-core/build/install/lib/cmake/depthai/dependencies/include)
include_directories(${ZED_INCLUDE_DIRS})
include_directories(/usr/local/depthai/lib/cmake/depthai/dependencies/include)
include_directories(${PCL_INCLUDE_DIRS})

add_executable(zed main.cpp cameras.cpp)
target_compile_definitions(zed PRIVATE -Dzed)
target_link_libraries(zed PRIVATE ${ZED_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(intel main.cpp cameras.cpp)
target_compile_definitions(intel PRIVATE -Dintel)
target_link_libraries(intel realsense2 ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(oak main.cpp cameras.cpp)
target_compile_definitions(oak PRIVATE -Doak)
target_link_libraries(oak PRIVATE depthai::opencv ${OpenCV_LIBS} ${PCL_LIBRARIES})
