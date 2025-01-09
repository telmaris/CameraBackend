#ifndef cameras_hpp
#define cameras_hpp

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

// OpenCV 4 for image conversion, and visualisation
#include <opencv2/opencv.hpp>
// OpenCV 4 DNN for setting up the neural network for detection
#include <opencv2/dnn.hpp>
// Intel RealSense API
#include <librealsense2/rs.hpp>
// ZED API
#include <sl/Camera.hpp>
// OAK API
#include <depthai/depthai.hpp>
// ASTRA API (OpenNI)
#include <openni/OpenNI.h>

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0),
                                        cv::Scalar(0, 255, 0),
                                        cv::Scalar(0, 255, 255), 
                                        cv::Scalar(255, 0, 0)};

class Camera
{
public:
    virtual cv::Mat getColorFrame() = 0;
    virtual cv::Mat getDepthFrame() = 0;
    virtual float getDistance(cv::Point) = 0;
    virtual void processFrame() = 0;
    virtual void close() = 0;

    cv::Size frameSize;
};

class ZED : public Camera
{
public:
    ZED();
    ~ZED() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    void processFrame() override;
    void close() override;

    sl::Camera zed;
    sl::Mat lastZedFrame;
    sl::Mat lastZedDepth;
    sl::Mat pointCloud;
};

class Realsense : public Camera
{
public:
    Realsense();
    ~Realsense() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    void processFrame() override;
    void close() override;

    rs2::pipeline pipe;
    rs2::config cfg;
    std::unique_ptr<rs2::align> align;
    rs2::frame lastColorFrame, lastDepthFrame;
};

class OAK : public Camera
{
public:
    OAK();
    ~OAK() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    void processFrame() override;
    void close() override;

    dai::Pipeline pipe;
    dai::Device device{pipe};

    // mono camera objects (represent physical, left and right)
    dai::node::MonoCamera* camLeft;
    dai::node::MonoCamera* camRight;

    // stereo handler object
    dai::node::StereoDepth* stereo;

    // XLinks handle video stream of a given type (color, depth)
    dai::node::XLinkOut* qRgb;
    dai::node::XLinkOut* qDepth;

    // output data streams
    dai::DataOutputQueue* qRgbOutput;
    dai::DataOutputQueue* qDepthOutput;

    // depth frame 
    cv::Mat lastDepthFrame;
};

class Astra : public Camera
{
public:
    Astra();
    ~Astra() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    void processFrame() override;
    void close() override;

    openni::Device device;

    // frame streams for color and depth
    openni::VideoStream rgbStream;
    openni::VideoStream depthStream;
};

class Network
{
public:

    int init();
    std::vector<cv::Rect> detect(cv::Mat);

    cv::dnn::Net net;
    std::unique_ptr<cv::dnn::DetectionModel> model;

    std::vector<std::string> classNames;
    std::string cfgPath = "../models/yolov4-tiny.cfg";
    std::string weightsPath = "../models/yolov4-tiny.weights";

    float tConfidence = 0.6;
};

class Backend
{
public:

    void init();
    void parseArgs(int argc, char** argv);

    void loop();

    std::unique_ptr<Camera> camera;
    std::unique_ptr<Network> net;

    bool run = true;
    bool visualMode = true;
    std::string windowName = "default";

};

#endif