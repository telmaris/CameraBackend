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
    virtual cv::Point3f getCartesianPoint(cv::Point target) = 0;
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
    cv::Point3f getCartesianPoint(cv::Point target) override;
    void processFrame() override;
    void close() override;

    sl::Camera zed;
    //sl::Mat lastZedFrame{sl::Resolution{1280, 720}, sl::MAT_TYPE::U8_C3};
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
    cv::Point3f getCartesianPoint(cv::Point target) override;
    void close() override;

    rs2::pipeline pipe;
    rs2::config cfg;
    std::unique_ptr<rs2::align> align;
    rs2::pointcloud pointCloud;
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
    cv::Point3f getCartesianPoint(cv::Point target) override;
    void processFrame() override;
    void close() override;

    dai::Pipeline pipe;
    std::unique_ptr<dai::Device> device;

    // mono camera objects (represent physical, left and right)
    std::shared_ptr<dai::node::MonoCamera> camLeft;
    std::shared_ptr<dai::node::MonoCamera> camRight;

    // color camera object

    std::shared_ptr<dai::node::ColorCamera> camRgb;

    // stereo handler object
    std::shared_ptr<dai::node::StereoDepth> stereo;

    std::shared_ptr<dai::node::PointCloud> pointcloud;

    // XLinks handle video stream of a given type (color, depth)
    std::shared_ptr<dai::node::XLinkOut> qRgb;
    std::shared_ptr<dai::node::XLinkOut> qDepth;
    std::shared_ptr<dai::node::XLinkOut> qDebug;
    std::shared_ptr<dai::node::XLinkOut> qPointCloud;

    // output data streams
    std::shared_ptr<dai::DataOutputQueue> qRgbOutput;
    std::shared_ptr<dai::DataOutputQueue> qDepthOutput;
    std::shared_ptr<dai::DataOutputQueue> qDebugMono;
    std::shared_ptr<dai::DataOutputQueue> qPointCloudOut;

    // depth frame 
    cv::Mat lastDepthFrame;
    std::shared_ptr<dai::PointCloudData> pointcloudData;

    float fx, fy, cx, cy; // focal lengths and principal points
};

class Astra : public Camera
{
public:
    Astra();
    ~Astra() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    cv::Point3f getCartesianPoint(cv::Point target) override;
    void processFrame() override;
    void close() override;

    openni::Device device;

    // frame streams for color and depth
    openni::VideoStream rgbStream;
    openni::VideoStream depthStream;

    cv::Mat lastDepthFrame;
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

    float tConfidence = 0.1;
};

class Backend
{
public:

    void init();
    void parseArgs(int argc, char** argv);

    void loop();

    std::unique_ptr<Camera> camera;
    std::unique_ptr<Network> net;

    // void setMeasurementPoint(int event, int x, int y, int flags, void* userdata);

    bool run = true;
    bool visualMode = true;
    std::string windowName = "default";
    cv::Point measurementLocation;

};

#endif