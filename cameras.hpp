#ifndef cameras_hpp
#define cameras_hpp

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

// OpenCV 4 for image conversion, and visualisation
#include <opencv2/opencv.hpp>
// OpenCV 4 DNN for setting up the neural network for detection
// #include <opencv2/dnn.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#define FILTER_LEN 30

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0),
                                        cv::Scalar(0, 255, 0),
                                        cv::Scalar(0, 255, 255), 
                                        cv::Scalar(255, 0, 0)};

class Camera
{
public:
    virtual ~Camera() = default;
    virtual cv::Mat getColorFrame() = 0;
    virtual cv::Mat getDepthFrame() = 0;
    virtual float getDistance(cv::Point) = 0;
    virtual cv::Point3f getCartesianPoint(cv::Point target) = 0;
    virtual pcl::PointCloud<pcl::PointXYZ> getPointCloud() = 0;
    virtual void processFrame() = 0;
    virtual void close() = 0;

    cv::Size frameSize;
};

#ifdef zed
// ZED API
#include <sl/Camera.hpp>
class ZED : public Camera
{
public:
    ZED();
    ~ZED() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    cv::Point3f getCartesianPoint(cv::Point target) override;
    pcl::PointCloud<pcl::PointXYZ> getPointCloud() override;
    void processFrame() override;
    void close() override;

    sl::Camera cam;
    //sl::Mat lastZedFrame{sl::Resolution{1280, 720}, sl::MAT_TYPE::U8_C3};
    sl::Mat lastZedFrame;
    sl::Mat lastZedDepth;
    sl::Mat pointCloud;
};
#endif
#ifdef intel
// Intel RealSense API
#include <librealsense2/rs.hpp>
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
    pcl::PointCloud<pcl::PointXYZ> getPointCloud() override;
    void close() override;

    rs2::pipeline pipe;
    rs2::config cfg;
    std::unique_ptr<rs2::align> align;
    rs2::pointcloud pointCloud;
    rs2::frame lastColorFrame, lastDepthFrame;
    rs2_intrinsics intr;

    // rs2::decimation_filter dec_filter;
    rs2::spatial_filter spat_filter;  
    rs2::temporal_filter temp_filter;
    rs2::hole_filling_filter hole_filter;
};
#endif
#ifdef oak
// OAK API
#include <depthai/depthai.hpp>
class OAK : public Camera
{
public:
    OAK();
    ~OAK() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    cv::Point3f getCartesianPoint(cv::Point target) override;
    pcl::PointCloud<pcl::PointXYZ> getPointCloud() override;

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
    std::shared_ptr<dai::node::SpatialLocationCalculator> spatial;

    // XLinks handle video stream of a given type (color, depth)
    std::shared_ptr<dai::node::XLinkOut> qRgb;
    std::shared_ptr<dai::node::XLinkOut> qDepth;
    std::shared_ptr<dai::node::XLinkOut> qDebug;
    std::shared_ptr<dai::node::XLinkOut> qPointCloud;
    std::shared_ptr<dai::node::XLinkOut> spatialOut;
    std::shared_ptr<dai::node::XLinkIn> spatialConfig;
    
    // output data streams
    std::shared_ptr<dai::DataOutputQueue> qRgbOutput;
    std::shared_ptr<dai::DataOutputQueue> qDepthOutput;
    std::shared_ptr<dai::DataOutputQueue> qDebugMono;
    std::shared_ptr<dai::DataOutputQueue> qPointCloudOut;
    std::shared_ptr<dai::DataOutputQueue> spatialData;
    std::shared_ptr<dai::DataInputQueue> spatialConfigQ;

    // depth frame 
    std::shared_ptr<dai::ImgFrame> depthFrame;
    cv::Mat lastDepthFrame;
    std::shared_ptr<dai::PointCloudData> pointcloudData;

    float fx, fy, cx, cy, fov; // focal lengths and principal points
    cv::Point lastTarget{0,0};
};
#endif

class Backend
{
public:

    void init();
    void parseArgs(std::string);

    void loop();
    void filterMeasurement();
    void savePointcloud();
    void saveMeasurement();
    void getCartesianLocationWithRoi();
    void updateRoiSize(int px);
    void updateRoiPosition(int x, int y);
    void tuneRoiPosition(int x, int y);

    std::shared_ptr<Camera> camera;
    // std::unique_ptr<Network> net;

    // void setMeasurementPoint(int event, int x, int y, int flags, void* userdata);

    bool run = true;
    bool visualMode = true;
    std::string windowName = "default";
    cv::Point measurementLocation{100,100};
    std::array<cv::Point3f, FILTER_LEN> pointVector;
    cv::Point3f coord{0,0,0};
    int frameCount = 0;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    int roiSize = 3;
    cv::Rect roi = {measurementLocation.x - (roiSize-1)/2,
                    measurementLocation.y - (roiSize-1)/2,
                    roiSize, roiSize};
    int savedMeasurements = 0;
    std::string cameraName;
    int verticalLinePos = 359;
    int horizontalLinePos = 639;
    int tuningMode = 0;
    int lineType = 0; // 0 is horizontal
    bool enableFilters = false;
    std::atomic<bool> isRunning = false;
    int measurementSeriesLength = 30;
    cv::Mat lastFrame;
    cv::Mat lastDepthFrame;
};

#endif
#ifdef gemini
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Utils.hpp>

class Gemini : public Camera
{
    public:
    Gemini();
    ~Gemini() = default;

    cv::Mat getColorFrame() override;
    cv::Mat getDepthFrame() override;
    float getDistance(cv::Point) override;
    cv::Point3f getCartesianPoint(cv::Point target) override;
    pcl::PointCloud<pcl::PointXYZ> getPointCloud() override;
    void processFrame() override;
    void close() override;

    ob::Pipeline pipeline;
    // ob::Config config;
    std::shared_ptr<ob::Frame> colorFrame;
    std::shared_ptr<ob::Frame> depthFrame;
    OBCameraIntrinsic intrinsic; 
    OBExtrinsic extrinsic;
    ob::CoordinateTransformHelper transformHelper;
    std::vector<std::shared_ptr<ob::Filter>> filters;

};

#endif