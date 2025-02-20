#include "cameras.hpp"
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace std::placeholders;

void Backend::init()
{
    // net = std::make_unique<Network>();
    // net->init();

#ifdef zed
    camera = std::make_unique<ZED>();
    windowName = "ZED object detection and distance measurement";
#endif
#ifdef intel
    camera = std::make_unique<Realsense>();
    windowName = "Realsense object detection and distance measurement";
    std::cout << "Realsense camera initialization\n";
#endif
#ifdef oak
    camera = std::make_unique<OAK>();
    windowName = "OAK object detection and distance measurement";
#endif
    // camera = std::make_unique<Realsense>();
    std::cout << "Window name: " << windowName << std::endl;
    cv::namedWindow(windowName);
}

void Backend::savePointcloud()
{
    std::stringstream ss;
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    ss << std::put_time(std::localtime(&in_time_t), "%H-%S-%M");
    ss << "_cloud.pcd";
    std::cout << "Saved a pointcloud with name " << ss.str() << std::endl;
    #ifdef intel
    cloud = camera->getPointCloud();
    pcl::io::savePCDFileASCII (ss.str(), cloud);
    #endif
}

void Backend::loop()
{
    if (camera == nullptr) return;
    std::cout << "Entering camera loop...\n";
    run = true;
    while (run)
    {
        camera->processFrame();
        auto color = camera->getColorFrame();
        auto depth = camera->getDepthFrame();
        auto pt = camera->getCartesianPoint(measurementLocation);
        pointVector[frameCount++ % FILTER_LEN] = pt;
        filterMeasurement();
        // auto boxes = net->detect(frame);

        // draw bounding boxes with distance measurement
        // for (int i = 0; i < boxes.size(); ++i)
        // {
        //     auto box = boxes[i];
        //     int centerX = box.x + box.width / 2;
        //     int centerY = box.y + box.height / 2;

        //     cv::rectangle(frame, box, colors[1], 3);
        //     cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), colors[1], cv::FILLED);
        // cv::putText(frame, "Distance: " + std::to_string(camera->getDistance(cv::Point(centerX, centerY))),
        //  cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        // }

        cv::Mat frame = color;

        std::stringstream location;

        location << std::fixed << std::setprecision(1) << "Measurement [" << coord.x << " , " << coord.y << " , " << coord.z << "]";
        //std::to_string(camera->getDistance(measurementLocation))

        // indicate the measurement point with the circle

        cv::circle(frame, measurementLocation, 5, colors[1], -1);
        cv::putText(frame, location.str(), {20, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[1]);
        // cv::applyColorMap(frame, frame, cv::COLORMAP_JET);
        cv::imshow(windowName, frame);

        auto key = cv::waitKey(1);

        switch(key)
        {
            case 's':
                std::cout << "saving pointcloud\n";
                savePointcloud();
                break;
            case 27:
                run = false;
                break;
        }
        
    }
    std::cout << "Exit camera backend loop...\n";
}

void Backend::filterMeasurement()
{
    // add a constant offset here!

    auto avgX = [this](){float r = 0; for(int i = 0; i < FILTER_LEN; i++){r += pointVector[i].x;} return (r/FILTER_LEN);};
    auto avgY = [this](){float r = 0; for(int i = 0; i < FILTER_LEN; i++){r += pointVector[i].y;} return (r/FILTER_LEN);};
    auto avgZ = [this](){float r = 0; for(int i = 0; i < FILTER_LEN; i++){r += pointVector[i].z;} return (r/FILTER_LEN);};

    coord = cv::Point3f{avgX(), avgY(), avgZ()};
}

/* ===================== REALSENSE ====================================*/
#ifdef intel
Realsense::Realsense()
{
    frameSize = cv::Size(640, 480);
    // Enable streams of color and depth
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    pipe.start(cfg);
    align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
}

void Realsense::processFrame()
{
    // Wait for frames
    rs2::frameset frames = pipe.wait_for_frames();

    // Align the frames
    frames = align->process(frames);
    lastColorFrame = frames.get_color_frame();
    lastDepthFrame = frames.get_depth_frame();
    
    intr = rs2::video_stream_profile(lastDepthFrame.get_profile()).get_intrinsics();
}

cv::Mat Realsense::getColorFrame()
{
    // left camera frame
    cv::Mat frame(frameSize, CV_8UC3, (void *)lastColorFrame.get_data(), cv::Mat::AUTO_STEP);
    return frame;
}

cv::Mat Realsense::getDepthFrame()
{
    cv::Mat frame(frameSize, CV_8UC3, (void *)lastDepthFrame.get_data(), cv::Mat::AUTO_STEP);
    return frame;
}

float Realsense::getDistance(cv::Point pt)
{
    return rs2::depth_frame(lastDepthFrame).get_distance(pt.x, pt.y);
}

pcl::PointCloud<pcl::PointXYZ> Realsense::getPointCloud()
{
    rs2::points points = pointCloud.calculate(lastDepthFrame);
    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width = sp.width();
    cloud.height = sp.height();
    cloud.is_dense = false;
    cloud.points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud.points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}

cv::Point3f Realsense::getCartesianPoint(cv::Point target)
{
    float coords[3];
    float xy[2] = {float(target.x), float(target.y)};
    auto dist = rs2::depth_frame(lastDepthFrame).get_distance(target.x, target.y);
    rs2_deproject_pixel_to_point(coords, &intr, xy, dist);
    return cv::Point3f(coords[0]*1000, coords[1]*1000, coords[2]*1000);
}

void Realsense::close()
{
    pipe.stop();
}
#endif
/* ======================== ZED =============================*/
#ifdef zed
ZED::ZED()
{
    // Initialization

    sl::InitParameters initParameters;
    initParameters.camera_resolution = sl::RESOLUTION::HD720;
    initParameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    initParameters.sdk_verbose = true;
    frameSize = cv::Size(1280, 720);
    try
    {
        if (cam.open(initParameters) != sl::ERROR_CODE::SUCCESS)
        {
            std::cout << "ERROR initializing zed!\n";
        }
        std::cout << "Initialized zed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void ZED::processFrame()
{
    // Frame acquisition by ZED - color frame and point cloud
    if (cam.grab() == sl::ERROR_CODE::SUCCESS)
    {
        cam.retrieveImage(lastZedFrame, sl::VIEW::LEFT);
        cam.retrieveMeasure(lastZedDepth, sl::MEASURE::DEPTH);
        cam.retrieveMeasure(pointCloud, sl::MEASURE::XYZRGBA);
    }
}

cv::Mat ZED::getColorFrame()
{
    cv::Mat frame(frameSize, CV_8UC4, (void *)lastZedFrame.getPtr<sl::uchar1>(), cv::Mat::AUTO_STEP);
    cv::Mat rgbFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGRA2BGR);

    return rgbFrame;
}

cv::Mat ZED::getDepthFrame()
{
    cv::Mat frame(frameSize, CV_8UC4, (void *)lastZedDepth.getPtr<sl::uchar1>(), cv::Mat::AUTO_STEP);
    return frame;
}

float ZED::getDistance(cv::Point pt)
{
    // get distance from the given point (centre of the bounding box)
    sl::float4 pointCloudValue;
    pointCloud.getValue(pt.x, pt.y, &pointCloudValue);
    float distance = sqrt(pointCloudValue.x * pointCloudValue.x +
                          pointCloudValue.y * pointCloudValue.y +
                          pointCloudValue.z * pointCloudValue.z);
    return distance;
}

cv::Point3f ZED::getCartesianPoint(cv::Point target)
{
    sl::float4 point;
    pointCloud.getValue(target.x, target.y, &point);

    return cv::Point3f(point.x, point.y, point.z);
}

pcl::PointCloud<pcl::PointXYZ> ZED::getPointCloud()
{
    pcl::PointCloud<pcl::PointXYZ> cloud;

    return cloud;
}

void ZED::close()
{
    // Close ZED camera
    cam.close();
}
#endif
/* ======================== OAK =============================*/
#ifdef oak
OAK::OAK()
{
    // Define a mono camera for each stream (left and right)
    camLeft = pipe.create<dai::node::MonoCamera>();
    camRight = pipe.create<dai::node::MonoCamera>();
    camRgb = pipe.create<dai::node::ColorCamera>();

    // Define stereo depth node
    stereo = pipe.create<dai::node::StereoDepth>();
    pointcloud = pipe.create<dai::node::PointCloud>();

    // Set up the cameras (MonoCamera nodes)
    // camLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    // camRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    camLeft->setCamera("left");
    camRight->setCamera("right");
    camLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    camRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);

    camRgb->setBoardSocket(dai::CameraBoardSocket::CENTER);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setPreviewSize(1280, 720);
    frameSize = cv::Size(1280, 720);

    // Set the stereo depth properties
    stereo->setLeftRightCheck(true);
    stereo->setExtendedDisparity(false);
    stereo->setSubpixel(true);
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);

    camLeft->out.link(stereo->left);
    camRight->out.link(stereo->right);

    // Create output queues
    qRgb = pipe.create<dai::node::XLinkOut>();
    qRgb->setStreamName("video");
    // color image comes from the left camera
    camRgb->preview.link(qRgb->input);

    qDepth = pipe.create<dai::node::XLinkOut>();
    qDepth->setStreamName("depth");
    stereo->disparity.link(qDepth->input);

    qDebug = pipe.create<dai::node::XLinkOut>();
    qDebug->setStreamName("debug");
    camLeft->out.link(qDebug->input);

    qPointCloud = pipe.create<dai::node::XLinkOut>();
    qPointCloud->setStreamName("pointCloud");
    stereo->depth.link(pointcloud->inputDepth);
    pointcloud->outputPointCloud.link(qPointCloud->input);
    pointcloud->initialConfig.setSparse(true);

    device = std::make_unique<dai::Device>(pipe);
    // Output queues for RGB and depth frames
    qRgbOutput = device->getOutputQueue("video", 8, false);
    qDepthOutput = device->getOutputQueue("depth", 8, false);
    qDebugMono = device->getOutputQueue("debug", 8, false);
    qPointCloudOut = device->getOutputQueue("pointCloud", 8, false);

    auto intr = device->readCalibration().getCameraIntrinsics(dai::CameraBoardSocket::CENTER);
    fx = intr[0][0];
    fy = intr[1][1];
    cx = intr[2][0];
    cy = intr[2][1];
}

cv::Mat OAK::getColorFrame()
{
    // Frame acquisition from output queue

    std::shared_ptr<dai::ImgFrame> frame = qRgbOutput->get<dai::ImgFrame>();
    // std::shared_ptr<dai::ImgFrame> frame = qDebugMono->get<dai::ImgFrame>();
    cv::Mat rgbMat = frame->getCvFrame();
    // cv::cvtColor(rgbMat, rgbMat, cv::COLOR_GRAY2BGR);
    return rgbMat;
}

cv::Mat OAK::getDepthFrame()
{

    std::shared_ptr<dai::ImgFrame> frame = qDepthOutput->get<dai::ImgFrame>();
    // cv::Mat depthMat(frame->getHeight(), frame->getWidth(), CV_8UC3, (void*)frame->getData());
    cv::Mat depthMat = frame->getCvFrame();
    //depthMat.convertTo(depthMat, CV_8UC1, 255.0 / stereo->initialConfig.getMaxDisparity());
    lastDepthFrame = depthMat;
    // pointcloudData = qPointCloudOut->get<dai::PointCloudData>();

    return depthMat;
}

float OAK::getDistance(cv::Point target)
{
    // DepthAI supports point cloud - dai::PointCloud, dai::PointCloudData
    // here we try with depth aquired from the depth frame
    cv::Mat depthDisplay;
    // Normalize the depth values for visualization
    lastDepthFrame.convertTo(depthDisplay, CV_8UC1, 255.0 / stereo->initialConfig.getMaxDisparity());
    float distance = depthDisplay.at<uint8_t>(target.x, target.y) / 1000.0f; // scale to mm

    return distance;
}

cv::Point3f OAK::getCartesianPoint(cv::Point target)
{
    return cv::Point3f(0, 0, 0);
}

pcl::PointCloud<pcl::PointXYZ> OAK::getPointCloud()
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    return cloud;
}

void OAK::processFrame()
{
    // this function is not necessary for OAK as frame acquisition is done by the API
}

void OAK::close()
{
    qDepthOutput->close();
    qRgbOutput->close();
}
#endif

// int Network::init()
// {
//     net = cv::dnn::readNetFromDarknet(cfgPath, weightsPath);
//     if (net.empty())
//     {
//         std::cerr << "Error: Failed to load YOLOv4 model!" << std::endl;
//         return 1;
//     }

//     std::ifstream classFile("../models/coco.names");
//     std::string line;
//     while (std::getline(classFile, line))
//     {
//         classNames.push_back(line);
//     }
//     if (classNames.empty())
//         std::cout << "ERROR loading class names!\n";

//     model = std::make_unique<cv::dnn::DetectionModel>(net);
//     if (model == nullptr)
//     {
//         std::cout << "Failed to load model\n";
//         return 1;
//     }

//     // model->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//     // model->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//     model->setInputParams(1.0 / 255, cv::Size(416, 416), cv::Scalar(), true);

//     return 0;
// }

// std::vector<cv::Rect> Network::detect(cv::Mat frame)
// {
//     std::vector<int> classIds;
//     std::vector<float> confidences;
//     std::vector<cv::Rect> boxes;
//     std::vector<cv::Rect> results;

//     // Detecting objects in the image
//     if (frame.empty())
//     {
//         std::cout << "ERROR: The frame is empty!!!\n";
//     }

//     model->detect(frame, classIds, confidences, boxes, 0.1, 0.1);

//     for (int i = 0; i < classIds.size(); i++)
//     {
//         if (confidences[i] > tConfidence && classIds[i] == 0)
//         {
//             auto box = boxes[i];
//             results.push_back(box);
//         }
//     }

//     return results;
// }