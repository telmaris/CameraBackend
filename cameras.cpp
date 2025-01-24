#include "cameras.hpp"

using namespace std::placeholders;

int Network::init()
{
    net = cv::dnn::readNetFromDarknet(cfgPath, weightsPath);
    if (net.empty())
    {
        std::cerr << "Error: Failed to load YOLOv4 model!" << std::endl;
        return 1;
    }

    std::ifstream classFile("../models/coco.names");
    std::string line;
    while (std::getline(classFile, line))
    {
        classNames.push_back(line);
    }
    if (classNames.empty())
        std::cout << "ERROR loading class names!\n";

    model = std::make_unique<cv::dnn::DetectionModel>(net);
    if (model == nullptr)
    {
        std::cout << "Failed to load model\n";
        return 1;
    }

    // model->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    // model->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model->setInputParams(1.0 / 255, cv::Size(416, 416), cv::Scalar(), true);

    return 0;
}

std::vector<cv::Rect> Network::detect(cv::Mat frame)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> results;

    // Detecting objects in the image
    if (frame.empty())
    {
        std::cout << "ERROR: The frame is empty!!!\n";
    }

    model->detect(frame, classIds, confidences, boxes, 0.1, 0.1);

    for (int i = 0; i < classIds.size(); i++)
    {
        if (confidences[i] > tConfidence && classIds[i] == 0)
        {
            auto box = boxes[i];
            results.push_back(box);
        }
    }

    return results;
}

void Backend::init()
{
    // net = std::make_unique<Network>();
    // net->init();
    cv::namedWindow(windowName);
}

void Backend::parseArgs(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "zed")
        {
            camera = std::make_unique<ZED>();
            windowName = "ZED object detection and distance measurement";
            return;
        }
        if (arg == "intel")
        {
            camera = std::make_unique<Realsense>();
            windowName = "Realsense object detection and distance measurement";
            return;
        }
        if (arg == "oak")
        {
            camera = std::make_unique<OAK>();
            windowName = "OAK object detection and distance measurement";
            return;
        }
        // if (arg == "astra")
        // {
        //     camera = std::make_unique<Astra>();
        //     windowName = "Astra object detection and distance measurement";
        //     return;
        // }

        // auto callback = std::bind(&Backend::setMeasurementPoint,this, _1, _2, _3, _4, _5);
        // cv::setMouseCallback(windowName, callback);
    }
}

void Backend::loop()
{
    std::cout << "Entering camera loop...\n";
    if (camera == nullptr)
        run = false;
    while (run)
    {
        camera->processFrame();
        auto color = camera->getColorFrame();
        auto depth = camera->getDepthFrame();
        auto coord = camera->getCartesianPoint(measurementLocation);

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
        location << "Measurement [" << coord.x << " , " << coord.y << " , " << coord.z << "]";

        

        // indicate the measurement point with the circle
        cv::circle(frame, measurementLocation, 5, colors[1], -1);
        // cv::putText(frame, location.str(), {20, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[1]);
        cv::putText(frame, std::to_string(camera->getDistance(measurementLocation)), {20, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[1]);
        cv::imshow(windowName, frame);

        if (cv::waitKey(1) == 27)
        {
            run = false;
        }
    }
    std::cout << "Exit camera backend loop...\n";
}

/* ===================== REALSENSE ====================================*/

Realsense::Realsense()
{
    frameSize = cv::Size(640, 480);
    // Enable streams of color and depth
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    pipe.start(cfg);
    align = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
    // pointCloud = rs2::context().create_pointcloud();
}

void Realsense::processFrame()
{
    // Wait for frames
    rs2::frameset frames = pipe.wait_for_frames();

    // Align the frames
    frames = align->process(frames);
    lastColorFrame = frames.get_color_frame();
    lastDepthFrame = frames.get_depth_frame();
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

cv::Point3f Realsense::getCartesianPoint(cv::Point target)
{
    return cv::Point3f(0, 0, 0);
}

void Realsense::close()
{
    pipe.stop();
}

/* ======================== ZED =============================*/

ZED::ZED()
{
    // Initialization

    sl::InitParameters initParameters;
    initParameters.camera_resolution = sl::RESOLUTION::HD720;
    initParameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    initParameters.sdk_verbose = true;
    frameSize = cv::Size(1280, 720);
    if (zed.open(initParameters) != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "ERROR initializing zed!\n";
    }
    std::cout << "Initialized zed\n";
}

void ZED::processFrame()
{
    // Frame acquisition by ZED - color frame and point cloud
    if (zed.grab() == sl::ERROR_CODE::SUCCESS)
    {
        zed.retrieveImage(lastZedFrame, sl::VIEW::LEFT);
        zed.retrieveMeasure(lastZedDepth, sl::MEASURE::DEPTH);
        zed.retrieveMeasure(pointCloud, sl::MEASURE::XYZRGBA);
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

void ZED::close()
{
    // Close ZED camera
    zed.close();
}

/* ======================== OAK =============================*/

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
    depthMat.convertTo(depthMat, CV_8UC1, 255 / stereo->initialConfig.getMaxDisparity());
    lastDepthFrame = depthMat;
    pointcloudData = qPointCloudOut->get<dai::PointCloudData>();

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
    // auto points = pointcloudData->getPoints();
    // int width = pointcloudData->getWidth();
    // int height = pointcloudData->getHeight();

    // // Create a blank 2D image to project the point cloud onto
    // cv::Mat projectionImage = cv::Mat::zeros(height, width, CV_8UC3);

    // // Iterate through the point cloud to project each 3D point onto 2D
    // for (int y = 0; y < height; ++y)
    // {
    //     for (int x = 0; x < width; ++x)
    //     {
    //         // Calculate the index of the 3D point in the raw vector
    //         int index = y * width + x;

    //         // Get the 3D point (X, Y, Z)
    //         dai::Point3f point3D = points[index];

    //         // Check if the depth (Z) is valid
    //         if (point3D.z != 0)
    //         {
    //             // Project 3D point onto 2D using camera intrinsics
    //             int u = static_cast<int>(fx * point3D.x / point3D.z + cx);
    //             int v = static_cast<int>(fy * point3D.y / point3D.z + cy);

    //             // // Ensure (u, v) is within the image bounds
    //             // if (u >= 0 && u < width && v >= 0 && v < height)
    //             // {
    //             //     // Color the projected pixel (this can be adjusted)
    //             //     projectionImage.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 255, 255); // White pixel
    //             // }
    //         }
    //     }
    // }

    return cv::Point3f(0, 0, 0);
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

/* ======================== ASTRA =============================*/

// Astra::Astra()
// {
//     // initialize OpenNI
//     if (openni::OpenNI::initialize() != openni::STATUS_OK) {
//         std::cerr << "OpenNI initialization failed: " << openni::OpenNI::getExtendedError() << std::endl;
//     }

//     // open an ASTRA physical camera
//     if (device.open(openni::ANY_DEVICE) != openni::STATUS_OK) {
//         std::cerr << "Failed to open device: " << openni::OpenNI::getExtendedError() << std::endl;
//     }

//     // create a color camera object
//     if (rgbStream.create(device, openni::SENSOR_COLOR) != openni::STATUS_OK) {
//         std::cerr << "Failed to create RGB stream: " << openni::OpenNI::getExtendedError() << std::endl;
//     }

//     // depth camera object
//     if (depthStream.create(device, openni::SENSOR_DEPTH) != openni::STATUS_OK) {
//         std::cerr << "Failed to create Depth stream: " << openni::OpenNI::getExtendedError() << std::endl;
//     }

// }

// cv::Mat Astra::getColorFrame()
// {
//     // Frame acquisition from color stream

//     openni::VideoFrameRef rgbFrame;
//     rgbStream.readFrame(&rgbFrame);

//     const openni::RGB888Pixel* rgbData = (const openni::RGB888Pixel*)rgbFrame.getData();
//     cv::Mat rgbMat(rgbFrame.getHeight(), rgbFrame.getWidth(), CV_8UC3, (void*)rgbData);
//     return rgbMat;
// }

// cv::Mat Astra::getDepthFrame()
// {
//     openni::VideoFrameRef depthFrame;
//     depthStream.readFrame(&depthFrame);

//     const uint16_t* depthData = (const uint16_t*)depthFrame.getData();
//     cv::Mat depthMat(depthFrame.getHeight(), depthFrame.getWidth(), CV_16U, (void*)depthData);
//     lastDepthFrame = depthMat;
//     return depthMat;
// }

// float Astra::getDistance(cv::Point target)
// {
//     float distance = lastDepthFrame.at<uint8_t>(target.x, target.y) / 1000.0f; // scale to mm
//     return distance;
// }

// void Astra::processFrame()
// {
//     // this function is not necessary for Astra as frame acquisition is done by the API
// }

// void Astra::close()
// {
//     // Close all streams, device and API
//     rgbStream.stop();
//     depthStream.stop();
//     device.close();

//     openni::OpenNI::shutdown();
// }