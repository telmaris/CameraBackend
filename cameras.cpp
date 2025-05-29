#include "cameras.hpp"
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>
#include <filesystem>

using namespace std::placeholders;
using namespace std::chrono_literals;

std::ostream &operator<<(std::ostream &os, const cv::Point3f &point)
{
    os << "(" << point.x << ", " << point.y << ", " << point.z << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const cv::Point &point)
{
    os << "(" << point.x << ", " << point.y << ")";
    return os;
}

int filter = 0;

void Backend::init()
{
    // net = std::make_unique<Network>();
    // net->init();

    if (enableFilters)
        filter = 1;

#ifdef zed
    camera = std::make_shared<ZED>();
    windowName = "ZED object detection and distance measurement";
    cameraName = "zed2i";
#endif
#ifdef intel
    camera = std::make_shared<Realsense>();
    windowName = "Realsense object detection and distance measurement";
    cameraName = "d455";
#endif
#ifdef oak
    camera = std::make_shared<OAK>();
    windowName = "OAK object detection and distance measurement";
    cameraName = "oak-d-pro";
#endif
#ifdef gemini
    camera = std::make_shared<Gemini>();
    windowName = "Gemini object detection and distance measurement";
    cameraName = "gemini-335l";
#endif
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
    pcl::io::savePCDFileASCII(ss.str(), cloud);
#endif
}

void Backend::saveMeasurement()
{
    static int counter = 0;
    std::string suffix{"_measurements_"};
    std::string name{cameraName + suffix + std::to_string(counter) + ".csv"};

    for (int i = 0; i < measurementSeriesLength; i++)
    {
        auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        if (savedMeasurements++ == 0)
        {
            while(std::filesystem::exists(name))
            {
                counter++;
                name = cameraName + suffix + std::to_string(counter) + ".csv";
            }
            std::fstream file(name, std::ios::out);

            // Check if the file opened successfully
            if (!file)
            {
                std::cerr << "Failed to open the file!" << std::endl;
                return;
            }
            file << cameraName << " measurements of depth on " << std::put_time(std::localtime(&in_time_t), "%d-%m-%Y") << std::endl;
        }
        std::fstream file(name, std::ios::out | std::ios::app);
        // file << coord << " at " << std::put_time(std::localtime(&in_time_t), "%S-%M-%H")
        // << " measurement position: " << measurementLocation << " ROI size: " << roiSize << std::endl;
        file << coord.x << "," << coord.y << "," << coord.z << std::endl;
        if (savedMeasurements % 10 == 0)
            file << std::endl;
        file.close();
        std::this_thread::sleep_for(100ms);
    }
    isRunning = false;
    std::cout << "Saved 10 measurements!\n";
    std::stringstream time;
    auto in_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    time << std::put_time(std::localtime(&in_time_t), "%S-%M-%H");
    cv::imwrite("../img/" + cameraName + "_frame_" + time.str() + ".jpg", lastFrame);
}

void Backend::getCartesianLocationWithRoi()
{
    int x = roi.x;
    int y = roi.y;
    std::vector<cv::Point3f> results;
    cv::Point3f res{0, 0, 0};

    for (int i = 0; i < roiSize; i++)
    {
        for (int j = 0; j < roiSize; j++)
        {
            cv::Point location{x + j, y + i};
            auto pt = camera->getCartesianPoint(location);
            results.push_back(pt);
            res.x += pt.x;
            res.y += pt.y;
            res.z += pt.z;
        }
    }

    // for (auto pt : results)
    // {
    //     res.x += pt.x;
    //     res.y += pt.y;
    //     res.z += pt.z;
    // }
    res.x /= results.size();
    res.y /= results.size();
    res.z /= results.size();
    coord = res;
}

void Backend::updateRoiSize(int px)
{
    roiSize += px;
    if (roiSize < 3)
    {
        roiSize += 2;
        return;
    }
#ifdef oak
    dai::SpatialLocationCalculatorConfigData config;
    config.roi = dai::Rect(dai::Point2f(measurementLocation.x - (roiSize - 1) / 2,
                                        measurementLocation.y - (roiSize - 1) / 2),
                           dai::Point2f(measurementLocation.x + (roiSize - 1) / 2,
                                        measurementLocation.y + (roiSize - 1) / 2));

    dai::SpatialLocationCalculatorConfig cfg;
    cfg.addROI(config);
    auto cam = std::static_pointer_cast<OAK>(camera);
    cam->spatialConfigQ->send(cfg);
    cam->lastTarget = measurementLocation;
#endif
    roi = {measurementLocation.x - (roiSize - 1) / 2,
           measurementLocation.y - (roiSize - 1) / 2,
           roiSize, roiSize};
}

void Backend::updateRoiPosition(int x, int y)
{
    roi = {x - (roiSize - 1) / 2,
           y - (roiSize - 1) / 2,
           roiSize, roiSize};
}

void Backend::tuneRoiPosition(int x, int y)
{
    roi.x += x;
    roi.y += y;
}

void Backend::loop()
{
    if (camera == nullptr)
        return;
    std::cout << "Entering camera loop...\n";
    run = true;
    while (run)
    {
        camera->processFrame();
        auto color = camera->getColorFrame();
        // std::cout << color.size() << std::endl;
        auto depth = camera->getDepthFrame();
        cv::Mat frame = color;
        lastFrame = frame;
        if (!tuningMode)
        {

#ifdef oak
            coord = camera->getCartesianPoint(measurementLocation);
#else
            getCartesianLocationWithRoi();
#endif

            pointVector[frameCount++ % FILTER_LEN] = coord;
            filterMeasurement();

            std::stringstream location;
            cv::rectangle(frame, {20, 20, 350, 20}, {255, 255, 255}, -1);
            cv::rectangle(frame, roi, {255, 0, 0}, 1);
            location << std::fixed << std::setprecision(1) << "Measurement [" << coord.x << " , " << coord.y << " , " << coord.z << "]";

            // indicate the measurement ROI with a square

            cv::putText(frame, location.str(), {21, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0});
        }
        else
        {
            cv::rectangle(frame, {lineType * verticalLinePos, (lineType ^ 1) * horizontalLinePos, (lineType ^ 1) * 1280 + 5, lineType * 720 + 5}, {0, 0, 255}, -1);
        }

        cv::imshow(windowName, frame);
        auto key = cv::waitKey(1);

        switch (key)
        {
        case 'x':
            std::cout << "saving pointcloud\n";
            savePointcloud();
            break;
        case 'p':
            updateRoiSize(2);
            std::cout << "ROI size: " << roiSize << std::endl;
            break;
        case 'o':
            updateRoiSize(-2);
            std::cout << "ROI size: " << roiSize << std::endl;
            break;
        case ' ':
            if (isRunning == false)
            {
                isRunning = true;
                std::thread([this]()
                            { this->saveMeasurement(); })
                    .detach();
            }
            // saveMeasurement();
            break;
        case 't':
            tuningMode = true;
            break;
        case 'l':
            if (tuningMode)
                lineType ^= 1;
            break;
        case 'w':
            if (--roi.y < 0)
                roi.y = 0;
            break;
        case 'a':
            if (--roi.x < 0)
                roi.x = 0;
            break;
        case 's':
            if (++roi.y > camera->frameSize.height)
            roi.y = camera->frameSize.height;
            break;
        case 'd':
            if (++roi.x > camera->frameSize.width)
            roi.x = camera->frameSize.width;
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

    auto avgX = [this]()
    {float r = 0; for(int i = 0; i < FILTER_LEN; i++){r += pointVector[i].x;} return (r/FILTER_LEN); };
    auto avgY = [this]()
    {float r = 0; for(int i = 0; i < FILTER_LEN; i++){r += pointVector[i].y;} return (r/FILTER_LEN); };
    auto avgZ = [this]()
    {float r = 0; for(int i = 0; i < FILTER_LEN; i++){r += pointVector[i].z;} return (r/FILTER_LEN); };

    coord = cv::Point3f{avgX(), avgY(), avgZ()};
}

/* ===================== REALSENSE ====================================*/
#ifdef intel
Realsense::Realsense()
{
    frameSize = cv::Size(1280, 720);
    // Enable streams of color and depth
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

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

    // lastDepthFrame = dec_filter.process(lastDepthFrame);
    if (filter)
    {
        lastDepthFrame = spat_filter.process(lastDepthFrame);
        lastDepthFrame = temp_filter.process(lastDepthFrame);
        lastDepthFrame = hole_filter.process(lastDepthFrame);
    }
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
    for (auto &p : cloud.points)
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
    return cv::Point3f(coords[0] * 1000, coords[1] * -1000, coords[2] * 1000);
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
    initParameters.camera_resolution = sl::RESOLUTION::HD1080;
    initParameters.depth_mode = sl::DEPTH_MODE::QUALITY;
    initParameters.sdk_verbose = true;
    frameSize = cv::Size(1920, 1080);
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

    return cv::Point3f(point.x, point.y, point.z - 0.017);
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
    // pointcloud = pipe.create<dai::node::PointCloud>();
    spatial = pipe.create<dai::node::SpatialLocationCalculator>();

    // Set up the cameras (MonoCamera nodes)
    // camLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    // camRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    camLeft->setCamera("left");
    camRight->setCamera("right");
    camLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);
    camRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_720_P);

    // camRgb->setBoardSocket(dai::CameraBoardSocket::CENTER);
    // camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    // camRgb->setPreviewSize(1280, 720);
    frameSize = cv::Size(1280, 720);

    // Set the stereo depth properties
    stereo->setLeftRightCheck(true);
    stereo->setExtendedDisparity(false);
    stereo->setSubpixel(true);
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->setDepthAlign(dai::CameraBoardSocket::LEFT);
    if (filter)
    {
        auto config = stereo->initialConfig.get();

        config.postProcessing.speckleFilter.enable = true;
        config.postProcessing.speckleFilter.speckleRange = 50;
        config.postProcessing.temporalFilter.enable = true;
        config.postProcessing.spatialFilter.enable = true;
        config.postProcessing.spatialFilter.holeFillingRadius = 2;
        config.postProcessing.spatialFilter.numIterations = 1;
        config.postProcessing.thresholdFilter.minRange = 400;
        config.postProcessing.thresholdFilter.maxRange = 15000;
        config.postProcessing.decimationFilter.decimationFactor = 1;
        stereo->initialConfig.set(config);
    }
    camLeft->out.link(stereo->left);
    camRight->out.link(stereo->right);

    // Create output queues
    // qRgb = pipe.create<dai::node::XLinkOut>();
    // qRgb->setStreamName("video");
    // color image comes from the left camera
    // camRgb->preview.link(qRgb->input);

    qDepth = pipe.create<dai::node::XLinkOut>();
    qDepth->setStreamName("depth");
    stereo->disparity.link(qDepth->input);

    qDebug = pipe.create<dai::node::XLinkOut>();
    qDebug->setStreamName("debug");
    camLeft->out.link(qDebug->input);

    dai::SpatialLocationCalculatorConfigData config;

    config.depthThresholds.lowerThreshold = 100;
    config.depthThresholds.upperThreshold = 20000;
    config.calculationAlgorithm = dai::SpatialLocationCalculatorAlgorithm::AVERAGE;
    config.roi = dai::Rect(dai::Point2f(0, 0), dai::Point2f(1, 1));
    spatial->inputConfig.setWaitForMessage(false);
    spatial->initialConfig.addROI(config);

    spatialOut = pipe.create<dai::node::XLinkOut>();
    spatialConfig = pipe.create<dai::node::XLinkIn>();
    spatialOut->setStreamName("spatial");
    spatialConfig->setStreamName("spatialConfig");
    stereo->depth.link(spatial->inputDepth);
    spatial->out.link(spatialOut->input);
    spatialConfig->out.link(spatial->inputConfig);

    // qPointCloud = pipe.create<dai::node::XLinkOut>();
    // qPointCloud->setStreamName("pointCloud");
    // stereo->depth.link(pointcloud->inputDepth);
    // pointcloud->outputPointCloud.link(qPointCloud->input);
    // pointcloud->initialConfig.setSparse(true);

    device = std::make_unique<dai::Device>(pipe);
    // Output queues for RGB and depth frames
    // qRgbOutput = device->getOutputQueue("video", 8, false);
    qDepthOutput = device->getOutputQueue("depth", 8, false);
    qDebugMono = device->getOutputQueue("debug", 8, false);
    // qPointCloudOut = device->getOutputQueue("pointCloud", 8, false);
    spatialData = device->getOutputQueue("spatial", 8, false);
    spatialConfigQ = device->getInputQueue("spatialConfig");
}

cv::Mat OAK::getColorFrame()
{
    // Frame acquisition from output queue

    // std::shared_ptr<dai::ImgFrame> frame = qRgbOutput->get<dai::ImgFrame>();
    std::shared_ptr<dai::ImgFrame> frame = qDebugMono->get<dai::ImgFrame>();
    cv::Mat rgbMat = frame->getCvFrame();
    // cv::cvtColor(rgbMat, rgbMat, cv::COLOR_GRAY2BGR);
    return rgbMat;
}

cv::Mat OAK::getDepthFrame()
{
    depthFrame = qDepthOutput->get<dai::ImgFrame>();
    cv::Mat depthMat = depthFrame->getCvFrame();
    // lastDepthFrame = depthMat;
    // pointcloudData = qPointCloudOut->get<dai::PointCloudData>();

    return depthMat;
}

float OAK::getDistance(cv::Point target)
{
    // DepthAI supports point cloud - dai::PointCloud, dai::PointCloudData
    // here we try with depth aquired from the depth frame
    cv::Mat depthDisplay;
    // Normalize the depth values for visualization
    // lastDepthFrame.convertTo(depthDisplay, CV_8UC1, 255.0 / stereo->initialConfig.getMaxDisparity());
    // float distance = depthDisplay.at<uint8_t>(target.x, target.y) / 1000.0f; // scale to mm

    return 0;
}

cv::Point3f OAK::getCartesianPoint(cv::Point target)
{
    if (target != lastTarget)
    {
        dai::SpatialLocationCalculatorConfigData config;
        config.roi = dai::Rect(dai::Point2f(target.x - 5, target.y - 5),
                               dai::Point2f(target.x + 5, target.y + 5));

        dai::SpatialLocationCalculatorConfig cfg;
        cfg.addROI(config);
        spatialConfigQ->send(cfg);
        lastTarget = target;
    }

    auto data = spatialData->get<dai::SpatialLocationCalculatorData>()->getSpatialLocations();

    float x = data[0].spatialCoordinates.x;
    float y = data[0].spatialCoordinates.y;
    float z = data[0].spatialCoordinates.z;

    return cv::Point3f(x, y, z - 0.005);
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
#ifdef gemini

Gemini::Gemini()
{
    try 
    {
        // filters.push_back(ob::SpatialAdvancedFilter{});
        // filters.push_back(ob::TemporalFilter{});
        // filters.push_back(ob::HoleFillingFilter{});

        auto sensor = pipeline.getDevice()->getSensor(OB_SENSOR_DEPTH);
        auto filters = sensor->createRecommendedFilters();
        std::cout << "filters amount: " << filters.size() << std::endl;
        for(auto& filter : filters)
        {
            std::cout << filter->getName() << filter->isEnabled() << std::endl;
            if(filter->getName() == "SpatialAdvancedFilter" || 
               filter->getName() == "TemporalFilter" || 
               filter->getName() == "HoleFillingFilter") 
               {
                filter->enable(true);
                std::cout << filter->getName() << " enabled\n";
               } else
               {
                filter->enable(false);
               }
        }

        std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();
        // Create pipeline and config
        config->enableVideoStream(OB_STREAM_COLOR, 1280, 720, 30, OB_FORMAT_BGR);
        config->enableVideoStream(OB_STREAM_DEPTH, 1280, 720, 30, OB_FORMAT_Y16);
        // config->enableStream(OB_STREAM_COLOR);
        // config->enableStream(OB_STREAM_DEPTH);
        config->setAlignMode(ALIGN_D2C_HW_MODE);
        config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);

        // to check the list of supported properties

        // auto am = pipeline.getDevice()->getSupportedPropertyCount();
        // for(int i = 0; i < am; i++)
        // {
        //     auto prop = pipeline.getDevice()->getSupportedProperty(i);
        //     std::cout << prop.name << std::endl;
        // }

        pipeline.getDevice()->setIntProperty(OB_PROP_LASER_CONTROL_INT, 0);
        pipeline.start(config);
        auto frameset = pipeline.waitForFrameset();
        depthFrame = frameset->getFrame(OB_FRAME_DEPTH);

        // Store intrinsics for later use
        auto profile = std::dynamic_pointer_cast<ob::VideoStreamProfile>(depthFrame->getStreamProfile());
        if (profile)
        {
            intrinsic = profile->getIntrinsic();
            extrinsic = profile->getExtrinsicTo(profile);
        }

        frameSize = cv::Size(1280, 720);
    } 
    catch (const std::exception &e) 
    {
        std::cerr << "Failed to initialize Gemini camera: " << e.what() << std::endl;
    }
}

void Gemini::processFrame()
{
        auto frameset = pipeline.waitForFrameset();
        if (!frameset) 
        {
            std::cout << "empty frameset!\n";
            return;
        }
        colorFrame = frameset->getFrame(OB_FRAME_COLOR);
        if(!colorFrame) std::cout << "color nullptr\n";
        depthFrame = frameset->getFrame(OB_FRAME_DEPTH);
        if(!depthFrame) std::cout << "depth nullptr\n";  
        
        for(auto& filter : filters)
        {
            if(filter->isEnabled()) filter->process(depthFrame);
        }
}

cv::Mat Gemini::getColorFrame()
{
    if (!colorFrame) 
    {
        std::cout << "color frame nullptr\n";
        std::cout << "desired frame size: " << frameSize << std::endl;
        return {};
    }
    // std::cout << "converting rgb frame to opencv mat\n";
    cv::Mat bgr(frameSize, CV_8UC3, (void *)colorFrame->data());
    // cv::Mat bgr;
    // cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

cv::Mat Gemini::getDepthFrame()
{
    if (!depthFrame) return {};
    return cv::Mat(frameSize, CV_16UC1, (void *)depthFrame->data()).clone();
}

float Gemini::getDistance(cv::Point pt)
{
    if (!depthFrame) return 0.0f;
    const uint16_t *depthData = reinterpret_cast<const uint16_t *>(depthFrame->data());
    return static_cast<float>(depthData[pt.y * frameSize.width + pt.x]) / 1000.0f; // mm to meters
}

cv::Point3f Gemini::getCartesianPoint(cv::Point pt)
{
    if (!depthFrame)
    {
        std::cout << "null depth frame\n";
        return {0,0,0};
    } 
        
    float scale  = depthFrame->as<ob::DepthFrame>()->getValueScale();
    float depth = reinterpret_cast<const uint16_t *>(depthFrame->data())[pt.y * frameSize.width + pt.x]*scale;
    // std::cout << "Depth at point: " << depth << std::endl;
    OBPoint3f point;
    point.z = depth;
    transformHelper.transformation2dto3d(OBPoint2f{pt.x, pt.y}, depth, intrinsic,
    extrinsic, &point);

    return cv::Point3f(point.x, point.y, depth); // in mm
}

pcl::PointCloud<pcl::PointXYZ> Gemini::getPointCloud()
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    
    return cloud;
}

void Gemini::close()
{
    pipeline.stop();
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