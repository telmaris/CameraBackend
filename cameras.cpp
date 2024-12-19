#include "cameras.hpp"

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
    model->setInputParams(1. / 255, cv::Size(416, 416), cv::Scalar(), true);

    return 0;
}

std::vector<cv::Rect> Network::detect(cv::Mat frame)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> results;

    model->detect(frame, classIds, confidences, boxes, 0.2, 0.4);

    for (int i = 0; i < classIds.size(); ++i)
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
    net = std::make_unique<Network>();
    net->init();
}

void Backend::parseArgs(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "zed")
        {
            // camera = std::make_unique<ZED>();
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
            // camera = std::make_unique<OAK>();
            windowName = "OAK object detection and distance measurement";
            return;
        }
    }
}

void Backend::loop()
{
    if (camera == nullptr)
        run = false;
    while (run)
    {
        camera->processFrame();
        auto frame = camera->getColorFrame();
        auto depth = camera->getDepthFrame();

        auto boxes = net->detect(frame);

        // draw bounding boxes with distance measurement
        for (int i = 0; i < boxes.size(); ++i)
        {
            auto box = boxes[i];
            int centerX = box.x + box.width / 2;
            int centerY = box.y + box.height / 2;

            cv::rectangle(frame, box, colors[1], 3);
            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), colors[1], cv::FILLED);
            cv::putText(frame, "Distance: " + std::to_string(camera->getDistance(cv::Point(centerX, centerY))), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

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

void Realsense::close()
{
    pipe.stop();
}

/* ======================== ZED =============================*/

ZED::ZED()
{
    sl::InitParameters initParameters;
    initParameters.camera_resolution = sl::RESOLUTION::HD720;
    initParameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    zed.open(initParameters);
}

void ZED::processFrame()
{
    if (zed.grab() == sl::ERROR_CODE::SUCCESS)
    {
        zed.retrieveImage(lastZedFrame, sl::VIEW::LEFT);
        zed.retrieveMeasure(lastZedDepth, sl::MEASURE::DEPTH);
        zed.retrieveMeasure(pointCloud, sl::MEASURE::XYZRGBA);
    }
}

cv::Mat ZED::getColorFrame()
{
    cv::Mat frame(frameSize, CV_8UC3, (void *)lastZedFrame.getPtr<sl::uchar4>(), cv::Mat::AUTO_STEP);
    return frame;
}

cv::Mat ZED::getDepthFrame()
{
    cv::Mat frame(frameSize, CV_8UC3, (void *)lastZedDepth.getPtr<sl::uchar4>(), cv::Mat::AUTO_STEP);
    return frame;
}

float ZED::getDistance(cv::Point pt)
{
    sl::float4 pointCloudValue;
    pointCloud.getValue(pt.x, pt.y, &pointCloudValue);
    float distance = sqrt(pointCloudValue.x * pointCloudValue.x + 
                            pointCloudValue.y * pointCloudValue.y + 
                            pointCloudValue.z * pointCloudValue.z);
    return distance;
}

void ZED::close()
{
    zed.close();
}