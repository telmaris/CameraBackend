#include "cameras.hpp"

#ifdef zed
const std::string camera = "zed";
#endif
#ifdef intel
const std::string camera = "intel";
#endif
#ifdef oak
const std::string camera = "oak";
#endif

void setMeasurementPoint(int event, int x, int y, int flags, void* userdata)
{
    if(event == cv::MouseEventTypes::EVENT_LBUTTONDBLCLK)
    {
        std::cout << "Changing the measurement location\n";
        static_cast<Backend*>(userdata)->measurementLocation = cv::Point{x,y};
    }
}

int main(int argc, char** argv)
{
    Backend base;

    // base.parseArgs(camera);
    base.init();
    cv::setMouseCallback(base.windowName, setMeasurementPoint, &base);
    base.loop();

    return 0;
}