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
        std::cout << "Changing the measurement location to [" << x << ", " << y << "]\n"; 
        auto backend = static_cast<Backend*>(userdata);
        backend->measurementLocation = cv::Point{x,y};
        backend->updateRoiPosition(x, y);
    }
}

int main(int argc, char** argv)
{
    Backend base;

    if(argc == 2)
    {
        std::cout << "Detected an arg..\n";
        if(std::string{argv[1]} == "f")
        {
            base.enableFilters = true;
            std::cout << "Filters enabled\n";
        }
    }
    base.init();
    cv::setMouseCallback(base.windowName, setMeasurementPoint, &base);
    base.loop();

    return 0;
}