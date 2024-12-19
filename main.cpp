#include "cameras.hpp"

int main(int argc, char** argv)
{
    Backend base;

    base.parseArgs(argc, argv);
    base.init();
    base.loop();

    return 0;
}