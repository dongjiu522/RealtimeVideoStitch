#ifndef IMAGE_STITCH_DETAILED_H
#define IMAGE_STITCH_DETAILED_H
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

int  ImageStitch(std::vector<std::string> img_names, cv::Mat & result);


#endif