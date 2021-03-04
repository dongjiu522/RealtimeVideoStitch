#ifndef IMAGE_STITCH_H
#define IMAGE_STITCH_H
#include <vector>
#include "opencv2/opencv.hpp"

cv::Mat ImageStitch(std::vector<cv::Mat> imgs);


#endif