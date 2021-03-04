#include<cstdint>
#include<string>
#include<vector>

#ifdef _MSC_VER
#include<Windows.h>
#endif

#include"opencv2/opencv.hpp"
#include "ImageStitch.h"

int main()
{
	std::string image1_path = "./data/image/Left_0001.jpg";
	std::string image2_path = "./data/image/Right_0001.jpg";

	cv::Mat image1 = cv::imread(image1_path);
	cv::Mat image2 = cv::imread(image2_path);
	cv::resize(image1, image1, cv::Size(480,270));
	cv::resize(image2, image2, cv::Size(480,270));
	std::vector<cv::Mat> imgs;
	imgs.push_back(image1);
	imgs.push_back(image2);

	cv::Mat result = ImageStitch(imgs);
	cv::imwrite("./result.jpg", result);
#ifdef _MSC_VER
	system("pause");
#endif
	return 0;
}
