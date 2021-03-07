#include<cstdint>
#include<string>
#include<vector>

#ifdef _MSC_VER
#include<Windows.h>
#endif

#include"opencv2/opencv.hpp"
#include "ImageStitch.h"
#include "stitching_detailed.h"

int main()
{
	std::string image1_path = "./data/image/Left_0001.jpg";
	std::string image2_path = "./data/image/Right_0001.jpg";
	//std::string image1_path = "./data/image/test1.jpg";
	//std::string image2_path = "./data/image/test2.jpg";
	cv::Mat image1 = cv::imread(image1_path);
	cv::Mat image2 = cv::imread(image2_path);
	//cv::resize(image1, image1, cv::Size(480,270));
	//cv::resize(image2, image2, cv::Size(480,270));
	std::vector<cv::Mat> inputs;
	inputs.push_back(image1);
	inputs.push_back(image2);

	self::ImageStitch stitch;
	stitch.init();
	bool ret = stitch.stitch(inputs);
#if 0
	std::vector<std::string> imgs;
	imgs.push_back(image1_path);
	imgs.push_back(image2_path);

	cv::Mat result;
	int ret = ImageStitch(imgs, result);
	if (ret == 0)
	{
		cv::imwrite("./result.jpg", result);
	}
#endif
#ifdef _MSC_VER
	//system("pause");
#endif
	return 0;
}
