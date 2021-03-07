#ifndef IMAGE_STITCH_H
#define IMAGE_STITCH_H
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

namespace self 
{
	class ImageStitch
	{
	public:
		ImageStitch();
		~ImageStitch();
		bool init();
		bool stitch(std::vector<cv::Mat> &imgs);
		bool destory();
	private:
		bool is_inited = false;


	};
}
#endif