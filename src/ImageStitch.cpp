
#include <cstdint>
#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip>
#include <vector>

#include "opencv2/opencv.hpp"

#include "ImageStitch.h"
using namespace cv;
using namespace std;
using namespace detail;

namespace self
{
	ImageStitch::ImageStitch()
	{

	}

	ImageStitch::~ImageStitch()
	{
		if (true == is_inited)
		{
			destory();
		}
	}


	bool ImageStitch::init()
	{
		if (true == is_inited)
		{
			destory();
		}

		return true;
	}
	bool ImageStitch::destory()
	{


		return true;
	}
	bool ImageStitch::stitch(std::vector<cv::Mat> &imgs)
	{	
		int32_t img_num = imgs.size();

		if (img_num < 0)
		{
			return false;
		}

		for (int i = 0 ; i < img_num; i++)
		{
			if (imgs[i].empty())
			{
				return false;
			}
		}

		cv::Ptr<ORB> detector = cv::ORB::create();
		vector<ImageFeatures> features(img_num);
		int img_w;
		int img_h;
		for (int i = 0; i < img_num; i++)
		{
			cv::Mat  input_img = imgs[i].clone();
#if 0
			
			std::vector<cv::KeyPoint> keyPoints;
			cv::Mat detect_key_points_mask = cv::Mat::ones(imgs[i].size(), CV_8UC1);
			
			cv::Mat descriptors;
			detector->detectAndCompute(input_img, detect_key_points_mask, keyPoints, descriptors);
			
#endif
			img_w = input_img.cols;
			img_h = input_img.rows;
			float scale_w = 2.0f / 3;
			float scale_h = 1.0f / 4;
			
			cv::Mat draw_key_points_img;
			cv::Mat detect_key_points_mask = cv::Mat::zeros(imgs[i].size(), CV_8UC1);
			
			cv::Rect roi;
			if (i == 0)
			{
				 roi = cv::Rect(img_w * 0.7, img_h *0.2, img_w - img_w * 0.7, img_h - img_h * 0.2);
			}
			if (i == 1)
			{
				roi = cv::Rect(0, img_h *0.2, img_w - img_w * 0.7, img_h - img_h * 0.2);
			}

			detect_key_points_mask(roi) = 255;
			cv::Scalar scalar(255,255,255);
			cv::rectangle(imgs[i], roi, scalar);
			computeImageFeatures(detector, input_img, features[i], detect_key_points_mask);
			//cv::drawKeypoints(input_img, features[i].keypoints, draw_key_points_img);

			//cv::imwrite("./data/image/draw_key_points_img_" + std::to_string(i) + "_.jpg", draw_key_points_img);

		}
		cv::detail::MatchesInfo matches_info;
		//cv::BFMatcher BRUTEFORCE
		Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
		
		//std::vector<std::vector<DMatch> > matches;
		std::vector <std::vector<DMatch>> img1_to_img2_matches;
		std::vector <std::vector<DMatch>> img2_to_img1_matches;
		std::set<std::pair<int,int>> matches;
		float ratio = 0.75;
		matcher->knnMatch(features[0].descriptors,features[1].descriptors, img1_to_img2_matches,2);
		for (int i = 0; i < img1_to_img2_matches.size(); i++)
		{
			std::vector<DMatch> & match = img1_to_img2_matches[i];
			if (match.size() != 2)
			{
				continue;
			}
			if (match[0].distance < match[1].distance * ratio)
			{
				matches_info.matches.push_back(match[0]);
				matches.insert(std::make_pair(match[0].queryIdx, match[0].trainIdx));
			}
		}
#if 0
		matcher->knnMatch(features[1].descriptors, features[0].descriptors, img2_to_img1_matches, 2);
		for (int i = 0; i < img2_to_img1_matches.size(); i++)
		{
			std::vector<DMatch> & match = img2_to_img1_matches[i];
			if (match.size() != 2)
			{
				continue;
			}

			if (match[0].distance <  match[1].distance *ratio )
			{
				if (matches.find(std::make_pair(match[0].trainIdx, match[0].queryIdx)) == matches.end())
				{
					matches_info.matches.push_back(match[0]);
				}
			}
		}
#endif
		if (matches_info.matches.size() < 4)
		{
			return false;
		}
		Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		for (int i = 0; i < matches_info.matches.size(); i++)
		{
			auto & features1 = features[0];
			auto & features2 = features[1];
			const DMatch& m = matches_info.matches[i];

			Point2f p = features1.keypoints[m.queryIdx].pt;
			p.x -= features1.img_size.width * 0.5f;
			p.y -= features1.img_size.height * 0.5f;
			src_points.at<Point2f>(0, static_cast<int>(i)) = p;

			p = features2.keypoints[m.trainIdx].pt;
			p.x -= features2.img_size.width * 0.5f;
			p.y -= features2.img_size.height * 0.5f;
			dst_points.at<Point2f>(0, static_cast<int>(i)) = p;


		}



		matches_info.H = findHomography(src_points, dst_points, cv::RANSAC);
		cv::Mat tmp;
		cv::warpAffine(imgs[1], tmp, matches_info.H, cv::Size(img_w, img_h));
#if 0
		// Construct point-point correspondences for homography estimation
		Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		for (size_t i = 0; i < matches_info.matches.size(); ++i)
		{
			auto & features1 = features[0];
			auto & features2 = features[1];
			const DMatch& m = matches_info.matches[i];

			Point2f p = features1.keypoints[m.queryIdx].pt;
			p.x -= features1.img_size.width * 0.5f;
			p.y -= features1.img_size.height * 0.5f;
			src_points.at<Point2f>(0, static_cast<int>(i)) = p;

			p = features2.keypoints[m.trainIdx].pt;
			p.x -= features2.img_size.width * 0.5f;
			p.y -= features2.img_size.height * 0.5f;
			dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
		}

		// Find pair-wise motion
		matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, RANSAC);
		if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
		{
			return false;
		}
		// Find number of inliers
		matches_info.num_inliers = 0;
		for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
			if (matches_info.inliers_mask[i])
				matches_info.num_inliers++;

		// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
		// using Invariant Features"
		matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

		// Set zero confidence to remove matches between too close images, as they don't provide
		// additional information anyway. The threshold was set experimentally.
		matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

		// Check if we should try to refine motion
		//if (matches_info.num_inliers < num_matches_thresh2_)
		//	return;

		// Construct point-point correspondences for inliers only
		src_points.create(1, matches_info.num_inliers, CV_32FC2);
		dst_points.create(1, matches_info.num_inliers, CV_32FC2);
		int inlier_idx = 0;
		for (size_t i = 0; i < matches_info.matches.size(); ++i)
		{
			if (!matches_info.inliers_mask[i])
				continue;
			auto & features1 = features[0];
			auto & features2 = features[1];
			const DMatch& m = matches_info.matches[i];

			Point2f p = features1.keypoints[m.queryIdx].pt;
			p.x -= features1.img_size.width * 0.5f;
			p.y -= features1.img_size.height * 0.5f;
			src_points.at<Point2f>(0, inlier_idx) = p;

			p = features2.keypoints[m.trainIdx].pt;
			p.x -= features2.img_size.width * 0.5f;
			p.y -= features2.img_size.height * 0.5f;
			dst_points.at<Point2f>(0, inlier_idx) = p;

			inlier_idx++;
		}

		// Rerun motion estimation on inliers only
		matches_info.H = findHomography(src_points, dst_points, RANSAC);
		cv::Mat tmp;
		cv::warpAffine(imgs[1], tmp, matches_info.H, imgs[1].size());
#endif

		int  panorama_w = img_w + img_w;
		int  panorama_h = std::max(img_h, img_h);
		Mat  resultImg = Mat(panorama_w, panorama_h, CV_8UC3, Scalar::all(0));
		Mat ROI_1 = resultImg(Rect(0, 0, img_w, img_h));
		Mat ROI_2 = resultImg(Rect(img_w, 0, img_w, img_h));
		imgs[0].copyTo(ROI_1);
		tmp.copyTo(ROI_2);
		cv::imwrite("./data/image/panorama__.jpg", resultImg);
	}
}