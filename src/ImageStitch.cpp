
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
using namespace cv::detail;





namespace self
{




	bool preview = false;
	bool try_cuda = false;
	double work_megapix = 1;
	double seam_megapix = 1;
	double compose_megapix = -1;
	float conf_thresh = 1.f;
	string matcher_type = "homography";
	string estimator_type = "homography";
	string ba_cost_func = "ray";
	string ba_refine_mask = "xxxxx";
	bool do_wave_correct = true;
	WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
	bool save_graph = false;
	std::string save_graph_to;
	string warp_type = "spherical";
	int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
	int expos_comp_nr_feeds = 1;
	int expos_comp_nr_filtering = 2;
	int expos_comp_block_size = 32;
	string seam_find_type = "gc_color";
	int blend_type = Blender::MULTI_BAND;
	float blend_strength = 5;
	string result_name = "result.jpg";
	bool timelapse = false;
	int range_width = -1;





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
	bool ImageStitch::stitch(std::vector<cv::Mat> &imgs,cv::Mat & out)
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
		int num_images = imgs.size();
		vector<Size> full_img_sizes(num_images);
		double seam_work_aspect = 1;


		cv::Ptr<SIFT> detector = cv::SIFT::create();
		vector<ImageFeatures> features(img_num);
		int img_w;
		int img_h;
		for (int i = 0; i < img_num; i++)
		{
			cv::Mat  input_img = imgs[i].clone();
			full_img_sizes[i] = imgs[i].size();
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
		
		//cv::BFMatcher BRUTEFORCE
		Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
		
		//std::vector<std::vector<DMatch> > matches;
		cv::detail::MatchesInfo matches_info;
		vector<MatchesInfo> pairwise_matches;
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
				
			}
		}
		cv::Mat draw_img__match;
		cv::drawMatches(imgs[0], features[0].keypoints, imgs[1], features[1].keypoints, matches_info.matches, draw_img__match,
			-1.0f, -1.0f,std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		cv::imwrite("./data/image/draw_match_key_points_img_.jpg", draw_img__match);
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
			auto & features0 = features[0];
			auto & features1 = features[1];
			const DMatch& m = matches_info.matches[i];

			Point2f p = features0.keypoints[m.queryIdx].pt;
			p.x -= features0.img_size.width * 0.5f;
			p.y -= features0.img_size.height * 0.5f;
			src_points.at<Point2f>(0, static_cast<int>(i)) = p;

			p = features1.keypoints[m.trainIdx].pt;
			p.x -= features1.img_size.width * 0.5f;
			p.y -= features1.img_size.height * 0.5f;
			dst_points.at<Point2f>(0, static_cast<int>(i)) = p;


		}



		matches_info.H = findHomography(src_points, dst_points, cv::RANSAC);
		cv::Mat tmp;
		cv::warpPerspective(imgs[0], tmp, matches_info.H, cv::Size(img_w*2, img_h));
		cv::imwrite("./data/image/tmp.jpg", tmp);
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

		pairwise_matches.push_back(matches_info);

		Ptr<Estimator> estimator;
		if (estimator_type == "affine")
			estimator = makePtr<AffineBasedEstimator>();
		else
			estimator = makePtr<HomographyBasedEstimator>();

		vector<CameraParams> cameras;
		if (!(*estimator)(features, pairwise_matches, cameras))
		{
			cout << "Homography estimation failed.\n";
			return -1;
		}


		for (size_t i = 0; i < cameras.size(); ++i)
		{
			Mat R;
			cameras[i].R.convertTo(R, CV_32F);
			cameras[i].R = R;
			//LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		}

		Ptr<detail::BundleAdjusterBase> adjuster;
		if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
		else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
		else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
		else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
		else
		{
			cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
			return -1;
		}



		adjuster->setConfThresh(conf_thresh);
		Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
		if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
		if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
		if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
		if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
		if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
		adjuster->setRefinementMask(refine_mask);
		if (!(*adjuster)(features, pairwise_matches, cameras))
		{
			cout << "Camera parameters adjusting failed.\n";
			return -1;
		}

		// Find median focal length

		vector<double> focals;
		for (size_t i = 0; i < cameras.size(); ++i)
		{
			//LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
			focals.push_back(cameras[i].focal);
		}

		sort(focals.begin(), focals.end());
		float warped_image_scale;
		if (focals.size() % 2 == 1)
			warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
		else
			warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

		if (do_wave_correct)
		{
			vector<Mat> rmats;
			for (size_t i = 0; i < cameras.size(); ++i)
				rmats.push_back(cameras[i].R.clone());
			waveCorrect(rmats, wave_correct);
			for (size_t i = 0; i < cameras.size(); ++i)
				cameras[i].R = rmats[i];
		}


		vector<Point> corners(num_images);
		vector<UMat> masks_warped(num_images);
		vector<UMat> images_warped(num_images);
		vector<Size> sizes(num_images);
		vector<UMat> masks(num_images);

		// Prepare images masks
		for (int i = 0; i < num_images; ++i)
		{
			masks[i].create(imgs[i].size(), CV_8U);
			masks[i].setTo(Scalar::all(255));
		}

		// Warp images and their masks

		Ptr<WarperCreator> warper_creator;
		{
			if (warp_type == "plane")
				warper_creator = makePtr<cv::PlaneWarper>();
			else if (warp_type == "affine")
				warper_creator = makePtr<cv::AffineWarper>();
			else if (warp_type == "cylindrical")
				warper_creator = makePtr<cv::CylindricalWarper>();
			else if (warp_type == "spherical")
				warper_creator = makePtr<cv::SphericalWarper>();
			else if (warp_type == "fisheye")
				warper_creator = makePtr<cv::FisheyeWarper>();
			else if (warp_type == "stereographic")
				warper_creator = makePtr<cv::StereographicWarper>();
			else if (warp_type == "compressedPlaneA2B1")
				warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
			else if (warp_type == "compressedPlaneA1.5B1")
				warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
			else if (warp_type == "compressedPlanePortraitA2B1")
				warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
			else if (warp_type == "compressedPlanePortraitA1.5B1")
				warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
			else if (warp_type == "paniniA2B1")
				warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
			else if (warp_type == "paniniA1.5B1")
				warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
			else if (warp_type == "paniniPortraitA2B1")
				warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
			else if (warp_type == "paniniPortraitA1.5B1")
				warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
			else if (warp_type == "mercator")
				warper_creator = makePtr<cv::MercatorWarper>();
			else if (warp_type == "transverseMercator")
				warper_creator = makePtr<cv::TransverseMercatorWarper>();
		}

		if (!warper_creator)
		{
			cout << "Can't create the following warper '" << warp_type << "'\n";
			return 1;
		}

		Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

		for (int i = 0; i < num_images; ++i)
		{
			Mat_<float> K;
			cameras[i].K().convertTo(K, CV_32F);
			float swa = (float)seam_work_aspect;
			K(0, 0) *= swa; K(0, 2) *= swa;
			K(1, 1) *= swa; K(1, 2) *= swa;

			corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
			sizes[i] = images_warped[i].size();

			warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
		}

		vector<UMat> images_warped_f(num_images);
		for (int i = 0; i < num_images; ++i)
			images_warped[i].convertTo(images_warped_f[i], CV_32F);



		Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
		if (dynamic_cast<GainCompensator*>(compensator.get()))
		{
			GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
			gcompensator->setNrFeeds(expos_comp_nr_feeds);
		}

		if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
		{
			ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
			ccompensator->setNrFeeds(expos_comp_nr_feeds);
		}

		if (dynamic_cast<BlocksCompensator*>(compensator.get()))
		{
			BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
			bcompensator->setNrFeeds(expos_comp_nr_feeds);
			bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
			bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
		}

		compensator->feed(corners, images_warped, masks_warped);



		Ptr<SeamFinder> seam_finder;
		if (seam_find_type == "no")
			seam_finder = makePtr<detail::NoSeamFinder>();
		else if (seam_find_type == "voronoi")
			seam_finder = makePtr<detail::VoronoiSeamFinder>();
		else if (seam_find_type == "gc_color")
		{

				seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
		}
		else if (seam_find_type == "gc_colorgrad")
		{
				seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		}
		else if (seam_find_type == "dp_color")
			seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
		else if (seam_find_type == "dp_colorgrad")
			seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
		if (!seam_finder)
		{
			cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
			return 1;
		}

		seam_finder->find(images_warped_f, corners, masks_warped);


		// Release unused memory
		//images.clear();
		images_warped.clear();
		images_warped_f.clear();
		masks.clear();



		Mat img_warped, img_warped_s;
		Mat dilated_mask, seam_mask, mask, mask_warped;
		Ptr<Blender> blender;
		//Ptr<Timelapser> timelapser;
		//double compose_seam_aspect = 1;
		double compose_work_aspect = 1;
		double work_scale = 1, seam_scale = 1, compose_scale = 1;
		bool is_work_scale_set = true, is_seam_scale_set = true, is_compose_scale_set = true;
		cv::Mat img;
		for (int img_idx = 0; img_idx < num_images; ++img_idx)
		{

			// Read image and resize it if necessary
			cv::Mat full_img = imgs[img_idx];
			if (!is_compose_scale_set)
			{
				if (compose_megapix > 0)
					compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
				is_compose_scale_set = true;

				// Compute relative scales
				//compose_seam_aspect = compose_scale / seam_scale;
				compose_work_aspect = compose_scale / work_scale;

				// Update warped image scale
				warped_image_scale *= static_cast<float>(compose_work_aspect);
				warper = warper_creator->create(warped_image_scale);

				// Update corners and sizes
				for (int i = 0; i < num_images; ++i)
				{
					// Update intrinsics
					cameras[i].focal *= compose_work_aspect;
					cameras[i].ppx *= compose_work_aspect;
					cameras[i].ppy *= compose_work_aspect;

					// Update corner and size
					Size sz = full_img_sizes[i];
					if (std::abs(compose_scale - 1) > 1e-1)
					{
						sz.width = cvRound(full_img_sizes[i].width * compose_scale);
						sz.height = cvRound(full_img_sizes[i].height * compose_scale);
					}

					Mat K;
					cameras[i].K().convertTo(K, CV_32F);
					Rect roi = warper->warpRoi(sz, K, cameras[i].R);
					corners[i] = roi.tl();
					sizes[i] = roi.size();
				}
			}
			if (abs(compose_scale - 1) > 1e-1)
				cv::resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
			else
				img = full_img;
			full_img.release();
			Size img_size = img.size();

			Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);

			// Warp the current image
			warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

			// Warp the current image mask
			mask.create(img_size, CV_8U);
			mask.setTo(Scalar::all(255));
			warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

			// Compensate exposure
			compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

			img_warped.convertTo(img_warped_s, CV_16S);
			img_warped.release();
			img.release();
			mask.release();

			cv::dilate(masks_warped[img_idx], dilated_mask, Mat());
			cv::resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
			mask_warped = seam_mask & mask_warped;

			if (!blender && !timelapse)
			{
				blender = Blender::createDefault(blend_type, try_cuda);
				Size dst_sz = resultRoi(corners, sizes).size();
				float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
				if (blend_width < 1.f)
					blender = Blender::createDefault(Blender::NO, try_cuda);
				else if (blend_type == Blender::MULTI_BAND)
				{
					MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
					mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
					//LOGLN("Multi-band blender, number of bands: " << mb->numBands());
				}
				else if (blend_type == Blender::FEATHER)
				{
					FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
					fb->setSharpness(1.f / blend_width);
					//LOGLN("Feather blender, sharpness: " << fb->sharpness());
				}
				blender->prepare(corners, sizes);
			}

			blender->feed(img_warped_s, mask_warped, corners[img_idx]);

		}

		if (!timelapse)
		{
			Mat result, result_mask;
			blender->blend(result, result_mask);

		}

	}
}