

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

cv::Mat ImageStitch(vector<Mat> imgs)
{
	int num_images = imgs.size();

	cv::Ptr<SIFT> detector = cv::SIFT::create();

	vector<ImageFeatures> features(num_images);    //表示图像特征


	for (int i = 0; i < num_images; i++)
	{
		features[i].img_idx = i;
		features[i].img_size = imgs[i].size();
		detector->detect(imgs[i], features[i].keypoints);    //特征检测
		detector->compute(imgs[i], features[i].keypoints, features[i].descriptors);
	}
	vector<MatchesInfo> pairwise_matches;    //表示特征匹配信息变量
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //定义特征匹配器，2NN方法
	matcher(features, pairwise_matches);    //进行特征匹配

	HomographyBasedEstimator estimator;    //定义参数评估器
	vector<CameraParams> cameras;    //表示相机参数
	estimator(features, pairwise_matches, cameras);    //进行相机参数评估

	for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

	Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
	//adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
	adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

	adjuster->setConfThresh(1);    //设置匹配置信度，该值设为1
	(*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数

	vector<Mat> rmats;

	//复制相机的旋转参数
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		rmats.push_back(cameras[i].R.clone());
	}

	//进行波形校正
	waveCorrect(rmats, WAVE_CORRECT_HORIZ); 

	//相机参数赋值
	for (size_t i = 0; i < cameras.size(); ++i)    
	{
		cameras[i].R = rmats[i];
	}

	rmats.clear();    //清变量

	//表示映射变换后图像的左上角坐标
	vector<Point> corners(num_images);    
	
	//表示映射变换后的图像掩码
	vector<Mat> masks_warped(num_images);    
	
	//表示映射变换后的图像
	vector<Mat> images_warped(num_images);    
	
	//表示映射变换后的图像尺寸
	vector<Size> sizes(num_images);    
	
	//表示源图的掩码
	vector<Mat> masks(num_images);    

	for (int i = 0; i < num_images; ++i)    //初始化源图的掩码
	{
		masks[i].create(imgs[i].size(), CV_8U);    //定义尺寸大小
		masks[i].setTo(Scalar::all(255));    //全部赋值为255，表示源图的所有区域都使用
	}

	Ptr<WarperCreator> warper_creator;    //定义图像映射变换创造器
	warper_creator = new cv::PlaneWarper();    //平面投影
	//warper_creator = new cv::CylindricalWarper();    //柱面投影
	//warper_creator = new cv::SphericalWarper();    //球面投影
	//warper_creator = new cv::FisheyeWarper();    //鱼眼投影
	//warper_creator = new cv::StereographicWarper();    //立方体投影

	//定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);    //转换相机内参数的数据类型
		//对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
		corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();    //得到尺寸
		//得到变换后的图像掩码
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	imgs.clear();    //清变量
	masks.clear();

	//创建曝光补偿器，应用增益补偿方法
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	{
		std::vector<cv::UMat> tmp_images_warped(images_warped.size());
		std::vector<cv::UMat> tmp_masks_warped(masks_warped.size());
		for (int i = 0; i < images_warped.size(); i++)
		{
			tmp_images_warped[i] = images_warped[i].getUMat(ACCESS_READ|ACCESS_WRITE);
		}
		for (int i = 0; i < masks_warped.size(); i++)
		{
			tmp_masks_warped[i] = masks_warped[i].getUMat(ACCESS_READ | ACCESS_WRITE);
		}
		compensator->feed(corners, tmp_images_warped, tmp_masks_warped);    //得到曝光补偿器
	}
	for (int i = 0; i < num_images; ++i)    //应用曝光补偿器，对图像进行曝光补偿
	{
		compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
	}

	//在后面，我们还需要用到映射变换图的掩码masks_warped，因此这里为该变量添加一个副本masks_seam
	vector<Mat> masks_seam(num_images);
	for (int i = 0; i < num_images; i++)
		masks_warped[i].copyTo(masks_seam[i]);

	Ptr<SeamFinder> seam_finder;    //定义接缝线寻找器
   //seam_finder = new NoSeamFinder();    //无需寻找接缝线
   //seam_finder = new VoronoiSeamFinder();    //逐点法
   //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //动态规范法
   //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
   //图割法
   //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
	seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);

	vector<Mat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)    //图像数据类型转换
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	images_warped.clear();    //清内存

	//得到接缝线的掩码图像masks_seam
	{
		std::vector<cv::UMat> tmp_images_warped_f(images_warped_f.size());
		std::vector<cv::UMat> tmp_masks_seam(masks_seam.size());
		for (int i = 0; i < images_warped_f.size(); i++)
		{
			tmp_images_warped_f[i] = images_warped_f[i].getUMat(ACCESS_READ | ACCESS_WRITE);
		}
		for (int i = 0; i < masks_seam.size(); i++)
		{
			tmp_masks_seam[i] = masks_seam[i].getUMat(ACCESS_READ | ACCESS_WRITE);
		}
		seam_finder->find(tmp_images_warped_f, corners, tmp_masks_seam);
	}
	vector<Mat> images_warped_s(num_images);
	Ptr<Blender> blender;    //定义图像融合器

	//blender = Blender::createDefault(Blender::NO, false);    //简单融合方法
	//羽化融合方法
	//blender = Blender::createDefault(Blender::FEATHER, false);
	//FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
	//fb->setSharpness(0.005);    //设置羽化锐度

	blender = Blender::createDefault(Blender::MULTI_BAND, false);    //多频段融合
	MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
	mb->setNumBands(8);   //设置频段数，即金字塔层数

	blender->prepare(corners, sizes);    //生成全景图像区域

	//在融合的时候，最重要的是在接缝线两侧进行处理，而上一步在寻找接缝线后得到的掩码的边界就是接缝线处，因此我们还需要在接缝线两侧开辟一块区域用于融合处理，这一处理过程对羽化方法尤为关键
	//应用膨胀算法缩小掩码面积
	vector<Mat> dilate_img(num_images);
	Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //定义结构元素
	for (int k = 0; k < num_images; k++)
	{
		images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //改变数据类型
		dilate(masks_seam[k], masks_seam[k], element);    //膨胀运算
		//映射变换图的掩码和膨胀后的掩码相“与”，从而使扩展的区域仅仅限于接缝线两侧，其他边界处不受影响
		masks_seam[k] = masks_seam[k] & masks_warped[k];
		blender->feed(images_warped_s[k], masks_seam[k], corners[k]);    //初始化数据
	}

	masks_seam.clear();    //清内存
	images_warped_s.clear();
	masks_warped.clear();
	images_warped_f.clear();

	Mat result, result_mask;
	//完成融合操作，得到全景图像result和它的掩码result_mask
	blender->blend(result, result_mask);

	return result;
}