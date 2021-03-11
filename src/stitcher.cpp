#include "stitcher.hpp"

namespace self {


 float Stitcher::ORIG_RESOL = -1.0;

Ptr<Stitcher> Stitcher::create()
{
    
	Ptr<Stitcher> stitcher = makePtr<Stitcher>();
    return stitcher;
}

bool Stitcher::init()
{
	is_inited = false;
	setRegistrationResol(0.6);
	setSeamEstimationResol(0.1);
	setCompositingResol(ORIG_RESOL);
	setPanoConfidenceThresh(1);
	setSeamFinder(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR));
	setBlender(makePtr<detail::MultiBandBlender>(false));
	setFeaturesFinder(ORB::create());
	setInterpolationFlags(INTER_LINEAR);
	work_scale_ = 1;
	seam_scale_ = 1;
	seam_work_aspect_ = 1;
	warped_image_scale_ = 1;


	setEstimator(makePtr<detail::HomographyBasedEstimator>());
	setWaveCorrection(true);
	setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
	setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false));
	setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());
	setWarper(makePtr<SphericalWarper>());
	setExposureCompensator(makePtr<detail::BlocksGainCompensator>());

	is_inited = true;
	return true;
}

bool Stitcher::destory()
{
	if (false == is_inited)
	{
		return true;
	}

}

Stitcher::Status Stitcher::composePanorama(OutputArray pano)
{

	return composePanorama(std::vector<UMat>(), pano);
}

Stitcher::Status Stitcher::composePanorama(InputArrayOfArrays images, OutputArray pano)
{



    std::vector<UMat> imgs;
    images.getUMatVector(imgs);
    if (!imgs.empty())
    {
        CV_Assert(imgs.size() == imgs_.size());

        UMat img;
        seam_est_imgs_.resize(imgs.size());

        for (size_t i = 0; i < imgs.size(); ++i)
        {
            imgs_[i] = imgs[i];
            resize(imgs[i], img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
            seam_est_imgs_[i] = img.clone();
        }

        std::vector<UMat> seam_est_imgs_subset;
        std::vector<UMat> imgs_subset;

        for (size_t i = 0; i < indices_.size(); ++i)
        {
            imgs_subset.push_back(imgs_[indices_[i]]);
            seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        }

        seam_est_imgs_ = seam_est_imgs_subset;
        imgs_ = imgs_subset;
    }

    UMat pano_;



    std::vector<Point> corners(imgs_.size());
    std::vector<UMat> masks_warped(imgs_.size());
    std::vector<UMat> images_warped(imgs_.size());
    std::vector<Size> sizes(imgs_.size());
    std::vector<UMat> masks(imgs_.size());

    // Prepare image masks
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        masks[i].create(seam_est_imgs_[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    Ptr<detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        Mat_<float> K;
        cameras_[i].K().convertTo(K, CV_32F);
        K(0,0) *= (float)seam_work_aspect_;
        K(0,2) *= (float)seam_work_aspect_;
        K(1,1) *= (float)seam_work_aspect_;
        K(1,2) *= (float)seam_work_aspect_;

        corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, interp_flags_, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        w->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }




    // Compensate exposure before finding seams
    exposure_comp_->feed(corners, images_warped, masks_warped);
    for (size_t i = 0; i < imgs_.size(); ++i)
        exposure_comp_->apply(int(i), corners[i], images_warped[i], masks_warped[i]);


	// Find seams
	std::vector<UMat> images_warped_f(imgs_.size());
	for (size_t i = 0; i < imgs_.size(); ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	seam_finder_->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	seam_est_imgs_.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();




	UMat img_warped, img_warped_s;
	UMat dilated_mask, seam_mask, mask, mask_warped;

	//double compose_seam_aspect = 1;
	double compose_work_aspect = 1;
	bool is_blender_prepared = false;

	double compose_scale = 1;
	bool is_compose_scale_set = false;

	std::vector<detail::CameraParams> cameras_scaled(cameras_);

	UMat full_img, img;
	for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
	{

		// Read image and resize it if necessary
		full_img = imgs_[img_idx];
		if (!is_compose_scale_set)
		{
			if (compose_resol_ > 0)
				compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
			is_compose_scale_set = true;

			// Compute relative scales
			//compose_seam_aspect = compose_scale / seam_scale_;
			compose_work_aspect = compose_scale / work_scale_;

			// Update warped image scale
			float warp_scale = static_cast<float>(warped_image_scale_ * compose_work_aspect);
			w = warper_->create(warp_scale);

			// Update corners and sizes
			for (size_t i = 0; i < imgs_.size(); ++i)
			{
				// Update intrinsics
				cameras_scaled[i].ppx *= compose_work_aspect;
				cameras_scaled[i].ppy *= compose_work_aspect;
				cameras_scaled[i].focal *= compose_work_aspect;

				// Update corner and size
				Size sz = full_img_sizes_[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
				}

				Mat K;
				cameras_scaled[i].K().convertTo(K, CV_32F);
				Rect roi = w->warpRoi(sz, K, cameras_scaled[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (std::abs(compose_scale - 1) > 1e-1)
		{

			resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);

		}
		else
			img = full_img;
		full_img.release();
		Size img_size = img.size();


		Mat K;
		cameras_scaled[img_idx].K().convertTo(K, CV_32F);


		// Warp the current image
		w->warp(img, K, cameras_[img_idx].R, interp_flags_, BORDER_REFLECT, img_warped);


		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		w->warp(mask, K, cameras_[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);


		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		// Make sure seam mask has proper size
		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);

		bitwise_and(seam_mask, mask_warped, mask_warped);



		if (!is_blender_prepared)
		{
			blender_->prepare(corners, sizes);
			is_blender_prepared = true;
		}


		// Blend the current image
		blender_->feed(img_warped_s, mask_warped, corners[img_idx]);

	}


	UMat result;
	blender_->blend(result, result_mask_);

	// Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
	// so convert it to avoid user confusing
	result.convertTo(pano, CV_8U);
    return OK;
}


Stitcher::Status Stitcher::stitch(InputArrayOfArrays images, OutputArray pano)
{
    return stitch(images, noArray(), pano);
}


Stitcher::Status Stitcher::stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano)
{

	Status status;
	//1.图像匹配
	images.getUMatVector(imgs_);
	masks.getUMatVector(masks_);
	if ((status = matchImages()) != OK)
	{
		return status;
	}

	//2.估计相机参数
	if ((status = estimateCameraParams()) != OK)
	{
		return status;
	}
	//##############################################


	//3.处理曝光参数



	//4.处理拼接缝

	status = composePanorama(pano);


    return status;
}


Stitcher::Status Stitcher::matchImages()
{
    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    work_scale_ = 1;
    seam_work_aspect_ = 1;
    seam_scale_ = 1;
    bool is_work_scale_set = false;
    bool is_seam_scale_set = false;
    features_.resize(imgs_.size());
    seam_est_imgs_.resize(imgs_.size());
    full_img_sizes_.resize(imgs_.size());



    std::vector<UMat> feature_find_imgs(imgs_.size());
    std::vector<UMat> feature_find_masks(masks_.size());

    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        full_img_sizes_[i] = imgs_[i].size();
        if (registr_resol_ < 0)
        {
            feature_find_imgs[i] = imgs_[i];
            work_scale_ = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img_sizes_[i].area()));
                is_work_scale_set = true;
            }
            resize(imgs_[i], feature_find_imgs[i], Size(), work_scale_, work_scale_, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img_sizes_[i].area()));
            seam_work_aspect_ = seam_scale_ / work_scale_;
            is_seam_scale_set = true;
        }

        if (!masks_.empty())
        {
            resize(masks_[i], feature_find_masks[i], Size(), work_scale_, work_scale_, INTER_NEAREST);
        }
        features_[i].img_idx = (int)i;

        resize(imgs_[i], seam_est_imgs_[i], Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
    }

    // find features possibly in parallel
    detail::computeImageFeatures(features_finder_, feature_find_imgs, features_, feature_find_masks);

    // Do it to save memory
    feature_find_imgs.clear();
    feature_find_masks.clear();


    (*features_matcher_)(features_, pairwise_matches_, matching_mask_);
    features_matcher_->collectGarbage();


    // Leave only images we are sure are from the same panorama
    indices_ = detail::leaveBiggestComponent(features_, pairwise_matches_, (float)conf_thresh_);
    std::vector<UMat> seam_est_imgs_subset;
    std::vector<UMat> imgs_subset;
    std::vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices_.size(); ++i)
    {
        imgs_subset.push_back(imgs_[indices_[i]]);
        seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
    }
    seam_est_imgs_ = seam_est_imgs_subset;
    imgs_ = imgs_subset;
    full_img_sizes_ = full_img_sizes_subset;

    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    return OK;
}


Stitcher::Status Stitcher::estimateCameraParams()
{
    // estimate homography in global frame
    if (!(*estimator_)(features_, pairwise_matches_, cameras_))
        return ERR_HOMOGRAPHY_EST_FAIL;

    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        Mat R;
        cameras_[i].R.convertTo(R, CV_32F);
        cameras_[i].R = R;

    }

    bundle_adjuster_->setConfThresh(conf_thresh_);
    if (!(*bundle_adjuster_)(features_, pairwise_matches_, cameras_))
        return ERR_CAMERA_PARAMS_ADJUST_FAIL;

    // Find median focal length and use it as final image scale
    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        focals.push_back(cameras_[i].focal);
    }

    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct_)
    {
        std::vector<Mat> rmats;
        for (size_t i = 0; i < cameras_.size(); ++i)
            rmats.push_back(cameras_[i].R.clone());
        detail::waveCorrect(rmats, wave_correct_kind_);
        for (size_t i = 0; i < cameras_.size(); ++i)
            cameras_[i].R = rmats[i];
    }

    return OK;
}


double Stitcher::registrationResol() 
{
	return registr_resol_;
}
void Stitcher::setRegistrationResol(double resol_mpx)
{
	registr_resol_ = resol_mpx;
}

double Stitcher::seamEstimationResol() 
{
	return seam_est_resol_;
}

void Stitcher::setSeamEstimationResol(double resol_mpx)
{
	Stitcher::seam_est_resol_ = resol_mpx;
}

double Stitcher::compositingResol() 
{
	return compose_resol_;
}
void Stitcher::setCompositingResol(double resol_mpx)
{
	compose_resol_ = resol_mpx;
}

double Stitcher::panoConfidenceThresh() 
{
	return conf_thresh_;
}
void Stitcher::setPanoConfidenceThresh(double conf_thresh)
{
	conf_thresh_ = conf_thresh;
}

bool Stitcher::waveCorrection() 
{
	return do_wave_correct_;
}
void Stitcher::setWaveCorrection(bool flag)
{
	do_wave_correct_ = flag;
}

InterpolationFlags Stitcher::interpolationFlags() 
{
	return interp_flags_;
}
void Stitcher::setInterpolationFlags(InterpolationFlags interp_flags)
{
	interp_flags_ = interp_flags;
}

detail::WaveCorrectKind Stitcher::waveCorrectKind() 
{
	return wave_correct_kind_;
}
void Stitcher::setWaveCorrectKind(detail::WaveCorrectKind kind)
{
	wave_correct_kind_ = kind;
}

Ptr<Feature2D> Stitcher::featuresFinder()
{
	return features_finder_;
}
void Stitcher::setFeaturesFinder(Ptr<Feature2D> features_finder)
{
	features_finder_ = features_finder;
}

Ptr<detail::FeaturesMatcher> Stitcher::featuresMatcher()
{
	return features_matcher_;
}

void Stitcher::setFeaturesMatcher(Ptr<detail::FeaturesMatcher> features_matcher)
{
	features_matcher_ = features_matcher;
}

cv::UMat& Stitcher::matchingMask() 
{
	return matching_mask_;
}
void Stitcher::setMatchingMask(cv::UMat &mask)
{
	CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
	matching_mask_ = mask.clone();
}

Ptr<detail::BundleAdjusterBase> Stitcher::bundleAdjuster()
{
	return bundle_adjuster_;
}

void Stitcher::setBundleAdjuster(Ptr<detail::BundleAdjusterBase> bundle_adjuster)
{
	bundle_adjuster_ = bundle_adjuster;
}

Ptr<detail::Estimator> Stitcher::estimator()
{
	return estimator_;
}
void Stitcher::setEstimator(Ptr<detail::Estimator> estimator)
{
	estimator_ = estimator;
}

Ptr<WarperCreator> Stitcher::warper()
{
	return warper_;
}

void Stitcher::setWarper(Ptr<WarperCreator> creator)
{
	warper_ = creator;
}

Ptr<detail::ExposureCompensator> Stitcher::exposureCompensator()
{
	return exposure_comp_;
}
void Stitcher::setExposureCompensator(Ptr<detail::ExposureCompensator> exposure_comp)
{
	exposure_comp_ = exposure_comp;
}

Ptr<detail::SeamFinder> Stitcher::seamFinder()
{
	return seam_finder_;
}

void Stitcher::setSeamFinder(Ptr<detail::SeamFinder> seam_finder)
{
	seam_finder_ = seam_finder;
}

Ptr<detail::Blender> Stitcher::blender()
{
	return blender_;
}

void Stitcher::setBlender(Ptr<detail::Blender> b)
{
	blender_ = b;
}

std::vector<int> Stitcher::component() 
{
	return indices_;
}
std::vector<detail::CameraParams> Stitcher::cameras() 
{
	return cameras_;
}
double Stitcher::workScale() 
{
	return work_scale_;
}
UMat Stitcher::resultMask() 
{
	return result_mask_;
}

}
