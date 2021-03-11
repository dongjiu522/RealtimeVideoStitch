#ifndef _STITCHER_HPP
#define _STITCHER_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"

using namespace cv;


namespace self {


class  Stitcher
{
public:


    enum Status
    {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    };

    enum Mode
    {
        PANORAMA = 0,
    };
public:
	static Ptr<Stitcher> create();
	Status stitch(InputArrayOfArrays images, OutputArray pano);
	Status stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano);
private:
	bool init();
	bool destory();
	Status matchImages();
	Status estimateCameraParams();
	Status composePanorama(OutputArray pano);
	Status composePanorama(InputArrayOfArrays images, OutputArray pano);
private:

	double registrationResol() ;
	void setRegistrationResol(double resol_mpx);

	double seamEstimationResol() ;
	void setSeamEstimationResol(double resol_mpx);

	double compositingResol() ;
	void setCompositingResol(double resol_mpx);

	double panoConfidenceThresh() ;
	void setPanoConfidenceThresh(double conf_thresh);

	bool waveCorrection() ;
	void setWaveCorrection(bool flag);

	InterpolationFlags interpolationFlags() ;
	void setInterpolationFlags(InterpolationFlags interp_flags);

	detail::WaveCorrectKind waveCorrectKind() ;
	void setWaveCorrectKind(detail::WaveCorrectKind kind);

	Ptr<Feature2D> featuresFinder();
	void setFeaturesFinder(Ptr<Feature2D> features_finder);

	Ptr<detail::FeaturesMatcher> featuresMatcher();
	void setFeaturesMatcher(Ptr<detail::FeaturesMatcher> features_matcher);

	cv::UMat& matchingMask() ;
	void setMatchingMask( cv::UMat &mask);

	Ptr<detail::BundleAdjusterBase> bundleAdjuster();
	void setBundleAdjuster(Ptr<detail::BundleAdjusterBase> bundle_adjuster);

	Ptr<detail::Estimator> estimator();
	void setEstimator(Ptr<detail::Estimator> estimator);

	Ptr<WarperCreator> warper();
	void setWarper(Ptr<WarperCreator> creator);

	Ptr<detail::ExposureCompensator> exposureCompensator();
	void setExposureCompensator(Ptr<detail::ExposureCompensator> exposure_comp);

	Ptr<detail::SeamFinder> seamFinder();
	void setSeamFinder(Ptr<detail::SeamFinder> seam_finder);

	Ptr<detail::Blender> blender();
	void setBlender(Ptr<detail::Blender> b);

	std::vector<int> component() ;
	std::vector<detail::CameraParams> cameras() ;
	double workScale();
	UMat resultMask();

private:
	bool is_inited = false;
	static float ORIG_RESOL;

	bool do_wave_correct_;

	//threshold
    double registr_resol_;
    double seam_est_resol_;
    double compose_resol_;
    double conf_thresh_;

	//scals
	double work_scale_;
	double seam_scale_;
	double seam_work_aspect_;
	double warped_image_scale_;

    InterpolationFlags interp_flags_;
	detail::WaveCorrectKind wave_correct_kind_;

	//woker
    Ptr<Feature2D> features_finder_;
	Ptr<detail::Estimator> estimator_;
	Ptr<detail::BundleAdjusterBase> bundle_adjuster_;
    Ptr<detail::FeaturesMatcher> features_matcher_;
	Ptr<WarperCreator> warper_;
	Ptr<detail::ExposureCompensator> exposure_comp_;
	Ptr<detail::SeamFinder> seam_finder_;
	Ptr<detail::Blender> blender_;

	//mask
	cv::UMat matching_mask_;
	cv::UMat result_mask_;

	//middle val
    std::vector<cv::UMat> imgs_;
    std::vector<cv::UMat> masks_;
    std::vector<cv::Size> full_img_sizes_;
    std::vector<detail::ImageFeatures> features_;
    std::vector<detail::MatchesInfo> pairwise_matches_;
    std::vector<cv::UMat> seam_est_imgs_;
    std::vector<int> indices_;
    std::vector<detail::CameraParams> cameras_;

};


}

#endif
