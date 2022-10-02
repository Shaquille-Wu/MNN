//
//  TensorStatistic.hpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <memory>
#include <vector>
#include <MNN/Tensor.hpp>
#include <string>

enum GET_THRESHOLD_METHOD {
    THRESHOLD_MAX = 0,
    THRESHOLD_KL  = 1,
};

class TensorStatistic {
public:
    TensorStatistic(const MNN::Tensor* tensor, std::string method, const std::string& name, float featureClampValue, int binNumber = 2048, GET_THRESHOLD_METHOD thresholdMethod = THRESHOLD_KL);
    ~TensorStatistic() {
        // Do nothing
    }

    void resetUpdatedDistributionFlag() {
        mUpdatedDistributionFlag = false;
    }
    void resetUpdatedRangeFlags() {
        mUpdatedRangeFlags = false;
    }
    void updateRange();
    void resetDistribution();
    void updateDistribution();

    void setThresholdMethod(GET_THRESHOLD_METHOD thresholdMethod);
    void setChannelWise(bool mergeChannel);

    std::vector<float> finishAndCompute();

    // only this one for ADMM
    std::vector<float> computeScaleADMM();

    std::string name() {
        return mName;
    }

    bool visited() {
        return mVisited;
    }

    void setVisited(bool visited) {
        mVisited = visited;
    }

//Shaquille, Modified 20210205 Start
#if 0
    std::pair<std::vector<float>, float> fakeQuantFeature();
#else
    std::pair<std::vector<float>, float>        fakeQuantFeature(bool generate_result = true, bool generate_overflow = true, bool apply_refine_scale = false);
    void                                        fakeQuantData(float* dst, int data_cnt, float scale) const;
    int                                         threshold() const { return threshold_; };
    std::pair<float, float> const&              range_per_channel(int idx) const { return mRangePerChannel[idx]; };
    float                                       caculate_scale(float threshold, int channel_idx) const;
    int                                         threshold_method() const { return mThresholdMethod; };
    void                                        set_refine_scale(std::vector<float> const& refine_scale) { refine_scale_ = refine_scale; } ;
    void                                        set_min_threshold(int threshold)  { min_threshold_ = threshold; };
    int                                         get_min_threshold() const         { return min_threshold_; };
#endif
//Shaquille, Modified 20210205 End

    float computeDistance(std::vector<float> const& fakeQuantedFeature);
    static float compute_data_distance(float const* std, float const* compare, int data_count);

//Shaquille, Added 20210201 Start
    void copy_param_from(TensorStatistic const& other);
//Shaquille, Added 20210201 End

private:
    int _computeThreshold(const std::vector<float>& distribution);
    std::vector<std::pair<float, float>> mRangePerChannel;
    std::vector<float> mIntervals;
    std::vector<bool> mValidChannel;
    std::vector<std::vector<float>> mDistribution;

    std::shared_ptr<MNN::Tensor> mHostTensor;
    const MNN::Tensor* mOriginTensor;
    int mBinNumber;
    bool mUpdatedDistributionFlag = false;
    bool mUpdatedRangeFlags       = false;

    bool mMergeChannel                    = true;
    std::string mName;
    GET_THRESHOLD_METHOD mThresholdMethod = THRESHOLD_KL;
    bool mVisited = false;
    std::vector<float> mScales;
    float mFeatureClampValue = 127.0f;
//Shaquille, Added 20210205 Start
    int                   threshold_     = 2047;
    int                   min_threshold_ = 512;
    std::vector<float>    refine_scale_;
//Shaquille, Added 20210205 End
};
