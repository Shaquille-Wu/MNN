//
//  DepthwiseConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DepthwiseConvExecution_hpp
#define DepthwiseConvExecution_hpp

#include "ConvExecution.hpp"

#define  ORION_DEPTHWISE_CONV_OPTIMIZE

namespace MNN {
namespace OpenCL {

class DepthwiseConvExecution : public ConvCommonExecution {
public:
    DepthwiseConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~DepthwiseConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
//Shaquille, Modified 20201030 Start    
#ifndef ORION_DEPTHWISE_CONV_OPTIMIZE
    std::vector<uint32_t> depthwiseConvLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);
#else
    std::vector<uint32_t> depthwiseConvLocalWS(std::string const&            tuning_name,
                                               cl::Kernel&                   kernel,
                                               const std::vector<uint32_t>&  gws, 
                                               const uint32_t                maxWorkGroupSize,
		                                       int&                          kernel_cost);
#endif    
//Shaquille, Modified 20201030 End 
private:
    const Convolution2DCommon *mConv2dCommonParams;
    const Convolution2D *mCon2dParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    OpenCLBackend *mOpenCLBackend;

//Shaquille, Added 20201024 Start
#ifdef ORION_DEPTHWISE_CONV_OPTIMIZE
    int            kx_s1_opt_      = 0;
    int            kx_s2_opt_      = 0;
    std::string    kernel_name_;
    float          leaky_relu_     = 0.0f;
	bool           bn_after_relu_  = false;
#endif    
//Shaquille, Added 20201024 End
};

} // namespace OpenCL
} // namespace MNN
#endif /* DepthwiseConvExecution_hpp */
