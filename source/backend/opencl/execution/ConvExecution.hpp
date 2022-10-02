//
//  ConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvExecution_hpp
#define ConvExecution_hpp

#include "core/Execution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#define ORION_CONV_OPTIMIZE

namespace MNN {
namespace OpenCL {

class ConvCommonExecution : public Execution {
public:
    ConvCommonExecution(const Convolution2D *op, Backend *backend);
    virtual ~ConvCommonExecution();

protected:
    std::shared_ptr<Tensor> mBias;
};

class ConvExecution : public ConvCommonExecution {
public:
    ConvExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static std::shared_ptr<Tensor> getBias(OpenCLBackend *backend, const Convolution2D *conv);

//Shaquille, Modified 20201011 Start
    std::vector<uint32_t> conv2d1x1LocalWS(std::string const&      tuning_name,
                                           std::string const&      tuning_ext_name,
                                           cl::Kernel&             kernel, 
                                           std::vector<uint32_t>&  gws, 
                                           const uint32_t          maxWorkGroupSize, 
                                           int&                    kernel_cost);
//Shaquille, Modified 20200111 End
    std::vector<uint32_t> conv2d1x1LocalWSOpt(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

    std::vector<uint32_t> conv2dGeneralLocalWS(const std::vector<uint32_t> &gws, const uint32_t kernelSize,
                                               const uint32_t maxWorkGroupSize);
//Shaquille, Added 20201024 Start                                         
    std::vector<uint32_t> conv2dGeneralLocalWSOpt(std::string const&            tuning_name,
                                                  cl::Kernel&                   kernel,
                                                  const std::vector<uint32_t>&  gws, 
                                                  const uint32_t                kernelSize,
                                                  const uint32_t                maxWorkGroupSize);
//Shaquille, Added 20201024 End

#ifdef ORION_CONV_OPTIMIZE
private:
    static std::string    get_conv1x1_tuning_size_by_shape(int w, int h, int input_channel, int output_channel);
    static int            select_conv1x1_opt_type(OpenCLBackend* ocl_bn, int n, int w, int h, int input_channel, int output_channel, int stride_x, int stride_y);
    int                   evaluate_performance_2d(cl::Kernel&                    kernel, 
                                                  std::vector<uint32_t> const&   global_work_space, 
                                                  std::vector<uint32_t> const&   local_work_space);
    static bool           is_zero_bias(Convolution2D const* conv_2d_param);
#endif

private:
    const Convolution2DCommon *mConv2dCommonParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<Tensor> mFilter;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    bool mIsTurn = false;
    OpenCLBackend *mOpenCLBackend;
    bool mConv1x1Opt{false};
    bool mUseLocalMem{false};
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;

//Shaquille, Added 20201010 Start
#ifdef ORION_CONV_OPTIMIZE
    bool                        zero_bias_flag_                = false;
    int                         conv_opt_stride_1x1_           = 0;
    int                         conv_opt_stride_1x1_dilation_  = 0;
    std::string                 conv_2d_tuning_ext_name_       = "";
    int                         conv_s2x2_opt_type_            = 0;
    int                         conv1x1_opt_type_              = 0;
    bool                        k1x1_conv_                     = false;
    float                       leaky_relu_                    = 0.0f;
	bool                        bn_after_relu_                 = false;
    cl::Kernel                  opt_kernel_;
#endif
//Shaquille, Added 20201010 End    
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvExecution_hpp */
