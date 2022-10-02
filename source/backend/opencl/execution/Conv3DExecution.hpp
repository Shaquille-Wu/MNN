//
//  Conv3DExecution.hpp
//  MNN
//
//  Created by Shaquille.Wu on 2021/02/14.
//  Copyright © 2021, OrionStar
//

#ifndef _CONV3D_EXECUTION_HPP_
#define _CONV3D_EXECUTION_HPP_

#include "core/Execution.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class Conv3DCommonExecution : public Execution {
public:
    Conv3DCommonExecution(const Convolution3D *conv3d_params, Backend *backend);
    virtual ~Conv3DCommonExecution();

protected:
    std::shared_ptr<Tensor> bias_tensor;
};

class Conv3DExecution : public Conv3DCommonExecution {
public:
    Conv3DExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    virtual ~Conv3DExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    std::vector<uint32_t> conv3dLocalWS(std::string const&            tuning_name,
                                        cl::Kernel&                   kernel,
                                        const std::vector<uint32_t>&  gws,
                                        const uint32_t                maxWorkGroupSize,
                                        int&                          kernel_cost);

    static void                    filter_nchw_to_nc4hw4(float const* src, int oc, int ic, int z, int y, int x, float* dst);
    static void                    filter_nchw_to_nc4hw4(half_float::half const* src, int oc, int ic, int z, int y, int x, half_float::half* dst);

    static void                    filter_nchw_to_nc4hw4_out_depth_as_channel(float const* src, int ic, int z, int y, int x, float* dst);
    static void                    filter_nchw_to_nc4hw4_out_depth_as_channel(half_float::half const* src, int ic, int z, int y, int x, half_float::half* dst);

private:
    const Convolution3DCommon*     conv3d_common_params_;
    std::vector<int>               strides_{ 1, 1, 1 };
    std::vector<int>               dilations_{ 1, 1, 1 };
    std::vector<int>               paddings_{ 0, 0, 0 };
    std::vector<int>               kernel_size_{ 1, 1, 1 };
    std::vector<uint32_t>          global_work_size_{1, 1, 1};
    std::vector<uint32_t>          local_work_size_{1, 1, 1};
    cl::Kernel                     kernel_;
    std::string                    kernel_name_ = std::string("");
    uint32_t                       max_work_group_size_;
    OpenCLBackend*                 ocl_backend_;
    float                          leaky_relu_ = 0.0f;

    std::shared_ptr<cl::Buffer>    kenel_buffer_;
    std::shared_ptr<cl::Buffer>    bias_buffer_;
    std::shared_ptr<Tensor>        filter_tensor_;

    int                            input_depth_ = 1;
    bool                           input_batch_as_depth_    = false;
    bool                           output_depth_as_channel_ = false;
    int                            conv3d_input_channels_   = 1;
    int                            conv3d_output_channels_  = 1;
    std::string                    conv3d_tunning_name_ext_ = std::string("");
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvExecution_hpp */
