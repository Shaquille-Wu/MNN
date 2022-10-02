//
//  Conv3DExecution.cpp
//  MNN
//
//  Created by Shaquille.Wu on 2021/02/14.
//  Copyright © 2021, OrionStar
//

#include "Conv3DExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

std::vector<uint32_t> Conv3DExecution::conv3dLocalWS(std::string const&            tuning_name,
                                                     cl::Kernel&                   kernel,
                                                     const std::vector<uint32_t>&  gws, 
                                                     const uint32_t                maxWorkGroupSize,
                                                     int&                          kernel_cost)
{
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);

    auto maxWorkItemSizes = ocl_backend_->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = ocl_backend_->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(tuning_name, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        std::vector<uint32_t> const& cost_time = (std::get<1>(tunedLws[info]));
        if (cost_time.size() > 0)
            kernel_cost = cost_time[0];
        return std::get<0>(tunedLws[info]);
    }

    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;
    while (lws[1] <= gws[1] * 2 || lws[1] <= 4) {
        lws[0] = 1;
        while (lws[0] <= gws[0] * 2 || lws[0] <= 4) {
            if (lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0] * lws[1] <= maxWorkGroupSize) {
                cl::Event event;
                std::vector<uint32_t> internalGlobalWS(2, 1);
                for (size_t i = 0; i < gws.size(); ++i) {
                    internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                }
                cl_int error = ocl_backend_->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                    kernel, cl::NullRange,
                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                    cl::NDRange(lws[0], lws[1]),
                    nullptr, &event);
                MNN_CHECK_CL_SUCCESS(error);

                int cost_time = (int)ocl_backend_->getOpenCLRuntime()->getCostTime(&event);
                if (cost_time < min_cost) {
                    min_cost = cost_time;
                    kernel_cost   = min_cost;
                    lws_prefer[0] = lws[0];
                    lws_prefer[1] = lws[1];
                }
            }
            lws[0] *= 2;
        }
        lws[1] *= 2;
    }

    if (tunedLws.find(info) == tunedLws.end()) {
        tunedLws.insert(std::make_pair(info, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>(lws_prefer, std::vector<uint32_t>({ (uint32_t)min_cost }))));
    }

    return lws_prefer;
#else
    auto maxWorkItemSizes       = ocl_backend_->getOpenCLRuntime()->getMaxWorkItemSizes();
    uint32_t deviceComputeUnits = ocl_backend_->getOpenCLRuntime()->deviceComputeUnits();

    std::vector<uint32_t> lws(4, 0);

    int coreNum = deviceComputeUnits;
    for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i) {
        int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
        if (remain == 0) {
            lws[i] = groupSize;
        } else {
            while(groupSize) {
                int remain = gws[i] % groupSize;
                if (remain == 0 && (i > 0 || groupSize <= maxWorkGroupSize)) {
                    lws[i] = groupSize;
                    break;
                }
                --groupSize;
            }
        }
        int limit = std::min<uint32_t>(maxWorkGroupSize / totalSizeNow, maxWorkItemSizes[i]);
        lws[i] = std::max<uint32_t>(std::min<uint32_t>(lws[i], limit), 1);
        totalSizeNow *= lws[i];
    }

    return lws;
#endif
}

Conv3DCommonExecution::Conv3DCommonExecution(const Convolution3D *conv3d_params, Backend *backend) : Execution(backend)
{
    auto          ocl_backend    = (OpenCLBackend *)backend;
    int           bias_size      = conv3d_params->bias()->size();
    const float*  bias_data_ptr  = conv3d_params->bias()->data();

    int buffer_size = ALIGN_UP4(bias_size);
    if (ocl_backend->getOpenCLRuntime()->isWeightCpuTransHalf())
        buffer_size *= sizeof(half_float::half);
    else
        buffer_size *= sizeof(float);
    cl::Buffer biasBuffer(ocl_backend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int     error;
    auto       bias_ptr_ocl = ocl_backend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(biasBuffer, 
                                                                                                true, 
                                                                                                CL_MAP_WRITE, 
                                                                                                0, 
                                                                                                buffer_size, nullptr, nullptr, &error);
    if (bias_ptr_ocl != nullptr && error == CL_SUCCESS) 
    {
        if (ocl_backend->getOpenCLRuntime()->isWeightCpuTransHalf()) 
        {
            for (int i = 0; i < bias_size; i++) 
                ((half_float::half*)bias_ptr_ocl)[i] = (half_float::half)(bias_data_ptr[i]);
            for (int i = bias_size; i < ALIGN_UP4(bias_size); i++)
                ((half_float::half*)bias_ptr_ocl)[i] = (half_float::half)(0.0f);
        }
        else 
        {
            ::memset(bias_ptr_ocl, 0, ALIGN_UP4(bias_size) * sizeof(float));
            ::memcpy(bias_ptr_ocl, bias_data_ptr, bias_size * sizeof(float));
        }
    }
    else
        MNN_ERROR("Map error bias_ptr_ocl == nullptr \n");

    ocl_backend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, bias_ptr_ocl);
    bias_tensor.reset(Tensor::createDevice<float>({ 1, 1, 1, bias_size }));
    backend->onAcquireBuffer(bias_tensor.get(), Backend::STATIC);
    copyBufferToImage(ocl_backend->getOpenCLRuntime(), biasBuffer, openCLImage(bias_tensor.get()), UP_DIV(bias_size, 4), 1);
}

Conv3DCommonExecution::~Conv3DCommonExecution()
{
    MNN_ASSERT(nullptr != bias_tensor);
    backend()->onReleaseBuffer(bias_tensor.get(), Backend::STATIC);
}

Conv3DExecution::Conv3DExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
	:Conv3DCommonExecution(op->main_as_Convolution3D(), backend)
{
#ifdef LOG_VERBOSE
    MNN_PRINT("Start Conv3DExecution init !\n");
#endif
    ocl_backend_ = static_cast<OpenCLBackend *>(backend);
    const auto *conv3d_params        = op->main_as_Convolution3D();
    const auto *conv3d_common_params = conv3d_params->common();
    conv3d_common_params_ = conv3d_common_params;
    kernel_size_ = {
        conv3d_common_params_->kernels()->Get(0),
        conv3d_common_params_->kernels()->Get(1),
        conv3d_common_params_->kernels()->Get(2)
    };
    strides_ = {
        conv3d_common_params_->strides()->Get(0),
        conv3d_common_params_->strides()->Get(1),
        conv3d_common_params_->strides()->Get(2)
    };
    dilations_ = {
        conv3d_common_params_->dilates()->Get(0),
        conv3d_common_params_->dilates()->Get(1),
        conv3d_common_params_->dilates()->Get(2)
    };

    int leaky_relu_int = (int)((0xFFFF0000U & ((uint32_t)(strides_[0]))) >> 16);
    strides_[0]        = (strides_[0] & 0x0000FFFFU);
    if (0 != leaky_relu_int)
        leaky_relu_ = (float)(((double)leaky_relu_int) * 0.001);

    int  dilation_z          = (dilations_[0] & 0x0000FFFF);
    int  dilation_y          = (dilations_[1] & 0x0000FFFF);
    int  dilation_x          = (dilations_[2] & 0x0000FFFF);
    input_depth_             = ((dilations_[0] & 0xFFFF0000) >> 16);
    input_batch_as_depth_    = (0 != ((dilations_[1] & 0xFFFF0000) >> 16) ? true : false);
    output_depth_as_channel_ = (0 != ((dilations_[2] & 0xFFFF0000) >> 16) ? true : false);
    dilations_[0] = dilation_z;
    dilations_[1] = dilation_y;
    dilations_[2] = dilation_x;

    paddings_ = {
        conv3d_common_params_->pads()->Get(0),
        conv3d_common_params_->pads()->Get(1),
        conv3d_common_params_->pads()->Get(2)
    };

    int           output_channels = conv3d_common_params_->outputCount();
    int           weight_size     = conv3d_params->weight()->size();
    const float*  filter_data_ptr = conv3d_params->weight()->data();
    int           input_channels  = weight_size / (kernel_size_[0] * kernel_size_[1] * kernel_size_[2] * output_channels);
    conv3d_input_channels_        = conv3d_common_params_->inputCount();
    conv3d_output_channels_       = output_channels;
    std::vector<int> filter_image_shape(4, 1);
    if (false == output_depth_as_channel_)
    {
        filter_image_shape = { ((int)((output_channels + 3) >> 2)) * kernel_size_[0] * kernel_size_[1] * kernel_size_[2],
                                (int)(UP_DIV(input_channels, 4) * 4 * 4) };
    }
    else
    {
        filter_image_shape = { kernel_size_[0] * kernel_size_[1] * kernel_size_[2],
                                (int)(UP_DIV(input_channels, 4) * 4) };
    }
    std::shared_ptr<Tensor> filter_buffer(
        Tensor::createDevice<float>({ filter_image_shape[0], filter_image_shape[1] }));
    int buffer_size = filter_buffer->elementSize();
    if (ocl_backend_->getOpenCLRuntime()->isWeightCpuTransHalf())
        buffer_size *= sizeof(half_float::half);
    else
        buffer_size *= sizeof(float);
    cl::Buffer filterBufferCL(ocl_backend_->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filter_buffer->buffer().device = (uint64_t)(&filterBufferCL);

    cl_int error;
    auto ptr_ocl = ocl_backend_->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if (ptr_ocl != nullptr && error == CL_SUCCESS) {
        ::memset(ptr_ocl, 0, buffer_size);
        if (false == output_depth_as_channel_)
        {
            if (ocl_backend_->getOpenCLRuntime()->isWeightCpuTransHalf())
                filter_nchw_to_nc4hw4((half_float::half*)filter_data_ptr, output_channels, input_channels, kernel_size_[0], kernel_size_[1], kernel_size_[2], (half_float::half*)ptr_ocl);
            else
                filter_nchw_to_nc4hw4(filter_data_ptr, output_channels, input_channels, kernel_size_[0], kernel_size_[1], kernel_size_[2], (float*)ptr_ocl);
        }
        else
        {
            if (ocl_backend_->getOpenCLRuntime()->isWeightCpuTransHalf())
                filter_nchw_to_nc4hw4_out_depth_as_channel((half_float::half*)filter_data_ptr, input_channels, kernel_size_[0], kernel_size_[1], kernel_size_[2], (half_float::half*)ptr_ocl);
            else
                filter_nchw_to_nc4hw4_out_depth_as_channel(filter_data_ptr, input_channels, kernel_size_[0], kernel_size_[1], kernel_size_[2], (float*)ptr_ocl);
        }
    }
    else {
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    ocl_backend_->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptr_ocl);

    filter_tensor_.reset(Tensor::createDevice<float>({ 1, filter_image_shape[0], 1, filter_image_shape[1] }));
    ocl_backend_->onAcquireBuffer(filter_tensor_.get(), Backend::STATIC);
    copyBufferToImage(ocl_backend_->getOpenCLRuntime(), filterBufferCL, *((cl::Image*)(filter_tensor_->buffer().device)), filter_image_shape[1] >> 2, filter_image_shape[0]);

    // Create Kernel
    std::set<std::string> build_options;
    conv3d_tunning_name_ext_ = std::string("");
    build_options.emplace("-DBIAS");
    if (conv3d_common_params_->relu()) {
        build_options.emplace("-DRELU");
    }
    else if (conv3d_common_params_->relu6()) {
        build_options.emplace("-DRELU6");
    }
    if (0 != leaky_relu_int)
    {
        char leaky_factor[64] = { 0 };
        sprintf(leaky_factor, "-DLEAKY_RELU=%.12f", leaky_relu_);
        build_options.emplace(std::string(leaky_factor));
        conv3d_tunning_name_ext_ += "LEAKY_RELU_";
    }
    if (input_channels >= 4)
    {
        build_options.emplace("-DINPUT_CHANNEL_2");
        build_options.emplace("-DINPUT_CHANNEL_3");
        build_options.emplace("-DINPUT_CHANNEL_4");
        conv3d_tunning_name_ext_ += "CHANNEL4_";
    }
    else if (3 == input_channels)
    {
        build_options.emplace("-DINPUT_CHANNEL_2");
        build_options.emplace("-DINPUT_CHANNEL_3");
        conv3d_tunning_name_ext_ += "CHANNEL3_";
    }
    else if (2 == input_channels)
    {
        build_options.emplace("-DINPUT_CHANNEL_2");
        conv3d_tunning_name_ext_ += "CHANNEL2_";
    }

    if (input_channels > 4)
    {
        build_options.emplace("-DIN_CHANNEL_LOOP");
        conv3d_tunning_name_ext_ += "CHANNEL_LOOP_";
    }
    char kernel_size_str[64] = { 0 };
    sprintf(kernel_size_str, "k%d_%d_%d_", kernel_size_[0], kernel_size_[1], kernel_size_[2]);
    conv3d_tunning_name_ext_ += std::string(kernel_size_str);
    std::string program_name  = "conv_3d_opt";
    if (((1 == strides_[0]) && (1 == strides_[1]) && (1 == strides_[2])) &&
        ((1 == dilations_[0]) && (1 == dilations_[1]) && (1 == dilations_[2])))
    {
        if(false == output_depth_as_channel_)
        {
            if (GpuType::OTHER != ocl_backend_->getOpenCLRuntime()->getGpuType() && 
                false == ocl_backend_->getOpenCLRuntime()->isSupportedFP16())
                kernel_name_ = "conv_3d_s1_out_4channel";
            else
                kernel_name_ = "conv_3d_s1";
        }
        else
            kernel_name_ = "conv_3d_s1_out_depth_as_channel";
    }

    kernel_              = ocl_backend_->getOpenCLRuntime()->buildKernel(program_name, kernel_name_, build_options);
    max_work_group_size_ = static_cast<uint32_t>(ocl_backend_->getOpenCLRuntime()->getMaxWorkGroupSize(kernel_));

#ifdef LOG_VERBOSE
    MNN_PRINT("end Conv3DExecution init !\n");
#endif
}

Conv3DExecution::~Conv3DExecution() {
    if(nullptr != filter_tensor_)
        ocl_backend_->onReleaseBuffer(filter_tensor_.get(), Backend::STATIC);
}

ErrorCode Conv3DExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start Conv3DExecution onResize !\n");
#endif
    
    auto kernel       = &kernel_;
    int  arg_idx      = 0;
    std::vector<int>  inputShape  = tensorShapeFormat(inputs[0]);
    int               height      = inputShape.at(1);
    const int         width       = inputShape.at(2);
    const int         depth       = input_depth_;

    if (true == input_batch_as_depth_)
    {
        ;
        MNN_ASSERT(inputShape.at(0) == input_depth_);
    }
    else
        height = height / input_depth_;

    std::vector<uint32_t>  gws              = { 1, 1 };
    int                    batch_size       = inputShape.at(0);
    if (true == input_batch_as_depth_)
        batch_size = 1;
    int                    in_channel_q     = ((conv3d_input_channels_ + 3) >> 2);
    int                    out_channel_q    = ((conv3d_output_channels_ + 3) >> 2);
    int                    image_shape[4]   = { depth, height, width, 1 };
    int                    kernel_shape[4]  = { kernel_size_[0], kernel_size_[1], kernel_size_[2], 1 };
    int                    padding_shape[4] = { paddings_[0], paddings_[1], paddings_[2], 1 };
    if (false == output_depth_as_channel_)
    {
        gws = {
            (uint32_t)(width * out_channel_q),
            (uint32_t)(batch_size * depth * ((height + 3) >> 2)),
        };
        if ("conv_3d_s1" == kernel_name_)
            gws[0] = (uint32_t)(width * ((out_channel_q + 1) >> 1));
        kernel->setArg(arg_idx++, static_cast<int>(gws[0]));
        kernel->setArg(arg_idx++, static_cast<int>(gws[1]));
        kernel->setArg(arg_idx++, openCLImage(inputs[0]));
        kernel->setArg(arg_idx++, openCLImage(filter_tensor_.get()));
        kernel->setArg(arg_idx++, openCLImage(bias_tensor.get()));
        kernel->setArg(arg_idx++, openCLImage(outputs[0]));
        kernel->setArg(arg_idx++, in_channel_q);
        kernel->setArg(arg_idx++, sizeof(image_shape), image_shape);
        kernel->setArg(arg_idx++, sizeof(kernel_shape), kernel_shape);
        kernel->setArg(arg_idx++, sizeof(padding_shape), padding_shape);
        kernel->setArg(arg_idx++, out_channel_q);
    }
    else
    {
        gws = {
            (uint32_t)width,
            (uint32_t)(batch_size * ((depth + 3) >> 2) * ((height + 3) >> 2)),
        };
        kernel->setArg(arg_idx++, static_cast<int>(gws[0]));
        kernel->setArg(arg_idx++, static_cast<int>(gws[1]));
        kernel->setArg(arg_idx++, openCLImage(inputs[0]));
        kernel->setArg(arg_idx++, openCLImage(filter_tensor_.get()));
        kernel->setArg(arg_idx++, openCLImage(bias_tensor.get()));
        kernel->setArg(arg_idx++, openCLImage(outputs[0]));
        kernel->setArg(arg_idx++, in_channel_q);
        kernel->setArg(arg_idx++, sizeof(image_shape), image_shape);
        kernel->setArg(arg_idx++, sizeof(kernel_shape), kernel_shape);
        kernel->setArg(arg_idx++, sizeof(padding_shape), padding_shape);
    }

    global_work_size_        = gws;
    char in_out_size[128]    = { 0 };
    sprintf(in_out_size, "%d_%d_%d_%d_%d", depth, height, width, conv3d_input_channels_, conv3d_output_channels_);
    std::string tunning_name = kernel_name_ +
                                std::string("_") +
                                conv3d_tunning_name_ext_ +
                                (std::string("") == conv3d_tunning_name_ext_ ? std::string("_") : std::string("")) +
                                std::string(in_out_size) +
                                std::string("localWS");
    int kernel_cost  = 0;
    local_work_size_ = conv3dLocalWS(tunning_name, kernel_, gws, max_work_group_size_, kernel_cost);

#ifdef LOG_VERBOSE
    MNN_PRINT("end Conv3DExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode Conv3DExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start Conv3DExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(kernel_, global_work_size_, local_work_size_,
                ocl_backend_->getOpenCLRuntime(), &event);
    int costTime = (int)ocl_backend_->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Conv2D\n", costTime);
#else
    runKernel2D(kernel_, global_work_size_, local_work_size_,
                ocl_backend_->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end Conv3DExecution onExecute !\n");
#endif
    return NO_ERROR;
}

void Conv3DExecution::filter_nchw_to_nc4hw4(float const* src, int oc, int ic, int z, int y, int x, float* dst)
{
    int out_channel_q = (oc + 3) >> 2;
    int dst_line_size = ((((ic + 3) >> 2) << 2) << 2);
    int ic4           = (((ic + 3) >> 2) << 2);
    memset(dst, 0, out_channel_q * dst_line_size * sizeof(float));
    for (int i = 0; i < out_channel_q; i++)
    {
        for (int j = 0; j < z; j++)
        {
            for (int k = 0; k < y; k++)
            {
                for (int m = 0; m < x; m++)
                {
                    for (int n = 0; n < ic4; n++)
                    {
                        int dst_pos = i * z * y * x * dst_line_size + j * y * x * dst_line_size + k * x * dst_line_size + m * dst_line_size + 4 * n;
                        int src_pos = 4 * i * ic * z * y * x + n * z * y * x + j * y * x + k * x + m;

                        if (n >= ic)
                            continue;

                        if ((4 * i + 3) < oc)
                        {
                            dst[dst_pos]     = src[src_pos];
                            dst[dst_pos + 1] = src[src_pos + ic * z * y * x];
                            dst[dst_pos + 2] = src[src_pos + 2 * ic * z * y * x];
                            dst[dst_pos + 3] = src[src_pos + 3 * ic * z * y * x];
                        }
                        else if ((4 * i + 2) < oc)
                        {
                            dst[dst_pos]     = src[src_pos];
                            dst[dst_pos + 1] = src[src_pos + ic * z * y * x];
                            dst[dst_pos + 2] = src[src_pos + 2 * ic * z * y * x];
                        }
                        else if ((4 * i + 1) < oc)
                        {
                            dst[dst_pos]    = src[src_pos];
                            dst[dst_pos + 1] = src[src_pos + ic * z * y * x];
                        }
                        else
                        {
                            dst[dst_pos]     = src[src_pos];
                        }
                    }
                }
            }
        }
    }
}

void Conv3DExecution::filter_nchw_to_nc4hw4(half_float::half const* src, int oc, int ic, int z, int y, int x, half_float::half* dst)
{
    int out_channel_q = (oc + 3) >> 2;
    int dst_line_size = ((((ic + 3) >> 2) << 2) << 2);
    int ic4           = (((ic + 3) >> 2) << 2);
    for (int i = 0; i < out_channel_q; i++)
    {
        for (int j = 0; j < z; j++)
        {
            for (int k = 0; k < y; k++)
            {
                for (int m = 0; m < x; m++)
                {
                    for (int n = 0; n < ic4; n++)
                    {
                        int dst_pos = i * z * y * x * dst_line_size + j * y * x * dst_line_size + k * x * dst_line_size + m * dst_line_size + 4 * n;
                        int src_pos = 4 * i * ic * z * y * x + n * z * y * x + j * y * x + k * x + m;

                        if (n >= ic)
                        {
                            ((half_float::half*)dst)[dst_pos]     = (half_float::half)0.0f;
                            ((half_float::half*)dst)[dst_pos + 1] = (half_float::half)0.0f;
                            ((half_float::half*)dst)[dst_pos + 2] = (half_float::half)0.0f;
                            ((half_float::half*)dst)[dst_pos + 3] = (half_float::half)0.0f;
                            continue;
                        }

                        if ((4 * i + 3) < oc)
                        {
                            ((half_float::half*)dst)[dst_pos]     = ((half_float::half*)src)[src_pos];
                            ((half_float::half*)dst)[dst_pos + 1] = ((half_float::half*)src)[src_pos + ic * z * y * x];
                            ((half_float::half*)dst)[dst_pos + 2] = ((half_float::half*)src)[src_pos + 2 * ic * z * y * x];
                            ((half_float::half*)dst)[dst_pos + 3] = ((half_float::half*)src)[src_pos + 3 * ic * z * y * x];
                        }
                        else if ((4 * i + 2) < oc)
                        {
                            ((half_float::half*)dst)[dst_pos]     = ((half_float::half*)src)[src_pos];
                            ((half_float::half*)dst)[dst_pos + 1] = ((half_float::half*)src)[src_pos + ic * z * y * x];
                            ((half_float::half*)dst)[dst_pos + 2] = ((half_float::half*)src)[src_pos + 2 * ic * z * y * x];
                            ((half_float::half*)dst)[dst_pos + 3] = (half_float::half)0.0f;
                        }
                        else if ((4 * i + 1) < oc)
                        {
                            ((half_float::half*)dst)[dst_pos]     = ((half_float::half*)src)[src_pos];
                            ((half_float::half*)dst)[dst_pos + 1] = ((half_float::half*)src)[src_pos + ic * z * y * x];
                            ((half_float::half*)dst)[dst_pos + 2] = (half_float::half)0.0f;
                            ((half_float::half*)dst)[dst_pos + 3] = (half_float::half)0.0f;
                        }
                        else
                        {
                            ((half_float::half*)dst)[dst_pos]     = ((half_float::half*)src)[src_pos];
                            ((half_float::half*)dst)[dst_pos + 1] = (half_float::half)0.0f;
                            ((half_float::half*)dst)[dst_pos + 2] = (half_float::half)0.0f;
                            ((half_float::half*)dst)[dst_pos + 3] = (half_float::half)0.0f;
                        }
                    }
                }
            }
        }
    }
}

void Conv3DExecution::filter_nchw_to_nc4hw4_out_depth_as_channel(float const* src, int ic, int z, int y, int x, float* dst)
{
    int ic4 = (((ic + 3) >> 2) << 2);
    for (int j = 0; j < z; j++)
    {
        for (int k = 0; k < y; k++)
        {
            for (int m = 0; m < x; m++)
            {
                for (int n = 0; n < ic4; n++)
                {
                    int dst_pos = j * y * x * ic4 + k * x * ic4 + m * ic4 + n;
                    int src_pos = n * z * y * x + j * y * x + k * x + m;

                    if (n >= ic)
                    {
                        dst[dst_pos] = 0.0f;
                        continue;
                    }
                    dst[dst_pos] = src[src_pos];
                }
            }
        }
    }
}

void Conv3DExecution::filter_nchw_to_nc4hw4_out_depth_as_channel(half_float::half const* src, int ic, int z, int y, int x, half_float::half* dst)
{
    int ic4 = (((ic + 3) >> 2) << 2);
    for (int j = 0; j < z; j++)
    {
        for (int k = 0; k < y; k++)
        {
            for (int m = 0; m < x; m++)
            {
                for (int n = 0; n < ic4; n++)
                {
                    int dst_pos = j * y * x * ic4 + k * x * ic4 + m * ic4 + n;
                    int src_pos = n * z * y * x + j * y * x + k * x + m;

                    if (n >= ic)
                    {
                        dst[dst_pos] = (half_float::half)0.0f;
                        continue;
                    }
                    ((half_float::half*)dst)[dst_pos] = ((half_float::half*)src)[src_pos];
                }
            }
        }
    }
}

class Convolution3DCreator : public OpenCLBackend::Creator {
public:
    virtual ~Convolution3DCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *>&    inputs, 
		                        const std::vector<Tensor *>&    outputs,
                                const MNN::Op*                  op, 
                                Backend*                        backend) const override {
        return new Conv3DExecution(inputs, outputs, op, backend);
    }
};

OpenCLCreatorRegister<Convolution3DCreator> __conv3d_op(OpType_Convolution3D);

} // namespace OpenCL
} // namespace MNN
