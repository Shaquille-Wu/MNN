//
//  DepthwiseConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/DepthwiseConvExecution.hpp"
#include "backend/opencl/execution/MultiInputConvExecution.hpp"
#include "core/Macro.h"
#include <string.h>
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace OpenCL {

static std::string  depth_conv_kx_s1_opt_kernel[] = {
    "none",
    "depthwise_conv2d_k3_s1_5in_opt",
    "depthwise_conv2d_s1_opt",
    "depthwise_conv2d_k3_s1_4row",
    "depthwise_conv2d_s1_4row"
};

static std::string  depth_conv_kx_s2_opt_kernel[] = {
    "none",
    "depthwise_conv2d_k3_s2_opt",
    "depthwise_conv2d_k5_s2_opt"
};

DepthwiseConvExecution::DepthwiseConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend) {
    mOpenCLBackend      = static_cast<OpenCLBackend *>(backend);
    mCon2dParams        = op->main_as_Convolution2D();
    mConv2dCommonParams = mCon2dParams->common();
    mStrides            = {mConv2dCommonParams->strideY(), mConv2dCommonParams->strideX()};
//Shaquille, Added 20210220 Start
    int  leaky_relu_int = (int)(((((uint32_t)(mStrides[0])) & 0xFFFF0000)) >> 16);
    mStrides[0]         = (mStrides[0] & 0x0000FFFF);
    leaky_relu_         = (float)(((double)leaky_relu_int) * 0.001);
//Shaquille, Added 20210220 End
    mDilations          = {mConv2dCommonParams->dilateY(), mConv2dCommonParams->dilateX()};

    mPaddings[0]    = mConv2dCommonParams->padY() * 2;
    mPaddings[1]    = mConv2dCommonParams->padX() * 2;
    PadMode padMode = mConv2dCommonParams->padMode();
    if (padMode == PadMode_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }

    int kernelWidth   = mConv2dCommonParams->kernelX();
    int kernelHeight  = mConv2dCommonParams->kernelY();
    int outputChannel = mConv2dCommonParams->outputCount();

//Shaquille, Modified 20210502 Start
	int bias_size_raw = mCon2dParams->bias()->size();
	if (bias_size_raw == 3 * outputChannel || bias_size_raw == (3 * outputChannel + 1))
	{
		bn_after_relu_ = true;
		leaky_relu_    = 0.0f;
		if (bias_size_raw == (3 * outputChannel + 1))
		{
			const float *bias_data_raw = mCon2dParams->bias()->data();
			leaky_relu_    = bias_data_raw[3 * outputChannel];
			leaky_relu_int = 0xFFFFFF;
		}
	}
//Shaquille, Modified 20210502 End

    std::vector<int> filterShape{1, outputChannel, kernelHeight, kernelWidth};
    std::vector<int> filterImageShape{(int)kernelHeight * kernelWidth, (int)UP_DIV(outputChannel, 4)};

        
    const float* filterDataPtr = nullptr;
    int filterDataSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, mCon2dParams, &filterDataPtr, &filterDataSize);

    mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
    std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>(filterShape));
        
    int buffer_size = filterBuffer->elementSize();
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }
    cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
    cl_int error;
    auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(ptrCL != nullptr && error == CL_SUCCESS){
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for (int i = 0; i < filterBuffer->elementSize(); i++) {
                ((half_float::half *)ptrCL)[i] = (half_float::half)(filterDataPtr[i]);
            }
        } else {
            ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
        }
    }else{
        MNN_ERROR("Map error ptrCL == nullptr \n");
    }
    mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

    mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);

    MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
    std::string buildOption = "";
    if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf() == false){
        buildOption = "-DBUFFER_INP_FP32";
    }
    imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::DW_CONV2D_FILTER, mFilter.get(), false, buildOption);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    std::set<std::string> buildOptions;
    std::string kernelName = "depthwise_conv2d";
//Shaquille, Modified 20201024 Start
#ifndef ORION_DEPTHWISE_CONV_OPTIMIZE
    if (mConv2dCommonParams->strideX() == 1 && mConv2dCommonParams->strideY() == 1 &&
        mConv2dCommonParams->dilateX() == 1 && mConv2dCommonParams->dilateY() == 1) {
        kernelName = "depthwise_conv2d_s1";
    }
#else
    std::string cl_programe_name = "depthwise_conv2d";
    if (mConv2dCommonParams->strideX() == 1 && mConv2dCommonParams->strideY() == 1 &&
        mConv2dCommonParams->dilateX() == 1 && mConv2dCommonParams->dilateY() == 1) {
        cl_programe_name = "depth_conv_opt";
        if (3 == mConv2dCommonParams->kernelX() && 3 == mConv2dCommonParams->kernelY())
            kx_s1_opt_ = 1;
        else
            kx_s1_opt_ = 2;
        if (mOpenCLBackend->getOpenCLRuntime()->rawTunedLwsMap().size() <= 0 ||
            true == mOpenCLBackend->getOpenCLRuntime()->is_kernel_in_raw_tuned_map(depth_conv_kx_s1_opt_kernel[kx_s1_opt_ + 2], false))
        {
            if (3 == mConv2dCommonParams->kernelX() && 3 == mConv2dCommonParams->kernelY())
                kx_s1_opt_ = 3;
            else
                kx_s1_opt_ = 4;
        }
        kernelName = depth_conv_kx_s1_opt_kernel[kx_s1_opt_];
    }
    else if((mConv2dCommonParams->strideX() == 2 && mConv2dCommonParams->strideY() == 2) &&
            (mConv2dCommonParams->dilateX() == 1 && mConv2dCommonParams->dilateY() == 1))
    {
        if((3 == mConv2dCommonParams->kernelX() && 3 == mConv2dCommonParams->kernelY()) &&
           (1 == mConv2dCommonParams->padX() && 1 == mConv2dCommonParams->padY()))
        {
            if (mOpenCLBackend->getOpenCLRuntime()->rawTunedLwsMap().size() <= 0 ||
                true == mOpenCLBackend->getOpenCLRuntime()->is_kernel_in_raw_tuned_map(depth_conv_kx_s2_opt_kernel[1], false))
            {
                cl_programe_name = "depth_conv_opt";
                kx_s2_opt_       = 1;
                kernelName       = depth_conv_kx_s2_opt_kernel[kx_s2_opt_];
            }
        } 
        else if ((5 == mConv2dCommonParams->kernelX() && 5 == mConv2dCommonParams->kernelY()) &&
                 (2 == mConv2dCommonParams->padX() && 2 == mConv2dCommonParams->padY()))
        {
            cl_programe_name = "depth_conv_opt";
            kx_s2_opt_       = 2;
            kernelName       = depth_conv_kx_s2_opt_kernel[kx_s2_opt_];
        }
    }
    kernel_name_ = kernelName;
#endif
//Shaquille, Modified 20201024 End
    if (mConv2dCommonParams->relu() == true || (true == bn_after_relu_ && 0.0f == leaky_relu_)) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6() == true) {
        buildOptions.emplace("-DRELU6");
    }

//Shaquille, Modified 20201024 Start
#ifndef ORION_DEPTHWISE_CONV_OPTIMIZE
    mKernel           = runtime->buildKernel("depthwise_conv2d", kernelName, buildOptions);
#else
    if (0 != leaky_relu_int)
    {
        char leaky_relu_factor[64] = { 0 };
        sprintf(leaky_relu_factor, "-DLEAKY_RELU=%.12f", leaky_relu_);
        buildOptions.emplace(std::string(leaky_relu_factor));
    }
	if (true == bn_after_relu_)
		buildOptions.emplace(std::string("-DBN_AFTER_RELU"));
    mKernel           = runtime->buildKernel(cl_programe_name.c_str(), kernelName, buildOptions);
#endif
//Shaquille, Modified 20201024 End
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

DepthwiseConvExecution::~DepthwiseConvExecution() {
    mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
}

ErrorCode DepthwiseConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input                   = inputs[0];
    auto output                  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

#ifdef ORION_DEPTHWISE_CONV_OPTIMIZE
    if(0 != kx_s1_opt_)
    {
        if (kx_s1_opt_ <= 2)
        {
            mGlobalWorkSize = { static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 5)),
                                static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1)) };
        }
        else
        {
            mGlobalWorkSize = { static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * outputShape.at(2)),
                                static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), 4)) };
        }
    }
    else
    {
#endif
        mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                        static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
#ifdef ORION_DEPTHWISE_CONV_OPTIMIZE
    }
#endif
    if (mConv2dCommonParams->padMode() == PadMode_SAME) {
        int kernelHeightSize = (mConv2dCommonParams->kernelY() - 1) * mConv2dCommonParams->dilateY() + 1;
        int padNeededHeight =
            (output->height() - 1) * mConv2dCommonParams->strideY() + kernelHeightSize - input->height();
        int kernelWidthSize = (mConv2dCommonParams->kernelX() - 1) * mConv2dCommonParams->dilateX() + 1;
        int padNeededWidth =
            (output->width() - 1) * mConv2dCommonParams->strideX() + kernelWidthSize - input->width();

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    const int filterHeight       = mCon2dParams->common()->kernelY();
    const int filterWidth        = mCon2dParams->common()->kernelX();
    uint32_t idx                 = 0;
    auto kernel                  = &mKernel;

    int inputImageShape[2]  = {inputHeight, inputWidth};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {mStrides[0], mStrides[1]};
    int paddingShape[2]     = {mPaddings[0] / 2, mPaddings[1] / 2};
    int kernelShape[2]      = {filterHeight, filterWidth};
    int dilationShape[2]    = {mDilations[0], mDilations[1]};

    kernel->setArg(idx++, mGlobalWorkSize[0]);
    kernel->setArg(idx++, mGlobalWorkSize[1]);
    kernel->setArg(idx++, openCLImage(input));
    kernel->setArg(idx++, openCLImage(mFilter.get()));
    kernel->setArg(idx++, openCLImage(mBias.get()));
    kernel->setArg(idx++, openCLImage(output));
    kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
    kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
    kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
//Shaquille, Modified 20201024 Start
#ifdef ORION_DEPTHWISE_CONV_OPTIMIZE
    if(0 == kx_s2_opt_)
    {
#endif
        kernel->setArg(idx++, sizeof(kernelShape), kernelShape);
        kernel->setArg(idx++, sizeof(paddingShape), paddingShape);
        if (mStrides[0] != 1 || mStrides[1] != 1 || mDilations[0] != 1 || mDilations[1] != 1) {
            kernel->setArg(idx++, sizeof(dilationShape), dilationShape);
            kernel->setArg(idx++, sizeof(strideShape), strideShape);
        }
#ifdef ORION_DEPTHWISE_CONV_OPTIMIZE
    }
#endif
//Shaquille, Modified 20201024 End

//Shaquille, Modified 20201030 Start
#ifndef ORION_DEPTHWISE_CONV_OPTIMIZE
    mLocalWorkSize  = depthwiseConvLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);
#else
    int  kernel_cost         = 0;
    char kernel_size_str[96] = { 0 };
    sprintf(kernel_size_str, "k%d_%d_", filterHeight, filterWidth);
    if (0.0f != leaky_relu_)
        strcat(kernel_size_str, "LEAKY_RELU_");
	if(true == bn_after_relu_)
		strcat(kernel_size_str, "BN_AFTER_RELU_");
    mLocalWorkSize  = depthwiseConvLocalWS(kernel_name_ + std::string("_") + std::string(kernel_size_str) + std::string("_localWS"),
                                           mKernel,
                                           mGlobalWorkSize, 
                                           mMaxWorkGroupSize,
                                           kernel_cost);
    //printf("depth_conv, k(%d, %d), s(%d, %d), in(%d, %d, %d), lws(%d, %d), gws(%d, %d), %s, cost(%d)\n",
    //       filterWidth, filterHeight, mStrides[1], mStrides[0],
    //       inputWidth, inputHeight, inputChannels, mLocalWorkSize[0], mLocalWorkSize[1], mGlobalWorkSize[0], mGlobalWorkSize[1],
    //       kernel_name_.c_str(),
    //       kernel_cost);
#endif    
//Shaquille, Modified 20201030 End
    return NO_ERROR;
}

//Shaquille, Modified 20201030 Start
#ifndef ORION_DEPTHWISE_CONV_OPTIMIZE
std::vector<uint32_t> DepthwiseConvExecution::depthwiseConvLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);

    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair("depthwiseConvLocalWS", gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("depthwiseConvLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
//Shaquille, Modified 20201118 Start
#if 0
		return tunedLws[info];
#else
		return std::get<0>(tunedLws[info]);
#endif
//Shaquille, Modified 20201118 End
    }
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;

    while(lws[1] <= gws[1]) {
        lws[0] = 1;
        while(lws[0] <= gws[0]) {
            if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                cl::Event event;
                std::vector<uint32_t> internalGlobalWS(2, 1);
                for (size_t i = 0; i < gws.size(); ++i) {
                    internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                }
                cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                mKernel, cl::NullRange,
                                cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                cl::NDRange(lws[0], lws[1]),
                                nullptr, &event);
                MNN_CHECK_CL_SUCCESS(error);

                int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                if(cost_time < min_cost) {
                    min_cost = cost_time;
                    lws_prefer[0] = lws[0];
                    lws_prefer[1] = lws[1];
                }
            }
            lws[0] *= 2;
        }
        lws[1] *= 2;
    }

    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("depthwiseConvLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, lws_prefer));
    }
    
    return lws_prefer;
#else
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    std::vector<uint32_t> lws(4, 0);

    int coreNum = deviceComputeUnits * 4;
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
#else
std::vector<uint32_t> DepthwiseConvExecution::depthwiseConvLocalWS(std::string const&            tuning_name,
                                                                   cl::Kernel&                   kernel,
                                                                   const std::vector<uint32_t>&  gws,
                                                                   const uint32_t                maxWorkGroupSize,
                                                                   int&                          kernel_cost) 
{
#ifdef MNN_OPENCL_LWS_TUNE
	MNN_ASSERT(gws.size() == 2);

	auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
	MNN_ASSERT(maxWorkItemSizes.size() >= 2);
	auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
	std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(tuning_name, gws);
	if (tunedLws.find(info) != tunedLws.end()) {
		//printf("depthwiseConvLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
//Shaquille, Modified 20201118 Start
#if 0
		return tunedLws[info];
#else
		std::vector<uint32_t> const& cost_time = (std::get<1>(tunedLws[info]));
		if (cost_time.size() > 0)
			kernel_cost = cost_time[0];
		return std::get<0>(tunedLws[info]);
#endif
//Shaquille, Modified 20201118 End
	}

	std::vector<uint32_t> lws(3, 1);
	std::vector<uint32_t> lws_prefer(4, 1);
	int min_cost = INT_MAX;

	while (lws[1] <= gws[1]) {
		lws[0] = 1;
		while (lws[0] <= gws[0]) {
			if (lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0] * lws[1] <= maxWorkGroupSize) {
				cl::Event event;
				std::vector<uint32_t> internalGlobalWS(2, 1);
				for (size_t i = 0; i < gws.size(); ++i) {
					internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
				}
				cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
					kernel, cl::NullRange,
					cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
					cl::NDRange(lws[0], lws[1]),
					nullptr, &event);
				MNN_CHECK_CL_SUCCESS(error);

				int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
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
		//printf("depthwiseConvLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
//Shaquille, Modified 20201118 Start
#if 0
		tunedLws.insert(std::make_pair(info, lws_prefer));
#else

		tunedLws.insert(std::make_pair(info, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>(lws_prefer, std::vector<uint32_t>({ (uint32_t)min_cost }))));
#endif
//Shaquille, Modified 20201118 End
	}

	return lws_prefer;
#else
	uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
	auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
	std::vector<uint32_t> lws(4, 0);

	int coreNum = deviceComputeUnits * 4;
	for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i) {
		int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
		if (remain == 0) {
			lws[i] = groupSize;
		}
		else {
			while (groupSize) {
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
#endif
//Shaquille, Modified 20201030 End

ErrorCode DepthwiseConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start DepthwiseConvExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime(),
                &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us DepthwiseConv\n",costTime);
#else
    runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end DepthwiseConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class DepthwiseConvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~DepthwiseConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        
        MNN_ASSERT(inputs.size() <= 3);
        if (inputs.size() == 2 || inputs.size() == 3) {
            return new MultiInputConvExecution(op, backend);
        }
        
        MNN_ASSERT(inputs.size() == 1);
        return new DepthwiseConvExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<DepthwiseConvolutionCreator> __DepthwiseConv_op(OpType_ConvolutionDepthwise);

//OpenCLCreatorRegister<TypedCreator<DepthwiseConvExecution>> __DepthwiseConv_op(OpType_ConvolutionDepthwise);

} // namespace OpenCL
} // namespace MNN
