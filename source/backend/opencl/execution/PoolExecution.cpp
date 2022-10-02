//
//  PoolExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/PoolExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

PoolExecution::PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mPoolParams    = op->main_as_Pool();
    mPoolType      = mPoolParams->type();

    mStrides[0] = mPoolParams->strideY();
    mStrides[1] = mPoolParams->strideX();
    mKernels[0] = mPoolParams->kernelY();
    mKernels[1] = mPoolParams->kernelX();

    mPaddings[0] = mPoolParams->padY() * 2;
    mPaddings[1] = mPoolParams->padX() * 2;
    mPadType     = mPoolParams->padType();
    if (mPadType == PoolPadType_VALID) {
        mPaddings[0] = 0;
        mPaddings[1] = 0;
    }
#ifndef ORION_POOLING_OPTIMIZE
    std::set<std::string> buildOptions;
    std::string kernelName = "pooling";
    auto runtime           = mOpenCLBackend->getOpenCLRuntime();

    if (mPoolType == PoolType_AVEPOOL) {
        buildOptions.emplace("-DPOOL_AVG");
    }

    mKernel           = runtime->buildKernel("pooling", kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
#endif
}

ErrorCode PoolExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];

    if (mPoolParams->isGlobal()) {
        std::vector<int> inputShape = tensorShapeFormat(inputs[0]);
        mKernels                    = {inputShape.at(1), inputShape.at(2)};
        mStrides                    = {inputShape.at(1), inputShape.at(2)};
        mPaddings                   = {0, 0};
    }

    if (mPadType == PoolPadType_SAME) {
        int padNeededHeight = std::max(0, (output->height() - 1) * mStrides[0] + mKernels[0] - input->height());
        int padNeededWidth  = std::max(0, (output->width() - 1) * mStrides[1] + mKernels[1] - input->width());

        mPaddings[0] = padNeededHeight;
        mPaddings[1] = padNeededWidth;
    }

    MNN_ASSERT(mDilations[0] == 1 && mDilations[1] == 1);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int batch        = outputShape.at(0);
    const int outputHeight = outputShape.at(1);
    const int outputWidth  = outputShape.at(2);
    const int channels     = outputShape.at(3);

    const int inputHeight = inputShape.at(1);
    const int inputWidth  = inputShape.at(2);

    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(batch * outputHeight),
    };

    int inputImageShape[2] = {inputHeight, inputWidth};
    int paddingShape[2]    = {mPaddings[0] / 2, mPaddings[1] / 2};
    int strideShape[2]     = {mStrides[0], mStrides[1]};
    int kernelShape[2]     = {mKernels[0], mKernels[1]};
/*
    printf("pooling: s(%d, %d), k(%d, %d), in(%d, %d, %d), out(%d, %d), %s\n", 
           strideShape[0], strideShape[1], kernelShape[0], kernelShape[1], 
           inputWidth, inputHeight, channels, outputWidth, outputHeight,
           mPoolType == PoolType_AVEPOOL ? "avg" : "max");
*/
    apply_opt_ = 0;
#ifdef ORION_POOLING_OPTIMIZE
    if((0 == paddingShape[0] && 0 == paddingShape[1]) && (1 == batch))
    {
        if((kernelShape[1] == inputWidth && kernelShape[0] == inputHeight) &&
           (strideShape[1] == inputWidth && strideShape[0] == inputHeight) &&
           (1 == outputWidth) && (1 == outputHeight)) //global average
        {
            apply_opt_ = 3;
        }
        else if((0 == (outputWidth & 1)) && (0 == (outputHeight & 1)))
        {
            if(2 == kernelShape[0] && 2 == kernelShape[1] && 2 == strideShape[0] && 2 == strideShape[1])
            {
                if((inputWidth == 2 * outputWidth) && (inputHeight == 2 * outputHeight))
                {
                    if(0 == ((outputWidth * channelBlocks) & 0x1F))
                        apply_opt_ = 1;
                    else
                        apply_opt_ = 2;
                }
            }
        }
    }
    std::set<std::string> buildOptions;
    std::string kernelName  = "pooling";
	std::string tuning_name = "";
    auto runtime            = mOpenCLBackend->getOpenCLRuntime();
    if (mPoolType == PoolType_AVEPOOL) {
        buildOptions.emplace("-DPOOL_AVG");
		tuning_name += std::string("_POOL_AVG");
    }
    if(0 == apply_opt_)
    {
        mKernel           = runtime->buildKernel("pooling", kernelName, buildOptions);
    }
    else if(1 == apply_opt_)
    {
        kernelName        = "pooling_2x2_opt";
        mKernel           = runtime->buildKernel("pooling_opt", kernelName, buildOptions);
    }
    else if(2 == apply_opt_)
    {
        kernelName        = "pooling_2x2_opt";
        buildOptions.emplace("-DCHECK_POOLING_BORDER");
        mKernel           = runtime->buildKernel("pooling_opt", kernelName, buildOptions);
    }
    else
    {
        kernelName        = "pooling_global_avg";
        mKernel           = runtime->buildKernel("pooling_opt", kernelName, buildOptions);
    }
	tuning_name = kernelName + tuning_name;
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
#endif
    if(0 == apply_opt_)
    {
        mLocalWorkSize = poolLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);
        uint32_t idx   = 0;
        mKernel.setArg(idx++, mGlobalWorkSize[0]);
        mKernel.setArg(idx++, mGlobalWorkSize[1]);
        mKernel.setArg(idx++, mGlobalWorkSize[2]);
        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, sizeof(inputImageShape), inputImageShape);
        mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
        mKernel.setArg(idx++, sizeof(paddingShape), paddingShape);
        mKernel.setArg(idx++, sizeof(strideShape), strideShape);
        mKernel.setArg(idx++, sizeof(kernelShape), kernelShape);
        mKernel.setArg(idx++, openCLImage(output));
    }
    else
    {
        if(3 == apply_opt_)  //global average
        {
            mLocalWorkSize  = { 8, 8,                       1 };
            mGlobalWorkSize = { 8, 8, (uint32_t)channelBlocks };
            mKernel.setArg(0, openCLImage(input));
            mKernel.setArg(1, openCLImage(output));
            mKernel.setArg(2, inputWidth);
            mKernel.setArg(3, inputHeight);
        }
        else
        {
            mGlobalWorkSize = { (uint32_t)(outputWidth * channelBlocks), (uint32_t)outputHeight };
            int  kernel_output_width = (int)outputWidth;
            mKernel.setArg(0, openCLImage(input));
            mKernel.setArg(1, openCLImage(output));
			mKernel.setArg(2, outputWidth * channelBlocks);
			mKernel.setArg(3, outputHeight);
            if(1 == apply_opt_)
            {
                if(0 == ((outputWidth * channelBlocks) & 0x3F))
                    mLocalWorkSize  = { 64, 1 };
                else
                    mLocalWorkSize  = { 32, 1 };
            }
			else
                mLocalWorkSize = pool2DLocalWS(tuning_name, mKernel, mGlobalWorkSize, mMaxWorkGroupSize);
        }
    }
/*
	if(mLocalWorkSize.size() >= 3)
		printf("lws: %d, %d, %d, %d\n", mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2], apply_opt_);
	else
		printf("lws: %d, %d, %d\n", mLocalWorkSize[0], mLocalWorkSize[1], apply_opt_);
*/
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onResize !\n");
#endif
    return NO_ERROR;
}

std::vector<uint32_t> PoolExecution::poolLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    std::vector<uint32_t> lws(4, 0);
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
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
}

//Shaquille, Added 20201122 Start
std::vector<uint32_t> PoolExecution::pool2DLocalWS(std::string const&            tuning_name, 
	                                               cl::Kernel&                   kernel, 
	                                               std::vector<uint32_t> const&  gws, 
	                                               const uint32_t                maxWorkGroupSize)
{
#ifdef MNN_OPENCL_LWS_TUNE
	MNN_ASSERT(gws.size() == 2);

	auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
	MNN_ASSERT(maxWorkItemSizes.size() >= 2);
	auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
	std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(tuning_name, gws);
	if (tunedLws.find(info) != tunedLws.end()) {
		//printf("pool2DLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
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
				cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
					kernel, cl::NullRange,
					cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
					cl::NDRange(lws[0], lws[1]),
					nullptr, &event);
				MNN_CHECK_CL_SUCCESS(error);

				int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
				if (cost_time < min_cost) {
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
		//printf("pool2DLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
		tunedLws.insert(std::make_pair(info, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>(lws_prefer, std::vector<uint32_t>({ (uint32_t)min_cost }))));
	}

	return lws_prefer;
#else
	std::vector<uint32_t> lws(4, 0);
	auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
	uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
	int coreNum = deviceComputeUnits;
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
//Shaquille, Added 20201122 End

ErrorCode PoolExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    if(0 == apply_opt_)
    {
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
    }
    else
    {
        if(3 == apply_opt_)
            run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        else
            runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
    }
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Pooling\n",costTime);
#else
    if(0 == apply_opt_)
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    else
    {
        if(3 == apply_opt_)
            run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
        else
            runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    }
        
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onExecute !\n");
#endif
    return NO_ERROR;
}

OpenCLCreatorRegister<TypedCreator<PoolExecution>> __Pool_op(OpType_Pooling);
} // namespace OpenCL
} // namespace MNN
