//
//  ArgMaxExecution.cpp
//  MNN
//
//  Created by MNN on 2020/09/10.
//  Copyright © 2020, OrionStar
//

#include "backend/opencl/execution/ArgMaxExecution.hpp"
#include "core/Macro.h"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

ArgMaxExecution::ArgMaxExecution(const std::vector<Tensor *> &inputs, Backend *backend, ArgMinOrMax mode, int topk, int outMaxVal, int softmaxThreshold, int axis)
    : Execution(backend) {
    mAxis          = axis;
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}

std::vector<uint32_t> ArgMaxExecution::argMaxLocalWS(const std::vector<uint32_t> &gws,
                                                     const uint32_t maxWorkGroupSize) 
{
#ifdef MNN_OPENCL_LWS_TUNE
	MNN_ASSERT(gws.size() == 2);

	auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
	MNN_ASSERT(maxWorkItemSizes.size() >= 2);
	auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
	std::pair<std::string, std::vector<uint32_t>> info = std::make_pair("argmaxLocalWS_" + build_option_, gws);
	if (tunedLws.find(info) != tunedLws.end()) {
		//printf("argmaxLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
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
					mKernel, cl::NullRange,
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
		//printf("argmaxLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
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

bool ArgMaxExecution::buildArgMaxKernel(int input_width, int input_channel)
{
    auto runtime  = mOpenCLBackend->getOpenCLRuntime();
	build_option_ = "";
    if (mKernel.get() == nullptr) {
        std::set<std::string> buildOptions;
        std::string kernelName;
        if (mAxis == 1) {
			if (input_channel >= 4)
			{
				buildOptions.emplace("-DCHANNEL_GE_4");
				buildOptions.emplace("-DPROC_CHANNEL_4");
				build_option_ += "-DCHANNEL_GE_4";
				build_option_ += "-DPROC_CHANNEL_4";
			}
			else
			{
				if (1 == input_channel)
				{
					buildOptions.emplace("-DPROC_CHANNEL_1");
					build_option_ += "-DPROC_CHANNEL_1";
				}
				else if (2 == input_channel)
				{
					buildOptions.emplace("-DPROC_CHANNEL_2");
					build_option_ += "-DPROC_CHANNEL_2";
				}
				else if (3 == input_channel)
				{
					buildOptions.emplace("-DPROC_CHANNEL_3");
					build_option_ += "-DPROC_CHANNEL_3";
				}
			}
			if (0 != (input_channel & 3))
			{
				buildOptions.emplace("-DPROC_CHANNEL_TAIL");
				build_option_ += "-DPROC_CHANNEL_TAIL";
				if (1 == (input_channel & 3))
				{
					buildOptions.emplace("-DPROC_CHANNEL_TAIL_1");
					build_option_ += "-DPROC_CHANNEL_TAIL_1";
				}
				else if (2 == (input_channel & 3))
				{
					buildOptions.emplace("-DPROC_CHANNEL_TAIL_2");
					build_option_ += "-DPROC_CHANNEL_TAIL_2";
				}
				else
				{
					MNN_ASSERT(3 == (input_channel & 3));
					buildOptions.emplace("-DPROC_CHANNEL_TAIL_3");
					build_option_ += "-DPROC_CHANNEL_TAIL_3";
				}
			}
            mKernel           = runtime->buildKernel("argmax", "argmax_channel", buildOptions);
        } else {
            MNN_ASSERT(2 == mAxis);
            mKernel           = runtime->buildKernel("argmax_common", "argmax_height", buildOptions);
        }
        mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
    }
    return true;
}

ErrorCode ArgMaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

	auto inputDimensionFromat = TensorUtils::getDescribe(input)->dimensionFormat;

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);
	const int input_channel  = inputShape.at(3);
    if (1 == mAxis) {

        mGlobalWorkSize = {static_cast<uint32_t>(outputWidth), static_cast<uint32_t>(outputHeight * outputBatch)};
		buildArgMaxKernel(outputWidth, input_channel);

        uint32_t idx    = 0;
        mKernel.setArg(idx++, openCLImage(input));
        mKernel.setArg(idx++, openCLImage(output));
		mKernel.setArg(idx++, outputWidth);
		mKernel.setArg(idx++, outputHeight*outputBatch);
		mKernel.setArg(idx++, input_channel);
        mLocalWorkSize = argMaxLocalWS(mGlobalWorkSize, mMaxWorkGroupSize);
    } else {
        MNN_ASSERT(2 == mAxis);
#if 0
        //FUNC_PRINT(mMaxWorkGroupSize);
        if (mMaxWorkGroupSize > 256) {
            mLocalWorkSize = {16, 16, 1};
        } else {
            mLocalWorkSize = {8, 8, 1};
        }
        mGlobalWorkSize = {(uint32_t)channelBlocks*outputWidth, (uint32_t)outputBatch, 1};
        int shape[] = {outputBatch, channelBlocks, outputHeight, outputWidth};
        mKernel.setArg(0, openCLImage(input));
        mKernel.setArg(1, openCLImage(output));
        mKernel.setArg(2, shape);
#endif
    }

    return NO_ERROR;
}

ErrorCode ArgMaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ArgMaxExecution onExecute !\n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
	runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us ArgMax\n",costTime);
#else
	runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end ArgMaxExecution onExecute !\n");
#endif

    return NO_ERROR;
}

class ArgMaxCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
		/*
        if(inputs[0]->dimensions() == 3 || outputs[0]->dimensions() == 3){
            MNN_PRINT("softmax not support dimensions == 3 \n");
            return nullptr;
        }
		*/
		auto argMax  = op->main_as_ArgMax();
        auto dimType = inputs[0]->getDimensionType();
		auto axis    = argMax->axis();
		if (op->type() == OpType_ArgMax) {
			if (dimType == Tensor::TENSORFLOW && inputs[0]->dimensions() == 4) {
				int index[4] = { 0, 2, 3, 1 };
				if (axis < 0) {
					axis = inputs[0]->dimensions() + axis;
				}

				axis = index[axis];
				//1 : channel //2 : height
				if (1 == axis/* || 2 == axis*/) {
					return new ArgMaxExecution(inputs, backend, ArgMaxExecution::ArgMinOrMax::ARGMAX,
						                       argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), axis);
				}
				return nullptr;
			}
			else {
				if (axis < 0) {
					axis = inputs[0]->dimensions() + axis;
				}

				if (1 == axis/* || 2 == axis*/) {
					return new ArgMaxExecution(inputs, backend, ArgMaxExecution::ArgMinOrMax::ARGMAX,
						argMax->topK(), argMax->outMaxVal(), argMax->softmaxThreshold(), axis);
				}
				return nullptr;
			}
		}
		else
			return nullptr;
    }
};
OpenCLCreatorRegister<ArgMaxCreator> __argmax_op(OpType_ArgMax);

} // namespace OpenCL
} // namespace MNN

