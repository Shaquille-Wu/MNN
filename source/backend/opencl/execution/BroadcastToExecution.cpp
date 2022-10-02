//
//  BroadcastToExecution.cpp
//  MNN
//
//  Created by MNN on 2020/09/10.
//  Copyright © 2020, OrionStart
//

#include "backend/opencl/execution/BroadcastToExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

BroadcastToExecution::BroadcastToExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start BroadcastToExecution init !\n");
#endif
    auto openclBackend        = (OpenCLBackend *)backend;
    mOpenCLBackend            = static_cast<OpenCLBackend *>(backend);
}

ErrorCode BroadcastToExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#if 0	
#ifdef LOG_VERBOSE
	MNN_PRINT("Start BroadcastToExecution onResize !\n");
#endif
	auto runtime = mOpenCLBackend->getOpenCLRuntime();

	if (mKernel.get() == nullptr) {
		mKernel = runtime->buildKernel("broadcast_to", "broadcast_to", {});
		mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
	}

	Tensor *input   = inputs[0];
	Tensor *output  = outputs[0];
	//std::vector<int> inputShape  = tensorShapeFormat(input);
	std::vector<int> outputShape   = tensorShapeFormat(output);
	const int        batch         = outputShape.at(0);
	const int        height        = outputShape.at(1);
	const int        width         = outputShape.at(2);
	const int        channels      = outputShape.at(3);
	const int        channelBlocks = UP_DIV(channels, 4);
	const std::vector<uint32_t> gws           = { static_cast<uint32_t>(channelBlocks),
									              static_cast<uint32_t>(width),
									              static_cast<uint32_t>(height * batch) };
    mLWS = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), std::string("broadcast_to"), mKernel);

    for (size_t i = 0; i < mLWS.size(); ++i) 
	{
		if (mLWS[i] != 0)
        	mGWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, mLWS[i]));
    }

	uint32_t idx = 0;
    mKernel.setArg(idx++, gws[0]);
    mKernel.setArg(idx++, gws[1]);
    mKernel.setArg(idx++, gws[2]);
	mKernel.setArg(idx++, openCLImage(input));
	mKernel.setArg(idx++, openCLImage(output));
#ifdef LOG_VERBOSE
	MNN_PRINT("end BroadcastToExecution onResize !\n");
#endif
#endif
	return NO_ERROR;
}

ErrorCode BroadcastToExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
	MNN_PRINT("Start BroadcastToExecution onExecute !\n");
#endif
	auto runtime = mOpenCLBackend->getOpenCLRuntime();
#ifdef ENABLE_OPENCL_TIME_PROFILER
	cl::Event event;
	auto error = runtime->commandQueue().enqueueNDRangeKernel(mKernel, cl::NullRange,
		                                                      cl::NDRange(mGWS[0], mGWS[1], mGWS[2]),
		                                                      cl::NDRange(mLWS[0], mLWS[1], mLWS[2]), nullptr, &event);

	int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
	MNN_PRINT("kernel cost:%d    us Resize\n", costTime);
#else
	auto error = runtime->commandQueue().enqueueNDRangeKernel(mKernel, cl::NullRange,
		                                                      cl::NDRange(mGWS[0], mGWS[1], mGWS[2]),
		                                                      cl::NDRange(mLWS[0], mLWS[1], mLWS[2]), nullptr, nullptr);
#endif

	MNN_CHECK_CL_SUCCESS(error);
#ifdef LOG_VERBOSE
	MNN_PRINT("end BroadcastToExecution onExecute !\n");
#endif
	return NO_ERROR;
}

class BroadcastToCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
		auto dimType = inputs[0]->getDimensionType();
        if(inputs[0]->dimensions() != 4 || outputs[0]->dimensions() != 4)
		{
            MNN_PRINT("BroadcastTo not support dimensions == %d \n", inputs[0]->dimensions());
            return nullptr;
        }
		std::vector<int> inputShape    = tensorShapeFormat(inputs[0]);
		const int        in_batch      = inputShape.at(0);
		const int        in_height     = inputShape.at(1);
		const int        in_width      = inputShape.at(2);
		const int        in_channels   = inputShape.at(3);
		if(1 != in_batch || 1 != in_height || 1 != in_width)
		{
            MNN_PRINT("BroadcastTo not support invalid input shape (%d, %d, %d, %d)\n", in_batch, in_height, in_width, in_channels);
            return nullptr;
		}
		std::vector<int> outputShape   = tensorShapeFormat(outputs[0]);
		const int        out_batch     = outputShape.at(0);
		const int        out_height    = outputShape.at(1);
		const int        out_width     = outputShape.at(2);
		const int        out_channels  = outputShape.at(3);
		if(1 != out_batch)
		{
            MNN_PRINT("BroadcastTo not support invalid output shape (%d, %d, %d, %d)\n", out_batch, out_height, out_width, out_channels);
            return nullptr;
		}


		return new BroadcastToExecution(inputs, op, backend);
    }
};

//OpenCLCreatorRegister<BroadcastToCreator> __broadcastto_op(OpType_BroadcastTo);

} // namespace OpenCL
} // namespace MNN
