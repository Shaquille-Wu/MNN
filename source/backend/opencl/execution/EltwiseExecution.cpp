//
//  EltwiseExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/EltwiseExecution.hpp"

#include "core/Macro.h"
#include <string.h>
#include <string>
#include "core/TensorUtils.hpp"

using std::string;
namespace MNN {
namespace OpenCL {

static const std::string  binary_kernel_name[] = {
    std::string("binary"),
    std::string("binary_opt")
};

static string swapComputeIn0In1(const string& computeOrigin) {
    string compute = computeOrigin;
    for (int i = 2; i < compute.length(); ++i) {
        if (compute.substr(i - 2, 2) == "in") {
            compute[i] = (compute[i] == '0' ? '1' : '0');
        }
    }
    return compute;
}

EltwiseExecution::EltwiseExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend,
                                   float operatorData, bool broadCast)
    : CommonExecution(backend), mCompute(compute), mBroadCast(broadCast), mOperatorData(operatorData) {
    mBuildOptions.emplace("-DOPERATOR=" + compute);
    mOp = op;
//Shaquille, Added 20201122 Start
	mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
//Shaquille, Added 20201122 End
}

ErrorCode EltwiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() >= 2);
    mUnits.resize(inputs.size() - 1);
    
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());

    auto nhwc0     = tensorShapeFormat(inputs[0]);
    auto nhwc       = tensorShapeFormat(outputs[0]);

    int nhwcArray[] = {nhwc[0], nhwc[1], nhwc[2], UP_DIV(nhwc[3], 4)};
    auto imageWidth  = nhwcArray[2] * nhwcArray[3];
    auto imageHeight = nhwcArray[0] * nhwcArray[1];

    int wh0[]             = {nhwc0[2], nhwc0[1]};
    int wh[]              = {nhwc[2], nhwc[1]};

    int input1Stride[]     = {1, 1, 1, 1};
//Shaquille, Modified 20201109 Start
#if 0    
    cl::NDRange localSize  = {16, 16};
    cl::NDRange globalSize = {(uint32_t)UP_DIV(imageWidth, 16) * 16, (uint32_t)UP_DIV(imageHeight, 16) * 16};
#else    
    cl::NDRange localSize  = {32, 1};
    cl::NDRange globalSize = {(uint32_t)UP_DIV(imageWidth, 32) * 32, (uint32_t)UP_DIV(imageHeight, 1) * 1};
#endif
//Shaquille, Modified 20201109 End
    if (inputs.size() > 2) {
        auto output = outputs[0];
        mTempOutput.reset(Tensor::createDevice(output->shape(), output->getType(), output->getDimensionType()));
        bool res = openCLBackend->onAcquireBuffer(mTempOutput.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        openCLBackend->onReleaseBuffer(mTempOutput.get(), Backend::DYNAMIC);
    }

    auto runTime     = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    bool useTempAsOutput = (inputs.size() % 2 != 0);
    for (int i = 0; i < inputs.size(); ++i) {
        if (i == 1)
            continue;

        auto &unit  = (i >= 2) ? mUnits[i - 1] : mUnits[i];
        int dimension = (i >= 2) ? inputs[i]->dimensions() : inputs[i + 1]->dimensions();
        int nums = 1;
        const auto& shape = (i >= 2) ? inputs[i]->shape() : inputs[i + 1]->shape();
        for (auto axis_len:shape) {
            nums*=axis_len;
        }
        /*
         DONT REMOVE THIS!!!!!
         When we do binary operation on many (>= 3) input image2d_t, we need:
         fun(outputs[0], inputs[i]) -> temp, then fun(temp, inputs[i+1]) -> outputs[0] and so on,
         instead of fun(outputs[0], inputs[i]) -> outputs[0]
         
         It's very very important for correctness on many common GPUs (Intel Iris GPU on MacBook Pro 15, for example) on Opencl 1.2.
         Opencl 1.2 do not guarantee correctness for kernel using same image2d_t as input and output, because Opencl 1.2 specification
         only support __read_only and __write_only, no include __read_write which is support on Opencl 2.x
         Your device may support it and get right result if remove this, but it is defined by the specification.
         If you insist on modifying this, please please contact hebin first. Thank you very much.
         */
        const Tensor* input0 = inputs[0];
        if (i >= 2) {
            input0 = useTempAsOutput ? outputs[0] : mTempOutput.get();
        }
        auto output = useTempAsOutput ? mTempOutput.get() : outputs[0];
        useTempAsOutput = !useTempAsOutput;
        
        if(dimension == 0 || nums == 1) {
            auto input = (i >= 2) ? inputs[i] : inputs[i + 1];
            unit.kernel = runTime->buildKernel("binary", "binary_value", mBuildOptions);
            unit.kernel.setArg(0, openCLImage(input0));
            unit.kernel.setArg(1, openCLImage(input));
            unit.kernel.setArg(2, openCLImage(output));
            unit.kernel.setArg(3, nhwcArray);
            unit.kernel.setArg(4, wh);
            unit.kernel.setArg(5, input1Stride);
        } else {
            const Tensor* input = (i >= 2) ? inputs[i] : inputs[i + 1];
            auto nhwc_0  = (i >= 2) ? nhwc : nhwc0;
            auto wh_v = (i >= 2) ? wh : wh0;
            int wh_0[] = {wh_v[0], wh_v[1]};
            auto nhwc_1 = tensorShapeFormat(input);
            int wh1[] = {nhwc_1[2], nhwc_1[1]};
            for (int dim = 0; dim < nhwc_0.size(); dim++) {
                if (nhwc_0[dim] != nhwc_1[dim]) {
                    mBroadCast = true;
                    break;
                }
            }

            if (mBroadCast) {
                if (nhwc_0[3] != nhwc_1[3]) {
                    if (nhwc_0[3] == 1) {
                        unit.kernel = (wh_0[0] != 1 && wh_0[1] != 1) ?
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_awh", mBuildOptions) :
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_1wh", mBuildOptions);
                        unit.kernel.setArg(0, openCLImage(input0));
                        unit.kernel.setArg(1, openCLImage(input));
                        unit.kernel.setArg(4, wh_0);
                        unit.kernel.setArg(5, wh1);
                    } else {
                        mBuildOptions.erase("-DOPERATOR=" + mCompute);
                        mBuildOptions.emplace("-DOPERATOR=" + swapComputeIn0In1(mCompute));
                        
                        unit.kernel = (wh1[0] != 1 && wh1[1] != 1) ?
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_awh", mBuildOptions) :
                            runTime->buildKernel("binary",
                                "binary_1toM_channel_broadcast_on_1wh", mBuildOptions);
                        unit.kernel.setArg(0, openCLImage(input));
                        unit.kernel.setArg(1, openCLImage(input0));
                        unit.kernel.setArg(4, wh1);
                        unit.kernel.setArg(5, wh_0);
                    }
                    unit.kernel.setArg(2, openCLImage(output));
                    unit.kernel.setArg(3, nhwcArray);
                    unit.kernel.setArg(6, wh);
                } else {
                    //printf("nhwc: %d, %d, %d, %d, binary_same_channel_broadcast\n", nhwcArray[0], nhwcArray[1], nhwcArray[2], nhwcArray[3]);
                    if (wh_0[0] == 1 || wh_0[1] == 1) {
                        if((wh_0[0] == wh_0[1]) && (1 == wh_0[0]))
                        {
                            mBuildOptions.emplace("-DINPUT0_X_1");
                            mBuildOptions.emplace("-DINPUT0_Y_1");
                        }
                        else if(1 == wh_0[0])
                        {
                            mBuildOptions.emplace("-DINPUT0_X_1");
                        }
                        else if(1 == wh_0[1])
                        {
                            mBuildOptions.emplace("-DINPUT0_Y_1");
                        }
                        unit.kernel = runTime->buildKernel("binary",
                                                           "binary_same_channel_broadcast", mBuildOptions);
                        unit.kernel.setArg(0, openCLImage(input0));
                        unit.kernel.setArg(1, openCLImage(input));
                        unit.kernel.setArg(4, wh_0);
                        unit.kernel.setArg(5, wh1);

                    } else {
                        mBuildOptions.erase("-DOPERATOR=" + mCompute);
                        mBuildOptions.emplace("-DOPERATOR=" + swapComputeIn0In1(mCompute));
                        if((wh1[0] == wh1[1]) && (1 == wh1[0]))
                        {
                            mBuildOptions.emplace("-DINPUT0_X_1");
                            mBuildOptions.emplace("-DINPUT0_Y_1");
                        }
                        else if(1 == wh1[0])
                        {
                            mBuildOptions.emplace("-DINPUT0_X_1");
                        }
                        else if(1 == wh1[1])
                        {
                            mBuildOptions.emplace("-DINPUT0_Y_1");
                        }
                        unit.kernel = runTime->buildKernel("binary",
                                                           "binary_same_channel_broadcast", mBuildOptions);
                        unit.kernel.setArg(0, openCLImage(input));
                        unit.kernel.setArg(1, openCLImage(input0));
                        unit.kernel.setArg(4, wh1);
                        unit.kernel.setArg(5, wh_0);
                    }
                    unit.kernel.setArg(2, openCLImage(output));
                    unit.kernel.setArg(3, nhwcArray);
                    unit.kernel.setArg(6, wh);
                }
            } else {
//Shaquille, Modified 20201122 Start
#if 0
                if((globalSize[0] != imageWidth) || (globalSize[1] != imageHeight) || (0 != (nhwc[3] & 3)))
                    mBuildOptions.emplace("-DCHECK_IMAGE_BORDER");
                unit.kernel = runTime->buildKernel("binary", "binary", mBuildOptions);
                unit.kernel.setArg(0, openCLImage(input0));
                unit.kernel.setArg(1, openCLImage(input));
                unit.kernel.setArg(2, openCLImage(output));
                unit.kernel.setArg(3, nhwcArray);
                unit.kernel.setArg(4, wh);
                unit.kernel.setArg(5, input1Stride);
#else
                std::string  tuning_name     = "binary";
                std::string  kernel_name     = binary_kernel_name[0];
                std::string  opt_tuning_name = binary_kernel_name[1] + std::string("_") + mCompute;
                bool         need_tuning     = false;
                bool         new_opt_kernel  = true;
                if (runTime->rawTunedLwsMap().size() > 0)
                {
                    if (false == runTime->is_kernel_in_raw_tuned_map(opt_tuning_name, false))
                        new_opt_kernel = false;
                }

                if (false == new_opt_kernel)
                {
                    if ((globalSize[0] != imageWidth) || (globalSize[1] != imageHeight) || (0 != (nhwc[3] & 3)))
                    {
                        mBuildOptions.emplace("-DCHECK_IMAGE_BORDER");
                        tuning_name += "_CHECK_IMAGE_BORDER";
                        need_tuning  = true;
                    }
                }
                else
                {
                    tuning_name = opt_tuning_name;
                    kernel_name = binary_kernel_name[1];
                    need_tuning = true;
                }

                unit.kernel = runTime->buildKernel("binary", kernel_name, mBuildOptions);
                unit.kernel.setArg(0, openCLImage(input0));
                unit.kernel.setArg(1, openCLImage(input));
                unit.kernel.setArg(2, openCLImage(output));
                unit.kernel.setArg(3, nhwcArray);
                unit.kernel.setArg(4, wh);
                unit.kernel.setArg(5, input1Stride);
                uint32_t     maxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(unit.kernel));
                if (true == need_tuning)
                {
                    std::vector<uint32_t>  tuning_gws(2, 1);
                    tuning_gws[0] = (globalSize.get())[0];
                    tuning_gws[1] = (globalSize.get())[1];
                    std::vector<uint32_t>  tuning_lws = binary2DLocalWS(tuning_name, unit.kernel, tuning_gws, maxWorkGroupSize);
                    (localSize.get())[0]  = tuning_lws[0];
                    (localSize.get())[1]  = tuning_lws[1];
                    (globalSize.get())[0] = (((globalSize.get())[0] + (localSize.get())[0] - 1) / (localSize.get())[0]) * (localSize.get())[0];
                    (globalSize.get())[1] = (((globalSize.get())[1] + (localSize.get())[1] - 1) / (localSize.get())[1]) * (localSize.get())[1];
                }
                else
                {
                    if(0 == (imageWidth & 0x3F))
                        (localSize.get())[0] = 64;
                }
#endif                
//Shaquille, Modified 20221122 End
                //printf("nhwc: %d, %d, %d, %d, binary, lws: %d, %d\n", nhwcArray[0], nhwcArray[1], nhwcArray[2], nhwcArray[3], (int)((localSize.get())[0]), (int)((localSize.get())[1]));
            }
        }
        unit.globalWorkSize = globalSize;
        unit.localWorkSize  = localSize;
    }
    return NO_ERROR;
}

//Shaquille, Added 20201122 Start
std::vector<uint32_t> EltwiseExecution::binary2DLocalWS(std::string const&            tuning_name,
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
        //printf("binary2DLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
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
        //printf("binary2DLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
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

ErrorCode EltwiseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs)
{
    return CommonExecution::onExecute(inputs, outputs);
}

class EltwiseCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        OpenCLBackend* ocl_backend = static_cast<OpenCLBackend*>(backend);
        OpenCLRuntime* ocl_rt      = ocl_backend->getOpenCLRuntime();
        if (op->type() == OpType_Eltwise) {
            switch (op->main_as_Eltwise()->type()) {
                case EltwiseType_SUM:
                    return new EltwiseExecution(inputs, "in0+in1", op, backend);
                case EltwiseType_PROD:
                    return new EltwiseExecution(inputs, "in0*in1", op, backend);
                case EltwiseType_MAXIMUM:
                    return new EltwiseExecution(inputs, "fmax(in0,in1)", op, backend);
				case EltwiseType_SUB:
					return new EltwiseExecution(inputs, "in0-in1", op, backend);
                default:
                    break;
            }
            return nullptr;
        }

        if (op->type() == OpType_BinaryOp) {
            MNN_ASSERT(inputs.size() > 1);

            switch (op->main_as_BinaryOp()->opType()) {
                case BinaryOpOperation_ADD:
                    return new EltwiseExecution(inputs, "in0+in1", op, backend);
                case BinaryOpOperation_SUB:
                    return new EltwiseExecution(inputs, "in0-in1", op, backend);
                case BinaryOpOperation_MUL:
                    return new EltwiseExecution(inputs, "in0*in1", op, backend);
                case BinaryOpOperation_POW:
                    return new EltwiseExecution(inputs, "pow(in0,in1)", op, backend);
                case BinaryOpOperation_DIV:
                    if(false == ocl_rt->isSupportedFP16())
                        return new EltwiseExecution(inputs, "sign(in1)*in0/fmax(fabs(in1),1e-9f)", op, backend);
                    else
                        return new EltwiseExecution(inputs, "sign(in1)*in0/fmax(fabs(in1),(FLOAT4)((FLOAT)(0.00001f)))", op, backend);
                case BinaryOpOperation_MAXIMUM:
                    return new EltwiseExecution(inputs, "fmax(in0,in1)", op, backend);
                case BinaryOpOperation_MINIMUM:
                    return new EltwiseExecution(inputs, "fmin(in0,in1)", op, backend);
                case BinaryOpOperation_REALDIV:
                    if (false == ocl_rt->isSupportedFP16())
                        return new EltwiseExecution(inputs, "sign(in1)*in0/fmax(fabs(in1),1e-9f)", op, backend);
                    else
                        return new EltwiseExecution(inputs, "sign(in1)*in0/fmax(fabs(in1),(FLOAT4)((FLOAT)(0.00001f)))", op, backend);
                default:
                    break;
            }
            return nullptr;
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<EltwiseCreator> __eltwise_op(OpType_Eltwise);
OpenCLCreatorRegister<EltwiseCreator> __binary_op(OpType_BinaryOp);

} // namespace OpenCL
} // namespace MNN
