//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/UnaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

UnaryExecution::UnaryExecution(const std::string& compute, Backend* backend) : Execution(backend) {
    auto openCLBackend = static_cast<OpenCLBackend*>(backend);
    std::set<std::string> buildOptions;
    buildOptions.emplace(" -DOPERATOR=" + compute);
    // FUNC_PRINT_ALL(buildOptions.begin()->c_str(), s);
    auto runtime      = openCLBackend->getOpenCLRuntime();
    mKernel           = runtime->buildKernel("unary", "unary", buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
//Shaquille, Added 20210219 Start
    op_compute_       = compute;
//Shaquille, Added 20210219 End
}
ErrorCode UnaryExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* input      = inputs[0];
    Tensor* output     = outputs[0];
    auto openCLBackend = static_cast<OpenCLBackend*>(backend());

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int batch        = outputShape.at(0);
    int outputHeight = outputShape.at(1);
    int outputWidth  = outputShape.at(2);
    int channels     = outputShape.at(3);

    int channelBlocks = (channels + 3) / 4;

    mGlobalWorkSize = {
        static_cast<uint32_t>(channelBlocks),
        static_cast<uint32_t>(outputWidth),
        static_cast<uint32_t>(batch * outputHeight),
    };

    uint32_t idx = 0;
    mKernel.setArg(idx++, mGlobalWorkSize[0]);
    mKernel.setArg(idx++, mGlobalWorkSize[1]);
    mKernel.setArg(idx++, mGlobalWorkSize[2]);
    mKernel.setArg(idx++, openCLImage(input));
    mKernel.setArg(idx++, openCLImage(output));

//Shaquille, Modified 20210219 Start
#if 0
    std::string name = "unary";
#else
    std::string name             = "unary";
    std::string new_tuning_name  = name + std::string("_") + op_compute_;
    bool        apply_new_tuning = true;
    if (openCLBackend->getOpenCLRuntime()->rawTunedLwsMap().size() > 0)
    {
        if (false == openCLBackend->getOpenCLRuntime()->is_kernel_in_raw_tuned_map(new_tuning_name, false))
            apply_new_tuning = false;
    }
    if (true == apply_new_tuning)
        name = new_tuning_name;
#endif
//Shaquille, Modified 20210219 End
    const std::vector<uint32_t> lws =
        localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, openCLBackend->getOpenCLRuntime(), name, mKernel);
    mLocalSize = lws;
    return NO_ERROR;
}

ErrorCode UnaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryExecution onExecute...");
#endif
    auto mOpenCLBackend = static_cast<OpenCLBackend*>(backend());
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Unary\n",costTime);
#else
    run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalSize,
                       mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryExecution onExecute...");
#endif
    return NO_ERROR;
}

class UnaryCreator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                case UnaryOpOperation_SQUARE:
                    return new UnaryExecution("in*in", backend);
                case UnaryOpOperation_ERF:
                    return new UnaryExecution("erf(convert_float4(in))", backend);
                case UnaryOpOperation_ERFC:
                    return new UnaryExecution("erfc(convert_float4(in))", backend);
                case UnaryOpOperation_SQRT:
                    return new UnaryExecution("sqrt(convert_float4(in))", backend);
                case UnaryOpOperation_RSQRT:
                    return new UnaryExecution("rsqrt(convert_float4(in))", backend);
                case UnaryOpOperation_ABS:
                    return new UnaryExecution("fabs(convert_float4(in))", backend);
                case UnaryOpOperation_SIN:
                    return new UnaryExecution("sin(convert_float4(in))", backend);
                case UnaryOpOperation_COS:
                    return new UnaryExecution("cos(convert_float4(in))", backend);
                case UnaryOpOperation_SIGN:
                    return new UnaryExecution("sign(convert_float4(in))", backend);
                case UnaryOpOperation_EXP:
                    return new UnaryExecution("exp(convert_float4(in))", backend);
                case UnaryOpOperation_NEG:
                    return new UnaryExecution("-(in)", backend);
                case UnaryOpOperation_TAN:
                    return new UnaryExecution("tan(convert_float4(in))", backend);
                case UnaryOpOperation_CEIL:
                    return new UnaryExecution("ceil(convert_float4(in))", backend);
                case UnaryOpOperation_LOG1P:
                    return new UnaryExecution("log1p(convert_float4(in))", backend);
                case UnaryOpOperation_FLOOR:
                    return new UnaryExecution("floor(convert_float4(in))", backend);
                case UnaryOpOperation_ROUND:
                    return new UnaryExecution("round(convert_float4(in))", backend);
                case UnaryOpOperation_SIGMOID:
                    return new UnaryExecution("native_recip((float4)1+native_exp(convert_float4(-in)))", backend);
                case UnaryOpOperation_TANH:
                    return new UnaryExecution("tanh(convert_float4(in))", backend);
                case UnaryOpOperation_RECIPROCAL:
                    return new UnaryExecution("native_recip(convert_float4(in))", backend);
                case UnaryOpOperation_LOG:
                    return new UnaryExecution("native_log(convert_float4(in+(FLOAT4)((FLOAT)0.0000001)))", backend);
                default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution("native_recip((float4)(1)+native_exp(convert_float4(-in)))", backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution("tanh(convert_float4(in))", backend);
        }
        return nullptr;
    }
};

OpenCLCreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
OpenCLCreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
OpenCLCreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
} // namespace OpenCL
} // namespace MNN
