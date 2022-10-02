//
//  ReductionExecution.cpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ReductionExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

ReductionExecution::ReductionExecution(const MNN::Op* op, Backend* backend) : CommonExecution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionExecution init !\n");
#endif
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto reduct = op->main_as_ReductionParam();
    if (nullptr != reduct->dim()) {
        for (int i = 0; i < reduct->dim()->size(); ++i) {
            mAxis.push_back(reduct->dim()->data()[i]);
        }
    }
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            mReductType = 0;
            break;
        case ReductionType_MAXIMUM:
            mReductType = 1;
            break;
        case ReductionType_MINIMUM:
            mReductType = 2;
            break;
        case ReductionType_PROD:
            mReductType = 3;
            break;
        case ReductionType_SUM:
            mReductType = 4;
            break;
        case ReductionType_L1:
            mReductType = 5;
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    mOp = op;
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionExecution init !\n");
#endif
}

ErrorCode ReductionExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    MNN_ASSERT(mAxis.size() == 1);
    MNN_ASSERT(mAxis[0] == 1);

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto input   = inputs[0];
    auto output  = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
//Shaquille, Modified 20210212 Start
#if 0
    //N=outside H=axis W=inside C=1
    MNN_ASSERT(inputShape[3] == 1);
    if(inputShape[1] >= 256) {
        mUseLocal = true;
    }
    if(!mUseLocal) {
        mGlobalWorkSize = {static_cast<uint32_t>(inputShape[0]), static_cast<uint32_t>(inputShape[2])};
        mLocalWorkSize = {1, 1, 1};
        
        switch (mReductType) {
            case 0:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mean", {});
                break;
            case 1:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_max", {});
                break;
            case 2:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_min", {});
                break;
            case 3:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mul", {});
                break;
            case 4:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_sum", {});
                break;
            default:
                MNN_ASSERT(false);
                break;
        }
    } else { //useLocal
        uint32_t global_x;
        int size = inputShape[1];
        if (size >= 1024) {
            global_x = 256;
        } else if(size >= 512) {
            global_x = 128;
        } else if (size >= 256) {
            global_x = 64;
        } else if (size >= 128) {
            global_x = 32;
        } else if (size >= 64) {
            global_x = 16;
        } else if (size >= 32) {
            global_x = 8;
        }
        mGlobalWorkSize = {global_x, static_cast<uint32_t>(inputShape[0]), static_cast<uint32_t>(inputShape[2])};
        mLocalWorkSize = {global_x, 1, 1 };
        
        switch (mReductType) {
            case 0:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mean_local", {});
                break;
            case 1:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_max_local", {});
                break;
            case 2:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_min_local", {});
                break;
            case 3:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_mul_local", {});
                break;
            case 4:
                mReduct1DKernel = runtime->buildKernel("reduction", "reduct_general_sum_local", {});
                break;
            default:
                MNN_ASSERT(false);
                break;
        }
    }
    //printf("reduce axis:%d , %d %d %d %d, useLocal:%d\n", mAxis[0], inputShape[0], inputShape[1], inputShape[2], inputShape[3], mUseLocal);

    mUnits.resize(1);
    uint32_t idx = 0;
    if(mUseLocal) {
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[1]);
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[2]);
    } else {
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[0]);
        mReduct1DKernel.setArg(idx++, mGlobalWorkSize[1]);
    }
    mReduct1DKernel.setArg(idx++, openCLImage(input));
    mReduct1DKernel.setArg(idx++, openCLImage(output));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[0]));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[1]));
    mReduct1DKernel.setArg(idx++, static_cast<int32_t>(inputShape[2]));
#else
    std::string kernel_name = "";
    switch (mReductType) {
    case 0:
        kernel_name = "reduct_general_mean_opt";
        break;
    case 1:
        kernel_name = "reduct_general_max_opt";
        break;
    case 2:
        kernel_name = "reduct_general_min_opt";
        break;
    case 3:
        kernel_name = "reduct_general_mul_opt";
        break;
    case 4:
        kernel_name = "reduct_general_sum_opt";
        break;
    case 5:
        kernel_name = "reduct_general_l1_opt";
        break;
    default:
        MNN_ASSERT(false);
        break;
    }

    std::set<std::string> build_options;
    int channel4_tail = (inputShape[3] & 3);
    if (0 != channel4_tail)
    {
        build_options.emplace("-DREDUCTION_TAIL");
        if (3 == channel4_tail)
            build_options.emplace("-DREDUCTION_TAIL_3");
        else if(2 == channel4_tail)
            build_options.emplace("-DREDUCTION_TAIL_2");
        else
            build_options.emplace("-DREDUCTION_TAIL_1");
    }

    if (0 == mReductType)
    {
        char avg_scale[64] = { 0 };
        sprintf(avg_scale, "-DINV_SCALE=%.8f", 1.0f / ((float)(inputShape.at(3))));
        build_options.emplace(std::string(avg_scale));
    }

    mReduct2DKernel         = runtime->buildKernel("reduction_opt", kernel_name, build_options);
    uint32_t max_group_size = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mReduct2DKernel));
    mGlobalWorkSize         = { (uint32_t)(inputShape[2]), uint32_t(inputShape[0] * inputShape[1]) };
    uint32_t idx = 0;
    mReduct2DKernel.setArg(idx++, openCLImage(input));
    mReduct2DKernel.setArg(idx++, openCLImage(output));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(inputShape[0]));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(inputShape[1]));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(inputShape[2]));
    mReduct2DKernel.setArg(idx++, static_cast<int32_t>(inputShape[3]));
    int      kernel_cost   = 0;
    mLocalWorkSize         = reductionLocalWS(kernel_name, mReduct2DKernel, mGlobalWorkSize, max_group_size, kernel_cost);
#endif
//Shaquille, Modified 20210212 End
    return NO_ERROR;
}

ErrorCode ReductionExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ReductionExecution onExecute !\n");
#endif
//Shaquille, Modified 20210212 Start
#if 0
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        if(mUseLocal) {
            run3DKernelDefault(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                               mOpenCLBackend->getOpenCLRuntime(), &event);
        } else {
            runKernel2D(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                               mOpenCLBackend->getOpenCLRuntime(), &event);
        }
        int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us Reduct1D\n",costTime);
    #else
    if(mUseLocal) {
        run3DKernelDefault(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    } else {
        runKernel2D(mReduct1DKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    }
    #endif
#else
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(mReduct2DKernel, mGlobalWorkSize, mLocalWorkSize,
        mOpenCLBackend->getOpenCLRuntime(), &event);
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Reduct1D\n", costTime);
#else
    runKernel2D(mReduct2DKernel, mGlobalWorkSize, mLocalWorkSize,
                mOpenCLBackend->getOpenCLRuntime());
#endif
#endif
//Shaquille, Modified 20210212 End
#ifdef LOG_VERBOSE
    MNN_PRINT("end ReductionExecution onExecute !\n");
#endif
    return NO_ERROR;
}

std::vector<uint32_t> ReductionExecution::reductionLocalWS(std::string const&            tuning_name,
	                                                       cl::Kernel&                   kernel,
	                                                       const std::vector<uint32_t>&  gws,
	                                                       const uint32_t                maxWorkGroupSize,
	                                                       int&                          kernel_cost)
{
    MNN_ASSERT(gws.size() == 2);

    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
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
                cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                    kernel, cl::NullRange,
                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                    cl::NDRange(lws[0], lws[1]),
                    nullptr, &event);
                MNN_CHECK_CL_SUCCESS(error);

                int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                if (cost_time < min_cost) {
                    min_cost = cost_time;
                    kernel_cost = min_cost;
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
}

class ReductionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ReductionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                 const MNN::Op *op, Backend *backend) const override {
//Shaquille, Modified 20210212 Start
#if 0
        if (inputs[0]->getDimensionType() == Tensor::TENSORFLOW) {
            auto openCLBackend = static_cast<OpenCLBackend *>(backend);
            auto reduct = op->main_as_ReductionParam();
            if (nullptr == reduct->dim()) {
                return NULL;
            }
            if(reduct->dim()->size() != 1) {
                return NULL;
            }
            switch (op->main_as_ReductionParam()->operation()) {
                case ReductionType_MEAN:
                    break;
                case ReductionType_MAXIMUM:
                    break;
                case ReductionType_MINIMUM:
                    break;
                case ReductionType_PROD:
                    break;
                case ReductionType_SUM:
                    break;
                default:
                    return NULL;
                    break;
            }
            return new ReductionExecution(op, backend);
        }
#else
        auto tensor_fmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto openCLBackend = static_cast<OpenCLBackend *>(backend);
        auto reduct = op->main_as_ReductionParam();
        if (nullptr == reduct->dim()) {
            return NULL;
        }
        if (reduct->dim()->size() != 1) {
            return NULL;
        }
        int axis_idx = reduct->dim()->data()[0];
        if (MNN_DATA_FORMAT_NHWC != tensor_fmt)
        {
            if (axis_idx != 1)
                return NULL;
        }
        else
        {
            if (axis_idx != 3)
                return NULL;
        }

        switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            break;
        case ReductionType_MAXIMUM:
            break;
        case ReductionType_MINIMUM:
            break;
        case ReductionType_PROD:
            break;
        case ReductionType_SUM:
            break;
        case ReductionType_L1:
            break;
        default:
            return NULL;
            break;
        }
        return new ReductionExecution(op, backend);
#endif
//Shaquille, Modified 20210212 End
        return NULL;
    }
};

OpenCLCreatorRegister<ReductionCreator> __reduction_op(OpType_Reduction);
} // namespace OpenCL
} // namespace MNN
