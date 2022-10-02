//
//  GridSampleExecution.cpp
//  OrionStar
//
//  Created by Shaquille.Wu on 2021/02/27.
//  Copyright © 2021, OrionStar
//

#include "backend/opencl/execution/GridSampleExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

GridSampleExecution::GridSampleExecution(const std::vector<Tensor *> &inputs, 
                                         const MNN::Op*               op, 
                                         Backend*                     backend)
    : Execution(backend) {
    ocl_backend_            = static_cast<OpenCLBackend *>(backend);
    auto runtime            = ocl_backend_->getOpenCLRuntime();
    auto grid_sample_param  = op->main_as_GridSample();
    mode_                   = grid_sample_param->mode();
    padding_mode_           = grid_sample_param->padding_mode();
    align_corners_          = grid_sample_param->align_corners();

    max_work_group_size_    = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernel_));
}

std::vector<uint32_t> GridSampleExecution::grid_sample_lws(std::string const&                tuning_name,
                                                           cl::Kernel&                       kernel,
                                                           const std::vector<uint32_t>&      gws,
                                                           const uint32_t                    maxWorkGroupSize,
                                                           int&                              kernel_cost) 
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
#else
    auto maxWorkItemSizes = ocl_backend_->getOpenCLRuntime()->getMaxWorkItemSizes();
    uint32_t deviceComputeUnits = ocl_backend_->getOpenCLRuntime()->deviceComputeUnits();

    std::vector<uint32_t> lws(4, 0);

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

std::vector<uint32_t> GridSampleExecution::compute_local_ws(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize, int gpu_type, int compute_units)
{
    std::vector<uint32_t> lws(4, 0);
    uint32_t deviceComputeUnits = compute_units;
    if (gpu_type == GpuType::ADRENO) {
        lws[0] = deviceComputeUnits * 4;
        lws[1] = 4;
        lws[2] = 1;
    }
    else {
        lws[0] = deviceComputeUnits * 2;
        lws[1] = 4;
        lws[2] = 1;
    }
    for (int i = 0; i < gws.size(); ++i) {
        while (gws[i] % lws[i] != 0) {
            --lws[i];
        }
    }
    return lws;
}

ErrorCode GridSampleExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start GridSampleExecution onResize !\n");
#endif
    std::set<std::string> build_options;
    std::vector<int>  input_shape   = tensorShapeFormat(inputs[0]);
    std::vector<int>  output_shape  = tensorShapeFormat(inputs[1]);
    int               height        = input_shape.at(1);
    const int         width         = input_shape.at(2);
    int               out_channel   = input_shape.at(3);
    double            y_inv_factor  = 0.5 * (double)(height);
    double            x_inv_factor  = 0.5 * (double)(width);
    if (true == align_corners_)
    {
        y_inv_factor = 0.5 * (double)(height - 1);
        x_inv_factor = 0.5 * (double)(width - 1);
    }

    if (3 == (out_channel & 3))
        build_options.emplace("-DCHANNEL_TAIL_3");
    else if(2 == (out_channel & 3))
        build_options.emplace("-DCHANNEL_TAIL_2");
    else if (1 == (out_channel & 3))
        build_options.emplace("-DCHANNEL_TAIL_1");

    if (0 == mode_)
    {
        if (0 == padding_mode_)
            kernel_name_ = std::string("grid_sample_bilinear_padding_zero");
        else if (1 == padding_mode_) //To be continue
        {
            ;
        }
        else if (2 == padding_mode_) //To be continue
        {
            ;
        }
        if (true == align_corners_)
            build_options.emplace("-DALIGN_CORNERS");
    }
    else if (1 == mode_)//To be continue
    {
        ;
    }
    else if (2 == mode_) //To be continue
    {
        ;
    }

    int  arg_idx = 0;
    kernel_              = ocl_backend_->getOpenCLRuntime()->buildKernel("grid_sample_opt", kernel_name_, build_options);
    max_work_group_size_ = static_cast<uint32_t>(ocl_backend_->getOpenCLRuntime()->getMaxWorkGroupSize(kernel_));

    std::vector<uint32_t>   gws = {
        (uint32_t)(output_shape.at(2) * ((input_shape.at(3) + 3) >> 2)),
        (uint32_t)(input_shape.at(0) * output_shape.at(1)),
    };
    int                     in_image_shape[4]  = { input_shape.at(0), input_shape.at(1), input_shape.at(2), input_shape.at(3) };
    int                     out_image_shape[2] = { output_shape.at(1), output_shape.at(2) };
    float                   inv_factor[2]      = { (float)x_inv_factor, (float)y_inv_factor };
    kernel_.setArg(arg_idx++, static_cast<int>(gws[0]));
    kernel_.setArg(arg_idx++, static_cast<int>(gws[1]));
    kernel_.setArg(arg_idx++, openCLImage(inputs[0]));
    kernel_.setArg(arg_idx++, openCLImage(inputs[1]));
    kernel_.setArg(arg_idx++, openCLImage(outputs[0]));
    kernel_.setArg(arg_idx++, sizeof(in_image_shape),  in_image_shape);
    kernel_.setArg(arg_idx++, sizeof(out_image_shape), out_image_shape);
    kernel_.setArg(arg_idx++, sizeof(inv_factor),      inv_factor);

    std::string    tuning_name   = kernel_name_ + std::string("_") + tuning_name_ext_ + std::string("localWS");
    int            cost_time     = 0;
    local_ws_                    = compute_local_ws(gws, max_work_group_size_, ocl_backend_->getOpenCLRuntime()->getGpuType(), ocl_backend_->getOpenCLRuntime()->deviceComputeUnits());// grid_sample_lws(tuning_name, kernel_, gws, max_work_group_size_, cost_time);
    global_ws_                   = gws;

#ifdef LOG_VERBOSE
    MNN_PRINT("end GridSampleExecution onResize !\n");
#endif

    return NO_ERROR;
}

ErrorCode GridSampleExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start GridSampleExecution onExecute... \n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    runKernel2D(kernel_, global_ws_, local_ws_, ocl_backend_->getOpenCLRuntime(), &event);
    int costTime = (int)ocl_backend_->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Conv2D\n", costTime);
#else
    runKernel2D(kernel_, global_ws_, local_ws_, ocl_backend_->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end GridSampleExecution onExecute... \n");
#endif

    return NO_ERROR;
}

class GridSampleCreator : public OpenCLBackend::Creator {
public:
    virtual ~GridSampleCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *>&    inputs,
                                const std::vector<Tensor *>&    outputs,
                                const MNN::Op*                  op,
                                Backend*                        backend) const override {

        auto grid_sample   = op->main_as_GridSample();
        int  mode          = grid_sample->mode();
        int  padding_mode  = grid_sample->padding_mode();

        if(0 != mode)
            return nullptr;
        if(0 != padding_mode)
            return nullptr;

        return new GridSampleExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<GridSampleCreator> __GridSample_op_(OpType_GridSample);

} // namespace OpenCL
} // namespace MNN
