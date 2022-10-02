//
//  GridSampleExecution.hpp
//  OrionStar
//
//  Created by Shaquille.Wu on 2021/02/27.
//  Copyright © 2021, OrionStar
//

#ifndef GridSampleExecution_hpp
#define GridSampleExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class GridSampleExecution : public Execution {
public:
    GridSampleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~GridSampleExecution() = default;

    virtual ErrorCode     onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<uint32_t> grid_sample_lws(std::string const&                tuning_name,
                                          cl::Kernel&                       kernel, 
                                          const std::vector<uint32_t>&      gws, 
                                          const uint32_t                    maxWorkGroupSize,
                                          int&                              kernel_cost);
    std::vector<uint32_t> compute_local_ws(const std::vector<uint32_t>&     gws, 
                                           const uint32_t                   maxWorkGroupSize, 
                                           int                              gpu_type, 
                                           int                              compute_units);
    virtual ErrorCode     onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    cl::Kernel              kernel_;
    std::string             kernel_name_;
    std::vector<uint32_t>   local_ws_{0, 0, 0, 0};
    std::vector<uint32_t>   global_ws_{0, 0, 0, 0};
    uint32_t                max_work_group_size_;
    OpenCLBackend*          ocl_backend_;

    int                     mode_            = 0;
    int                     padding_mode_    = 0;
    bool                    align_corners_   = false;
    std::string             tuning_name_ext_ = std::string("");
};

} // namespace OpenCL
} // namespace MNN
#endif /* GridSampleExecution_hpp */
