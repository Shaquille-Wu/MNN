//
//  InterpExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef InterpExecution_hpp
#define InterpExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
//#define  ORION_INTERP_OPTIMIZE

namespace MNN {
namespace OpenCL {

class InterpExecution : public Execution {
public:
    InterpExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~InterpExecution() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<uint32_t> interpLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    cl::Kernel mKernel;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    float mCordTransform[4];
//Shaquille, Added 20201104 Start
#ifdef ORION_INTERP_OPTIMIZE
    int            upsample_opt_type_ = 0;
    int            resize_type_ = 0;
    bool           half_pixel_center_ = false;
    cl::Kernel     upsample_opt_kernel_;
#endif    
//Shaquille, Added 20201104 End
};

} // namespace OpenCL
} // namespace MNN
#endif /* InterpExecution_hpp */
