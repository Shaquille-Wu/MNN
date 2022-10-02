//
//  BroadcastToExecution.hpp
//  MNN
//
//  Created by MNN on 2020/09/10.
//  Copyright © 2020, OrionStart
//

#ifndef BroadcastToExecution_hpp
#define BroadcastToExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class BroadcastToExecution : public Execution {
public:
	BroadcastToExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~BroadcastToExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScale;
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    std::vector<uint32_t> mGWS{1, 1, 1, 1};
    std::vector<uint32_t> mLWS{1, 1, 1, 1};
	cl::Kernel mImageToBufferKernel;
	cl::Kernel mBufferToImageKernel;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ScaleExecution_hpp */
