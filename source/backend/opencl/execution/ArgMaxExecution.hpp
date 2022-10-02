//
//  ArgMaxExecution.hpp
//  MNN
//
//  Created by MNN on 2020/09/10.
//  Copyright © 2020, OrionStar
//

#ifndef ArgMaxExecution_hpp
#define ArgMaxExecution_hpp

#include <vector>
#include "core/Execution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class ArgMaxExecution : public Execution {
public:
	enum ArgMinOrMax {
		ARGMIN,
		ARGMAX
	};
	ArgMaxExecution(const std::vector<Tensor *> &inputs, Backend *backend, ArgMinOrMax mode, int topk, int outMaxVal, int softmaxThreshold, int axis);

    virtual ~ArgMaxExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    bool buildArgMaxKernel(int input_width, int input_channel);
    std::vector<uint32_t> argMaxLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

private:
    cl::Kernel mKernel;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    std::vector<uint32_t> mGlobalWorkSize{1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1};
	std::string           build_option_;
    int mAxis;
};
} // namespace OpenCL
} // namespace MNN
#endif /* SoftmaxExecution_hpp */
