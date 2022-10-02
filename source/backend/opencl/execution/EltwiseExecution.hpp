//
//  EltwiseExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef EltwiseExecution_hpp
#define EltwiseExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class EltwiseExecution : public CommonExecution {
public:
    EltwiseExecution(const std::vector<Tensor *> &inputs, const std::string &compute, const MNN::Op *op, Backend *backend, float operatorData = 0.0001f, bool broadCast = false);
    virtual ~EltwiseExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

//Shaquille, Added 20201122 Start
    virtual ErrorCode     onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
	std::vector<uint32_t> binary2DLocalWS(std::string const&            tuning_name,
                                          cl::Kernel&                   kernel,
                                          std::vector<uint32_t> const&  gws,
                                          const uint32_t                maxWorkGroupSize);
//Shaquille, Added 20201122 End

private:
    bool mBroadCast;
    float mOperatorData;
    std::string mCompute;
    std::set<std::string> mBuildOptions;
    std::shared_ptr<Tensor> mTempOutput;
//Shaquille, Added 20201122 Start
    OpenCLBackend *mOpenCLBackend;
//Shaquille, Added 20201122 End
};

} // namespace OpenCL
} // namespace MNN
#endif /* EltwiseExecution_hpp */
