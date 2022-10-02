//
//  RasterExecution.hpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RaterExecution_hpp
#define RaterExecution_hpp
#include <array>
#include <memory>
#include <vector>
#include "CommonExecution.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

namespace MNN {
namespace OpenCL {

class RasterExecution : public CommonExecution {
public:
    RasterExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~RasterExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
    std::vector<uint32_t> rasterLocalWorkSize(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize, std::string &kernelName, cl::Kernel &mKernel);

//Shaquille, Added 20201117 Start
	ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
//Shaquille, Added 20201117 End

private:
    std::map<Tensor*, cl::Buffer *> mTempInput;
    cl::Buffer *mTempOutput;
    OpenCLBackend *mOpenCLBackend;
    bool mNeedZero = false;
    bool mFast = false;

//Shaquille, Added 20201117 Start
	bool                                 apply_img_cpy_ = false;
	std::vector<size_t >                 src_offfset_;
	std::vector<size_t >                 dst_offfset_;
	std::vector<std::vector<size_t> >    copy_size_;
	std::vector<Tensor* >                input_tensor_;
	std::vector<Tensor* >                output_tensor_;
//Shaquille, Added 20201117 End
};

} // namespace OpenCL
} // namespace MNN

#endif
