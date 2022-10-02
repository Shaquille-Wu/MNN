//
//  ConvWinograd.hpp
//  MNN
//
//  Created by MNN on 2019/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef conv_winograd_hpp
#define conv_winograd_hpp

#include "core/Execution.hpp"

#include <array>
#include <memory>
#include <vector>
#include "backend/opencl/execution/ConvExecution.hpp"

#define ORION_OPTIMIZE

namespace MNN {
namespace OpenCL {
class ConvWinograd : public Execution {
public:
    virtual ~ConvWinograd() = default;

    ConvWinograd(const MNN::Convolution2D* op, Backend* backend);

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    static bool valid(const Convolution2DCommon* common, const Tensor* input, int limit = 8192);
    std::vector<uint32_t> getLocalWS(std::string kernelName, int index, std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize, cl::Kernel mKernel);
//Shaquille, Added 20200921 Start
#ifdef ORION_OPTIMIZE
	typedef struct tag_orion_select_info {
		bool   select;
		int    slice_count;
        int    kernel_type;
		int    block_size_x;
        int    block_size_y;
	}ORION_SELECT_INFO, *PORION_SELECT_INFO;
    typedef enum tag_gemm_dual_data_type{
        DUAL_DATA_NONE = 0,   //normal 8x8
        DUAL_INPUT_DATA,
        DUAL_FILTER_DATA,
        DUAL_INPUT_DATA_CHECK_SRC_BORDER,
        DUAL_FILTER_DATA_CHECK_SRC_BORDER,
    }GEMM_DUAL_DATA_TYPE;
	static void               select_block_size(int                 split_count,
		                                        int                 compute_height,
		                                        int                 input_channel_q,
		                                        int                 output_channel_q,
		                                        int                 max_group_size,
		                                        int                 max_local_mem_size,
		                                        int                 gpu_type,
		                                        ORION_SELECT_INFO&  select_info);
	static int                find_aligned_height(int w, int h, int cur_split_count, int aligned_magic, int min_height, int max_try_count);
	static ORION_SELECT_INFO  select_orion_adpative_height(int    wUnit,
		                                                   int    hUnit,
		                                                   int    max_group_size,
		                                                   int    max_local_mem_size,
		                                                   int    cur_mnn_select,
		                                                   int    input_channel,
		                                                   int    output_channel,
		                                                   int    gpu_type);
    int                       select_opt_16x_kernel(cl::Kernel&   k_8x, 
                                                    cl::Kernel&   k_16x, 
                                                    int           gemm_width, 
                                                    int           gemm_height, 
	                                                int           compute_height,
	                                                int           input_channel_q,
	                                                int           output_channel_q,
                                                    int           alpha);
#endif
//Shaquille, Added 20200921 End

private:
    OpenCLBackend* mOpenCLBackend;
    const Convolution2DCommon* mCommon;
    int mKernelX;
    int mKernelY;
    int mPadX;
    int mPadY;
    int mStrideX;
    int mStrideY;
    MNN::PadMode mPadMode;
    std::shared_ptr<cl::Image2D> mWeight;
    std::shared_ptr<cl::Image2D> mBias;

    std::shared_ptr<Tensor> mSource;
    std::shared_ptr<Tensor> mDest;

    std::vector<cl::Kernel> mSourceTransform;
    std::vector<cl::Kernel> mDestTransform;
    std::vector<cl::Kernel> mMatMul;

    std::vector<uint32_t> mMaxWGS_S;
    std::vector<uint32_t> mMaxWGS_D;
    std::vector<uint32_t> mMaxWGS_M;

    std::vector<std::vector<uint32_t> > mGWS_S;
    std::vector<std::vector<uint32_t> > mGWS_D;
    std::vector<std::vector<uint32_t> > mGWS_M;
    
    std::vector<std::vector<uint32_t> > mLWS_S;
    std::vector<std::vector<uint32_t> > mLWS_D;
    std::vector<std::vector<uint32_t> > mLWS_M;

    int mSliceNumber;

#ifdef ORION_OPTIMIZE
    std::vector<cl::Kernel>             mMatMul_orion;
    std::vector<uint32_t>               mMaxWGS_M_orion;
    std::vector<std::vector<uint32_t> > mGWS_M_orion;
    std::vector<std::vector<uint32_t> > mLWS_M_orion;
	std::vector<ORION_SELECT_INFO>      mMatMulOptInfo;
    float                               leaky_relu_ = 0.0f;
#endif
};

} // namespace OpenCL
} // namespace MNN

#endif /* conv_winograd_hpp */
