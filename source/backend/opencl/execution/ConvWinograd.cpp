//
//  ConvWinograd.cpp
//  MNN
//
//  Created by MNN on 2019/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/ConvWinograd.hpp"
#include <string.h>
#include "core/Backend.hpp"
#include "core/ConvolutionCommon.hpp"
#include "math/WingoradGenerater.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#define UNIT 2
#define INTERP 1
namespace MNN {
namespace OpenCL {
bool ConvWinograd::valid(const Convolution2DCommon* common, const Tensor* input, int limit) {
//Shaquille, Modified 20210220 Start
#ifndef ORION_OPTIMIZE
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
#else
    int32_t  stride_x = common->strideX();
    int32_t  stride_y = common->strideY();
    stride_y          = (stride_y & 0x0000FFFF);
    if (stride_x != 1 || stride_y != 1) {
        return false;
    }
#endif
//Shaquille, Modified 20210220 End
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if (input->channel() < 8 || common->outputCount() < 8) {
        return false;
    }
    return (common->kernelX() == 3 && common->kernelY() == 3) || (common->kernelX() == 5 && common->kernelY() == 5);
}

ConvWinograd::ConvWinograd(const MNN::Convolution2D* op, Backend* backend) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mCommon        = op->common();
    MNN_ASSERT((3 == mCommon->kernelY() && 3 == mCommon->kernelX()) ||
               (5 == mCommon->kernelX() && 5 == mCommon->kernelY()));
//Shaquille, Modifed 20210220 Start
#ifndef ORION_OPTIMIZE
    MNN_ASSERT(1 == mCommon->strideX() && 1 == mCommon->strideY());
#else
    int32_t  stride_x = mCommon->strideX();
    int32_t  stride_y = mCommon->strideY();
    stride_y = (stride_y & 0x0000FFFF);
    MNN_ASSERT(1 == stride_x && 1 == stride_y);
#endif
//Shaquille, Modified 20210220 End
    MNN_ASSERT(1 == mCommon->dilateX() && 1 == mCommon->dilateY());
    auto runTime = mOpenCLBackend->getOpenCLRuntime();
    int ky       = mCommon->kernelY();
    int kx       = mCommon->kernelX();

    int weightSize             = 0;
    const float* filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != op->quanParameter()) {
        quanCommon = ConvolutionCommon::load(op->quanParameter(), true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution \n");
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    }

    if (nullptr == filterDataPtr) {
        weightSize    = op->weight()->size();
        filterDataPtr = op->weight()->data();
    }

    int co     = mCommon->outputCount();
    int ci     = weightSize / co / mCommon->kernelX() / mCommon->kernelY();
    auto coC4  = UP_DIV(co, 4);
    auto ciC4  = UP_DIV(ci, 4);
    auto queue = runTime->commandQueue();

    auto imageChannelType = CL_HALF_FLOAT;
    if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_High) {
        imageChannelType = CL_FLOAT;
    }
    // Create Image
    {
        mBias.reset(new cl::Image2D(runTime->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                    UP_DIV(co, 4), 1, 0, nullptr, nullptr));
        
        int buffer_size = ALIGN_UP4(co);
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        std::shared_ptr<cl::Buffer> biasBuffer(
            new cl::Buffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));

        cl_int error;
        auto biasC = queue.enqueueMapBuffer(*biasBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(biasC != nullptr && error == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
                for(int i=0; i<co; i++) {
                    ((half_float::half*)biasC)[i] = (half_float::half)(op->bias()->data()[i]);
                }
                for(int i=co; i<ALIGN_UP4(co); i++) {
                    ((half_float::half*)biasC)[i] = (half_float::half)(0.0f);
                }
            }else{
                ::memset(biasC, 0, buffer_size);
                ::memcpy(biasC, op->bias()->data(), co * sizeof(float));
            }
        }else{
            MNN_ERROR("Map error biasC == nullptr \n");
        }
        queue.enqueueUnmapMemObject(*biasBuffer, biasC);
        copyBufferToImage(runTime, *biasBuffer, *mBias, coC4, 1);

        std::shared_ptr<Tensor> sourceWeight(
            Tensor::create<float>(std::vector<int>{co, ci, ky, kx}, (void*)(filterDataPtr), Tensor::CAFFE));

        int unit       = UNIT;
        int kernelSize = kx;
        Math::WinogradGenerater generator(unit, kernelSize, INTERP);
        int alpha       = unit + kernelSize - 1;
        auto weightDest = generator.allocTransformWeight(sourceWeight.get());
        generator.transformWeight(weightDest.get(), sourceWeight.get());
        auto weightDestSize = weightDest->size();
        
        buffer_size = weightDest->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        cl::Buffer weightBuffer(runTime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        {
            cl_int error;
            auto weightPtr = queue.enqueueMapBuffer(weightBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
            if(weightPtr != nullptr && error == CL_SUCCESS){
                if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
                    for(int i=0; i<weightDest->elementSize(); i++) {
                        ((half_float::half*)weightPtr)[i] = (half_float::half)(weightDest->host<float>()[i]);
                    }
                }else{
                    ::memcpy(weightPtr, weightDest->host<float>(), buffer_size);
                }
            } else{
                MNN_ERROR("Map error weightPtr == nullptr \n");
            }

            queue.enqueueUnmapMemObject(weightBuffer, weightPtr);
        }
        mWeight.reset(new cl::Image2D(runTime->context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, imageChannelType),
                                      ciC4 * 4, coC4 * alpha * alpha, 0, nullptr, nullptr));
        copyBufferToImage(runTime, weightBuffer, *mWeight, ciC4 * 4, coC4 * alpha * alpha);
    }
}

#ifndef ORION_OPTIMIZE
ErrorCode ConvWinograd::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    mKernelX    = mCommon->kernelX();
    mKernelY    = mCommon->kernelY();
    mPadX       = mCommon->padX();
    mPadY       = mCommon->padY();
    mStrideX    = mCommon->strideX();
    mStrideY    = mCommon->strideY();
    mPadMode    = mCommon->padMode();
    
    int alpha  = mCommon->kernelX() + UNIT - 1;
    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);
    int padX   = mPadX;
    int padY   = mPadY;
    if (mPadMode == PadMode_SAME) {
        int kernelWidthSize  = (mKernelX - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mKernelY - 1) * mCommon->dilateY() + 1;
        int padNeededWidth   = (output->width() - 1) * mStrideX + kernelWidthSize - input->width();
        int padNeededHeight  = (output->height() - 1) * mStrideY + kernelHeightSize - input->height();
        padX                 = padNeededWidth / 2;
        padY                 = padNeededHeight / 2;
    }

    auto runTime = mOpenCLBackend->getOpenCLRuntime();

    int maxWidth  = runTime->getMaxImage2DSize()[0];
    int maxHeight = runTime->getMaxImage2DSize()[1];

    int sourceWidth  = UP_DIV(input->channel(), 4) * 4;
    int sourceHeight = alpha * alpha * UP_DIV(wUnit * hUnit, 4);

    int sliceNumber    = 1;
    const int maxSlice = 100;

    if (maxWidth < sourceWidth || maxHeight < sourceHeight) {
        for (int i = 2; i < maxSlice; ++i) {
            int realWidth  = (size_t)UP_DIV(input->channel(), 4) * 4;
            int readHeight = (size_t)alpha * alpha * UP_DIV(UP_DIV(wUnit, i) * UP_DIV(hUnit, i), 4);

            if (realWidth < maxWidth && readHeight < maxHeight) {
                sliceNumber = i;
                break;
            }
        }
    }

    mSliceNumber = sliceNumber;

    int wPiece = UP_DIV(wUnit, sliceNumber);
    int hPiece = UP_DIV(hUnit, sliceNumber);

    auto bn = backend();
    mSource.reset(Tensor::createDevice<float>(
        std::vector<int>{alpha * alpha, input->channel(), UP_DIV(wPiece * hPiece, 4), 4}, Tensor::CAFFE_C4));
    mDest.reset(Tensor::createDevice<float>(
        std::vector<int>{4, wPiece * hPiece, UP_DIV(output->channel(), 4), alpha * alpha}, Tensor::CAFFE_C4));

    bn->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
    bn->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

    auto icC4 = UP_DIV(input->channel(), 4);
    auto ocC4 = UP_DIV(output->channel(), 4);
    
    uint32_t total_num = input->batch()*mSliceNumber*mSliceNumber;
    mSourceTransform.resize(total_num);
    mMatMul.resize(total_num);
    mDestTransform.resize(total_num);
    mMaxWGS_S.resize(total_num);
    mMaxWGS_D.resize(total_num);
    mMaxWGS_M.resize(total_num);
    
    std::set<std::string> basic;
    /*Create Kernel*/
    for(int i = 0; i < input->batch()*mSliceNumber*mSliceNumber; i++) {
        char format[20];
        ::memset(format, 0, sizeof(format));
        sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
        auto formatStr = std::string(format);
        mSourceTransform[i] =
            runTime->buildKernel("winogradTransformSource" + formatStr,
                                 "winogradTransformSource", basic);
        mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mSourceTransform[i]));
        {
            std::set<std::string> buildOptions = basic;
            if (mCommon->relu()) {
                buildOptions.emplace("-DRELU");
            }
            if (mCommon->relu6()) {
                buildOptions.emplace("-DRELU6");
            }
            mDestTransform[i] =
                runTime->buildKernel("winogradTransformDest" + formatStr,
                                     "winogradTransformDest", buildOptions);
            mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mDestTransform[i]));
        }
        mMatMul[i] = runTime->buildKernel("gemm", "gemm", basic);
        mMaxWGS_M[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mMatMul[i]));
    }
    
    mGWS_S.resize(total_num);
    mGWS_D.resize(total_num);
    mGWS_M.resize(total_num);
    mLWS_S.resize(total_num);
    mLWS_D.resize(total_num);
    mLWS_M.resize(total_num);

    for (int b = 0; b < input->batch(); ++b) {
        std::vector<int> offsetData;
        offsetData.push_back(0);
        offsetData.push_back(0);

        for (int y = 0; y < mSliceNumber; ++y) {
            int hCount = hPiece;
            if (y == mSliceNumber - 1) {
                hCount = hUnit - (mSliceNumber - 1) * hPiece;
            }
            offsetData[1] = y * hPiece;

            for (int x = 0; x < mSliceNumber; ++x) {
                int wCount = wPiece;
                if (x == mSliceNumber - 1) {
                    wCount = wUnit - (mSliceNumber - 1) * wPiece;
                }
                offsetData[0] = x * wPiece;

                auto dest = mDest.get();
                int index = b*mSliceNumber*mSliceNumber + y*mSliceNumber + x;

                mSourceTransform[index].setArg(0, openCLImage(input));
                mSourceTransform[index].setArg(1, openCLImage(mSource.get()));
                mSourceTransform[index].setArg(4, padX);
                mSourceTransform[index].setArg(5, padY);
                mSourceTransform[index].setArg(6, input->width());
                mSourceTransform[index].setArg(7, input->height());
                mSourceTransform[index].setArg(8, icC4);

                mMatMul[index].setArg(0, openCLImage(mSource.get()));
                mMatMul[index].setArg(1, *mWeight);
                mMatMul[index].setArg(4, ocC4);
                mMatMul[index].setArg(5, icC4);
                mMatMul[index].setArg(6, alpha*alpha);

                mDestTransform[index].setArg(1, *mBias);
                mDestTransform[index].setArg(2, openCLImage(output));
                mDestTransform[index].setArg(5, output->width());
                mDestTransform[index].setArg(6, output->height());
                mDestTransform[index].setArg(7, ocC4);
                
                
                mSourceTransform[index].setArg(2, wCount);
                mSourceTransform[index].setArg(3, hCount);
                mSourceTransform[index].setArg(9, offsetData[0]);
                mSourceTransform[index].setArg(10, offsetData[1]);
                mSourceTransform[index].setArg(11, b);

                auto gemmWidth = UP_DIV(wCount * hCount, 4);
                mMatMul[index].setArg(2, openCLImage(dest));
                mMatMul[index].setArg(3, gemmWidth);

                mDestTransform[index].setArg(0, openCLImage(dest));
                mDestTransform[index].setArg(3, wCount);
                mDestTransform[index].setArg(4, hCount);
                mDestTransform[index].setArg(8, offsetData[0]);
                mDestTransform[index].setArg(9, offsetData[1]);
                mDestTransform[index].setArg(10, b);

                /*Source Transform*/
                {
                    mGWS_S[index] = {static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(icC4)};
                    mLWS_S[index] = getLocalWS("winogradTransformSource", index,
                                               mGWS_S[index], mMaxWGS_S[index], mSourceTransform[index]);
                }

                /*MatMul*/
                {
                    auto gemmHeight = ocC4;
                    mGWS_M[index] = {static_cast<uint32_t>(gemmWidth*gemmHeight), static_cast<uint32_t>(alpha * alpha)};
                    mLWS_M[index] = getLocalWS("gemm", index,
                                               mGWS_M[index], mMaxWGS_M[index], mMatMul[index]);
                }

                // Dest Transform
                {
                    mGWS_D[index] = {static_cast<uint32_t>(wCount*hCount), static_cast<uint32_t>(ocC4)};
                    mLWS_D[index] = getLocalWS("winogradTransformDest", index,
                                               mGWS_D[index], mMaxWGS_D[index], mDestTransform[index]);
                }
            }
        }
    }
    return NO_ERROR;
}
#else
void ConvWinograd::select_block_size(int                 split_count,
	                                 int                 compute_height,
	                                 int                 input_channel_q,
	                                 int                 output_channel_q,
	                                 int                 max_group_size,
	                                 int                 max_local_mem_size,
	                                 int                 gpu_type,
	                                 ORION_SELECT_INFO&  select_info)
{
	int required_local_mem = 0;

	if (GpuType::OTHER != gpu_type)
		required_local_mem = 8 * 8 * 4 * sizeof(float);
	else
		required_local_mem = 8 * (8 + 1) * 4 * sizeof(float); //this is nvidia

	if ((required_local_mem <= max_local_mem_size) &&
		(max_group_size >= 64) &&
		(0 == (input_channel_q & 0x1)) &&
		(0 == (output_channel_q & 0x7)) &&
		(0 == (compute_height & 0x7)))
	{
		select_info.select        = true;
		select_info.slice_count   = split_count;
        select_info.kernel_type   = 0;
		select_info.block_size_x  = 8;
        select_info.block_size_y  = 8;
	}
}

int ConvWinograd::find_aligned_height(int w, int h, int cur_split_count, int aligned_magic, int min_height, int max_try_count)
{
	bool found     = false;
	int  try_count = 0;
	while (try_count < max_try_count)
	{
		try_count++;
		int compute_height = UP_DIV(UP_DIV(w, (cur_split_count + try_count)) * UP_DIV(h, (cur_split_count + try_count)), 4);
		if ((compute_height >= min_height) && (0 == (compute_height & aligned_magic)))
		{
			found = true;
			return try_count;
		}
	}
	return 0;
}

ConvWinograd::ORION_SELECT_INFO ConvWinograd::select_orion_adpative_height(int    wUnit,
	                                                                       int    hUnit, 
	                                                                       int    max_group_size,
	                                                                       int    max_local_mem_size,
	                                                                       int    cur_mnn_select,
	                                                                       int    input_channel,
	                                                                       int    output_channel,
	                                                                       int    gpu_type)
{
	ORION_SELECT_INFO select_info;
	memset(&select_info, 0, sizeof(ORION_SELECT_INFO));
	int   input_channel_q    = UP_DIV(input_channel, 4);
	int   output_channel_q   = UP_DIV(output_channel, 4);
	int   required_local_mem = 0;
    int   compute_height     = UP_DIV(UP_DIV(wUnit, cur_mnn_select) * UP_DIV(hUnit, cur_mnn_select), 4);
	if ((0 != (input_channel_q & 0x1)) || (0 != (output_channel_q & 0x7)))
        return select_info;

	const int  kMinComputeHeight  = 32;
	const int  kMaxTryCount       = 3;
	int        orion_select       = cur_mnn_select;

	int  try_count = 0;
	if (0 != (compute_height & 0xF)) //we should find 16x, if compute_height is not 16x
		try_count = find_aligned_height(wUnit, hUnit, cur_mnn_select, 0xF, kMinComputeHeight, kMaxTryCount);
    else
    {
		select_block_size(cur_mnn_select,
			              compute_height,
			              input_channel_q,
			              output_channel_q,
			              max_group_size,
			              max_local_mem_size,
			              gpu_type,
			              select_info);
        return select_info;
    }

	if (try_count > 0)
	{
		compute_height = UP_DIV(UP_DIV(wUnit, cur_mnn_select + try_count) * UP_DIV(hUnit, cur_mnn_select + try_count), 4);
		select_block_size(cur_mnn_select + try_count,
			              compute_height,
			              input_channel_q,
			              output_channel_q,
			              max_group_size,
			              max_local_mem_size,
			              gpu_type,
			              select_info);
	}
    else
    {
        try_count = 0;
        if (0 == (compute_height & 0x7))
        {
            select_block_size(cur_mnn_select + try_count,
                              compute_height,
                              input_channel_q,
                              output_channel_q,
                              max_group_size,
                              max_local_mem_size,
                              gpu_type,
                              select_info);
        }
        else
        {
            if (0 != (compute_height & 0x7)) //we should find 8x, if compute_height is not 8x
                try_count = find_aligned_height(wUnit, hUnit, cur_mnn_select, 0x7, kMinComputeHeight, kMaxTryCount);
            
            if (try_count > 0)
            {
                compute_height = UP_DIV(UP_DIV(wUnit, cur_mnn_select + try_count) * UP_DIV(hUnit, cur_mnn_select + try_count), 4);
                select_block_size(cur_mnn_select + try_count,
                                compute_height,
                                input_channel_q,
                                output_channel_q,
                                max_group_size,
                                max_local_mem_size,
                                gpu_type,
                                select_info);
            }
        }
    }

	return select_info;
}

int ConvWinograd::select_opt_16x_kernel(cl::Kernel&   k_8x, 
                                        cl::Kernel&   k_16x, 
                                        int           gemm_width, 
                                        int           gemm_height, 
	                                    int           compute_height,
	                                    int           input_channel_q,
	                                    int           output_channel_q,
                                        int           alpha)
{
    cl::Event cost_event_8x;
    cl::Event cost_event_16x;
    int       cost_time_8x  = 0;
    int       cost_time_16x = 0;
    uint32_t  global_work_size0 = (uint32_t)gemm_width;
    uint32_t  global_work_size1 = (uint32_t)gemm_height;
    uint32_t  global_work_size2 = (uint32_t)(alpha * alpha);

    if((0 != (compute_height & 0xF)) ||
       (0 != (input_channel_q & 0xF)) ||
       (0 != (output_channel_q & 0xF)))
       return 0;

    cl_int error_8x = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(k_8x, cl::NullRange,
                                                                                             cl::NDRange(global_work_size0, global_work_size1, global_work_size2),
                                                                                             cl::NDRange(8, 8, 1),
                                                                                             nullptr, &cost_event_8x);
    MNN_CHECK_CL_SUCCESS(error_8x);
    if(0 == error_8x)
    {
        cost_time_8x = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&cost_event_8x);
    }

    cl_int error_16x = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(k_16x, cl::NullRange,
                                                                                             cl::NDRange(global_work_size0, global_work_size1, global_work_size2),
                                                                                             cl::NDRange(16, 16, 1),
                                                                                             nullptr, &cost_event_16x);
    MNN_CHECK_CL_SUCCESS(error_16x);
    if(0 == error_16x)
    {
        cost_time_16x = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&cost_event_16x);
    }

    //printf("cost_time_8x: %d, cost_time_16x: %d\n", cost_time_8x, cost_time_16x);

    if(cost_time_16x < cost_time_8x)
        return 1;
    
    return 0;
}

ErrorCode ConvWinograd::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
	auto input = inputs[0];
	auto output = outputs[0];
	mKernelX = mCommon->kernelX();
	mKernelY = mCommon->kernelY();
	mPadX = mCommon->padX();
	mPadY = mCommon->padY();
	mStrideX = mCommon->strideX();
	mStrideY = mCommon->strideY();
//Shaquille, Added 20210220 Start
#ifdef ORION_OPTIMIZE
    int32_t leaky_relu_int = (int32_t)((((uint32_t)mStrideY) & 0xFFFF0000) >> 16);
    mStrideY               = (mStrideY & 0x0000FFFF);
    leaky_relu_            = (float)(((double)(leaky_relu_int)) * 0.001);
#endif
//Shaquille, Added 20210220 End
	mPadMode = mCommon->padMode();

	int alpha = mCommon->kernelX() + UNIT - 1;
	auto wUnit = UP_DIV(output->width(), UNIT);
	auto hUnit = UP_DIV(output->height(), UNIT);
	int padX = mPadX;
	int padY = mPadY;
	if (mPadMode == PadMode_SAME) {
		int kernelWidthSize = (mKernelX - 1) * mCommon->dilateX() + 1;
		int kernelHeightSize = (mKernelY - 1) * mCommon->dilateY() + 1;
		int padNeededWidth = (output->width() - 1) * mStrideX + kernelWidthSize - input->width();
		int padNeededHeight = (output->height() - 1) * mStrideY + kernelHeightSize - input->height();
		padX = padNeededWidth / 2;
		padY = padNeededHeight / 2;
	}

	auto runTime = mOpenCLBackend->getOpenCLRuntime();

	int maxWidth = runTime->getMaxImage2DSize()[0];
	int maxHeight = runTime->getMaxImage2DSize()[1];

	int sourceWidth = UP_DIV(input->channel(), 4) * 4;
	int sourceHeight = alpha * alpha * UP_DIV(wUnit * hUnit, 4);

	int sliceNumber = 1;
	const int maxSlice = 100;

	if (maxWidth < sourceWidth || maxHeight < sourceHeight) {
		for (int i = 2; i < maxSlice; ++i) {
			int realWidth = (size_t)UP_DIV(input->channel(), 4) * 4;
			int readHeight = (size_t)alpha * alpha * UP_DIV(UP_DIV(wUnit, i) * UP_DIV(hUnit, i), 4);

			if (realWidth < maxWidth && readHeight < maxHeight) {
				sliceNumber = i;
				break;
			}
		}
	}
	mSliceNumber = sliceNumber;
	int               gpu_type       = mOpenCLBackend->getOpenCLRuntime()->getGpuType();
	int               max_group_size = mOpenCLBackend->getOpenCLRuntime()->getDevMaxWorkGroupSize();
	int               local_mem_size = (int)(mOpenCLBackend->getOpenCLRuntime()->getMaxLocalMem());
	ORION_SELECT_INFO slice_opt_info = select_orion_adpative_height(wUnit,
		                                                            hUnit, 
		                                                            max_group_size,
		                                                            local_mem_size,
		                                                            sliceNumber, 
		                                                            input->channel(), 
		                                                            output->channel(), 
		                                                            gpu_type);
	if (true == slice_opt_info.select)
		sliceNumber = slice_opt_info.slice_count;

	mSliceNumber = sliceNumber;
	int wPiece   = UP_DIV(wUnit, sliceNumber);
	int hPiece   = UP_DIV(hUnit, sliceNumber);

	auto bn = backend();
	mSource.reset(Tensor::createDevice<float>(
		std::vector<int>{alpha * alpha, input->channel(), UP_DIV(wPiece * hPiece, 4), 4}, Tensor::CAFFE_C4));
	mDest.reset(Tensor::createDevice<float>(
		std::vector<int>{4, wPiece * hPiece, UP_DIV(output->channel(), 4), alpha * alpha}, Tensor::CAFFE_C4));

	bn->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
	bn->onAcquireBuffer(mDest.get(), Backend::DYNAMIC);
	bn->onReleaseBuffer(mSource.get(), Backend::DYNAMIC);
	bn->onReleaseBuffer(mDest.get(), Backend::DYNAMIC);

	auto icC4 = UP_DIV(input->channel(), 4);
	auto ocC4 = UP_DIV(output->channel(), 4);

	//printf("wUnit: %d, hUnit: %d, sliceNumber: %d, wPiece: %d, hPiece: %d, (wPiece*hPiece): %d, icC4: %d, ocC4: %d, alpha2: %d\n", wUnit, hUnit, sliceNumber, wPiece, hPiece, wPiece*hPiece, icC4, ocC4, alpha * alpha);
    //printf("select: %d, %d, %d, %d\n", slice_opt_info.select, slice_opt_info.kernel_type, slice_opt_info.block_size_x, slice_opt_info.block_size_y);
	uint32_t total_num = input->batch()*mSliceNumber*mSliceNumber;
	mSourceTransform.resize(total_num);
	mMatMul.resize(total_num);
	mDestTransform.resize(total_num);
	mMaxWGS_S.resize(total_num);
	mMaxWGS_D.resize(total_num);
	mMaxWGS_M.resize(total_num);

    mMaxWGS_M_orion.resize(total_num);
    mMatMul_orion.resize(total_num);
	mMatMulOptInfo.resize(total_num);

	std::set<std::string> basic;
	/*Create Kernel*/
	for (int i = 0; i < input->batch()*mSliceNumber*mSliceNumber; i++) {
		char format[20];
		::memset(format, 0, sizeof(format));
		sprintf(format, "%d_%d_%d", UNIT, mKernelX, INTERP);
		auto formatStr = std::string(format);
		mSourceTransform[i] =
			runTime->buildKernel("winogradTransformSource" + formatStr,
				"winogradTransformSource", basic);
		mMaxWGS_S[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mSourceTransform[i]));
		{
			std::set<std::string> buildOptions = basic;
			if (mCommon->relu()) {
				buildOptions.emplace("-DRELU");
			}
			if (mCommon->relu6()) {
				buildOptions.emplace("-DRELU6");
			}
            if (0 != leaky_relu_int)
            {
                char leaky_relu_factor[64] = { 0 };
                sprintf(leaky_relu_factor, "-DLEAKY_RELU=%.12f", leaky_relu_);
                buildOptions.emplace(std::string(leaky_relu_factor));
            }
			mDestTransform[i] =
				runTime->buildKernel("winogradTransformDest" + formatStr,
					"winogradTransformDest", buildOptions);
			mMaxWGS_D[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mDestTransform[i]));
		}
		memset(&(mMatMulOptInfo[i]), 0, sizeof(ORION_SELECT_INFO));
/*
		mMatMul[i]   = runTime->buildKernel("gemm", "gemm", basic);
		mMaxWGS_M[i] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mMatMul[i]));
*/
	}

	mGWS_S.resize(total_num);
	mGWS_D.resize(total_num);
	mGWS_M.resize(total_num);
	mLWS_S.resize(total_num);
	mLWS_D.resize(total_num);
	mLWS_M.resize(total_num);
    mGWS_M_orion.resize(total_num);
    mLWS_M_orion.resize(total_num);

	for (int b = 0; b < input->batch(); ++b) {
		std::vector<int> offsetData;
		offsetData.push_back(0);
		offsetData.push_back(0);

		for (int y = 0; y < mSliceNumber; ++y) {
			int hCount = hPiece;
			if (y == mSliceNumber - 1) {
				hCount = hUnit - (mSliceNumber - 1) * hPiece;
			}
			offsetData[1] = y * hPiece;

			for (int x = 0; x < mSliceNumber; ++x) {
				int wCount = wPiece;
				if (x == mSliceNumber - 1) {
					wCount = wUnit - (mSliceNumber - 1) * wPiece;
				}
				offsetData[0] = x * wPiece;

				auto dest = mDest.get();
				int index = b * mSliceNumber*mSliceNumber + y * mSliceNumber + x;

				mSourceTransform[index].setArg(0, openCLImage(input));
				mSourceTransform[index].setArg(1, openCLImage(mSource.get()));
				mSourceTransform[index].setArg(4, padX);
				mSourceTransform[index].setArg(5, padY);
				mSourceTransform[index].setArg(6, input->width());
				mSourceTransform[index].setArg(7, input->height());
				mSourceTransform[index].setArg(8, icC4);

				int real_computation_height = UP_DIV((hCount * wCount), 4);
				ORION_SELECT_INFO  cur_opt_info;
				memset(&cur_opt_info, 0, sizeof(ORION_SELECT_INFO));
				select_block_size(mSliceNumber, 
					              real_computation_height, 
					              icC4, 
					              ocC4, 
					              max_group_size, 
					              local_mem_size, 
					              gpu_type,
					              cur_opt_info);
                if(false == cur_opt_info.select)
                {
                    if((0 == (icC4 & 1)) && (0 == (ocC4 & 7)))
                    {
                        cur_opt_info.select        = true;
                        cur_opt_info.kernel_type   = 1;
		                cur_opt_info.block_size_x  = 8;
                        cur_opt_info.block_size_y  = 8;
                    }
                }
				if (true == cur_opt_info.select)
				{
                    /*
					std::set<std::string>  opt_build_opt = basic;
					if(GpuType::OTHER == gpu_type)
						opt_build_opt.emplace("-DAVOID_MEM_BANK_CONFLICT");
                    switch(slice_opt_info.kernel_type)
                    {
                        case 0:
                            mMatMul_orion[index] = runTime->buildKernel("gemm_local_mem", "gemm_local_mem_8x", opt_build_opt);
                            break;
                        case 2:
                            mMatMul_orion[index] = runTime->buildKernel("gemm_local_mem", "gemm_local_mem_opt_4x16", opt_build_opt);
                            break;
                        default:
                            break;
                    }
					mMaxWGS_M_orion[index]   = slice_opt_info.block_size_x * slice_opt_info.block_size_y;
					mMatMul_orion[index].setArg(0, openCLImage(mSource.get()));
					mMatMul_orion[index].setArg(1, *mWeight);
					memcpy(&(mMatMulOptInfo[index]), &cur_opt_info, sizeof(ORION_SELECT_INFO));
                    */
				}
                else
                {
                    mMatMul[index]   = runTime->buildKernel("gemm", "gemm", basic);
                    mMaxWGS_M[index] = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mMatMul[index]));
                    mMatMul[index].setArg(0, openCLImage(mSource.get()));
                    mMatMul[index].setArg(1, *mWeight);
                    mMatMul[index].setArg(4, ocC4);
                    mMatMul[index].setArg(5, icC4);
                    mMatMul[index].setArg(6, alpha*alpha);
                }


				mDestTransform[index].setArg(1, *mBias);
				mDestTransform[index].setArg(2, openCLImage(output));
				mDestTransform[index].setArg(5, output->width());
				mDestTransform[index].setArg(6, output->height());
				mDestTransform[index].setArg(7, ocC4);

				mSourceTransform[index].setArg(2, wCount);
				mSourceTransform[index].setArg(3, hCount);
				mSourceTransform[index].setArg(9, offsetData[0]);
				mSourceTransform[index].setArg(10, offsetData[1]);
				mSourceTransform[index].setArg(11, b);

				auto gemmWidth = UP_DIV(wCount * hCount, 4);
				mMatMul[index].setArg(2, openCLImage(dest));
				if (true == cur_opt_info.select)
                {
                    ;
                }
				else
				    mMatMul[index].setArg(3, gemmWidth);

				mDestTransform[index].setArg(0, openCLImage(dest));
				mDestTransform[index].setArg(3, wCount);
				mDestTransform[index].setArg(4, hCount);
				mDestTransform[index].setArg(8, offsetData[0]);
				mDestTransform[index].setArg(9, offsetData[1]);
				mDestTransform[index].setArg(10, b);

				/*Source Transform*/
				{
					mGWS_S[index] = { static_cast<uint32_t>(wCount * hCount), static_cast<uint32_t>(icC4) };
					mLWS_S[index] = getLocalWS("winogradTransformSource", index,
						mGWS_S[index], mMaxWGS_S[index], mSourceTransform[index]);
				}

				/*MatMul*/
				{
					auto gemmHeight = ocC4;
					if (true == cur_opt_info.select)
					{
                        std::set<std::string>  opt_build_opt = basic;
                        cl::Kernel   k_8x;
                        int          dual_data_sel = DUAL_DATA_NONE;
                        if(GpuType::OTHER == gpu_type)
                            opt_build_opt.emplace("-DAVOID_MEM_BANK_CONFLICT");
                        switch(cur_opt_info.kernel_type)
                        {
                            case 0:
                                cur_opt_info.block_size_x = 8;
                                cur_opt_info.block_size_y = 8;
                                if(0 == (gemmWidth & 0xF))
                                {
                                    dual_data_sel        = DUAL_INPUT_DATA;
                                    mMatMul_orion[index] = runTime->buildKernel("gemm_local_mem", "gemm_local_mem_opt_8x8_dual_input", opt_build_opt);
                                }
                                else if(0 == (gemmHeight & 0xF))
                                {
                                    dual_data_sel        = DUAL_FILTER_DATA;
                                    mMatMul_orion[index] = runTime->buildKernel("gemm_local_mem", "gemm_local_mem_opt_8x_dual_filter", opt_build_opt);
                                }
                                else
                                {
                                    k_8x  = runTime->buildKernel("gemm_local_mem", "gemm_local_mem_8x", opt_build_opt);
                                    k_8x.setArg(0, openCLImage(mSource.get()));
                                    k_8x.setArg(1, *mWeight);
                                    k_8x.setArg(2, openCLImage(dest));
                                    k_8x.setArg(3, 4 * icC4);
                                    cur_opt_info.block_size_x = 8;
                                    cur_opt_info.block_size_y = 8;
                                    mMatMul_orion[index]      = k_8x;
                                }
                                break;
                            case 1:
                                opt_build_opt.emplace("-DCHECK_SRC_BORDER");
                                cur_opt_info.kernel_type  = 1;
                                cur_opt_info.block_size_x = 8;
                                cur_opt_info.block_size_y = 8;
                                dual_data_sel             = DUAL_INPUT_DATA_CHECK_SRC_BORDER;
                                {
                                    int  gemmWidth_8 = (((gemmWidth + 7) >> 3) << 3);
                                    int  delta       = gemmWidth_8 - gemmWidth;
                                    if(0 == (gemmHeight & 0xF))
                                        if((7 * delta) > gemmWidth)
                                            dual_data_sel = DUAL_FILTER_DATA_CHECK_SRC_BORDER;
                                }
                                if(DUAL_INPUT_DATA_CHECK_SRC_BORDER == dual_data_sel)
                                    mMatMul_orion[index]  = runTime->buildKernel("gemm_local_mem", "gemm_local_mem_opt_8x8_dual_input", opt_build_opt);
                                else
                                    mMatMul_orion[index]  = runTime->buildKernel("gemm_local_mem", "gemm_local_mem_opt_8x_dual_filter", opt_build_opt);
                                break;
                            default:
                                break;
                        }
                        mMatMul_orion[index].setArg(0, openCLImage(mSource.get()));
                        mMatMul_orion[index].setArg(1, *mWeight);
                        mMatMul_orion[index].setArg(2, openCLImage(dest));
                        if(DUAL_DATA_NONE == dual_data_sel)
                            mMatMul_orion[index].setArg(3, 4 * icC4);
                        else
                        {
                            mMatMul_orion[index].setArg(3, gemmWidth);
                            mMatMul_orion[index].setArg(4, 4 * icC4);
                        }
                        mMaxWGS_M_orion[index]   = cur_opt_info.block_size_x * cur_opt_info.block_size_y;
                        memcpy(&(mMatMulOptInfo[index]), &cur_opt_info, sizeof(ORION_SELECT_INFO));
						mGWS_M_orion[index] = { (uint32_t)gemmWidth,                            
                                                (uint32_t)gemmHeight,                           
                                                (uint32_t)(alpha * alpha) };
						if(DUAL_INPUT_DATA == dual_data_sel)
                            mGWS_M_orion[index][0] = (mGWS_M_orion[index][0] >> 1);
                        else if(DUAL_FILTER_DATA == dual_data_sel)
                            mGWS_M_orion[index][1] = (mGWS_M_orion[index][1] >> 1);
                        else if(DUAL_INPUT_DATA_CHECK_SRC_BORDER == dual_data_sel)
                        {
                            mGWS_M_orion[index][0] = (((mGWS_M_orion[index][0] + 0xF) >> 4) << 3);
                        }
                        else if(DUAL_FILTER_DATA_CHECK_SRC_BORDER == dual_data_sel)
                        {
                            mGWS_M_orion[index][0] = (((mGWS_M_orion[index][0] + 7) >> 3) << 3);
                            mGWS_M_orion[index][1] = (mGWS_M_orion[index][1] >> 1);
                        }
                        mLWS_M_orion[index] = { (uint32_t)(mMatMulOptInfo[index].block_size_x), (uint32_t)(mMatMulOptInfo[index].block_size_y), 1 };
					}
					else
					{
						mGWS_M[index] = { static_cast<uint32_t>(gemmWidth*gemmHeight), static_cast<uint32_t>(alpha * alpha) };
						mLWS_M[index] = getLocalWS("gemm", index,
							mGWS_M[index], mMaxWGS_M[index], mMatMul[index]);
					}
				}

				// Dest Transform
				{
					mGWS_D[index] = { static_cast<uint32_t>(wCount*hCount), static_cast<uint32_t>(ocC4) };
					mLWS_D[index] = getLocalWS("winogradTransformDest", index,
						mGWS_D[index], mMaxWGS_D[index], mDestTransform[index]);
				}
			}
		}
	}

	return NO_ERROR;
}
#endif

std::vector<uint32_t> ConvWinograd::getLocalWS(std::string kernelName, int index, std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize, cl::Kernel mKernel) {

#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);

    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("ConvWinograd Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
//Shaquille, Modified 20201118 Start
#if 0
		return tunedLws[info];
#else
		return std::get<0>(tunedLws[info]);
#endif
//Shaquille, Modified 20201118 End
    }
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;
    while(lws[1] <= gws[1]) {
        lws[0] = 1;
        while(lws[0] <= gws[0]) {
            if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                cl::Event event;
                std::vector<uint32_t> internalGlobalWS(2, 1);
                for (size_t i = 0; i < gws.size(); ++i) {
                    internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                }
                cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                                mKernel, cl::NullRange,
                                cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                cl::NDRange(lws[0], lws[1]),
                                nullptr, &event);
                MNN_CHECK_CL_SUCCESS(error);
                int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                if(cost_time < min_cost) {
                    min_cost = cost_time;
                    lws_prefer[0] = lws[0];
                    lws_prefer[1] = lws[1];
                }
            }
            lws[0] *= 2;
        }
        lws[1] *= 2;
    }
    
    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("ConvWinograd %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
//Shaquille, Modified 20201118 Start
#if 0
		tunedLws.insert(std::make_pair(info, lws_prefer));
#else

		tunedLws.insert(std::make_pair(info, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>(lws_prefer, std::vector<uint32_t>({ (uint32_t)min_cost }))));
#endif
//Shaquille, Modified 20201118 End
    }
    
    return lws_prefer;
#else
    uint32_t cu = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();
    int waveSize = 16; //could be 8, 16, 32, 64, 128 in Adreno GPU
    std::vector<uint32_t> lws(4, 0);

    int coreNum   = cu*2;
    int groupSize = ROUND_UP(gws[0] / coreNum, waveSize);

    lws[0] = groupSize;
    lws[0] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize, lws[0]), 1);

    int remain = ((maxWorkGroupSize - lws[0]) / waveSize) * waveSize;
    groupSize = ROUND_UP(gws[1] / coreNum, waveSize);
    lws[1] = groupSize;
    lws[1] = std::max<uint32_t>(std::min<uint32_t>(remain / lws[0], lws[1]), 1);
    return lws;
#endif
}

ErrorCode ConvWinograd::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    #ifdef ENABLE_OPENCL_TIME_PROFILER
    int costTime = 0;
    #endif
    for (int b = 0; b < input->batch(); ++b) {
        for (int y = 0; y < mSliceNumber; ++y) {
            for (int x = 0; x < mSliceNumber; ++x) {
                int index = b*mSliceNumber*mSliceNumber + y*mSliceNumber + x;

                /*Source Transform*/
                {
                #ifdef ENABLE_OPENCL_TIME_PROFILER
                    cl::Event event;
                    runKernel2D(mSourceTransform[index], mGWS_S[index], mLWS_S[index],
                                mOpenCLBackend->getOpenCLRuntime(), &event);
                    
                    int costTime0 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    costTime += costTime0;
                    MNN_PRINT("kernel cost:%d    us ConvWino0\n",costTime0);
                #else
                    runKernel2D(mSourceTransform[index], mGWS_S[index], mLWS_S[index],
                                mOpenCLBackend->getOpenCLRuntime());
                #endif
                }

#if 1
                /*MatMul*/
                {
                #ifdef ENABLE_OPENCL_TIME_PROFILER
                    cl::Event event;
#ifdef ORION_OPTIMIZE
					if (true == mMatMulOptInfo[index].select)
					{
						run3DKernelDefault(mMatMul_orion[index], mGWS_M_orion[index], mLWS_M_orion[index],
							               mOpenCLBackend->getOpenCLRuntime(), &event);
					}
					else
#endif
                    {
                        runKernel2D(mMatMul[index], mGWS_M[index], mLWS_M[index],
                                    mOpenCLBackend->getOpenCLRuntime(), &event);
                    }
                    
                    int costTime1 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    costTime += costTime1;
                    MNN_PRINT("kernel cost:%d    us ConvWino1\n",costTime1);
                #else
#ifdef ORION_OPTIMIZE
					if (true == mMatMulOptInfo[index].select)
					{
                        run3DKernelDefault(mMatMul_orion[index], mGWS_M_orion[index], mLWS_M_orion[index],
                                        mOpenCLBackend->getOpenCLRuntime());
					}
					else
#endif
                    {
                        runKernel2D(mMatMul[index], mGWS_M[index], mLWS_M[index],
                                    mOpenCLBackend->getOpenCLRuntime());
                    }
                #endif
                }
#endif
                // Dest Transform
                {
                #ifdef ENABLE_OPENCL_TIME_PROFILER
                    cl::Event event;
                    runKernel2D(mDestTransform[index], mGWS_D[index], mLWS_D[index],
                                mOpenCLBackend->getOpenCLRuntime(), &event);
                    
                    int costTime2 = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                    costTime += costTime2;
                    MNN_PRINT("kernel cost:%d    us ConvWino2\n",costTime2);
                #else
                    runKernel2D(mDestTransform[index], mGWS_D[index], mLWS_D[index],
                                mOpenCLBackend->getOpenCLRuntime());
                #endif
                }
            }
        }
    }
    #ifdef ENABLE_OPENCL_TIME_PROFILER
    MNN_PRINT("kernel cost:%d    us ConvWino total\n",costTime);
    #endif

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN
