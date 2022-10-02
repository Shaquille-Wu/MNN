//
//  RasterExecution.cpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/RasterExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

std::vector<uint32_t> RasterExecution::rasterLocalWorkSize(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize, std::string &kernelName, cl::Kernel &mKernel) {
    
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);
    
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2d1x1LocalWSOpt Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
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

    while(lws[1] <= gws[1]*2 || lws[1] <= 4) {
        lws[0] = 1;
        while(lws[0] <= gws[0]*2  || lws[0] <= 4) {
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
        //printf("conv2d1x1LocalWSOpt %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
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
    std::vector<uint32_t> lws(4, 1);
    return lws;
#endif
}

RasterExecution::RasterExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend) {
    mOpenCLBackend = (OpenCLBackend *)backend;
    mOp = op;
    //nothing to do
}

ErrorCode RasterExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RasterExecution onResize !\n");
#endif
    mTempInput.clear();
    mTempOutput = nullptr;
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input     = inputs[0];
    auto output    = outputs[0];
    auto des       = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero      = !TensorUtils::regionIsFull(input);
    auto regionNum = des->regions.size();
    auto runtime   = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    
//Shaquille, Added 20201117 Start
	src_offfset_.resize(0);
	dst_offfset_.resize(0);
	copy_size_.resize(0);
	input_tensor_.resize(0);
	output_tensor_.resize(0);
	apply_img_cpy_         = false;
	bool apply_img_cpy_all = false;
//Shaquille, Added 20201117 End

    mFast = false;
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output)) {
                mFast = false;
                break;
            }
        }
    }

    if(mFast)
    {
//Shaquille, Added 20201117 Start
		if (false == mNeedZero)
		{
			auto outputShape = tensorShapeFormat(output);
			int  slice_idx   = 0;
			for (slice_idx = 0; slice_idx < (int)(des->regions.size()); slice_idx++)
			{
				auto& slice     = des->regions[slice_idx];
				auto sliceShape = tensorShapeFormat(slice.origin);
				if (1 != slice.size[0] ||
                    output->buffer().dimensions != slice.origin->buffer().dimensions ||
					output->buffer().dimensions != 4 ||
					sliceShape[0] != outputShape[0] ||   //N
					sliceShape[1] != outputShape[1] ||   //H
					sliceShape[2] != outputShape[2])     //W
					break;
			}
			if (slice_idx >= (int)(des->regions.size()))
			{
				if (1 == (int)(des->regions.size()))
				{
					auto& slice     = des->regions[0];
					auto sliceShape = tensorShapeFormat(slice.origin);
					if (sliceShape[3] == outputShape[3])
						apply_img_cpy_all = true;
				}
				apply_img_cpy_ = true;
			}
		}
//Shaquille, Added 20201117 End

        mUnits.resize(regionNum);
        int kernel_idx = 0;
        
        if(mNeedZero)
        {
            mUnits.resize(regionNum + 1);
            auto outputShape    = tensorShapeFormat(output);
            int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};//nhwc
            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster", "image_set_zero", {});
            unit.localWorkSize  = {16, 16};
            unit.globalWorkSize = {(uint32_t)UP_DIV((region[1] * region[3]), 16)*16,
                                   (uint32_t)UP_DIV((region[0] * region[2]), 16)*16};

            int global_dim0 = region[1] * region[3];
            int global_dim1 = region[0] * region[2];

            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel.setArg(idx++, global_dim0);
            ret |= unit.kernel.setArg(idx++, global_dim1);
            ret |= unit.kernel.setArg(idx++, openCLImage(output));
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
        }
        
//Shaquille, Modified 20201117 Start
#if 0
        // image raster
        for (auto& slice : des->regions)
        {
            Tensor::InsideDescribe::Region C4Region;
            OpCommonUtils::turnToPackRegion(slice, C4Region, output, 4);

            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster", "raster_image", {});

            const std::vector<uint32_t> gws =  {(uint32_t)C4Region.size[2],
                                                    (uint32_t)C4Region.size[1],
                                                    (uint32_t)C4Region.size[0]};
            uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

            auto outputShape    = tensorShapeFormat(output);
            auto sliceShape    = tensorShapeFormat(slice.origin);

            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel.setArg(idx++, gws[0]);
            ret |= unit.kernel.setArg(idx++, gws[1]);
            ret |= unit.kernel.setArg(idx++, gws[2]);
            ret |= unit.kernel.setArg(idx++, openCLImage(slice.origin));
            ret |= unit.kernel.setArg(idx++, C4Region.src.offset);
            ret |= unit.kernel.setArg(idx++, C4Region.src.stride[0]);
            ret |= unit.kernel.setArg(idx++, C4Region.src.stride[1]);
            ret |= unit.kernel.setArg(idx++, C4Region.src.stride[2]);
            ret |= unit.kernel.setArg(idx++, sliceShape[1]);
            ret |= unit.kernel.setArg(idx++, sliceShape[2]);
            ret |= unit.kernel.setArg(idx++, sliceShape[3]);
            ret |= unit.kernel.setArg(idx++, openCLImage(output));
            ret |= unit.kernel.setArg(idx++, C4Region.dst.offset);
            ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[0]);
            ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[1]);
            ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[2]);
            ret |= unit.kernel.setArg(idx++, outputShape[1]);
            ret |= unit.kernel.setArg(idx++, outputShape[2]);
            ret |= unit.kernel.setArg(idx++, outputShape[3]);
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
            std::string name = "rasterImage";
            const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel);
            
            unit.localWorkSize = {lws[0], lws[1], lws[2]};
            
            unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                                   ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                                   ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
        }
#else
// image raster
		if (false == apply_img_cpy_)
		{
			for (auto& slice : des->regions)
			{
				Tensor::InsideDescribe::Region C4Region;
				OpCommonUtils::turnToPackRegion(slice, C4Region, output, 4);

				Unit &unit = mUnits[kernel_idx++];
				unit.kernel = runtime->buildKernel("raster", "raster_image", {});

				const std::vector<uint32_t> gws = { (uint32_t)C4Region.size[2],
					                                (uint32_t)C4Region.size[1],
					                                (uint32_t)C4Region.size[0] };
				uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

				auto outputShape = tensorShapeFormat(output);
				auto sliceShape  = tensorShapeFormat(slice.origin);

				uint32_t idx = 0;
				cl_int ret   = CL_SUCCESS;
				ret |= unit.kernel.setArg(idx++, gws[0]);
				ret |= unit.kernel.setArg(idx++, gws[1]);
				ret |= unit.kernel.setArg(idx++, gws[2]);
				ret |= unit.kernel.setArg(idx++, openCLImage(slice.origin));
				ret |= unit.kernel.setArg(idx++, C4Region.src.offset);
				ret |= unit.kernel.setArg(idx++, C4Region.src.stride[0]);
				ret |= unit.kernel.setArg(idx++, C4Region.src.stride[1]);
				ret |= unit.kernel.setArg(idx++, C4Region.src.stride[2]);
				ret |= unit.kernel.setArg(idx++, sliceShape[1]);
				ret |= unit.kernel.setArg(idx++, sliceShape[2]);
				ret |= unit.kernel.setArg(idx++, sliceShape[3]);
				ret |= unit.kernel.setArg(idx++, openCLImage(output));
				ret |= unit.kernel.setArg(idx++, C4Region.dst.offset);
				ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[0]);
				ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[1]);
				ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[2]);
				ret |= unit.kernel.setArg(idx++, outputShape[1]);
				ret |= unit.kernel.setArg(idx++, outputShape[2]);
				ret |= unit.kernel.setArg(idx++, outputShape[3]);
				if (ret != CL_SUCCESS)
				{
					MNN_PRINT("setArg err %d\n", (int)ret);
				}
				std::string name = "rasterImage";
				const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel);

				unit.localWorkSize = { lws[0], lws[1], lws[2] };

				unit.globalWorkSize = { ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
					                    ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
					                    ROUND_UP(gws[2], std::max((uint32_t)1, lws[2])) };
			}
		}
		else
		{
			src_offfset_.resize(regionNum, 0);
			dst_offfset_.resize(regionNum, 0);
			copy_size_.resize(regionNum, { 0, 0 });
			input_tensor_.resize(regionNum, nullptr);
			output_tensor_.resize(regionNum, nullptr);
			int  slice_idx = 0;
			for (slice_idx = 0; slice_idx < (int)(des->regions.size()) ; slice_idx ++)
			{
				auto& slice = des->regions[slice_idx];
				Tensor::InsideDescribe::Region C4Region;
				OpCommonUtils::turnToPackRegion(slice, C4Region, output, 4);

				Unit&                       unit  = mUnits[kernel_idx++];
				const std::vector<uint32_t> gws   = { (uint32_t)C4Region.size[2],
					                                  (uint32_t)C4Region.size[1],
					                                  (uint32_t)C4Region.size[0] };
				uint32_t mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getDevMaxWorkGroupSize());

				auto outputShape = tensorShapeFormat(output);
				auto sliceShape  = tensorShapeFormat(slice.origin);

				if (true == apply_img_cpy_all)
				{
					src_offfset_[slice_idx] = 0;
					dst_offfset_[slice_idx] = 0;
					copy_size_[slice_idx]   = { (size_t)(sliceShape[2] * ((sliceShape[3] + 3) >> 2)),
						                        (size_t)(sliceShape[0] * sliceShape[1]) };
				}
				else
				{
					src_offfset_[slice_idx] = sliceShape[2] * ((C4Region.src.offset / (sliceShape[1] * sliceShape[2])) >> 2);
					dst_offfset_[slice_idx] = outputShape[2] * ((C4Region.dst.offset / (outputShape[1] * outputShape[2])) >> 2);
					copy_size_[slice_idx]   = { (size_t)(sliceShape[2] * C4Region.size[1]),
						                        (size_t)(sliceShape[0] * sliceShape[1]) };
				}

				input_tensor_[slice_idx]  = slice.origin;
				output_tensor_[slice_idx] = output;

				unit.localWorkSize  = { 8, 8, 1 };
				unit.globalWorkSize = { ROUND_UP(gws[0], 8),
					                    ROUND_UP(gws[1], 8),
					                    ROUND_UP(gws[2], 8) };
			}
		}
#endif
//Shaquille, Modified 20201117 End

        if(mNeedZero)
        {
            MNN_ASSERT((regionNum+1==kernel_idx));
        }
        else
        {
            MNN_ASSERT((regionNum==kernel_idx));
        }
        return NO_ERROR;
    }

    // Alloc Temp buffer
    auto bufferPool     = ((OpenCLBackend *)backend())->getBufferPool();
    auto bufferUnitSize = runtime->isSupportedFP16() ? sizeof(half_float::half) : sizeof(float);
    for(int i=0; i< regionNum; ++i)
    {
        auto origin = des->regions[i].origin;
        if(mTempInput.find(origin) != mTempInput.end())
        {
            continue;
        }

        auto buffer = bufferPool->alloc(origin->elementSize()*bufferUnitSize);
        mTempInput.insert(std::make_pair(origin, buffer));
    }
    mTempOutput         = bufferPool->alloc(output->elementSize() * bufferUnitSize);

    for(auto& iter : mTempInput)
    {
        bufferPool->recycle(iter.second);
    }
    bufferPool->recycle(mTempOutput);
    
    auto originNum = mTempInput.size();
    mUnits.resize(regionNum + originNum + 1);
    
    int kernel_idx = 0;
    if(mNeedZero)
    {
        mUnits.resize(regionNum + originNum + 2);
        auto outputShape    = tensorShapeFormat(output);
        int region[] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//nhwc
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "buffer_set_zero", {});

        std::vector<uint32_t> gws = {(uint32_t)(region[2] * region[3]),
                                     (uint32_t)(region[0] * region[1])};
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, gws[0]);
        ret |= unit.kernel.setArg(idx++, gws[1]);
        ret |= unit.kernel.setArg(idx++, *mTempOutput);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_buffer_set_zero";
        std::vector<uint32_t> lws = rasterLocalWorkSize(gws, mMaxWorkGroupSize, kernelName, unit.kernel);
        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
    }

    //image to buffer
    for(auto& iter : mTempInput)
    {
        Tensor* origin = iter.first;
        std::vector<int> regionShape = tensorShapeFormat(origin);
        int inputWH[]      = {regionShape[2], regionShape[1]};
        int region[]       = {regionShape[0], UP_DIV(regionShape[3], 4), regionShape[1], regionShape[2]};
                
        Unit &unit          = mUnits[kernel_idx++];
        if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC)// Image to nhwc buffer
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "image_to_nhwc_buffer", {});
        }
        else //Image to nchw buffer
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "image_to_nchw_buffer", {});
        }

        std::vector<uint32_t> gws = {(uint32_t)(region[3] * region[1]),
                                     (uint32_t)(region[2] * region[0])};
        //MNN_CHECK_CL_SUCCESS
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, gws[0]);
        ret |= unit.kernel.setArg(idx++, gws[1]);
        ret |= unit.kernel.setArg(idx++, *(iter.second));
        ret |= unit.kernel.setArg(idx++, inputWH[1]);
        ret |= unit.kernel.setArg(idx++, inputWH[0]);
        ret |= unit.kernel.setArg(idx++, regionShape[3]);
        ret |= unit.kernel.setArg(idx++, openCLImage(origin));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_image_to_buffer";
        std::vector<uint32_t> lws = rasterLocalWorkSize(gws, mMaxWorkGroupSize, kernelName, unit.kernel);
        
        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
    }
    
    // buffer raster
    for (auto& slice : des->regions)
    {
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "raster_buffer", {});

        unit.globalWorkSize = {(uint32_t)slice.size[2],
                               (uint32_t)slice.size[1],
                               (uint32_t)slice.size[0]};

        const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2],
                                                (uint32_t)slice.size[1],
                                                (uint32_t)slice.size[0]};
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, gws[0]);
        ret |= unit.kernel.setArg(idx++, gws[1]);
        ret |= unit.kernel.setArg(idx++, gws[2]);
        ret |= unit.kernel.setArg(idx++, *(mTempInput[slice.origin]));
        ret |= unit.kernel.setArg(idx++, slice.src.offset);
        ret |= unit.kernel.setArg(idx++, slice.src.stride[0]);
        ret |= unit.kernel.setArg(idx++, slice.src.stride[1]);
        ret |= unit.kernel.setArg(idx++, slice.src.stride[2]);
        ret |= unit.kernel.setArg(idx++, *mTempOutput);
        ret |= unit.kernel.setArg(idx++, slice.dst.offset);
        ret |= unit.kernel.setArg(idx++, slice.dst.stride[0]);
        ret |= unit.kernel.setArg(idx++, slice.dst.stride[1]);
        ret |= unit.kernel.setArg(idx++, slice.dst.stride[2]);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        std::string name = "rasterBuffer";
        const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel);
        
        unit.localWorkSize = {lws[0], lws[1], lws[2]};
        
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                               ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                               ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
    }
    
    //buffer to image
    {
        auto outputShape    = tensorShapeFormat(output);
        int wh[]     = {outputShape[2], outputShape[1]};
        int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};

        Unit &unit          = mUnits[kernel_idx++];
        if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NHWC)//nhwc buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "nhwc_buffer_to_image", {});
        }
        else //nchw buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "nchw_buffer_to_image", {});
        }
        
        std::vector<uint32_t> gws = {(uint32_t)(region[3] * region[1]),
                                     (uint32_t)(region[2] * region[0])};
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, gws[0]);
        ret |= unit.kernel.setArg(idx++, gws[1]);
        ret |= unit.kernel.setArg(idx++, *mTempOutput);
        ret |= unit.kernel.setArg(idx++, wh[1]);
        ret |= unit.kernel.setArg(idx++, wh[0]);
        ret |= unit.kernel.setArg(idx++, outputShape[3]);
        ret |= unit.kernel.setArg(idx++, openCLImage(output));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_buffer_to_image";
        std::vector<uint32_t> lws = rasterLocalWorkSize(gws, mMaxWorkGroupSize, kernelName, unit.kernel);
        
        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
    }
    
    //kernel num check
    if(mNeedZero)
    {
        MNN_ASSERT((kernel_idx==regionNum + originNum + 2));
    }
    else
    {
        MNN_ASSERT((kernel_idx==regionNum + originNum + 1));
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end RasterExecution onResize !\n");
#endif
    return NO_ERROR;
}

//Shaquille, Added 20201117 Start
ErrorCode RasterExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) 
{
	auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
	int  idx     = 0;

	if (false == apply_img_cpy_)
	{
		for (idx = 0; idx < (int)(mUnits.size()); idx++)
		{
			Unit& unit = mUnits[idx];
#ifdef ENABLE_OPENCL_TIME_PROFILER
			cl::Event event;
			auto errorCode = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
				                                                          cl::NullRange,
				                                                          unit.globalWorkSize,
				                                                          unit.localWorkSize,
				                                                          nullptr,
				                                                          &event);

			int costTime = (int)runtime->getCostTime(&event);
			MNN_PRINT("kernel cost:%d    us %s%d\n", costTime, EnumNameOpType(mOp->type()), idx++);
#else
			auto errorCode = runtime->commandQueue().enqueueNDRangeKernel(unit.kernel,
				                                                          cl::NullRange,
				                                                          unit.globalWorkSize,
				                                                          unit.localWorkSize);
#endif
			MNN_CHECK_CL_SUCCESS(errorCode);
		}
	}
	else
	{
		for (idx = 0; idx < (int)(mUnits.size()); idx++)
		{
			Unit& unit = mUnits[idx];
			if (false == output_tensor_[idx]->IsUsedFlag() && true == TensorUtils::getDescribe(output_tensor_[idx])->fake_used)
                continue;

#ifdef ENABLE_OPENCL_TIME_PROFILER
			cl::Event event;
			auto errorCode = runtime->commandQueue().enqueueCopyImage(openCLImage(input_tensor_[idx]),
				                                                      openCLImage(output_tensor_[idx]),
				                                                      { src_offfset_[idx], 0, 0 },
				                                                      { dst_offfset_[idx], 0, 0 },
				                                                      { copy_size_[idx][0], copy_size_[idx][1], 1 },
				                                                      nullptr, &event);
			int costTime = (int)runtime->getCostTime(&event);
			MNN_PRINT("kernel cost:%d    us CommonExe%d\n", costTime, idx);
#else
			auto errorCode = runtime->commandQueue().enqueueCopyImage(openCLImage(input_tensor_[idx]),
				                                                      openCLImage(output_tensor_[idx]),
				                                                      { src_offfset_[idx], 0, 0 },
				                                                      { dst_offfset_[idx], 0, 0 },
				                                                      { copy_size_[idx][0], copy_size_[idx][1], 1 },
				                                                      nullptr, nullptr);
#endif
			MNN_CHECK_CL_SUCCESS(errorCode);
		}
	}

	return NO_ERROR;
}
//Shaquille, Added 20201117 End

OpenCLCreatorRegister<TypedCreator<RasterExecution>> __Raster_op(OpType_Raster);
} // namespace OpenCL
} // namespace MNN
