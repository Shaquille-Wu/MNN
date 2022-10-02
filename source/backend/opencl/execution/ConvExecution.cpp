//
//  ConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvExecution.hpp"
#include "MultiInputConvExecution.hpp"
#include "ConvWinograd.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"

#define UNIT 4
namespace MNN {
namespace OpenCL {

static std::string  conv_2d_s1_opt_kernel[] = {
    std::string("none"),
    std::string("conv_2d_stride_1x1"),
    std::string("conv_2d_stride_1x1"),
    std::string("conv_2d_s1x1_4row")
};

static std::string  conv_2d_s1_dilation_kernel[] = {
    std::string("none"),
    std::string("conv_2d_s1x1_dilation_4row")
};

static std::string  conv1x1_opt_kernel[] = {
    std::string("none"),
    std::string("conv_2d_kernel1x1_stride1x1_opt"),
    std::string("conv_2d_kernel1x1_stride1x1_opt"),
    std::string("conv_2d_kernel1x1_stride1x1_opt"),
    std::string("conv_2d_kernel1x1_stride1x1_opt"),
    std::string("conv_2d_k1_s1_dual_output"),
    std::string("conv_2d_k1_s1_2_in_row_2_col_4_dual_out"),
    std::string("conv_2d_k1_s1_4row_dual_output")
};

std::vector<uint32_t> ConvExecution::conv2d1x1LocalWSOpt(std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize) {
    
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);
    
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair("conv2d1x1LocalWSOpt", gws);
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
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();

    std::vector<uint32_t> lws(4, 1);

    int coreNum   = deviceComputeUnits * 2;
    for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i) {
        int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
        if (remain == 0) {
            lws[i] = groupSize;
        } else {
            while(groupSize) {
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

std::vector<uint32_t> ConvExecution::conv2d1x1LocalWS(std::string const&      tuning_name,
                                                      std::string const&      tuning_ext_name,
                                                      cl::Kernel&             kernel, 
                                                      std::vector<uint32_t>&  gws, 
                                                      const uint32_t          maxWorkGroupSize, 
                                                      int&                    kernel_cost) 
{
    
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);
    
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::string  full_tuning_name = tuning_name + std::string("_") + tuning_ext_name;
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(full_tuning_name, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2d1x1LocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
//Shaquille, Modified 20201118 Start
#if 0
		return tunedLws[info];
#else
		std::vector<uint32_t> const& cost_time = (std::get<1>(tunedLws[info]));
		if(cost_time.size() > 0)
			kernel_cost = cost_time[0];
		return std::get<0>(tunedLws[info]);
#endif
//Shaquille, Modified 20201118 End
    }
    else
    {
        info = std::make_pair(tuning_name, gws);
        if (tunedLws.find(info) != tunedLws.end()) {
            std::vector<uint32_t> const& cost_time = (std::get<1>(tunedLws[info]));
            if (cost_time.size() > 0)
                kernel_cost = cost_time[0];
            return std::get<0>(tunedLws[info]);
        }
        //MNN_PRINT("warning, cannot find tunning result\n");
        info = std::make_pair(full_tuning_name, gws);
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
                                kernel, cl::NullRange,
                                cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                cl::NDRange(lws[0], lws[1]),
                                nullptr, &event);
                MNN_CHECK_CL_SUCCESS(error);
                
                int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
                if(cost_time < min_cost) {
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
        //printf("conv2d1x1LocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
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

std::vector<uint32_t> ConvExecution::conv2dGeneralLocalWS(const std::vector<uint32_t> &gws, const uint32_t kernelSize,
                                                          const uint32_t maxWorkGroupSize) {
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 2);
    
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair("conv2dGeneralLocalWS", gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2dGeneralLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
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
        //printf("conv2dGeneralLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
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
    auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
    uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();

    std::vector<uint32_t> lws(4, 0);

    int coreNum = deviceComputeUnits;
    for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i) {
        int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
        if (remain == 0) {
            lws[i] = groupSize;
        } else {
            while(groupSize) {
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

//Shaquille, Added 20201024 Start 
std::vector<uint32_t> ConvExecution::conv2dGeneralLocalWSOpt(std::string const&            tuning_name,
                                                             cl::Kernel&                   kernel,
                                                             const std::vector<uint32_t>&  gws, 
                                                             const uint32_t                kernelSize,
                                                             const uint32_t                maxWorkGroupSize)
{
#ifdef MNN_OPENCL_LWS_TUNE
	MNN_ASSERT(gws.size() == 2);

	auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
	MNN_ASSERT(maxWorkItemSizes.size() >= 2);
	auto& tunedLws = mOpenCLBackend->getOpenCLRuntime()->tunedLwsMap();
	std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(tuning_name, gws);
	if (tunedLws.find(info) != tunedLws.end()) {
		//printf("conv2dGeneralLocalWS Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
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
	while (lws[1] <= gws[1] * 2 || lws[1] <= 4) {
		lws[0] = 1;
		while (lws[0] <= gws[0] * 2 || lws[0] <= 4) {
			if (lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0] * lws[1] <= maxWorkGroupSize) {
				cl::Event event;
				std::vector<uint32_t> internalGlobalWS(2, 1);
				for (size_t i = 0; i < gws.size(); ++i) {
					internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
				}
				cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
					kernel, cl::NullRange,
					cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
					cl::NDRange(lws[0], lws[1]),
					nullptr, &event);
				MNN_CHECK_CL_SUCCESS(error);

				int cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
				if (cost_time < min_cost) {
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
		//printf("conv2dGeneralLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
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
	auto maxWorkItemSizes = mOpenCLBackend->getOpenCLRuntime()->getMaxWorkItemSizes();
	uint32_t deviceComputeUnits = mOpenCLBackend->getOpenCLRuntime()->deviceComputeUnits();

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
//Shaquille, Added 20201024 End

ConvCommonExecution::ConvCommonExecution(const Convolution2D *conv2dParams, Backend *backend) : Execution(backend) {
    auto openclBackend       = (OpenCLBackend *)backend;
    const float *biasDataPtr = conv2dParams->bias()->data();
	int          biasSize    = conv2dParams->bias()->size();
	const auto*  conv2dCommonParams  = conv2dParams->common();
	int          outputChannel       = conv2dCommonParams->outputCount();
	int          bias_size_raw       = outputChannel;
	int          buffer_size         = ALIGN_UP4(bias_size_raw);
	int          bn_after_relu_flag  = 0;
	
	if ((biasSize == 3 * bias_size_raw || biasSize == (3 * bias_size_raw + 1)) && bias_size_raw > 0)
	{
		bn_after_relu_flag = 1;
		buffer_size        = 3 * ALIGN_UP4(bias_size_raw);
		if (biasSize == (3 * outputChannel + 1))
			bn_after_relu_flag = 2;
		buffer_size = ALIGN_UP4(buffer_size);
	}

    if(openclBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
        buffer_size *= sizeof(half_float::half);
    } else {
        buffer_size *= sizeof(float);
    }

    cl::Buffer biasBuffer(openclBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
    cl_int error;
    auto biasPtrCL = openclBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
        biasBuffer, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
    if(biasPtrCL != nullptr && error == CL_SUCCESS){
//Shaquille, Modified 20210502 Start
#if 0
        if(openclBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
            for(int i=0; i<biasSize; i++) {
                ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
            }
            for(int i=biasSize; i<ALIGN_UP4(biasSize); i++) {
                ((half_float::half*)biasPtrCL)[i] = (half_float::half)(0.0f);
            }
        }else{
            ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
            ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
        }
#else
		if (openclBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
			if (0 == bn_after_relu_flag)
			{
				for (int i = 0; i < biasSize; i++) {
					((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
				}
				for (int i = biasSize; i < ALIGN_UP4(biasSize); i++) {
					((half_float::half*)biasPtrCL)[i] = (half_float::half)(0.0f);
				}
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < bias_size_raw; j++)
						((half_float::half*)biasPtrCL)[j + i * (ALIGN_UP4(bias_size_raw))] = (half_float::half)(biasDataPtr[j + i * bias_size_raw]);
					for (int j = 0; j < bias_size_raw; j++)
						((half_float::half*)biasPtrCL)[j + i * (ALIGN_UP4(bias_size_raw))] = (half_float::half)(0.0f);
				}
				biasSize = (buffer_size >> 2);
			}
		}
		else {
			if (0 == bn_after_relu_flag)
			{
				::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
				::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
			}
			else
			{
				::memset(biasPtrCL, 0, buffer_size);
				for (int i = 0; i < 3; i++)
				{
					::memcpy(((float*)biasPtrCL) + i * (ALIGN_UP4(bias_size_raw)), 
						     ((float*)biasDataPtr) + i * bias_size_raw, 
						     bias_size_raw * sizeof(float));
				}
				biasSize = (buffer_size >> 2);
			}
		}
#endif
//Shaquille, Modified 20210502 End
    }else{
        MNN_ERROR("Map error biasPtrCL == nullptr \n");
    }
    openclBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(biasBuffer, biasPtrCL);
    mBias.reset(Tensor::createDevice<float>({1, 1, 1, biasSize}));
    backend->onAcquireBuffer(mBias.get(), Backend::STATIC);
    copyBufferToImage(openclBackend->getOpenCLRuntime(), biasBuffer, openCLImage(mBias.get()), UP_DIV(biasSize, 4), 1);
}
ConvCommonExecution::~ConvCommonExecution() {
    MNN_ASSERT(nullptr != mBias);
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

ConvExecution::ConvExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend)
    : ConvCommonExecution(op->main_as_Convolution2D(), backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dCommonParams            = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
//Shaquille, Added 20210220 Start
    int  leaky_relu_int            = (int)(((((uint32_t)(mStrides[0])) & 0xFFFF0000)) >> 16);
    mStrides[0]                    = (mStrides[0] & 0x0000FFFF);
    leaky_relu_                    = (float)(((double)leaky_relu_int) * 0.001);
//Shaquille, Added 20210220 End
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};

    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;

    int kernelWidth   = conv2dCommonParams->kernelX();
    int kernelHeight  = conv2dCommonParams->kernelY();
    int outputChannel = conv2dCommonParams->outputCount();

//Shaquille, Added 20210502 Start
	int bias_size_raw = conv2dParams->bias()->size();
	if ((bias_size_raw == 3 * outputChannel || bias_size_raw == (3 * outputChannel + 1)) && bias_size_raw > 0)
	{
		bn_after_relu_ = true;
		leaky_relu_    = 0.0f;
		if (bias_size_raw == (3 * outputChannel + 1))
		{
			const float *bias_data_raw = conv2dParams->bias()->data();
			leaky_relu_    = bias_data_raw[3 * outputChannel];
			leaky_relu_int = 0xFFFFFF;
		}
	}
//Shaquille, Added 20210502 End

    int weightSize             = 0;
    const float *filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2dParams->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2dParams->quanParameter(), true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    } else if (nullptr == conv2dParams->weight() || nullptr == conv2dParams->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", op->name()->c_str());
    }

    if (nullptr == filterDataPtr) {
        weightSize    = conv2dParams->weight()->size();
        filterDataPtr = conv2dParams->weight()->data();
    }
    int inputChannel = weightSize / (kernelWidth * kernelHeight * outputChannel);

    auto gpuType = mOpenCLBackend->getOpenCLRuntime()->getGpuType();

    //select opt conv method
    std::string kernelName = "conv_2d";
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 &&
        mPaddings[1] == 0) {
        mConv1x1Opt = (mStrides[0] == 1 && mStrides[1] == 1 && gpuType == GpuType::MALI);
#if 0
        if((gpuType == GpuType::ADRENO)){
            uint64_t useLocalSize = UNIT*UNIT*4*sizeof(float)*4;
            if(useLocalSize >= mOpenCLBackend->getOpenCLRuntime()->getMaxLocalMem()){
                mUseLocalMem = false;
            }else{
                kernelName = "conv_2d_1x1_local";
                mUseLocalMem=true;
            }
        }
#endif
        if(!mUseLocalMem){
            if(mConv1x1Opt){
                kernelName = "conv_2d_1x1_mali";
            }else{
                kernelName = "conv_2d_1x1";
            }
        }
    }

    if(mConv1x1Opt && !mUseLocalMem){
        cl_int error;
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({UP_DIV(outputChannel, 4)*4, UP_DIV(inputChannel, 4)*4, kernelWidth, kernelHeight}));
        
        int buffer_size = filterBuffer->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mKernelBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto kernelBufferPtr = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mKernelBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(kernelBufferPtr != nullptr && error == CL_SUCCESS){
            ::memset(kernelBufferPtr, 0, buffer_size);
            for(int o = 0; o < outputChannel; o++){
                for(int i = 0 ; i < inputChannel; i++){
                    int bufferIdx = (o/4) * ROUND_UP(inputChannel, 4)*4 + (i/4)*16 + (o%4)*4 + (i%4);
                    int filterIdx = o*inputChannel + i;
                    if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                        ((half_float::half*)kernelBufferPtr)[bufferIdx] = (half_float::half)(filterDataPtr[filterIdx]);
                    }else{
                        ((float*)kernelBufferPtr)[bufferIdx] = (float)(filterDataPtr[filterIdx]);
                    }
                }
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mKernelBuffer.get()), kernelBufferPtr);

        //bias
        int biasSize             = conv2dParams->bias()->size();
        const float *biasDataPtr = conv2dParams->bias()->data();
        
        buffer_size = ALIGN_UP4(biasSize);
        if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        
        mBiasBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size));
        auto biasPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
            *(mBiasBuffer.get()), true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(biasPtrCL != nullptr && error == CL_SUCCESS){
            if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                for (int i = 0; i < biasSize; i++)
                {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(biasDataPtr[i]);
                }
                for(int i=biasSize; i<ALIGN_UP4(biasSize); i++) {
                    ((half_float::half*)biasPtrCL)[i] = (half_float::half)(0.0f);
                }
            }else{
                ::memset(biasPtrCL, 0, ALIGN_UP4(biasSize) * sizeof(float));
                ::memcpy(biasPtrCL, biasDataPtr, biasSize * sizeof(float));
            }
        }else{
            MNN_ERROR("Map error biasPtrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*(mBiasBuffer.get()), biasPtrCL);

    }else{
        std::vector<int> filterImageShape{(int)inputChannel, (int)(UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight)};
        std::shared_ptr<Tensor> filterBuffer(
            Tensor::createDevice<float>({outputChannel, inputChannel, kernelWidth, kernelHeight}));
        
        int buffer_size = filterBuffer->elementSize();
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()) {
            buffer_size *= sizeof(half_float::half);
        } else {
            buffer_size *= sizeof(float);
        }
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);

        cl_int error;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(ptrCL != nullptr && error == CL_SUCCESS) {
            ::memset(ptrCL, 0, buffer_size);
            if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf()){
                for(int i = 0 ; i < filterBuffer->elementSize(); i++){
                    ((half_float::half*)ptrCL)[i] = (half_float::half)(filterDataPtr[i]);
                }
            }else{
                ::memcpy(ptrCL, filterDataPtr, filterBuffer->size());
            }
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);

        mFilter.reset(Tensor::createDevice<float>({1, filterImageShape[1], 1, 4 * filterImageShape[0]}));
        mOpenCLBackend->onAcquireBuffer(mFilter.get(), Backend::STATIC);
        MNN::OpenCL::ImageBufferConvertor imageBufferConvertor{mOpenCLBackend->getOpenCLRuntime()};

        std::string buildOption = "";
        if(mOpenCLBackend->getOpenCLRuntime()->isWeightCpuTransHalf() == false){
            buildOption = "-DBUFFER_INP_FP32";
        }
        imageBufferConvertor.convertBufferToImage(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mFilter.get(), false, buildOption);
    }

    // Create Kernel
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DBIAS");
    if (mConv2dCommonParams->relu() || (true == bn_after_relu_ && 0.0f == leaky_relu_)) {
        buildOptions.emplace("-DRELU");
    } else if (mConv2dCommonParams->relu6()) {
        buildOptions.emplace("-DRELU6");
    }
    std::string program_name = "conv_2d";
//Shaquille, Added 20201012 Start
	zero_bias_flag_          = is_zero_bias(conv2dParams);
	conv_2d_tuning_ext_name_ = "";
    if (0 != leaky_relu_int)
    {
        char leaky_relu_factor[64] = { 0 };
        sprintf(leaky_relu_factor, "-DLEAKY_RELU=%.12f", leaky_relu_);
        buildOptions.emplace(std::string(leaky_relu_factor));
        conv_2d_tuning_ext_name_ += "LEAKY_RELU_";
    }
	if (true == bn_after_relu_)
	{
		buildOptions.emplace(std::string("-DBN_AFTER_RELU"));
		conv_2d_tuning_ext_name_ += "BN_AFTER_RELU_";
	}
    if(conv2dCommonParams->inputCount() >= 4)
    {
        buildOptions.emplace("-DINPUT_CHANNEL_2");
        buildOptions.emplace("-DINPUT_CHANNEL_3");
        buildOptions.emplace("-DINPUT_CHANNEL_4");
		conv_2d_tuning_ext_name_ += "CHANNEL4_";
    }
    else if(3 == conv2dCommonParams->inputCount())
    {
        buildOptions.emplace("-DINPUT_CHANNEL_2");
        buildOptions.emplace("-DINPUT_CHANNEL_3");
		conv_2d_tuning_ext_name_ += "CHANNEL3_";
    }
    else if(2 == conv2dCommonParams->inputCount())
    {
        buildOptions.emplace("-DINPUT_CHANNEL_2");
		conv_2d_tuning_ext_name_ += "CHANNEL2_";
    }

	if (conv2dCommonParams->inputCount() > 4)
	{
		buildOptions.emplace("-DIN_CHANNEL_LOOP");
		conv_2d_tuning_ext_name_ += "CHANNEL_LOOP_";
	}

    if((mStrides[0] == mStrides[1]) && (1 == mStrides[0]) &&
       (mDilations[0] == mDilations[1]) && (1 == mDilations[1]) &&
       (conv2dCommonParams->kernelX() == conv2dCommonParams->kernelY()) && (conv2dCommonParams->kernelX() > 1))
    {
        if(true == zero_bias_flag_)
            buildOptions.erase("-DBIAS");
        kernelName           = conv_2d_s1_opt_kernel[1];
        int  out_channel         = conv2dCommonParams->outputCount();
        int  out_channel_q       = (out_channel + 3) >> 2;
        char kernel_size_str[64] = { 0 };
        sprintf(kernel_size_str, "k%d_", conv2dCommonParams->kernelX());
        conv_2d_tuning_ext_name_ += std::string(kernel_size_str);
        if (mOpenCLBackend->getOpenCLRuntime()->rawTunedLwsMap().size() <= 0 ||
            true == mOpenCLBackend->getOpenCLRuntime()->is_kernel_in_raw_tuned_map(conv_2d_s1_opt_kernel[3], false))
        {
            conv_opt_stride_1x1_ = 3;
            kernelName           = conv_2d_s1_opt_kernel[conv_opt_stride_1x1_];
            program_name         = "conv_2d_opt";
        }
        else
        {
            if (0 != (out_channel_q & 1))
                conv_opt_stride_1x1_ = 1;
            else
            {
                conv_2d_tuning_ext_name_ += "DUAL_OUTPUT_DATA_";
                conv_opt_stride_1x1_      = 2;
                buildOptions.emplace("-DDUAL_OUTPUT_DATA");
            }
        }
    }
	else if ((mStrides[0] == mStrides[1]) && (1 == mStrides[0]) &&
             (mDilations[0] == mDilations[1]) &&
             (conv2dCommonParams->kernelX() == conv2dCommonParams->kernelY()) && (conv2dCommonParams->kernelX() > 1))
	{
        conv_opt_stride_1x1_dilation_ = 1;
        kernelName                    = conv_2d_s1_dilation_kernel[conv_opt_stride_1x1_dilation_];
        program_name                  = "conv_2d_dilation_opt";
	}
//Shaquille, Added 20201012 End

    mKernel           = mOpenCLBackend->getOpenCLRuntime()->buildKernel(program_name, kernelName, buildOptions);
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ConvExecution::~ConvExecution() {
    if(mUseLocalMem || !mConv1x1Opt){
        mOpenCLBackend->onReleaseBuffer(mFilter.get(), Backend::STATIC);
    }
}

ErrorCode ConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    const int height             = outputShape.at(1);
    const int width              = outputShape.at(2);

    const int inputHeight   = inputShape.at(1);
    const int inputWidth    = inputShape.at(2);
    const int inputChannels = inputShape.at(3);

    const int inputChannelBlocks = UP_DIV(inputChannels, 4);
    int kernelHeight = mConv2dCommonParams->kernelY();
    int kernelWidth  = mConv2dCommonParams->kernelX();
    
    auto pad = ConvolutionCommon::convolutionPad(input, output, mConv2dCommonParams);
    mPaddings[0] = pad.second;
    mPaddings[1] = pad.first;
    k1x1_conv_ = false;
    if (kernelHeight == kernelWidth && kernelHeight == 1 && mPaddings[0] == 0 && mPaddings[1] == 0) {
        k1x1_conv_ = true;
        if(mConv1x1Opt){

            auto kernel             = &mKernel;
            uint32_t idx            = 0;

            if(mUseLocalMem){
                mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4)), static_cast<uint32_t>(UP_DIV(outputShape.at(2), 4)),
                static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
                std::vector<uint32_t> lws{UNIT, UNIT, 1};
                mLocalWorkSize = lws;
                kernel->setArg(idx++, mGlobalWorkSize[0]);
                kernel->setArg(idx++, mGlobalWorkSize[1]);
                kernel->setArg(idx++, mGlobalWorkSize[2]);
                kernel->setArg(idx++, openCLImage(input));
                kernel->setArg(idx++, openCLImage(mFilter.get()));
                kernel->setArg(idx++, openCLImage(mBias.get()));
                kernel->setArg(idx++, openCLImage(output));
                kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
                kernel->setArg(idx++, height);
                kernel->setArg(idx++, width);
            }else{
                mGlobalWorkSize = {static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                           static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
                kernel->setArg(idx++, mGlobalWorkSize[0]);
                kernel->setArg(idx++, mGlobalWorkSize[1]);
                kernel->setArg(idx++, UP_DIV(width, 4));
                kernel->setArg(idx++, openCLImage(input));
                kernel->setArg(idx++, *mKernelBuffer.get());
                kernel->setArg(idx++, *mBiasBuffer.get());
                kernel->setArg(idx++, openCLImage(output));
                kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
                kernel->setArg(idx++, height);
                kernel->setArg(idx++, width);
                
                mLocalWorkSize          = conv2d1x1LocalWSOpt(mGlobalWorkSize, mMaxWorkGroupSize);
            }


        }else{
//Shaquille, Modified 20201010 Start
            mGlobalWorkSize = {
            static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * static_cast<uint32_t>(UP_DIV(outputShape.at(2), 4))),
            static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))};
            
            auto kernel             = &mKernel;
            uint32_t idx            = 0;
            int inputImageShape[2]  = {inputHeight, inputWidth};
            int outputImageShape[2] = {height, width};
            int stideShape[2]       = {mStrides[0], mStrides[1]};
            kernel->setArg(idx++, mGlobalWorkSize[0]);
            kernel->setArg(idx++, mGlobalWorkSize[1]);
            kernel->setArg(idx++, openCLImage(input));
            kernel->setArg(idx++, openCLImage(mFilter.get()));
            kernel->setArg(idx++, openCLImage(mBias.get()));
            kernel->setArg(idx++, openCLImage(output));
            kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
            kernel->setArg(idx++, static_cast<int>(inputChannelBlocks));
            kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
            kernel->setArg(idx++, sizeof(stideShape), stideShape);
            kernel->setArg(idx++, UP_DIV(width, 4));

            int  kernel_cost = 0;
            std::string   prg_name    = "conv_2d_opt";
            std::string   tuning_size = get_conv1x1_tuning_size_by_shape(inputWidth, inputHeight, inputChannels, outputShape.at(3));
            mLocalWorkSize            = conv2d1x1LocalWS("conv2d1x1LocalWS", tuning_size, mKernel, mGlobalWorkSize, mMaxWorkGroupSize, kernel_cost);
#ifdef ORION_CONV_OPTIMIZE
            int opt_type    = select_conv1x1_opt_type(mOpenCLBackend,
                                                      outputShape.at(0),
                                                      inputWidth, 
                                                      inputHeight, 
                                                      inputChannels, 
                                                      outputShape.at(3), 
                                                      mStrides[0],
                                                      mStrides[1]);
            if(0 != opt_type)
            {
                std::vector<uint32_t> global_work_size;
                std::vector<uint32_t> local_work_size;
                int                   output_height_q  = ((outputShape.at(1) + 3) >> 2);
                int                   output_width_q   = ((outputShape.at(2) + 3) >> 2);
                int                   output_channel_q = ((outputShape.at(3) + 3) >> 2);
                std::string           kernel_name      = conv1x1_opt_kernel[opt_type];
                std::set<std::string> buildOptions;
                if(false == zero_bias_flag_)
                    buildOptions.emplace("-DBIAS");
                if (mConv2dCommonParams->relu() || (true == bn_after_relu_ && 0.0f == leaky_relu_)) {
                    buildOptions.emplace("-DRELU");
                } else if (mConv2dCommonParams->relu6()) {
                    buildOptions.emplace("-DRELU6");
                }
                if (0.0f != leaky_relu_)
                {
                    char leaky_relu_factor[64] = { 0 };
                    sprintf(leaky_relu_factor, "-DLEAKY_RELU=%.12f", leaky_relu_);
                    buildOptions.emplace(std::string(leaky_relu_factor));
                }
				if (true == bn_after_relu_)
					buildOptions.emplace(std::string("-DBN_AFTER_RELU"));

                if(1 == opt_type || 2 == opt_type || 3 == opt_type || 4 == opt_type)
                {
                    global_work_size = {
                        static_cast<uint32_t>((((inputWidth * inputHeight + 31) >> 5) << 5) >> 2),
                        static_cast<uint32_t>((output_channel_q + 1) >> 1)
                    };
                    if(2 == opt_type)
                        buildOptions.emplace("-DCHECK_IMG_BORDER");
                    else if(3 == opt_type)
                        buildOptions.emplace("-DCHECK_OUT_CHANNEL");
                    else if(4 == opt_type)
                    {
                        buildOptions.emplace("-DCHECK_IMG_BORDER");
                        buildOptions.emplace("-DCHECK_OUT_CHANNEL");
                    }  
                }
                else
                {
                    if(5 == opt_type)
                    {
                        global_work_size = {
                            static_cast<uint32_t>(((output_channel_q + 1) >> 1) * static_cast<uint32_t>(output_width_q)),
                            static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))
                        };
                    }
                    else if(6 == opt_type)
                    {
                        global_work_size = {
                            static_cast<uint32_t>(output_channel_q * static_cast<uint32_t>(output_width_q)),
                            static_cast<uint32_t>(outputShape.at(0) * ((outputShape.at(1) + 1) >> 1))
                        };
                    }
                    else
                    {
                        global_work_size = {
                            static_cast<uint32_t>(((output_channel_q + 1) >> 1) * static_cast<uint32_t>(outputShape.at(2))),
                            static_cast<uint32_t>(outputShape.at(0) * output_height_q)
                        };
                    }
                }
                    
                opt_kernel_       = mOpenCLBackend->getOpenCLRuntime()->buildKernel(prg_name, kernel_name, buildOptions);
                uint32_t  max_grp_size = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(opt_kernel_));
                
                //
                uint32_t idx             = 0;
                int     cur_kernel_cost  = 0;    
                if(1 == opt_type || 2 == opt_type || 3 == opt_type || 4 == opt_type)
                {
                    opt_kernel_.setArg(idx++, openCLImage(input));
                    opt_kernel_.setArg(idx++, openCLImage(mFilter.get()));
                    if(false == zero_bias_flag_)
                        opt_kernel_.setArg(idx++, openCLImage(mBias.get()));
                    opt_kernel_.setArg(idx++, openCLImage(output));
                    opt_kernel_.setArg(idx++, inputWidth);
                    opt_kernel_.setArg(idx++, inputHeight);
                    opt_kernel_.setArg(idx++, inputChannels);
                    if(3 == opt_type || 4 == opt_type)
                        opt_kernel_.setArg(idx++, output_channel_q);
                    local_work_size     = { 8, 8 };
                    mMaxWorkGroupSize   = max_grp_size;
                    mLocalWorkSize      = local_work_size;
                    mGlobalWorkSize     = global_work_size;
                    conv1x1_opt_type_   = opt_type;

                    if(0 == kernel_cost)
                        kernel_cost = evaluate_performance_2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
                    if(0 == cur_kernel_cost)
                        cur_kernel_cost = evaluate_performance_2d(opt_kernel_, global_work_size, local_work_size);

                    //printf("%s, cost_time %d, %d\n", kernel_name.c_str(), kernel_cost, cur_kernel_cost);
                }
                else
                {
                    opt_kernel_.setArg(idx++, global_work_size[0]);
                    opt_kernel_.setArg(idx++, global_work_size[1]);
                    opt_kernel_.setArg(idx++, openCLImage(input));
                    opt_kernel_.setArg(idx++, openCLImage(mFilter.get()));
                    if(false == zero_bias_flag_)
                        opt_kernel_.setArg(idx++, openCLImage(mBias.get()));
                    opt_kernel_.setArg(idx++, openCLImage(output));
                    opt_kernel_.setArg(idx++, sizeof(inputImageShape), inputImageShape);
                    opt_kernel_.setArg(idx++, inputChannelBlocks);
                    opt_kernel_.setArg(idx++, sizeof(outputImageShape), outputImageShape);
                    opt_kernel_.setArg(idx++, output_channel_q);
                    if(7 != opt_type)
                        opt_kernel_.setArg(idx++, output_width_q);  
                    else
                        opt_kernel_.setArg(idx++, output_height_q);
                    std::string lws_tuning_name = kernel_name + std::string("_LocalWS");   
                    local_work_size = conv2d1x1LocalWS(lws_tuning_name, tuning_size, opt_kernel_, global_work_size, max_grp_size, cur_kernel_cost); 
                    if(0 == kernel_cost)
                        kernel_cost = evaluate_performance_2d(mKernel, mGlobalWorkSize, mLocalWorkSize);
                    if(0 == cur_kernel_cost)
                        cur_kernel_cost = evaluate_performance_2d(opt_kernel_, global_work_size, local_work_size);

                    //printf("%s, cost_time %d, %d\n", kernel_name.c_str(), kernel_cost, cur_kernel_cost);

                    if(kernel_cost > cur_kernel_cost)
                    {
                        conv1x1_opt_type_ = opt_type;
                        mMaxWorkGroupSize = max_grp_size;
                        mGlobalWorkSize   = global_work_size;
                        mLocalWorkSize    = local_work_size;
                    }
                }
/*
                if(0 != conv1x1_opt_type_)
                    printf("kernel_name %s\n", kernel_name.c_str());
*/
            }
#endif
//Shaquille, Modified 20201010 End
            //printf("conv_1x1, mLocalWorkSize %d, %d, mGlobalWorkSize %d, %d, w %d, h %d, icq: %d, ocq: %d, opt_type: %d\n", 
            //       mLocalWorkSize[0], mLocalWorkSize[1], mGlobalWorkSize[0], mGlobalWorkSize[1], inputWidth, inputHeight, inputChannelBlocks, UP_DIV(outputShape.at(3), 4), conv1x1_opt_type_);
        }
    } else {
//Shaquille, Added 20201024 Start
        int           conv_s2x2_opt           = 0;
        std::string   conv_s2x2_kernel_name   = "";
        if((mConv2dCommonParams->kernelX() == mConv2dCommonParams->kernelY()) &&
           ((mStrides[0] == mStrides[1]) && (2 == mStrides[0])) &&
           ((mDilations[0] == mDilations[1]) && (1 == mDilations[1])))
        {
            if((3 == mConv2dCommonParams->kernelX()) && (1 == mConv2dCommonParams->padX()))
            {
                conv_s2x2_opt          = 1;
                conv_s2x2_kernel_name  = "conv_2d_k3_s2x2";
            }
            else if((5 == mConv2dCommonParams->kernelX()) && (2 == mConv2dCommonParams->padX()))
            {
                conv_s2x2_opt          = 2;
                conv_s2x2_kernel_name  = "conv_2d_kN_s2x2";
            }
            else if((7 == mConv2dCommonParams->kernelX()) && (3 == mConv2dCommonParams->padX()))
            {
                conv_s2x2_opt          = 3;
                conv_s2x2_kernel_name  = "conv_2d_kN_s2x2";
            }
        }
        
        int inputImageShape[2]      = {inputHeight, inputWidth};
        int outputImageShape[2]     = {height, width};
        uint32_t idx                = 0;
        if(0 == conv_s2x2_opt)
        {
//Shaquille, Added 20201024 End
            mGlobalWorkSize         = { static_cast<uint32_t>(UP_DIV(outputShape.at(3), 4) * UP_DIV(outputShape.at(2), 4)),
                                        static_cast<uint32_t>(outputShape.at(0) * outputShape.at(1))
			};
//Shaquille, Added 20201020 Start
            if(2 == conv_opt_stride_1x1_)
                mGlobalWorkSize[0]  = (mGlobalWorkSize[0] >> 1);
            else if (3 == conv_opt_stride_1x1_ || 0 != conv_opt_stride_1x1_dilation_)
            {
                int out_channel_q = UP_DIV(outputShape.at(3), 4);
                mGlobalWorkSize = { static_cast<uint32_t>(((out_channel_q + 1) >> 1) * outputShape.at(2)),
                                    static_cast<uint32_t>(outputShape.at(0) * UP_DIV(outputShape.at(1), 4))
                                  };
            }
//Shaquille, Added 20201020 End

			int inputImageShape[2]  = {inputHeight, inputWidth};
			int outputImageShape[2] = {height, width};
			int kernelShape[2]      = {kernelHeight, kernelWidth};
			int strideShape[2]      = {mStrides[0], mStrides[1]};
			int paddingShape[2]     = {mPaddings[0], mPaddings[1]};
			int dilationShape[2]    = {mDilations[0], mDilations[1]};
			uint32_t idx            = 0;
			auto kernel             = &mKernel;
			kernel->setArg(idx++, mGlobalWorkSize[0]);
			kernel->setArg(idx++, mGlobalWorkSize[1]);
			kernel->setArg(idx++, openCLImage(input));
			kernel->setArg(idx++, openCLImage(mFilter.get()));
            if(false == zero_bias_flag_)
			    kernel->setArg(idx++, openCLImage(mBias.get()));
			kernel->setArg(idx++, openCLImage(output));
			kernel->setArg(idx++, sizeof(inputImageShape), inputImageShape);
			kernel->setArg(idx++, inputChannelBlocks);
			kernel->setArg(idx++, sizeof(outputImageShape), outputImageShape);
			kernel->setArg(idx++, sizeof(kernelShape), kernelShape);
//Shaquille, Added 20201020 Start
            if(0 == conv_opt_stride_1x1_ && 0 == conv_opt_stride_1x1_dilation_)
                kernel->setArg(idx++, sizeof(strideShape), strideShape);
//Shaquille, Added 20201020 End
			kernel->setArg(idx++, sizeof(paddingShape), paddingShape);
//Shaquille, Added 20201020 Start
            if(0 == conv_opt_stride_1x1_ || 0 != conv_opt_stride_1x1_dilation_)
                kernel->setArg(idx++, sizeof(dilationShape), dilationShape);
//Shaquille, Added 20201020 End
            if(3 != conv_opt_stride_1x1_ && 0 == conv_opt_stride_1x1_dilation_)
                kernel->setArg(idx++, UP_DIV(width, 4));
            else
                kernel->setArg(idx++, UP_DIV(outputShape.at(3), 4));
//Shaquille, Modified 20201030 Start
#ifndef ORION_CONV_OPTIMIZE
			mLocalWorkSize     = conv2dGeneralLocalWS(mGlobalWorkSize, kernelHeight * kernelWidth, mMaxWorkGroupSize);
#else
            if(0 == conv_opt_stride_1x1_ && 0 == conv_opt_stride_1x1_dilation_)
                mLocalWorkSize = conv2dGeneralLocalWS(mGlobalWorkSize, kernelHeight * kernelWidth, mMaxWorkGroupSize);
            else
            {
                std::string  tuning_kernl_name = conv_2d_s1_opt_kernel[conv_opt_stride_1x1_];
				if(1 == conv_opt_stride_1x1_dilation_)
					tuning_kernl_name = conv_2d_s1_dilation_kernel[conv_opt_stride_1x1_dilation_];
                mLocalWorkSize = conv2dGeneralLocalWSOpt(tuning_kernl_name + "_" + conv_2d_tuning_ext_name_ + "localWS",
                                                         mKernel, 
                                                         mGlobalWorkSize, 
                                                         kernelHeight * kernelWidth, 
                                                         mMaxWorkGroupSize);
            }
#endif
//Shaquille, Modified 20201030 End
        }
        else
        {
            int           output_channel_q        = mConv2dCommonParams->outputCount();
            output_channel_q                      = ((output_channel_q + 3) >> 2);
            int           batch_count             = outputShape.at(0);
            int           output_w_q              = ((width + 3) >> 2);
            int           global_size0            = ((output_channel_q + 1) >> 1) * output_w_q;
            int           global_size1            = batch_count * height;
			std::string   tuning_ext_name         = "";
            std::set<std::string> buildOptions;
            buildOptions.emplace("-DBIAS");
            if (mConv2dCommonParams->relu() || (true == bn_after_relu_ && 0.0f == leaky_relu_)) {
                buildOptions.emplace("-DRELU");
            } else if (mConv2dCommonParams->relu6()) {
                buildOptions.emplace("-DRELU6");
            }
            if (0.0f != leaky_relu_)
            {
                char leaky_relu_factor[64] = { 0 };
                sprintf(leaky_relu_factor, "-DLEAKY_RELU=%.12f", leaky_relu_);
                buildOptions.emplace(std::string(leaky_relu_factor));
                tuning_ext_name += "LEAKY_RELU_";
            }
			if (true == bn_after_relu_)
			{
				buildOptions.emplace(std::string("-DBN_AFTER_RELU"));
				tuning_ext_name += "BN_AFTER_RELU_";
			}
            if(mConv2dCommonParams->inputCount() >= 4)
            {
                buildOptions.emplace("-DINPUT_CHANNEL_2");
                buildOptions.emplace("-DINPUT_CHANNEL_3");
                buildOptions.emplace("-DINPUT_CHANNEL_4");
				tuning_ext_name += "CHANNEL4_";
            }
            else if(3 == mConv2dCommonParams->inputCount())
            {
                buildOptions.emplace("-DINPUT_CHANNEL_2");
                buildOptions.emplace("-DINPUT_CHANNEL_3");
				tuning_ext_name += "CHANNEL3_";
            }
            else if(2 == mConv2dCommonParams->inputCount())
            {
                buildOptions.emplace("-DINPUT_CHANNEL_2");
				tuning_ext_name += "CHANNEL2_";
            }
			if (mConv2dCommonParams->inputCount() > 4)
			{
				buildOptions.emplace("-DIN_CHANNEL_LOOP");
				tuning_ext_name += "IN_CHANNEL_LOOP_";
			}
                
            if(1 == conv_s2x2_opt)
            {
                buildOptions.emplace("-DKERNEL_SHAPE_SIZE=3");
                buildOptions.emplace("-DKERNEL_PADDING_SIZE=1");
				tuning_ext_name += "K3P1_";
            }
            else if(2 == conv_s2x2_opt)
            {
                buildOptions.emplace("-DKERNEL_SHAPE_SIZE=5");
                buildOptions.emplace("-DKERNEL_PADDING_SIZE=2");
				tuning_ext_name += "K5P2_";
            }
            else if(3 == conv_s2x2_opt)
            {
                buildOptions.emplace("-DKERNEL_SHAPE_SIZE=7");
                buildOptions.emplace("-DKERNEL_PADDING_SIZE=3");
				tuning_ext_name += "K7P3_";
            }
            opt_kernel_          = mOpenCLBackend->getOpenCLRuntime()->buildKernel("conv_2d_opt", conv_s2x2_kernel_name, buildOptions);
            conv_s2x2_opt_type_  = conv_s2x2_opt;
            
            opt_kernel_.setArg(idx++, openCLImage(input));
            opt_kernel_.setArg(idx++, openCLImage(mFilter.get()));
            opt_kernel_.setArg(idx++, openCLImage(mBias.get()));
            opt_kernel_.setArg(idx++, openCLImage(output));
            opt_kernel_.setArg(idx++, sizeof(inputImageShape), inputImageShape);
            opt_kernel_.setArg(idx++, inputChannelBlocks);
            opt_kernel_.setArg(idx++, sizeof(outputImageShape), outputImageShape);
            opt_kernel_.setArg(idx++, output_channel_q);
            opt_kernel_.setArg(idx++, global_size1);

            mGlobalWorkSize   = { static_cast<uint32_t>(global_size0),
                                  static_cast<uint32_t>(global_size1)};
            mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(opt_kernel_));
            mLocalWorkSize    = conv2dGeneralLocalWSOpt(conv_s2x2_kernel_name + "_" + tuning_ext_name + std::string("_localWS"),
                                                        opt_kernel_, 
                                                        mGlobalWorkSize, 
                                                        kernelHeight * kernelWidth, 
                                                        mMaxWorkGroupSize);
        }
/*
    printf("conv, mLocalWorkSize %d, %d, mGlobalWorkSize %d, %d, w %d, h %d, k: %d, s: %d, ic: %d, oc: %d\n", 
                mLocalWorkSize[0], mLocalWorkSize[1], 
                mGlobalWorkSize[0], mGlobalWorkSize[1], 
                inputWidth, inputHeight, 
                kernelWidth, mStrides[0],
                inputChannels, outputShape.at(3));
*/
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode ConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onExecute !\n");
#endif
    if(mUseLocalMem){
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime(), &event);
        
        float costTime = mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
        MNN_PRINT("kernel cost:%f    us Conv UseLocalMem\n",costTime);
    #else
        run3DKernelDefault(mKernel, mGlobalWorkSize, mLocalWorkSize,
                           mOpenCLBackend->getOpenCLRuntime());
    #endif
    }
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
#ifdef ORION_CONV_OPTIMIZE
    if(0 != conv1x1_opt_type_ || 0 != conv_s2x2_opt_type_)
    {
        runKernel2D(opt_kernel_, mGlobalWorkSize, mLocalWorkSize,
                    mOpenCLBackend->getOpenCLRuntime(), &event);
    }
    else
#endif
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                    mOpenCLBackend->getOpenCLRuntime(), &event);
    int costTime = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    MNN_PRINT("kernel cost:%d    us Conv2D\n",costTime);
#else
#ifdef ORION_CONV_OPTIMIZE
    if(0 != conv1x1_opt_type_ || 0 != conv_s2x2_opt_type_)
    {
        runKernel2D(opt_kernel_, mGlobalWorkSize, mLocalWorkSize,
                    mOpenCLBackend->getOpenCLRuntime());
    }
    else
#endif
        runKernel2D(mKernel, mGlobalWorkSize, mLocalWorkSize,
                    mOpenCLBackend->getOpenCLRuntime());
#endif
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

//Shaquille, Added 20201010 Start
#ifdef ORION_CONV_OPTIMIZE
std::string ConvExecution::get_conv1x1_tuning_size_by_shape(int w, int h, int input_channel, int output_channel)
{
	char src_tuning_size[128] = { 0 };
	sprintf(src_tuning_size, "%d_%d_%d_%d", w, h, (input_channel + 3) >> 2, (output_channel + 3) >> 2);
	std::string   tuning_size = std::string(src_tuning_size);
	return tuning_size;
}

int ConvExecution::select_conv1x1_opt_type(OpenCLBackend*    ocl_bn,
                                           int               n, 
                                           int               w, 
                                           int               h, 
                                           int               input_channel, 
                                           int               output_channel, 
                                           int               stride_x, 
                                           int               stride_y)
{
    if(1 != n)                            return 0;
    if(stride_x != stride_y)              return 0;
    if(stride_x != 1 && 0 != stride_x)    return 0;
    if(0 != (w & 3))
    {
        if(w < 8)
            return 0;
    }

    int  image_size        = w * h;
    int  input_channel_q   = ((input_channel + 3) >> 2);
    int  output_channel_q  = ((output_channel + 3) >> 2);
    int  opt_type          = 5;
    int  image_size_q      = ((image_size + 3) >> 2);
    int  image_size_q2     = ((image_size_q + 7) >> 3);
    int  output_channel_q2 = ((output_channel_q + 0xF) >> 4);

    if((0 == (image_size & 0x1F)) &&
       (0 == (input_channel_q & 1)) &&
       (0 == (output_channel_q & 0xF)))
    {
        if(image_size >= 48 * 32)
            opt_type = 1;
    }
    else if((0 != (image_size & 0x1F)) &&
            (0 == (input_channel_q & 1)) &&
            (0 == (output_channel_q & 0xF)))
    {
        if(image_size >= 48 * 32)
        {
            int  delta         = (image_size_q2 << 3) - image_size_q;
            if(10 * delta <= image_size_q)
                opt_type = 2;
        }
    }
    else if((0 == (image_size & 0x1F)) &&
            (0 == (input_channel_q & 1)) &&
            (0 != (output_channel_q & 0xF)))
    {
        if(image_size >= 48 * 32)
        {
            int  delta         = (output_channel_q2 << 4) - output_channel_q;
            if((7 * delta) <= output_channel_q)
                opt_type = 3;
        }
    }
    else if((0 != (image_size & 0x1F)) &&
            (0 == (input_channel_q & 1)) &&
            (0 != (output_channel_q & 0xF)))
    {
        if(image_size >= 48 * 32)
        {
            int  delta0  = (image_size_q2 << 3)     - image_size_q;
            int  delta1  = (output_channel_q2 << 4) - output_channel_q;
            if(((10 * delta0) <= image_size_q) && ((7 * delta1) <= output_channel_q))
                opt_type = 4;//output 2 channel
        }
    }
    else
        opt_type = 5;

    if(5 == opt_type)
    {
        if (image_size < 48 * 32)
            opt_type = 6;
    }

    int new_opt_type = 7;
    if (image_size < 48 * 32)
        new_opt_type = 6;
    else if (image_size < 64 * 64)
        new_opt_type = 5;

    if(opt_type != new_opt_type)
    {
        if (ocl_bn->getOpenCLRuntime()->rawTunedLwsMap().size() > 0)
        {
            if (false == ocl_bn->getOpenCLRuntime()->is_kernel_in_raw_tuned_map(conv1x1_opt_kernel[new_opt_type], false)) //let's return, if we cannot find 7 in tuned_map
                return opt_type;
        }
    }

    return new_opt_type;
}

int ConvExecution::evaluate_performance_2d(cl::Kernel&                    kernel, 
                                           std::vector<uint32_t> const&   global_work_space, 
                                           std::vector<uint32_t> const&   local_work_space)
{
    cl::Event event;
    std::vector<uint32_t> internalGlobalWS(2, 1);
    for (size_t i = 0; i < global_work_space.size(); ++i) {
        internalGlobalWS[i] = ROUND_UP(global_work_space[i], std::max((uint32_t)1, local_work_space[i]));
    }
    cl_int error = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueNDRangeKernel(
                    kernel, cl::NullRange,
                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                    cl::NDRange(local_work_space[0], local_work_space[1]),
                    nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);
    
    int cost_time = 0;
    if(0 == error)
        cost_time = (int)mOpenCLBackend->getOpenCLRuntime()->getCostTime(&event);
    return cost_time;
}

bool ConvExecution::is_zero_bias(Convolution2D const* conv_2d_param)
{
    float const* bias_ptr   = conv_2d_param->bias()->data();
    int          bias_size  = conv_2d_param->bias()->size();
    bool         zero_value = true;
    if(bias_size > 0)
    {
        for(int i = 0; i < bias_size ; i ++)
        {
            if(0.0f != bias_ptr[i])
            {
                zero_value = false;
                break;
            }
        }
    }
    return zero_value;
}

#endif
//Shaquille, Added 20201010 End

class ConvolutionCreator : public OpenCLBackend::Creator {
public:
    virtual ~ConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }
        
        if (inputs.size() == 3) {
            return new MultiInputConvExecution(op, backend);
        }

        auto conv2D = op->main_as_Convolution2D();
        if (ConvWinograd::valid(conv2D->common(), inputs[0])) {
//Shaquille, Added 20210221 Start
            int input_c          = conv2D->common()->inputCount();
            int output_c         = conv2D->common()->outputCount();
            auto ocl_backend     = (OpenCLBackend *)backend;
            bool select_winograd = true;
            if (input_c <= 32 && output_c <= 32 &&
                (false == ocl_backend->getOpenCLRuntime()->isSupportedFP16())) {
                select_winograd = false;
            }
//Shaquille, Added 20210221 End
            if(true == select_winograd)
                return new ConvWinograd(conv2D, backend);
        }

        return new ConvExecution(inputs, outputs, op, backend);
    }
};

OpenCLCreatorRegister<ConvolutionCreator> __conv_op(OpType_Convolution);

} // namespace OpenCL
} // namespace MNN
