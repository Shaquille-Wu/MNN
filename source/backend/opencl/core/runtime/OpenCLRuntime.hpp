//
//  OpenCLRuntime.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLRuntime_hpp
#define OpenCLRuntime_hpp


#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <sstream>
#include <string>
#include <vector>
#include "core/Macro.h"
#include "Type_generated.h"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"

namespace MNN {

#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM 0x40C3
#define CL_PERF_HINT_NORMAL_QCOM 0x40C4
#define CL_PERF_HINT_LOW_QCOM 0x40C5
#define CL_CONTEXT_PRIORITY_HINT_QCOM 0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM 0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM 0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM 0x40CC

#define CL_KERNEL_WAVE_SIZE_QCOM 0xAA02

enum GpuType { MALI = 0, ADRENO = 1, RADEON = 2, OTHER = 3 };

class OpenCLRuntime {
public:
    OpenCLRuntime(bool permitFloat16);
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime &) = delete;
    OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

    bool isSupportedFP16() const;
    bool isWeightCpuTransHalf() const;
    bool isDeviceSupportedFP16() const;
    bool isSupportedDotInt8() const;
    bool isSupportedDotAccInt8() const;
    ::cl::Context &context();
    ::cl::CommandQueue &commandQueue();
    uint64_t deviceGlobalMemeryCacheSize() const;
    uint32_t deviceComputeUnits() const;
    uint32_t maxFreq() const;
//Shaquille, Added 20200921 Start
    uint64_t getDevMaxWorkGroupSize() const { return mDevMaxGroupSize; };
//Shaquille, Added 20200921 End
    uint64_t getMaxWorkGroupSize(const ::cl::Kernel &kernel);
    uint64_t GetKernelWaveSize(const cl::Kernel &kernel);
	std::vector<uint32_t> getMaxWorkItemSizes();
//Shaquille, Modified 20200921 Start
    uint64_t getMaxLocalMem() const         { return mLocalMemSize ; } ;
//Shaquille, Modified 20200921 End
    GpuType getGpuType();
    uint64_t maxAllocSize() const;
    void setCommandQueueProfileEnable();
    void setCommandQueueProfileDisable();
//Shaquille, Added 20201107 Start
	bool     is_support_qcom_host_ptr_iocoherent() const    { return qcom_host_ptr_iocoherent_ ; } ;
	int      get_qcom_ext_mem_padding_size() const          { return qcom_ext_mem_padding_size_ ; } ;
	int      get_qcom_page_size() const                     { return qcom_page_size_ ; } ;
//Shaquille, Added 20201107 End
    unsigned int mQueueCount = 0;
    unsigned int getQueueNum();
    
    unsigned int mKernelTime = 0;

    std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t> > >& tunedLwsMap();
//Shaquille, Added 20210127 Start
	std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t> > > const& rawTunedLwsMap() const;
	bool is_kernel_in_raw_tuned_map(std::string const& kernel_name, bool exactly_match) const ;
//Shaquille, Added 20210127 End
    ::cl::Kernel buildKernel(const std::string &programName, const std::string &kernelName,
                             const std::set<std::string> &buildOptions);

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const;

    float flops() const {
        return mFlops;
    }

    double getCostTime(const cl::Event *event);
    double getQueuedTime(const cl::Event *event);
    double getSubmitTime(const cl::Event *event);

    std::pair<const void*, size_t> makeCache();
    void setCache(std::pair<const void*, size_t> cache);
private:
    bool loadProgram(const std::string &programName, cl::Program *program);
    bool buildProgram(const std::string &buildOptionsStr, cl::Program *program);
    bool getDeviceSupportsExtension(const cl::Device &device, const char *extensionName);

    static uint64_t   get_dev_group_size(const cl::Device &device);

private:
    std::shared_ptr<::cl::Context> mContext;
    std::shared_ptr<::cl::Device> mFirstGPUDevicePtr;
    std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
    std::map<std::pair<std::string, std::string>, ::cl::Program> mBuildProgramMap;
    uint64_t mGPUGlobalMemeryCacheSize;
    uint32_t mGPUComputeUnits;
    uint32_t mMaxFreq;
    uint32_t mMaxMemAllocSize;
    uint64_t mMaxLocalMemSize;
    bool mIsSupportedFP16     = false;
    bool mIsDeviceSupportedFP16     = false;
    bool mSupportDotInt8 = false;
    bool mSupportDotAccInt8 = false;
    GpuType mGpuType;
    bool isSetWorkGroupAttribute = true;
    std::string mDefaultBuildParams;
    float mFlops = 4.0f;
    bool mIsCreateError{false};
    
    double mStartNanos;
    double mStopNanos;
//Shaquille, Modified 20201118 Start
#if 0
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::vector<uint32_t>> mTunedLws;
#else
	std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t> > > mTunedLws;
    std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t> > > raw_tuned_lws_;
#endif
//Shaquille, Modified 20201118 End
    std::vector<uint8_t> mBuffer;
    const void* mCacheOutside = nullptr;
    size_t mCacheOutsideSize = 0;
	
	uint64_t                                                                         mDevMaxGroupSize = 0ULL;
	uint64_t                                                                         mLocalMemSize = 0ULL;

//Shaquille, Added 20201107 Start
	bool              qcom_host_ptr_iocoherent_   = false;
	int               qcom_ext_mem_padding_size_  = 0;
	int               qcom_page_size_             = 4096;
//Shaquille, Added 20201107 End
};

} // namespace MNN
#endif  /* OpenCLRuntime_hpp */
