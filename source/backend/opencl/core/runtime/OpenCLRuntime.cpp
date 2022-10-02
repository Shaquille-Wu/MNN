//
//  OpenCLRuntime.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/Macro.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "CLCache_generated.h"
using namespace CLCache;
//#define ENABLE_OPENCL_TIME_PROFILER
namespace MNN {

extern const std::map<std::string, std::vector<unsigned char>> OpenCLProgramMap;

bool OpenCLRuntime::getDeviceSupportsExtension(const cl::Device &device, const char *extensionName) {
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    auto pos               = extensions.find(extensionName);
    return (pos != std::string::npos);
}

GpuType OpenCLRuntime::getGpuType() {
    return mGpuType;
}

bool OpenCLRuntime::isCreateError() const {
    return mIsCreateError;
}

/*
void GetDeviceExtensions(cl::Device* dev)
{
    std::string  dev_str("");
    dev->getInfo(CL_DEVICE_EXTENSIONS, &dev_str);
    if (dev_str != "") 
    {
        printf("CL_DEVICE_EXTENSIONS:");
        std::string stdDevString;
        stdDevString = std::string(dev_str);
        size_t szOldPos = 0;
        size_t szSpacePos = stdDevString.find(' ', szOldPos);
        while (szSpacePos != stdDevString.npos)
        {
            if (szOldPos > 0)
            {
                printf("\t\t");
            }
            printf("\t\t\t%s\n", stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str());
            
            do {
                szOldPos = szSpacePos + 1;
                szSpacePos = stdDevString.find(' ', szOldPos);
            } while (szSpacePos == szOldPos);
        }
        printf("\n");
    }
    else 
    {

        printf("  CL_DEVICE_EXTENSIONS: None\n");
    }
}

static size_t GetPECount(const cl::Device& device) 
{
    cl::Program program(cl::Context({device}), "__kernel void test(){}", true);
    std::vector<cl::Kernel> kernels;
    program.createKernels(std::addressof(kernels));
    return  kernels[0].getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
}
*/

OpenCLRuntime::OpenCLRuntime(bool permitFloat16) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start OpenCLRuntime !\n");
#endif
    mDefaultBuildParams = " -cl-mad-enable";
    std::vector<cl::Platform> platforms;
    cl_int res = cl::Platform::get(&platforms);
    MNN_CHECK_CL_SUCCESS(res);
	static const int PLAT_SEL_IDX = 1;
    if(platforms.size() > 0 && res == CL_SUCCESS){
        cl::Platform::setDefault(platforms[0]);
        std::vector<cl::Device> gpuDevices;
        res = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);

        if(1 <= gpuDevices.size() && res == CL_SUCCESS){
            mFirstGPUDevicePtr              = std::make_shared<cl::Device>(gpuDevices[0]);
            const std::string deviceName    = mFirstGPUDevicePtr->getInfo<CL_DEVICE_NAME>();
            const std::string deviceVersion = mFirstGPUDevicePtr->getInfo<CL_DEVICE_VERSION>();
            static std::map<std::string, float> gFlopsMap {
                {"Mali-T860", 6.83f},
                {"Mali-T880", 6.83f},
                {"Mali-G51", 6.83f},
                {"Mali-G52", 6.83f},
                {"Mali-G71", 31.61f},
                {"Mali-G72", 31.61f},
                {"Mali-G76", 31.61f},
                {"Adreno (TM) 505", 3.19f},
                {"Adreno (TM) 506", 4.74f},
                {"Adreno (TM) 512", 14.23f},
                {"Adreno (TM) 530", 25.40f},
                {"Adreno (TM) 540", 42.74f},
                {"Adreno (TM) 615", 16.77f},
                {"Adreno (TM) 616", 18.77f},
                {"Adreno (TM) 618", 18.77f},
                {"Adreno (TM) 630", 42.74f},
                {"Adreno (TM) 640", 42.74f},
            };
        
            if (gFlopsMap.find(deviceName) != gFlopsMap.end()) {
                mFlops = gFlopsMap[deviceName];
            }
            const std::string deviceVendor  = mFirstGPUDevicePtr->getInfo<CL_DEVICE_VENDOR>();
            cl_command_queue_properties properties = 0;

        #ifdef ENABLE_OPENCL_TIME_PROFILER
            properties |= CL_QUEUE_PROFILING_ENABLE;
        #endif
            cl_int err;
            // if device is QUALCOMM's and version is 2.0 , set spacial optimized param

            if (deviceName == "QUALCOMM Adreno(TM)" && deviceVersion.substr(0, deviceVersion.find('2')) == "OpenCL ") {
                mGpuType = ADRENO;
                
                //if Adreno version is less than Adreno512, donot set WorkGroupAttribute option
                std::string adrenoVersion = deviceVersion.substr(deviceVersion.size()-3);
                //printf("Adreno Version:%s\n", adrenoVersion.c_str());
                if(adrenoVersion > "300" && adrenoVersion < "512") {
                    isSetWorkGroupAttribute = false;
                }
            } else if (deviceName.find("Mali") != std::string::npos) {
                mGpuType = MALI;
            } else if (deviceVendor.find("Advanced Micro Devices") != std::string::npos) {
                // Radeon series GPU is main product of Advanced Micro Devices (AMD)
                mGpuType = RADEON;
            } else {
                mGpuType = OTHER;
            }

//Shaquille, Added 20201107 Start
			if (ADRENO == mGpuType)
			{
				qcom_host_ptr_iocoherent_  = getDeviceSupportsExtension(*mFirstGPUDevicePtr, "cl_qcom_ext_host_ptr_iocoherent");
				cl_int cl_err              = mFirstGPUDevicePtr->getInfo(CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, &qcom_ext_mem_padding_size_);
				cl_err                    |= mFirstGPUDevicePtr->getInfo(CL_DEVICE_PAGE_SIZE_QCOM, &qcom_page_size_);
			}
//Shaquille, Added 20201107 End

            const std::string extensions = platforms[0].getInfo<CL_PLATFORM_EXTENSIONS>();
            if(mGpuType == ADRENO/* && " " != extensions*/)
            {
                std::vector<cl_context_properties> context_properties;
                context_properties.reserve(5);
                context_properties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
                context_properties.push_back(CL_PERF_HINT_HIGH_QCOM);
                context_properties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
                context_properties.push_back(CL_PRIORITY_HINT_LOW_QCOM);
                context_properties.push_back(0);
                mContext = std::shared_ptr<cl::Context>(new cl::Context({*mFirstGPUDevicePtr}, context_properties.data(), nullptr, nullptr, &err));
            }
            else
            {
                mContext = std::shared_ptr<cl::Context>(new cl::Context({*mFirstGPUDevicePtr}, nullptr, nullptr, nullptr, &err));
            }

            MNN_CHECK_CL_SUCCESS(err);

            mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &err);
            MNN_CHECK_CL_SUCCESS(err);

            mFirstGPUDevicePtr->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &mGPUGlobalMemeryCacheSize);
            mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &mGPUComputeUnits);
            mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &mMaxFreq);
            cl_device_fp_config fpConfig;
            auto success = mFirstGPUDevicePtr->getInfo(CL_DEVICE_HALF_FP_CONFIG, &fpConfig);
            mIsSupportedFP16     = CL_SUCCESS == success && fpConfig > 0;
            mIsDeviceSupportedFP16 = mIsSupportedFP16;
            mIsSupportedFP16     = mIsSupportedFP16 && permitFloat16;

            if(getDeviceSupportsExtension(*(mFirstGPUDevicePtr.get()), "cl_arm_integer_dot_product_int8")){
                mSupportDotInt8 = true;
            }
            if(getDeviceSupportsExtension(*(mFirstGPUDevicePtr.get()), "cl_arm_integer_dot_product_accumulate_int8")){
                mSupportDotAccInt8 = true;
            }
//Shaquille, Added 20200921 Start
			cl_device_local_mem_type local_mem_type = 0;
			cl_ulong                 mem_size       = 0;
			mLocalMemSize = 0;
			mFirstGPUDevicePtr->getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &local_mem_type);
			if (1 == local_mem_type)
				mFirstGPUDevicePtr->getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &mem_size);
			mLocalMemSize = (uint64_t)(mem_size);
            mDevMaxGroupSize = get_dev_group_size(*(mFirstGPUDevicePtr.get()));
            //printf("mLocalMemSize: %lu, mDevMaxGroupSize %lu\n", mLocalMemSize, mDevMaxGroupSize);
            //GetDeviceExtensions(mFirstGPUDevicePtr.get());
//Shaquille, Added 20200921 Ends
        }else{
            mIsCreateError = true;
            MNN_ASSERT(1 <= gpuDevices.size());
        }
    }else{
        mIsCreateError = true;
        MNN_ASSERT(platforms.size() > 0);
    }
}

void OpenCLRuntime::setCommandQueueProfileEnable() {
    mCommandQueuePtr->finish();
    mCommandQueuePtr.reset();
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

    cl_int err;
    mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &err);
    MNN_CHECK_CL_SUCCESS(err);
}

void OpenCLRuntime::setCommandQueueProfileDisable() {
    mCommandQueuePtr->finish();
    mCommandQueuePtr.reset();
#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
#else
    cl_command_queue_properties properties = 0;
#endif    

    cl_int err;
    mCommandQueuePtr = std::make_shared<cl::CommandQueue>(*mContext, *mFirstGPUDevicePtr, properties, &err);
    MNN_CHECK_CL_SUCCESS(err);
}

unsigned int OpenCLRuntime::getQueueNum() {
    mQueueCount++;
    return mQueueCount;
}

std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t> > >& OpenCLRuntime::tunedLwsMap() {
    return mTunedLws;
}

std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t> > > const& OpenCLRuntime::rawTunedLwsMap() const
{
	return raw_tuned_lws_;
}

using lws_tuning_map = std::map<std::pair<std::string, std::vector<uint32_t>>, std::tuple<std::vector<uint32_t>, std::vector<uint32_t> > >;

bool OpenCLRuntime::is_kernel_in_raw_tuned_map(std::string const& kernel_name, bool exactly_match) const
{
    lws_tuning_map::const_iterator  lws_iter   = raw_tuned_lws_.begin();
    bool                            exist_flag = false;
    while (raw_tuned_lws_.end() != lws_iter)
    {
        auto const&  kernel_gws_pair = lws_iter->first;
        auto const&  kernel_name_lws = kernel_gws_pair.first;
        if(true == exactly_match)
        {
            if(kernel_name == kernel_name_lws)
            {
                exist_flag = true;
                break;
            }
        }
        else
        {
            if (0 == kernel_name_lws.find(kernel_name))
            {
                exist_flag = true;
                break;
            }
        }
        lws_iter++;
    }

    return exist_flag;
}

OpenCLRuntime::~OpenCLRuntime() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ~OpenCLRuntime !\n");
#endif
    mBuildProgramMap.clear();
    mCommandQueuePtr.reset();
    mContext.reset();
    mFirstGPUDevicePtr.reset();
#ifdef LOG_VERBOSE
    MNN_PRINT("end ~OpenCLRuntime !\n");
#endif
}

std::vector<size_t> OpenCLRuntime::getMaxImage2DSize() {
    size_t max_height, max_width;
    cl_int err = mFirstGPUDevicePtr->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &max_height);
    MNN_CHECK_CL_SUCCESS(err);
    err = mFirstGPUDevicePtr->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &max_width);
    MNN_CHECK_CL_SUCCESS(err);
    return {max_height, max_width};
}

bool OpenCLRuntime::isSupportedFP16() const {
    return mIsSupportedFP16;
}
bool OpenCLRuntime::isWeightCpuTransHalf() const {
#ifdef USE_HALF_WEIGHT_MEMORY
    return mIsSupportedFP16;
#else
    return false;
#endif
}

bool OpenCLRuntime::isDeviceSupportedFP16() const {
    return mIsDeviceSupportedFP16;
}

bool OpenCLRuntime::isSupportedDotInt8() const {
    return mSupportDotInt8;
}

bool OpenCLRuntime::isSupportedDotAccInt8() const {
    return mSupportDotAccInt8;
}


cl::Context &OpenCLRuntime::context() {
    return *mContext;
}

cl::CommandQueue &OpenCLRuntime::commandQueue() {
    return *mCommandQueuePtr;
}

uint64_t OpenCLRuntime::deviceGlobalMemeryCacheSize() const {
    return mGPUGlobalMemeryCacheSize;
}

uint32_t OpenCLRuntime::deviceComputeUnits() const {
    return mGPUComputeUnits;
}

uint32_t OpenCLRuntime::maxFreq() const {
    return mMaxFreq;
}

uint64_t OpenCLRuntime::maxAllocSize() const {
    return mMaxMemAllocSize;
}

bool OpenCLRuntime::loadProgram(const std::string &programName, cl::Program *program) {
    auto it_source = OpenCLProgramMap.find(programName);
    if (it_source != OpenCLProgramMap.end()) {
        cl::Program::Sources sources;
        std::string source(it_source->second.begin(), it_source->second.end());
        sources.push_back(source);
        *program = cl::Program(context(), sources);
        return true;
    } else {
        MNN_PRINT("Can't find kernel source !\n");
        return false;
    }
}

bool OpenCLRuntime::buildProgram(const std::string &buildOptionsStr, cl::Program *program) {
    AUTOTIME;
    cl_int ret = program->build({*mFirstGPUDevicePtr}, buildOptionsStr.c_str());
    if (ret != CL_SUCCESS) {
        if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(*mFirstGPUDevicePtr) == CL_BUILD_ERROR) {
            std::string buildLog = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*mFirstGPUDevicePtr);
            MNN_PRINT("Program build log: %s \n", buildLog.c_str());
        }
        MNN_PRINT("Build program failed ! \n");
        return false;
    }
    return true;
}

cl::Kernel OpenCLRuntime::buildKernel(const std::string &programName, const std::string &kernelName,
                                      const std::set<std::string> &buildOptions) {
    std::string buildOptionsStr;
    if (mIsSupportedFP16) {
        buildOptionsStr = "-DFLOAT=half -DFLOAT4=half4 -DFLOAT16=half16 -DRI_F=read_imageh -DWI_F=write_imageh -DCONVERT_FLOAT4=convert_half4 -DMNN_SUPPORT_FP16";
    } else {
        buildOptionsStr = "-DFLOAT=float -DFLOAT4=float4 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT4=convert_float4";
    }
    
    if(isSetWorkGroupAttribute) {
        buildOptionsStr += " -DSET_ATTRIBUTE=true";
    } else {
        buildOptionsStr += " -DSET_ATTRIBUTE=false";
    }
    for (auto &option : buildOptions) {
        buildOptionsStr += " " + option;
    }
    buildOptionsStr += mDefaultBuildParams;
    auto key = std::make_pair(programName, buildOptionsStr);

    auto buildProgramInter = mBuildProgramMap.find(key);
    cl::Program program;
    if (buildProgramInter != mBuildProgramMap.end()) {
        program = buildProgramInter->second;
    } else {
        this->loadProgram(programName, &program);
        auto status = this->buildProgram(buildOptionsStr, &program);
        if (!status) {
            FUNC_PRINT_ALL(programName.c_str(), s);
        }
        mBuildProgramMap.emplace(key, program);
    }

    cl_int err;
    cl::Kernel kernel = cl::Kernel(program, kernelName.c_str(), &err);
    MNN_CHECK_CL_SUCCESS(err);
    return kernel;
}

//Shaquille, Added 20200921 Start
uint64_t OpenCLRuntime::get_dev_group_size(const cl::Device& device)
{
    cl_device_id    ocl_dev_id  = device.get();
    cl::Context     ocl_ctx     = cl::Context({ device });
    cl_context      ctx         = ocl_ctx.get();
    cl_int          err         = 0;
    std::string     source      = std::string("__kernel void empty_test(){}");
    const char*     str         = source.c_str();
    const size_t    src_size    = source.size();
    cl_program      program     = clCreateProgramWithSource(ctx, 1, &str, &src_size, &err);
    if(0 != err)
    {
        printf("clCreateProgramWithSource %d\n", err);
        return -1;
    }
    err                         = clBuildProgram(program, 1, &ocl_dev_id, NULL, NULL, NULL);
    if(0 != err)
    {
        printf("clBuildProgram %d\n", err);
        return -1;
    }
    cl_kernel       func_kernel = clCreateKernel(program, "empty_test", &err);
    if(0 != err)
    {
        printf("clCreateKernel %d\n", err);
        return -1;
    }
	size_t          maxWorkGroupSize = 0 ;
	clGetKernelWorkGroupInfo(func_kernel,
                             ocl_dev_id,
		                     CL_KERNEL_WORK_GROUP_SIZE,
		                     sizeof(size_t),
		                     &maxWorkGroupSize,
		                     NULL);
	err = clReleaseKernel(func_kernel);
	err = clReleaseProgram(program);

    return  maxWorkGroupSize; 
}
//Shaquille, Added 20200921 End

uint64_t OpenCLRuntime::getMaxWorkGroupSize(const cl::Kernel &kernel) {
    uint64_t maxWorkGroupSize = 0;
    kernel.getWorkGroupInfo(*mFirstGPUDevicePtr, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize);
    return maxWorkGroupSize;
}

uint64_t OpenCLRuntime::GetKernelWaveSize(const cl::Kernel &kernel) {
    uint64_t kernelWaveSize = 0;
    kernel.getWorkGroupInfo(*mFirstGPUDevicePtr, CL_KERNEL_WAVE_SIZE_QCOM, &kernelWaveSize);
    return kernelWaveSize;
}

std::vector<uint32_t> OpenCLRuntime::getMaxWorkItemSizes() {
    cl::vector<cl::size_type> _workItems;
    mFirstGPUDevicePtr->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &_workItems);
    std::vector<uint32_t> workItems;
    for (int i = 0; i < _workItems.size(); ++i) {
        workItems.push_back(_workItems[i]);
    }
    return workItems;
}

double OpenCLRuntime::getCostTime(const cl::Event *event){
    //cl_int res = mCommandQueuePtr->finish();
    cl_int res = event->wait();
    MNN_CHECK_CL_SUCCESS(res);
    mStartNanos = event->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    mStopNanos = event->getProfilingInfo<CL_PROFILING_COMMAND_END>();
    mKernelTime += (unsigned int)((mStopNanos - mStartNanos) / 1000.0);
    return (mStopNanos - mStartNanos) / 1000.0;
}

double OpenCLRuntime::getQueuedTime(const cl::Event *event){
    //cl_int res = mCommandQueuePtr->finish();
    cl_int res = event->wait();
    MNN_CHECK_CL_SUCCESS(res);
    return (event->getProfilingInfo<CL_PROFILING_COMMAND_START>() - event->getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / 1000.0;
}

double OpenCLRuntime::getSubmitTime(const cl::Event *event){
    //cl_int res = mCommandQueuePtr->finish();
    cl_int res = event->wait();
    MNN_CHECK_CL_SUCCESS(res);
    return (event->getProfilingInfo<CL_PROFILING_COMMAND_START>() - event->getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()) / 1000.0;
}


std::pair<const void*, size_t> OpenCLRuntime::makeCache() {
    if (nullptr != mCacheOutside) {
        return std::make_pair(mCacheOutside, mCacheOutsideSize);
    }
    std::unique_ptr<CacheT> cache(new CacheT);
    // Get All program's binary
    for (auto& iter : mBuildProgramMap) {
        std::unique_ptr<ShaderT> pro(new ShaderT);
        auto program = iter.second;
        auto devicesNumber = program.getInfo<CL_PROGRAM_NUM_DEVICES>();
        auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
        auto binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
        if (binSizes.empty() || devices.empty()) {
            MNN_ERROR("Can't load binary\n");
            continue;
        }
        // Only use first one
        pro->key = iter.first.first;
        pro->buildInfo = iter.first.second;
        pro->buffer.resize(binSizes[0]);
        auto proRaw = program.get();
        auto c = pro->buffer.data();
        clGetProgramInfo(proRaw, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &c, nullptr);
        cache->programs.emplace_back(std::move(pro));
    }
    // Get All Autotuning cache
    for (auto& iter : mTunedLws) {
        std::unique_ptr<AutotuningT> tuning(new AutotuningT);
        tuning->gloablSize = iter.first.second;
//Shaquille, Modified 20201118 Start
#if 0
        tuning->localSize  = iter.second;
#else
		tuning->localSize  = std::get<0>(iter.second);
		tuning->costTime   = std::get<1>(iter.second);
#endif
//Shaquille, Modified 20201118 End
        tuning->key = iter.first.first;
        cache->tunings.emplace_back(std::move(tuning));
    }

    flatbuffers::FlatBufferBuilder builder;
    auto lastOffset = Cache::Pack(builder, cache.get());
    builder.Finish(lastOffset);
    mBuffer.resize(builder.GetSize());
    ::memcpy(mBuffer.data(), builder.GetBufferPointer(), builder.GetSize());
    return std::make_pair(mBuffer.data(), mBuffer.size());
}

void OpenCLRuntime::setCache(std::pair<const void*, size_t> cache) {
    if (nullptr == cache.first) {
        mCacheOutside = nullptr;
        mCacheOutsideSize = 0;
        mBuffer.clear();
        return;
    }
    mCacheOutsideSize = cache.second;
    mCacheOutside = cache.first;
    auto cacheBuffer = GetCache(cache.first);
    // Load Program
    if (nullptr != cacheBuffer->programs()) {
        auto programs = cacheBuffer->programs();
        for (int i=0; i<programs->size(); ++i) {
            auto shaderInfo = programs->GetAs<Shader>(i);
            if (nullptr == shaderInfo->key() || nullptr == shaderInfo->buffer()) {
                continue;
            }
            auto key = shaderInfo->key()->str();
            // Builder Info may be empty
            std::string buildinfo;
            if (shaderInfo->buildInfo()) {
                buildinfo = shaderInfo->buildInfo()->str();
            }
            auto buffer = shaderInfo->buffer()->data();
            size_t bufferSize = shaderInfo->buffer()->size();
            auto deviceId = mFirstGPUDevicePtr->get();
            auto programRaw = clCreateProgramWithBinary(context().get(), 1, &deviceId, &bufferSize, (const unsigned char**)(&buffer), nullptr, nullptr);
            if (!programRaw) {
                MNN_ERROR("Can't load %s - %s load program\n", key.c_str(), buildinfo.c_str());
                continue;
            }
            auto pro = cl::Program(programRaw);
            auto res = buildProgram(buildinfo, &pro);
            if (!res) {
                MNN_ERROR("Can't build %s - %s load program\n", key.c_str(), buildinfo.c_str());
                continue;
            }
            mBuildProgramMap.insert(std::make_pair(std::make_pair(key, buildinfo), pro));
        }
    }

    // Load Auto Turning
    if (nullptr != cacheBuffer->tunings()) {
        auto tuningInfo = cacheBuffer->tunings();
        for (int i=0; i<tuningInfo->size(); ++i) {
            auto tun = tuningInfo->GetAs<Autotuning>(i);
            if (nullptr == tun->gloablSize() || nullptr == tun->localSize() || nullptr == tun->key()) {
                MNN_ERROR("Error tunning info\n");
                continue;
            }
            std::vector<uint32_t> glo(tun->gloablSize()->size());
            for (int v=0; v<glo.size(); ++v) {
                glo[v] = tun->gloablSize()->data()[v];
            }
            std::vector<uint32_t> loc(tun->localSize()->size());
            for (int v=0; v<loc.size(); ++v) {
                loc[v] = tun->localSize()->data()[v];
            }
//Shaquille, Added 20201118 Start
			std::vector<uint32_t> cost_time(tun->costTime()->size());
			for (int v = 0; v<cost_time.size(); ++v) {
				cost_time[v] = tun->costTime()->data()[v];
			}
//Shaquille, Added 20201118 End
            mTunedLws.insert(std::make_pair(std::make_pair(tun->key()->str(), glo), std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>(loc, cost_time)));
        }
//Shaquille, Added 20210127 Start
        raw_tuned_lws_ = mTunedLws;
//Shaquille, Added 20210127 End
    }
}

} // namespace MNN
