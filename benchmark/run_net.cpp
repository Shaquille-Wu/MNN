//
//  benchmark.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif
#include "string_func.h"
#include <getopt.h>
#include "core/Backend.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "revertMNNModel.hpp"
/**
 TODOs:
 1. dynamically get CPU related info.
 2. iOS support
 */
struct Model {
    std::string name;
    std::string model_file;
};

#if !defined(_MSC_VER)
inline bool file_exist(const char* file) {
    struct stat buffer;
    return stat(file, &buffer) == 0;
}
#endif

Model getModelFile(const char* model_file) {
    Model m;
    std::string   model_file_name = std::string(model_file);
    int           dot_pos         = model_file_name.rfind(".mnn");
    std::string   model_name      = model_file_name.substr(0, dot_pos);
    int           dir_pos         = model_name.rfind('/');
    if(dir_pos < 0)
        dir_pos = 0;
    if(dot_pos > 0)
    {
        m.name       = model_name.substr(dir_pos + 1);
        m.model_file = model_file;
    }

    return m;
}

void setInputData(MNN::Tensor* tensor) {
    float* data = tensor->host<float>();
    for (int i = 0; i < tensor->elementSize(); i++) {
        data[i] = Revert::getRandValue();
    }
}

static inline uint64_t getTimeInUs() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

std::vector<float> run_net(Model&                      model, 
                           int                         loop, 
                           int                         warmup = 10, 
                           int                         forward = MNN_FORWARD_CPU, 
                           bool                        only_inference = true,
                           int                         numberThread = 4, 
                           int                         precision = 2,
                           std::vector<std::string>    tensor_in_names = std::vector<std::string>(0),
                           std::vector<std::string>    tensor_out_names = std::vector<std::string>(0)) 
{
    auto revertor = std::unique_ptr<Revert>(new Revert(model.model_file.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    MNN::ScheduleConfig config;
    config.numThread = numberThread;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    char*          pwd          = new char[1024];
    getcwd(pwd, 1023);
    std::string    pwd_str      = pwd;
    delete[] pwd;

    std::vector<float> costs;
    MNN::Session* session = net->createSession(config);
    net->releaseModel();
    MNN::Tensor* input    = net->getSessionInput(session, NULL);

    // if the model has not the input dimension, umcomment the below code to set the input dims
    // std::vector<int> dims{1, 3, 224, 224};
    // net->resizeTensor(input, dims);
    // net->resizeSession(session);

    const MNN::Backend* inBackend = net->getBackend(session, input);
    std::shared_ptr<MNN::Tensor> givenTensor(MNN::Tensor::createHostTensorFromDevice(input, false));
    MNN::Tensor*   tensor_in_test    = givenTensor.get();
    if(tensor_in_names.size() > 0)
    {
        std::string  tensor_in_file_name = pwd_str + "/" + tensor_in_names[0] + ".raw";
        FILE*  tensor_in_file = fopen(tensor_in_file_name.c_str(), "rb");
        if(nullptr != tensor_in_file)
        {
            fseek(tensor_in_file, 0, SEEK_END);
            int tensor_in_file_size     = (int)(ftell(tensor_in_file));
            fseek(tensor_in_file, 0, SEEK_SET);
            MNN::Tensor*   tensor_in    = givenTensor.get();
            auto      type              = tensor_in->getType();
            float*    buf_in_ptr   = tensor_in->host<float>();
            int       buf_in_size  = tensor_in->size();
            int       N_in         = tensor_in->batch();
            int       C_in         = tensor_in->channel();
            int       H_in         = tensor_in->height();
            int       W_in         = tensor_in->width();
            memset(buf_in_ptr, 0, buf_in_size);
            if(tensor_in_file_size <= buf_in_size)
            {
                MNN::Tensor::InsideDescribe* tensor_desc = MNN::TensorUtils::getDescribe(tensor_in);
                printf("tensor_in: %d\n", tensor_desc->dimensionFormat);
                if(MNN::MNN_DATA_FORMAT_NC4HW4 == tensor_desc->dimensionFormat)
                {
                    float*    buf_in_raw   = new float[tensor_in_file_size >> 2];
                    fread(buf_in_raw, 1, tensor_in_file_size, tensor_in_file);
                    fflush(tensor_in_file);
                    fclose(tensor_in_file);
                    int   i               = 0;
                    int   raw_ele_cnt     = (tensor_in_file_size >> 2);
                    int   channel_ele_cnt = raw_ele_cnt / 3;
                    for(i = 0 ; i < channel_ele_cnt ; i ++)
                    {
                        buf_in_ptr[4 * i]     = buf_in_raw[i];
                        buf_in_ptr[4 * i + 1] = buf_in_raw[i + channel_ele_cnt];
                        buf_in_ptr[4 * i + 2] = buf_in_raw[i + 2 * channel_ele_cnt];
                    }
                    delete[] buf_in_raw;
                }
                else
                {
                    fread(buf_in_ptr, 1, tensor_in_file_size, tensor_in_file);
                    fflush(tensor_in_file);
                    fclose(tensor_in_file);
                }
            }
            else
            {
                fflush(tensor_in_file);
                fclose(tensor_in_file);
                return std::vector<float>(0) ;
            }
        }
    }

    std::vector<MNN::Tensor*>                    tensor_outs(0);
    std::vector<std::shared_ptr<MNN::Tensor> >   tensor_outs_host(0);
    if(tensor_out_names.size() > 0)
    {
        tensor_outs.resize(tensor_out_names.size());
        tensor_outs_host.resize(tensor_out_names.size());
        int tensor_idx = 0;
        for(tensor_idx = 0 ; tensor_idx < (int)(tensor_out_names.size()) ; tensor_idx ++)
        {
            std::string const& tensor_name = tensor_out_names[tensor_idx];
            printf("tensor_out%d: %s\n", tensor_idx, tensor_name.c_str());
            tensor_outs[tensor_idx]      = net->getSessionOutput(session, tensor_name.c_str());
            tensor_outs_host[tensor_idx] = std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(tensor_outs[tensor_idx], false));
        }
    }
    else
    {
        tensor_outs.push_back(net->getSessionOutput(session, NULL));
        tensor_outs_host.push_back(std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(tensor_outs[0], false)));
    }

    // Warming up...
    for (int i = 0; i < warmup; ++i) {
        input->copyFromHostTensor(givenTensor.get());
        net->runSession(session);
        for(int j = 0 ; j < (int)(tensor_outs.size()) ; j ++)
            tensor_outs[j]->copyToHostTensor(tensor_outs_host[j].get());
    }

    for (int round = 0; round < loop; round++) {
        auto timeBegin = getTimeInUs();

        input->copyFromHostTensor(givenTensor.get());
        net->runSession(session);
        for(int j = 0 ; j < (int)(tensor_outs.size()) ; j ++)
            tensor_outs[j]->copyToHostTensor(tensor_outs_host[j].get());

        auto timeEnd = getTimeInUs();
        costs.push_back((timeEnd - timeBegin) / 1000.0);
    }

    MNN::Tensor*   tensor_out_test    = tensor_outs_host[0].get();
    if(tensor_out_names.size() > 0)
    {
        for(int i = 0 ; i < (int)(tensor_out_names.size()); i ++)
        {
            MNN::Tensor*   tensor_out    = tensor_outs_host[i].get();
            float*         buf_out_ptr   = tensor_out->host<float>();
            int            buf_out_size  = tensor_out->size();
            int            N_out         = tensor_out->batch();
            int            C_out         = tensor_out->channel();
            int            H_out         = tensor_out->height();
            int            W_out         = tensor_out->width();

            std::string    tensor_out_file_name = pwd_str + "/" + tensor_out_names[i] + ".raw";
            FILE*  file_out_tensor = fopen(tensor_out_file_name.c_str(), "wb");
            if(nullptr != file_out_tensor)
            {
                std::cout << "tensor_out: " << tensor_out_names[i] << ", buf_len: " << buf_out_size << std::endl;
                printf("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
                       ((float*)(buf_out_ptr))[0], ((float*)(buf_out_ptr))[1], 
                       ((float*)(buf_out_ptr))[2], ((float*)(buf_out_ptr))[3],
                       ((float*)(buf_out_ptr))[4], ((float*)(buf_out_ptr))[5],
                       ((float*)(buf_out_ptr))[6], ((float*)(buf_out_ptr))[7]);
                fwrite(buf_out_ptr, 1, buf_out_size, file_out_tensor);
                fflush(file_out_tensor);
                fclose(file_out_tensor);
            }
        }
    }

    return costs;
}

void displayStats(const std::string& name, const std::vector<float>& costs) {
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs) {
        max = fmax(max, v);
        min = fmin(min, v);
        sum += v;
        //printf("[ - ] cost：%f ms\n", v);
    }
    avg = costs.size() > 0 ? sum / costs.size() : 0;
    printf("[ - ] %-24s    max = %8.3fms  min = %8.3fms  avg = %8.3fms\n", name.c_str(), max, avg == 0 ? 0 : min, avg);
}
static inline std::string forwardType(MNNForwardType type) {
    switch (type) {
        case MNN_FORWARD_CPU:
            return "CPU";
        case MNN_FORWARD_VULKAN:
            return "Vulkan";
        case MNN_FORWARD_OPENCL:
            return "OpenCL";
        case MNN_FORWARD_OPENGL:
            return "OpenGL";
        case MNN_FORWARD_METAL:
            return "Metal";
        default:
            break;
    }
    return "N/A";
}



#ifdef __ANDROID__
#include <errno.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#define BUFFER_SIZE 1024

static uint32_t getNumberOfCPU() {
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp) {
        return 1;
    }
    uint32_t number = 0;
    char buffer[BUFFER_SIZE];
    while (!feof(fp)) {
        char* str = fgets(buffer, BUFFER_SIZE, fp);
        if (!str) {
            break;
        }
        if (memcmp(buffer, "processor", 9) == 0) {
            number++;
        }
    }
    fclose(fp);
    if (number < 1) {
        number = 1;
    }
    return number;
}

static int getCPUMaxFreqKHz(int cpuID) {
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuID);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuID);
        fp = fopen(path, "rb");
        if (!fp) {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuID);
            fp = fopen(path, "rb");
            if (!fp) {
                return -1;
            }
            int maxfrequency = -1;
            fscanf(fp, "%d", &maxfrequency);
            fclose(fp);
            return maxfrequency;
        }
    }
    int maxfrequency = 0;
    while (!feof(fp)) {
        int frequency = 0;
        int history   = fscanf(fp, "%d %*d", &frequency);
        if (history != 1) {
            break;
        }
        if (frequency > maxfrequency) {
            maxfrequency = frequency;
        }
    }
    fclose(fp);
    return maxfrequency;
}

static int sortCPUIDByMaxFrequency(std::vector<int>& cpuIDs, int* littleClusterOffset) {
    const int cpuNumbers = cpuIDs.size();
    *littleClusterOffset = 0;
    if (cpuNumbers == 0) {
        return 0;
    }
    std::vector<int> cpusFrequency;
    cpusFrequency.resize(cpuNumbers);
    for (int i = 0; i < cpuNumbers; ++i) {
        int frequency    = getCPUMaxFreqKHz(i);
        cpuIDs[i]        = i;
        cpusFrequency[i] = frequency;
        // MNN_PRINT("cpu fre: %d, %d\n", i, frequency);
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        for (int j = i + 1; j < cpuNumbers; ++j) {
            if (cpusFrequency[i] < cpusFrequency[j]) {
                // id
                int temp  = cpuIDs[i];
                cpuIDs[i] = cpuIDs[j];
                cpuIDs[j] = temp;
                // frequency
                temp             = cpusFrequency[i];
                cpusFrequency[i] = cpusFrequency[j];
                cpusFrequency[j] = temp;
            }
        }
    }
    int midMaxFrequency = (cpusFrequency.front() + cpusFrequency.back()) / 2;
    if (midMaxFrequency == cpusFrequency.back()) {
        return 0;
    }
    for (int i = 0; i < cpuNumbers; ++i) {
        if (cpusFrequency[i] < midMaxFrequency) {
            *littleClusterOffset = i;
            break;
        }
    }
    return 0;
}


//#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))

#endif

void set_cpu_affinity()
{
#ifdef __ANDROID__
    int cpu_core_num = sysconf(_SC_NPROCESSORS_CONF);
    //LOG_MCNN_CL_INF("cpu core num = %d\n", cpu_core_num);
    int cpu_id = 0;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    
    auto numberOfCPUs = getNumberOfCPU();
    static std::vector<int> sortedCPUIDs;
    static int littleClusterOffset = 0;
    if (sortedCPUIDs.empty()) {
        sortedCPUIDs.resize(numberOfCPUs);
        for (int i = 0; i < numberOfCPUs; ++i) {
            sortedCPUIDs[i] = i;
        }
        sortCPUIDByMaxFrequency(sortedCPUIDs, &littleClusterOffset);
    }

    printf("max core:");
    for (cpu_id = 0; cpu_id < littleClusterOffset; cpu_id++)
    {
        printf("%d ", sortedCPUIDs[cpu_id]);
        CPU_SET(sortedCPUIDs[cpu_id], &mask);
    }
    printf("\n");


    int sys_call_res = syscall(__NR_sched_setaffinity, gettid(), sizeof(mask), &mask);
    //LOG_MCNN_CL_INF("sys call res = %d\n", sys_call_res);
    if (sys_call_res)
    {
        printf("set_cpu_affinity errno = %d\n", (int)errno);
    }
#endif
}


int main(int argc, char* argv[]) 
{
    std::cout << "MNN benchmark" << std::endl;
    std::string  model_file_str    = "";
    std::string  loop_count_str    = "";
    std::string  warm_up_str       = "";
    std::string  forward_type_str  = "";
    std::string  thread_num_str    = "";
    std::string  precision_str     = "";
    std::string  in_tensors_str    = "";
    std::string  out_tensors_str   = "";

    int loop               = 10;
    int warmup             = 10;
    MNNForwardType forward = MNN_FORWARD_CPU;
    int numberThread       = 4;
    int opt                = 0;
    while ((opt = getopt(argc, argv, "hm:l:w:f:t:p:i:o:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
               std::cout << "\nDESCRIPTION:\n"
                         << "  -h  help.\n"
                         << "  -m  model file.\n"
                         << "  -l  loop count.\n"
                         << "  -w  warm-up count.\n"
                         << "  -f  forward_type, 0 for cpu, 3 for opencl, other is invalid\n"
                         << "  -t  thread num, it useful for cpu.\n"
                         << "  -p  precision mode, o for normal, 1 for high, 2 for low, default is 2.\n"
                         << "  -i  input tensor name, sperator is \":\"\n"
                         << "  -o  output tensor name, sperator is \":\".\n"
                         << std::endl;
                std::exit(0);
                break;
            case 'm':
                model_file_str   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;      
            case 'l':
                loop_count_str   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;   
            case 'w':
                warm_up_str      = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;   
            case 'f':
                forward_type_str = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;   
            case 't':
                thread_num_str   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;   
            case 'p':
                precision_str    = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;   
            case 'i':
                in_tensors_str   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;  
            case 'o':
                out_tensors_str  = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break; 
            default:
                std::cout  << "Invalid parameter specified." << std::endl;
                std::exit(-1);
        }
    }

    if ("" != loop_count_str) 
        loop = atoi(loop_count_str.c_str());

    if ("" != warm_up_str) 
        warmup = atoi(warm_up_str.c_str());

    if ("" != forward_type_str) 
        forward = static_cast<MNNForwardType>(atoi(forward_type_str.c_str()));

    if ("" != thread_num_str) 
        numberThread = atoi(thread_num_str.c_str());

    int precision = 2;
    if ("" != precision_str) 
        precision = atoi(precision_str.c_str());

    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numberThread << "** precision=" <<precision << std::endl;
    Model model = getModelFile(model_file_str.c_str());
    std::cout << "--------> Benchmarking... loop = " << loop << ", warmup = " << warmup << std::endl;
    
    std::vector<std::string>    tensor_in_names  = split_string(in_tensors_str, ':');
    std::vector<std::string>    tensor_out_names = split_string(out_tensors_str, ':');
    /* not called yet */
    // set_cpu_affinity();
    
    std::vector<float> costs = run_net(model, 
                                       loop, 
                                       warmup, 
                                       forward, 
                                       false, 
                                       numberThread, 
                                       precision,
                                       tensor_in_names,
                                       tensor_out_names);
    displayStats(model.name, costs);
}
