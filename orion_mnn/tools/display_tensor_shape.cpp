#if !defined(_MSC_VER)
#include <getopt.h>
#endif
#include "string_func.h"
#include <iostream>
#include <utility>
#include "MNN/Interpreter.hpp"
#include "../print_tensor.h"

static const int            FAILURE = 1;
static const int            SUCCESS = 0;

static const std::string    kCoreTypes[] = {
    std::string("cpu"),
    std::string("metal"),
    std::string("mps"),
    std::string("opencl"),
    std::string("auto"),
    std::string("nn"),
    std::string("opengl"),
    std::string("vulkan"),
};

static std::string kTensorDimensionType[] = {
    "TENSORFLOW",
    "CAFFE",
    "CAFFE_C4",
};

static std::string kTensorFormatType[] = {
    "NCHW",
    "NHWC",
    "NC4HW4",
    "NHWC4",
    "UNKNOWN",
};

int main(int argc, char** argv)
{
    int            i                   = 0 ;
    std::string    model_file_name     = "" ;
    std::string    core_type_str       = "" ;
    std::string    input_tensor_name   = "" ;
    std::string    output_tensor_name  = "" ;
    int            forward_loop        = 0 ;
    int opt = 0;
#if !defined(_MSC_VER)
    while ((opt = getopt(argc, argv, "hm:c:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
                std::cout
                        << "\nDESCRIPTION:\n"                  
                        << "  -m  model file name, \".mnn\" file.\n"
                        << "  -c  core type.\n"
                        << "\n"
                        << std::endl;
                std::exit(SUCCESS);
                break;
            case 'm':
                model_file_name    = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 'c':
                core_type_str      = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;  
            case 'i':
                input_tensor_name  = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 'o':
                output_tensor_name = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            default:
                std::cout << "Invalid parameter specified." << std::endl;
                std::exit(FAILURE);
        }
    }
#else
	model_file_name    = trim_string(argv[1]);
	core_type_str      = trim_string(argv[2]);
	input_tensor_name  = trim_string(argv[3]);
	output_tensor_name = trim_string(argv[4]);
#endif
    int  core_type = atoi(core_type_str.c_str());
    if(core_type < 0 || core_type > MNN_FORWARD_VULKAN)
        core_type = 0;

    std::vector<std::string>  input_tensors  = split_string(input_tensor_name, ':');
    std::vector<std::string>  output_tensors = split_string(output_tensor_name, ':');

    printf("mnn file:        %s\r\n",       model_file_name.c_str());
    printf("core type:       %s(%d)\r\n",   kCoreTypes[core_type].c_str(), core_type);
    printf("input tensors:   %s\r\n",       input_tensor_name.c_str());
    printf("output tensors:  %s\r\n",       output_tensor_name.c_str());

    MNN::Interpreter*   net     = MNN::Interpreter::createFromFile(model_file_name.c_str());
    if(nullptr == net)
    {
        printf("create model failed\r\n");
        return 0;
    }

    MNN::ScheduleConfig config;
    config.numThread        = 0;
    config.type             = static_cast<MNNForwardType>(core_type);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    backendConfig.power     = MNN::BackendConfig::Power_High;
    config.backendConfig    = &backendConfig;
    MNN::Session*   session = net->createSession(config);
    if(nullptr == session)
    {
        delete net;
        printf("create session failed\r\n");
        return 0;
    }
    net->releaseModel();

    std::map<std::string, MNN::Tensor*> input_tensor_map  = net->getSessionInputAll(session);
    std::map<std::string, MNN::Tensor*> output_tensor_map = net->getSessionOutputAll(session);
    std::vector<std::pair<std::string, MNN::Tensor*> >   input_tensor_list;
    std::vector<std::pair<std::string, MNN::Tensor*> >   output_tensor_list;

    if(input_tensor_name.size() == 0)
    {
        input_tensor_list.resize(input_tensor_map.size());
        i = 0 ;
        std::map<std::string, MNN::Tensor*>::iterator   in_iter  = input_tensor_map.begin();
        while(input_tensor_map.end() != in_iter)
        {
            input_tensor_list[i].first  = in_iter->first;
            input_tensor_list[i].second = in_iter->second;
            i ++;
            in_iter ++;
        }
    }
    else
    {
        input_tensor_list.resize(input_tensors.size());
        for(i = 0 ; i < (int)(input_tensors.size()) ; i ++)
        {
            std::map<std::string, MNN::Tensor*>::iterator in_iter = input_tensor_map.find(input_tensors[i]);
            if(input_tensor_map.end() == in_iter)
            {
                input_tensor_list[i].first  = input_tensors[i];
                input_tensor_list[i].second = nullptr;
            }
            else
            {
                input_tensor_list[i].first  = input_tensors[i];
                input_tensor_list[i].second = in_iter->second;
            }
        }
    }

    if(output_tensor_name.size() == 0)
    {
        output_tensor_list.resize(output_tensor_map.size());
        i = 0;
        std::map<std::string, MNN::Tensor*>::iterator   out_iter = output_tensor_map.begin();
        while(output_tensor_map.end() != out_iter)
        {
            output_tensor_list[i].first  = out_iter->first;
            output_tensor_list[i].second = out_iter->second;
            i ++;
            out_iter ++;
        }
    }
    else
    {
        output_tensor_list.resize(output_tensors.size());
        for(i = 0 ; i < (int)(output_tensors.size()) ; i ++)
        {
            std::map<std::string, MNN::Tensor*>::iterator out_iter = output_tensor_map.find(output_tensors[i]);
            if(output_tensor_map.end() == out_iter)
            {
                output_tensor_list[i].first  = output_tensors[i];
                output_tensor_list[i].second = nullptr;
            }
            else
            {
                output_tensor_list[i].first  = output_tensors[i];
                output_tensor_list[i].second = out_iter->second;
            }
        }
    }

    printf("input_tensors:\n");
    for(i = 0 ; i < ((int)(input_tensor_list.size())) ; i ++)
    {
        printf("input_tensor%d details\r\n",  i);
        printf("    tensor_name: %s\r\n", input_tensor_list[i].first.c_str());
        MNN::Tensor*   cur_tensor = input_tensor_list[i].second;
        PrintTensor(cur_tensor, input_tensor_list[i].first.c_str(), "    ");
    }
    if(input_tensor_list.size() < 1)
        printf("    no input tensors");

    printf("output_tensors:\n");
    for(i = 0 ; i < ((int)(output_tensor_list.size())) ; i ++)
    {
        printf("output_tensor%d details\r\n", i);
        MNN::Tensor*   cur_tensor = output_tensor_list[i].second;
        PrintTensor(cur_tensor, output_tensor_list[i].first.c_str(), "    ");
    }
    if(output_tensor_list.size() < 1)
        printf("    no output tensors");


    if(nullptr != net)
    {
        if(nullptr != session)
            net->releaseSession(session);
        session = nullptr;
        net->releaseModel();
        delete net;
        net = nullptr;
    }
    
    printf("end\r\n");

    return 0;
}