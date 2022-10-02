#include <string>
#include <stdio.h>
#if !defined(_MSC_VER)
#include <getopt.h>
#endif
#include <vector>
#include <utility>
#include <fstream>
#include "MNN/Interpreter.hpp"
#include "string_func.h"
#include <nlohmann/json.hpp>
#include <nlohmann/fifo_map.hpp>
#include "../print_tensor.h"

template<class K, class V, class dummy_compare, class A>
using json_fifo_map = nlohmann::fifo_map<K, V, nlohmann::fifo_map_compare<K>, A>;
using OrderJSON     = nlohmann::basic_json<json_fifo_map>;

std::vector<std::pair<std::string, MNN::Tensor*> >  GetSortedTensors(MNN::Interpreter* net, MNN::Session* session, bool input_or_output)
{
    std::map<std::string, MNN::Tensor*>  src_tensor_map;
    if(true == input_or_output)
        src_tensor_map = net->getSessionInputAll(session);
    else
        src_tensor_map = net->getSessionOutputAll(session);
    int                                                                   tensor_count = (int)(src_tensor_map.size());
    std::vector<std::pair<std::string, MNN::Tensor*> >                    tensors_list(tensor_count);
    std::vector<std::pair<int, std::pair<std::string, MNN::Tensor*> > >   sorted_idx(tensor_count);
    int                                                                   i = 0;
    std::map<std::string, MNN::Tensor*>::iterator iter_in  = src_tensor_map.begin();
    while(src_tensor_map.end() != iter_in)
    {
        std::string const&   cur_tensor_name = iter_in->first;
        int                  tensor_idx      = 0;
        if(true == input_or_output)
            tensor_idx = net->GetSessionInputTensorIdx(session, cur_tensor_name.c_str());
        else
            tensor_idx = net->GetSessionOutputTensorIdx(session, cur_tensor_name.c_str());
        sorted_idx[i].first           = tensor_idx;
        (sorted_idx[i].second).first  = cur_tensor_name;
        (sorted_idx[i].second).second = iter_in->second;
        i ++;
        iter_in ++;
    }

    std::sort(sorted_idx.begin(), sorted_idx.end());
    for(i = 0 ; i < tensor_count ; i ++)
    {
        tensors_list[i].first  = (sorted_idx[i].second).first;
        tensors_list[i].second = (sorted_idx[i].second).second;
    }

    return tensors_list;
}

std::vector<int>   GetTensorShape(MNN::Tensor* cur_tensor, bool specify_input_opcl = false)
{
    std::vector<int>  tensor_shape(4, 1);
    std::vector<int>  cur_shape(4, 1);
    int               dims = cur_tensor->dimensions();
    dims = (dims > 4 ? 4 : dims);
    for(int i = 0 ; i < dims ; i ++)
        cur_shape[i] = cur_tensor->length(i);

    MNN::Tensor::InsideDescribe* desc = MNN::TensorUtils::getDescribe(cur_tensor);
    if(false == specify_input_opcl)
    {
        if(cur_tensor->getDimensionType() == MNN::Tensor::TENSORFLOW) //NHWC
        {
            tensor_shape[0] = cur_shape[0];
            tensor_shape[1] = cur_shape[2];
            tensor_shape[2] = cur_shape[3];
            tensor_shape[3] = cur_shape[1];
        }
        else
        {
            if(MNN::MNN_DATA_FORMAT_NC4HW4 == desc->dimensionFormat)
            {
                tensor_shape[0] = cur_shape[0];
                tensor_shape[1] = cur_shape[2];
                tensor_shape[2] = cur_shape[3];
                tensor_shape[3] = cur_shape[1];
            }
            else
            {
                tensor_shape[0] = cur_shape[0];
                tensor_shape[1] = cur_shape[1];
                tensor_shape[2] = cur_shape[2];
                tensor_shape[3] = cur_shape[3];
            }
        }
    }
    else
    {
        //input_tensor's shape should be NC4HW4, if core_type is open_cl
        tensor_shape[0] = cur_shape[0];
        tensor_shape[1] = cur_shape[2];
        tensor_shape[2] = cur_shape[3];
        tensor_shape[3] = cur_shape[1];
    }


    return tensor_shape;
}

int GetMNNTensorInfo(std::string const&                                             model_file, 
                     MNNForwardType                                                 forward_type,
                     MNN::BackendConfig::PrecisionMode                              precision_mode,
                     MNN::BackendConfig::PowerMode                                  power_mode,
                     std::vector<std::tuple<std::string, int, std::vector<int> > >& input_tensors,
                     std::vector<std::tuple<std::string, int, std::vector<int> > >& output_tensors)
{
    input_tensors.clear();
    output_tensors.clear();
    
    MNN::Interpreter* net  = MNN::Interpreter::createFromFile(model_file.c_str());
    if(nullptr == net)
        return -1;

    MNN::ScheduleConfig config;
    config.numThread        = 0;
    config.type             = forward_type;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = precision_mode;
    backendConfig.power     = power_mode;
    config.backendConfig    = &backendConfig;
    MNN::Session*      session = net->createSession(config);
    if(nullptr == session)
    {
        net->releaseModel();
        delete net;
        return -1;
    }
        
    net->releaseModel();

    std::vector<std::pair<std::string, MNN::Tensor*> >  input_tensor_list  = GetSortedTensors(net, session, true);
    std::vector<std::pair<std::string, MNN::Tensor*> >  output_tensor_list = GetSortedTensors(net, session, false);
    int                                                 i                  = 0;
    printf("there %d input_tensor(s):\n", (int)(input_tensor_list.size()));
    input_tensors.resize(input_tensor_list.size());
    for(i = 0 ; i < (int)(input_tensor_list.size()); i ++)
    {
        std::vector<int>     tensor_shape(4);
        std::string const&   cur_tensor_name = input_tensor_list[i].first;
        MNN::Tensor*         cur_tensor      = input_tensor_list[i].second;
        PrintTensor(cur_tensor, cur_tensor_name.c_str(), "    ");
        tensor_shape = GetTensorShape(cur_tensor, MNN_FORWARD_OPENCL == forward_type);
        MNN::Tensor::InsideDescribe* desc = MNN::TensorUtils::getDescribe(cur_tensor);
        std::get<0>(input_tensors[i]) = cur_tensor_name;
        std::get<1>(input_tensors[i]) = MNN_FORWARD_OPENCL != forward_type ? desc->dimensionFormat : MNN::MNN_DATA_FORMAT_NC4HW4;
        std::get<2>(input_tensors[i]) = tensor_shape;
    }

    printf("there %d output_tensor(s):\n", (int)(output_tensor_list.size()));
    output_tensors.resize(output_tensor_list.size());
    for(i = 0 ; i < (int)(output_tensor_list.size()); i ++)
    {
        std::vector<int>     tensor_shape(4);
        std::string const&   cur_tensor_name = output_tensor_list[i].first;
        MNN::Tensor*         cur_tensor      = output_tensor_list[i].second;
        PrintTensor(cur_tensor, cur_tensor_name.c_str(), "    ");
        tensor_shape = GetTensorShape(cur_tensor);
        MNN::Tensor::InsideDescribe* desc = MNN::TensorUtils::getDescribe(cur_tensor);
        std::get<0>(output_tensors[i]) = cur_tensor_name;
        std::get<1>(output_tensors[i]) = desc->dimensionFormat;
        std::get<2>(output_tensors[i]) = tensor_shape;
    }

    if(nullptr != net)
    {
        if(nullptr != session)
            net->releaseSession(session);
        session = nullptr;
        net->releaseModel();
        delete net;
    }

    return 0;
}

OrderJSON  NewToTensorOpJson(bool hwc2chw)
{
    OrderJSON  totensor_op_json;
    totensor_op_json["type"]               = "totensor";
    OrderJSON  totensor_op_param_json;
    totensor_op_param_json["hwc2chw"]      = hwc2chw;
    totensor_op_param_json["swapchannel"]  = false;
    totensor_op_param_json["dtype"]        = "float32";
    totensor_op_json["param"]              = totensor_op_param_json;
    return totensor_op_json;
}

std::vector<OrderJSON>  GenerateTensorJson(std::vector<std::tuple<std::string, int, std::vector<int> > > const&  input_tensors)
{
    std::vector<OrderJSON>   tensor_jsons(input_tensors.size());
    int                      i            = 0 ;
    int                      tensor_count = (int)(input_tensors.size());
    for(i = 0 ; i < tensor_count ; i ++)
    {
        tensor_jsons[i]["name"]  = std::get<0>(input_tensors[i]);
        tensor_jsons[i]["shape"] = std::get<2>(input_tensors[i]);
    }

    return tensor_jsons;
}

int ModifyPreprocessOp(OrderJSON& preprocess_json, bool hwc2chw)
{
    bool  found_to_tensor = false;
    if(true == preprocess_json.contains("ops"))
    {
        std::vector<OrderJSON>    op_jsons = preprocess_json.at("ops");
        int                       op_count = (int)(op_jsons.size());
        for(int i = 0 ; i < op_count ; i ++)
        {
            OrderJSON&          cur_op_json = op_jsons[i];
            std::string const&  cur_type    = cur_op_json.at("type").get<std::string>();
            if("totensor" == cur_type)
            {
                OrderJSON&  cur_param_json     = cur_op_json.at("param");
                bool        swapchannel        = cur_param_json.at("swapchannel").get<bool>();
                cur_param_json["hwc2chw"]      = hwc2chw;
                cur_param_json["swapchannel"]  = swapchannel;
                cur_param_json["dtype"]        = "float32";
                found_to_tensor                = true;
                break;
            }
        }
        if(false == found_to_tensor)
            op_jsons.push_back(NewToTensorOpJson(hwc2chw));
        preprocess_json["ops"] = op_jsons;
    }
    else
    {
        OrderJSON  totensor_op_json = NewToTensorOpJson(hwc2chw);
        std::vector<OrderJSON>   op_jsons( { totensor_op_json } );
        preprocess_json["ops"] = op_jsons;
    }

    return 0;
}

int RewriteJSON(std::string const&                                                    src_json,
                std::string const&                                                    dst_json,
                std::string const&                                                    model_name,
                std::string const                                                     core_type,
                int                                                                   thread_count,
                int                                                                   precision_type,
                int                                                                   power_mode,
                std::vector<std::tuple<std::string, int, std::vector<int> > > const&  input_tensors,
                std::vector<std::tuple<std::string, int, std::vector<int> > > const&  output_tensors)
{
    int             i = 0;
    OrderJSON       cfg_json;
    if("" != src_json)
    {
        std::ifstream   input_stream(src_json);
        if(false == input_stream.is_open())
            return -1 ;
        input_stream >> cfg_json;
    }

    bool  hwc2chw[64] = { false };
    memset(hwc2chw, 0, sizeof(hwc2chw));
    for(i = 0 ; i < (int)(input_tensors.size()); i ++)
    {
        if(MNN::MNN_DATA_FORMAT_NCHW == std::get<1>(input_tensors[i]))
        {
            hwc2chw[i] = true;
            std::vector<int> cur_shape = std::get<2>(input_tensors[i]);
            if(1 == cur_shape[1])
                hwc2chw[i] = false;
        }
        else if(MNN::MNN_DATA_FORMAT_NHWC == std::get<1>(input_tensors[i]))
            hwc2chw[i] = false;
        else if(MNN::MNN_DATA_FORMAT_NC4HW4 == std::get<1>(input_tensors[i]))
            hwc2chw[i] = false;
    }

    OrderJSON    new_postprocess_json;
    if(true == cfg_json.contains("preprocess"))
    {
        if(true == cfg_json.at("preprocess").is_array())
        {
            std::vector<OrderJSON>  preprocess_json = cfg_json.at("preprocess");
            int                     preprocess_cnt  = (int)(preprocess_json.size());
            for(i = 0 ; i < preprocess_cnt ; i ++)
                ModifyPreprocessOp(preprocess_json[i], hwc2chw[i]);
            cfg_json["preprocess"] = preprocess_json;
        }
        else
        {
            ModifyPreprocessOp(cfg_json.at("preprocess"), hwc2chw[0]);
        }
    }
    else
    {
        OrderJSON  totensor_op_json = NewToTensorOpJson(hwc2chw[0]);
        std::vector<OrderJSON>   op_jsons( { totensor_op_json } );
        OrderJSON                preprocess_json;
        preprocess_json["debug"] = false;
        preprocess_json["ops"]   = op_jsons;
        cfg_json["preprocess"]   = preprocess_json;
    }

    OrderJSON   inference_json;
    OrderJSON   engine_param_json;
    std::string model_cache_file = "";
    int         model_last_dot   = model_name.rfind(".mnn");
    if(model_last_dot > 0)
    {
        std::string   model_name_without_dot = model_name.substr(0, model_last_dot);
        model_cache_file                     = model_name_without_dot + ".mnn_cache";
    }


    inference_json["debug"]                  = false;
    engine_param_json["core_type"]           = ("cpu" == core_type ? 0 : 1);
    engine_param_json["thread_count"]        = thread_count;
    engine_param_json["precision"]           = precision_type;
    engine_param_json["power_mode"]          = power_mode;
    engine_param_json["print_tensor_shape"]  = 0;
    engine_param_json["model_cache"]         = ("cpu" == core_type ? "" : model_cache_file);
    inference_json["engine_param"]           = engine_param_json;
    inference_json["engine"]                 = "liborion_mnn.so";
    inference_json["model"]                  = model_name;
    inference_json["inputs"]                 = GenerateTensorJson(input_tensors);
    inference_json["outputs"]                = GenerateTensorJson(output_tensors);
    if(true == cfg_json.contains("postprocess"))
        new_postprocess_json = cfg_json["postprocess"];
    else
    {
        new_postprocess_json["debug"] = false;
        new_postprocess_json["ops"]   = std::vector<OrderJSON>(0) ;
    }
        
    OrderJSON new_cfg_json;
    new_cfg_json["preprocess"]  = cfg_json["preprocess"];
    new_cfg_json["inference"]   = inference_json;
    new_cfg_json["postprocess"] = new_postprocess_json;

    std::ofstream output_stream(dst_json);
    if(false == output_stream.is_open())
        return -1;
    output_stream << new_cfg_json.dump(4) << std::endl;

    return 0;
}

int main(int argc, char* argv[])
{
    std::string  model_file_str     = "";
    std::string  src_json_file_str  = "";
    std::string  dst_json_file_str  = "";
    std::string  core_type_str      = "";
    std::string  precision_type_str = "";
    int          opt                = 0;
#if !defined(_MSC_VER)
    while ((opt = getopt(argc, argv, "hm:s:d:c:p:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
               std::cout << "\nDESCRIPTION:\n"
                         << "  -h  help.\n"
                         << "  -m  model file.\n"
                         << "  -s  src json file.\n"
                         << "  -d  dst json file.\n"
                         << "  -c  core type.\n"
                         << "  -p  precision mode\n"
                         << std::endl;
                std::exit(0);
                break;
            case 'm':
                model_file_str     = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;      
            case 's':
                src_json_file_str  = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;   
            case 'd':
                dst_json_file_str  = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 'c':
                core_type_str      = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 'p':
                precision_type_str = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            default:
                std::cout  << "Invalid parameter specified." << std::endl;
                std::exit(-1);
        }
    }
#else
	model_file_str     = trim_string(argv[1]);
	src_json_file_str  = trim_string(argv[2]);
	dst_json_file_str  = trim_string(argv[3]);
	core_type_str      = trim_string(argv[4]);
	precision_type_str = trim_string(argv[5]);
#endif
    int precision_type = atoi(precision_type_str.c_str());
    printf("model file:     %s\n", model_file_str.c_str());
    printf("src json file:  %s\n", src_json_file_str.c_str());
    printf("dst json file:  %s\n", dst_json_file_str.c_str());
    printf("core type:      %s\n", core_type_str.c_str());
    printf("precision type: %d\n", precision_type);

    MNNForwardType forward_type = MNN_FORWARD_CPU;
    if("cpu" == core_type_str)
        forward_type = MNN_FORWARD_CPU;
    else if("gpu" == core_type_str)
        forward_type = MNN_FORWARD_OPENCL;
    else
        core_type_str = "cpu";

    std::vector<std::tuple<std::string, int, std::vector<int> > > input_tensors;
    std::vector<std::tuple<std::string, int, std::vector<int> > > output_tensors;
    int res = GetMNNTensorInfo(model_file_str, 
                               forward_type, 
                               (MNN::BackendConfig::PrecisionMode)precision_type, 
                               MNN::BackendConfig::Power_High,
                               input_tensors,
                               output_tensors);
    if(0 != res)
    {
        printf("mnn model load error\n");
        return -1;
    }

    int          model_name_pos = model_file_str.rfind("/");
    std::string  model_name     = model_file_str;
    if(model_name_pos >= 0)
        model_name = model_file_str.substr(model_name_pos + 1);
    res = RewriteJSON(src_json_file_str, 
                      dst_json_file_str, 
                      model_name, 
                      core_type_str, 
                      forward_type == MNN_FORWARD_CPU ? 4 : 0, 
                      precision_type, 
                      MNN::BackendConfig::Power_High, 
                      input_tensors, 
                      output_tensors);
    if(0 != res)
    {
        printf("rewrite json file failed\n");
        return -1;
    }

    return res;
}