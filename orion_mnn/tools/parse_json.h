#ifndef PARSE_JSON_H_
#define PARSE_JSON_H_

#include <nlohmann/json.hpp>
#include <nlohmann/fifo_map.hpp>
#include <string>
#include <vector>

static const std::string    kCoreTypes[] = {
    std::string("cpu"),
    std::string("gpu")
};

static const std::string  kProfileJSONNode                   = std::string("inference");
static const std::string  kProfileJSONNodeEngine             = std::string("engine_param");
static const std::string  kProfileJSONNodeCore               = std::string("core_type");
static const std::string  kProfileJSONNodeThreadCount        = std::string("thread_count");
static const std::string  kProfileJSONNodePrecision          = std::string("precision");
static const std::string  kProfileJSONNodePowerMode          = std::string("power_mode");
static const std::string  kProfileJSONNodePrintTensorShape   = std::string("print_tensor_shape");
static const std::string  kProfileJSONNodeModelCache         = std::string("model_cache");
static const std::string  kProfileJSONNodeModel              = std::string("model");
static const std::string  kProfileJSONNodeInTensor           = std::string("inputs");
static const std::string  kProfileJSONNodeOutTensor          = std::string("outputs");
static const std::string  kProfileJSONNodeTensorName         = std::string("name");
static const std::string  kProfileJSONNodeTensorShape        = std::string("shape");

static const std::string  kTensorFormat[] = {
    "NHWC",
    "NCHW",
    "NC4HW4"
};

class InputOption
{
public:
    InputOption(std::string const& json_file)
    {
        nlohmann::json  cfg_json;
        std::ifstream   input_stream(json_file);
        if(false == input_stream.is_open())
            return ;
        input_stream >> cfg_json;

        nlohmann::json  inference_json   = cfg_json.at(kProfileJSONNode);
        nlohmann::json  engine_json      = inference_json.at(kProfileJSONNodeEngine);
        core_type_                       = engine_json.at(kProfileJSONNodeCore);

        thread_count_       = engine_json.at(kProfileJSONNodeThreadCount);
        precision_          = engine_json.at(kProfileJSONNodePrecision);
        power_mode_         = engine_json.at(kProfileJSONNodePowerMode);
        print_tensor_shape_ = engine_json.at(kProfileJSONNodePrintTensorShape);
        model_cache_file_   = engine_json.at(kProfileJSONNodeModelCache);

        model_file_                                 = inference_json.at(kProfileJSONNodeModel);
        std::vector<nlohmann::json>  tensor_in_json = inference_json.at(kProfileJSONNodeInTensor);
        in_tensor_.resize(tensor_in_json.size());
        for(int i = 0 ; i < (int)(tensor_in_json.size()) ; i ++)
        {
            std::get<0>(in_tensor_[i])    = tensor_in_json[i].at(kProfileJSONNodeTensorName);
            std::get<2>(in_tensor_[i])    = tensor_in_json[i].at(kProfileJSONNodeTensorShape).get<std::vector<int> >();
        }

        std::vector<nlohmann::json>  tensor_out_jsons     = inference_json.at(kProfileJSONNodeOutTensor);
        out_tensor_.resize(tensor_out_jsons.size());
        for(int i = 0 ; i < (int)(tensor_out_jsons.size()) ; i ++)
        {
            std::get<0>(out_tensor_[i]) = tensor_out_jsons[i].at(kProfileJSONNodeTensorName);
            std::get<2>(out_tensor_[i]) = tensor_out_jsons[i].at(kProfileJSONNodeTensorShape).get<std::vector<int> >();
        }
        parse_ok_ = true;
    };

    virtual ~InputOption()
    {

    };

    static std::string GetModelPathFromJsonPath(std::string const& json_file, std::string const& model_file)
    {
        struct stat buffer;
        bool model_exist = (stat(model_file.c_str(), &buffer) == 0) ;
        std::string   model_exact_file = model_file;
        if(true == model_exist)
            return model_exact_file;
        
        int json_dir_pos = json_file.rfind("/");
		//windows seperator include '\' and '/', so we should search another seperator
#ifdef _MSC_VER
		if (json_dir_pos < 0)
			json_dir_pos = json_file.rfind("\\");
#endif
        if(json_dir_pos < 0)
            return model_exact_file;
        
        std::string  json_dir = json_file.substr(0, json_dir_pos + 1);
        model_exact_file      = json_dir + model_file;

        return model_exact_file;
    }

public:
    std::string                                                     model_file_;
    int                                                             core_type_;
    int                                                             thread_count_;
    int                                                             precision_;
    int                                                             power_mode_;
    int                                                             print_tensor_shape_;
    std::string                                                     model_cache_file_;
    std::vector<std::tuple<std::string, int, std::vector<int> > >   in_tensor_;
    std::vector<std::tuple<std::string, int, std::vector<int> > >   out_tensor_;
    bool                                                            parse_ok_;
};

#endif