#include "../orion_mnn_impl.h"
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#if !defined(_MSC_VER)
#include <sys/time.h>
#include <getopt.h>
#include <unistd.h>
#endif
#include <vector>
#include <utility>
#include <fstream>
#include <chrono>
#include "string_func.h"
#include "parse_json.h"


static const int            FAILURE = 1;
static const int            SUCCESS = 0;

static int GetTensorFormat(std::string const& format)
{
    int  fmt = 0;
    if("NHWC" == format)
        fmt = 0;
    else if("NCHW" == format)
        fmt = 1;
    else if("NC4HW4" == format)
        fmt = 2;
    else
        fmt = 0;
    
    return fmt;
}

int main(int argc, char** argv)
{
    int            i                     = 0 ;
    std::string    json_file_name        = "" ;
    std::string    input_file_name       = "" ;
    std::string    output_file_name      = "";
    int            warm_up_loop          = 0 ;
    int            forward_loop          = 0 ;
    bool           is_directory          = false;
#if !defined(_MSC_VER)
	int opt = 0;
    while ((opt = getopt(argc, argv, "hj:i:d:w:n:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
                std::cout
                        << "\nDESCRIPTION:\n"                  
                        << "  -j  json file path name, .json file.\n"
                        << "  -i  input data file.\n"
                        << "  -d  is directory style"
                        << "  -w  loop count for warm-up"
                        << "  -n  execute loop count of forward"
                        << "\n"
                        << std::endl;
                std::exit(SUCCESS);
                break;
            case 'j':
                json_file_name   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 'i':
                input_file_name  = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            case 'd':
                is_directory    = (nullptr == optarg ? false : atoi(trim_string(std::string(optarg)).c_str()));
                break;
            case 'w':
                warm_up_loop    = (nullptr == optarg ? 0 : atoi(trim_string(std::string(optarg)).c_str()));
                break;
            case 'n':
                forward_loop    = (nullptr == optarg ? 0 : atoi(trim_string(std::string(optarg)).c_str()));
                break;                            
            default:
                std::cout << "Invalid parameter specified." << std::endl;
                std::exit(FAILURE);
        }
    }
#else
	json_file_name  = trim_string(argv[1]);
	input_file_name = trim_string(argv[2]);
	warm_up_loop    = atoi(trim_string(argv[3]).c_str());
	forward_loop    = atoi(trim_string(argv[4]).c_str());
#endif
    if(warm_up_loop <= 0)
        warm_up_loop = 0;

    if(forward_loop <= 0)
        forward_loop = 1;

    InputOption    inputOpt(json_file_name) ;
    printf("json file:          %s\r\n",           json_file_name.c_str());
    if(false == inputOpt.parse_ok_)
    {
        printf("json parse error\n");
        std::exit(FAILURE);
    }
    printf("mnn file:           %s\r\n",           inputOpt.model_file_.c_str());
    printf("core type:          %s(%d)\r\n",       kCoreTypes[inputOpt.core_type_].c_str(), inputOpt.core_type_);
    printf("thread count:       %d\r\n",           inputOpt.thread_count_);
    printf("precision:          %d\r\n",           inputOpt.precision_);
    printf("power mode:         %d\r\n",           inputOpt.power_mode_);
    printf("print tensor shape: %d\r\n",           inputOpt.print_tensor_shape_);
    printf("model_cache:        %s\r\n",           inputOpt.model_cache_file_.c_str());
    printf("input tensor:       { %s, [ %d, %d, %d, %d ] }\r\n", std::get<0>(inputOpt.in_tensor_[0]).c_str(), 
                                                                     std::get<2>(inputOpt.in_tensor_[0])[0], 
                                                                     std::get<2>(inputOpt.in_tensor_[0])[1], 
                                                                     std::get<2>(inputOpt.in_tensor_[0])[2],
                                                                     std::get<2>(inputOpt.in_tensor_[0])[3]);
    for(i = 1 ; i < (int)(inputOpt.in_tensor_.size()) ; i ++)
        printf("                    { %s, [ %d, %d, %d, %d ] }\r\n", std::get<0>(inputOpt.in_tensor_[i]).c_str(), 
                                                                     std::get<2>(inputOpt.in_tensor_[i])[0],
                                                                     std::get<2>(inputOpt.in_tensor_[i])[1],
                                                                     std::get<2>(inputOpt.in_tensor_[i])[2],
                                                                     std::get<2>(inputOpt.in_tensor_[i])[3]);
    printf("output tensor:      { %s,[ %d, %d, %d, %d ] }\r\n", std::get<0>(inputOpt.out_tensor_[0]).c_str(), 
                                                                     std::get<2>(inputOpt.out_tensor_[0])[0],
                                                                     std::get<2>(inputOpt.out_tensor_[0])[1],
                                                                     std::get<2>(inputOpt.out_tensor_[0])[2],
                                                                     std::get<2>(inputOpt.out_tensor_[0])[3]);
    for(i = 1 ; i < (int)(inputOpt.out_tensor_.size()) ; i ++)
        printf("                    { %s, [ %d, %d, %d, %d ] }\r\n", std::get<0>(inputOpt.out_tensor_[i]).c_str(), 
                                                                     std::get<2>(inputOpt.out_tensor_[i])[0],
                                                                     std::get<2>(inputOpt.out_tensor_[i])[1],
                                                                     std::get<2>(inputOpt.out_tensor_[i])[2],
                                                                     std::get<2>(inputOpt.out_tensor_[i])[3]);
    printf("input data file:    %s\r\n", input_file_name.c_str());
    printf("directory style:    %d\r\n", is_directory);
    printf("warm_up loop:       %d\r\n", warm_up_loop);
    printf("forward loop:       %d\r\n", forward_loop);

    vision::IInference*             orion_mnn_interface   = new vision::OrionMNNImpl;
    int                             in_element_size       = sizeof(float);
    int                             res                   = 0;
    std::vector<unsigned char*>     input_bufs(inputOpt.in_tensor_.size(), nullptr);
    std::vector<unsigned char*>     output_buf(inputOpt.out_tensor_.size());
    std::vector<std::vector<int>>   input_dims(inputOpt.in_tensor_.size());
    std::vector<int>                input_sizes(inputOpt.in_tensor_.size());
    for (i = 0; i < (int)(inputOpt.in_tensor_.size()); i++)
        input_dims[i] = std::get<2>(inputOpt.in_tensor_[i]);
    vision::EngineFeature   features[8];
    int                     feature_cnt           = 0;
	std::vector<std::string>        input_file_list;
	std::string                     exact_model_file      = InputOption::GetModelPathFromJsonPath(json_file_name, inputOpt.model_file_);
	std::chrono::time_point<std::chrono::system_clock> fwd_start;
	std::chrono::time_point<std::chrono::system_clock> fwd_end;
	std::chrono::microseconds                          forward_cost;
    memset(features, 0, sizeof(features));
    feature_cnt = orion_mnn_interface->INF_enum_engine_feature(features, 8);
    printf("orion_mnn there are %d features:\r\n", feature_cnt);
    for(i = 0 ; i < feature_cnt ; i ++)
    {
        printf("feature%d, %s, type %d\r\n", i, features[i].keyName, features[i].valueType);
    }

    features[0].value.nValue = inputOpt.core_type_;
    orion_mnn_interface->INF_set_engine_feature(features + 0);
    features[1].value.nValue = inputOpt.thread_count_ ;
    orion_mnn_interface->INF_set_engine_feature(features + 1);
    features[2].value.nValue = inputOpt.precision_;
    orion_mnn_interface->INF_set_engine_feature(features + 2);
    features[3].value.nValue = inputOpt.power_mode_;
    orion_mnn_interface->INF_set_engine_feature(features + 3);
    features[4].value.nValue = inputOpt.print_tensor_shape_;
    orion_mnn_interface->INF_set_engine_feature(features + 4);
    strcpy((char*)(features[5].value.ucValue), inputOpt.model_cache_file_.c_str());
    orion_mnn_interface->INF_set_engine_feature(features + 5);

    for (i = 0; i < (int)(inputOpt.in_tensor_.size()); i++)
    {
        input_sizes[i] = input_dims[i][0] * input_dims[i][1] * input_dims[i][2] * input_dims[i][3];
        input_bufs[i]  = new unsigned char[input_sizes[i] * in_element_size];
        memset(input_bufs[i], 0, input_sizes[i] * in_element_size);
    }

    bool  read_file_ok = false;
    if(false == is_directory && "" != input_file_name)
    {
        for (i = 0; i < (int)(inputOpt.in_tensor_.size()); i++)
        {
            FILE* input_file = fopen(input_file_name.c_str(), "rb");
            if (nullptr != input_file)
            {
                read_file_ok = true;
                size_t read_size = fread(input_bufs[i], input_sizes[i] * in_element_size, 1, input_file);
                (void)read_size;
                fflush(input_file);
                fclose(input_file);
            }
            else
                printf("read data from file failed, and then, run mode with random data\r\n");
        }
    }
    else
	{
        if (true == is_directory && "" != input_file_name)
            input_file_list = travel_image_dir(input_file_name, { std::string(".raw") }, std::string(""));
	}

    if (input_file_list.size() > 0)
    {
        printf("input_file_dir is not empty, we will run as image_batch style according to the image file directory\n");
        forward_loop = (int)(input_file_list.size());
        printf("input image count is %d, and we will make loop count as it\n", forward_loop);
    }

    for (i = 0; i < (int)(inputOpt.in_tensor_.size()); i++)
    {
        res = orion_mnn_interface->INF_set_data(std::get<0>(inputOpt.in_tensor_[i]).c_str(), input_bufs[i], input_dims[i], vision::INPUT);
        if (0 != res)
        {
            printf("set in data failed\r\n");
            goto ORION_MNN_PROFILE;
        }
    }
        
    for(i = 0 ; i < (int)(inputOpt.out_tensor_.size()) ; i ++)
    {
        std::vector<int>  output_dim  = std::get<2>(inputOpt.out_tensor_[i]);
        int               output_size = output_dim[0] * output_dim[1] * output_dim[2] * output_dim[3] ;
        output_buf[i]   = new unsigned char[output_size * sizeof(float)];
        memset(output_buf[i], 0, output_size * sizeof(float));
        res = orion_mnn_interface->INF_set_data(std::get<0>(inputOpt.out_tensor_[i]).c_str(), output_buf[i], output_dim, vision::OUTPUT);
        if(0 != res)
        {
            printf("set out data failed\r\n");
            goto ORION_MNN_PROFILE;
        }
    }

    res = orion_mnn_interface->INF_load_model(exact_model_file.c_str(), 0);
    if(0 != res)
    {
        printf("mode load failed\r\n");
        goto ORION_MNN_PROFILE;
    }

    printf("warm_up start\n");
    for(i = 0 ; i < warm_up_loop ; i ++)
        res = orion_mnn_interface->INF_forward(1);
    printf("warm_up end\n");

    fwd_start = std::chrono::system_clock::now();
    for(i = 0 ; i < forward_loop ; i ++)
    {
        FILE*        input_file          = nullptr;
        std::string  cur_input_file_name = std::string("");
        bool         img_read_ok         = true;
        if (input_file_list.size() > 0)
        {
            img_read_ok         = false;
            cur_input_file_name = input_file_name + "/" + input_file_list[i];
            input_file          = fopen(cur_input_file_name.c_str(), "rb");
            if (nullptr != input_file)
            {
                fseek(input_file, 0, SEEK_END);
                int  file_len = ftell(input_file);
                fseek(input_file, 0, SEEK_SET);
                if (file_len <= input_sizes[0] * in_element_size)
                {
                    printf("%d, %s read ok\n", i, cur_input_file_name.c_str());
                    fread(input_bufs[0], 1, file_len, input_file);
                    fflush(input_file);
                    fclose(input_file);
                    img_read_ok = true;
                }
                else
                    printf("%d, %s read failed, image size is too large, %d\n", i, cur_input_file_name.c_str(), file_len);
            }
            else
            {
                printf("%d, %s read failed\n", i, cur_input_file_name.c_str());
            }
        }
        res = orion_mnn_interface->INF_forward(1);
        if(0 != res)
        {
            printf("forward failed\r\n");
            goto ORION_MNN_PROFILE;
        }
        else
        {
            if ("" != cur_input_file_name && true == img_read_ok)
            {
                int              last_dot     = cur_input_file_name.rfind(".raw");
                for (int j = 0; j < (int)(inputOpt.out_tensor_.size()); j++)
                {
                    std::string  result_name;
                    if (last_dot >= 0)
                    result_name = cur_input_file_name.substr(0, last_dot);
                    std::string      out_tensor_name = result_name + std::string(".") + std::get<0>(inputOpt.out_tensor_[j]) + std::string(".result_raw");
                    std::vector<int> output_dim      = std::get<2>(inputOpt.out_tensor_[j]);
                    int              out_tensor_size = output_dim[0] * output_dim[1] * output_dim[2] * output_dim[3] * sizeof(float);
                    FILE*            out_tensor_file = fopen(out_tensor_name.c_str(), "wb");
                    if (nullptr != out_tensor_file)
                    {
                        fwrite(output_buf[j], out_tensor_size, 1, out_tensor_file);
                        printf("%s saved\r\n", out_tensor_name.c_str());
                        fflush(out_tensor_file);
                        fclose(out_tensor_file);
                    }
                }
            }
        }
    }
    fwd_end      = std::chrono::system_clock::now();
    forward_cost = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
    printf("forward cost: %lld\r\n", (long long int)(forward_cost.count()));

    if (input_file_list.size() <= 0)
    {
        for (i = 0; i < (int)(inputOpt.out_tensor_.size()); i++)
        {
            std::string      out_tensor_name = std::get<0>(inputOpt.out_tensor_[i]) + std::string(".result_raw");
			while(-1 != out_tensor_name.find('/'))
				out_tensor_name = out_tensor_name.replace(out_tensor_name.find("/"), 1, "_");
			while(-1 != out_tensor_name.find(':'))
				out_tensor_name = out_tensor_name.replace(out_tensor_name.find(":"), 1, "_");
            std::vector<int> output_dim = std::get<2>(inputOpt.out_tensor_[i]);
            int              out_tensor_size = output_dim[0] * output_dim[1] * output_dim[2] * output_dim[3] * sizeof(float);
            FILE*            out_tensor_file = fopen(out_tensor_name.c_str(), "wb");
            if (nullptr != out_tensor_file)
            {
                fwrite(output_buf[i], out_tensor_size, 1, out_tensor_file);
                printf("%s saved\r\n", out_tensor_name.c_str());
                fflush(out_tensor_file);
                fclose(out_tensor_file);
            }
        }
    }

ORION_MNN_PROFILE:
    if(nullptr != orion_mnn_interface)
        delete orion_mnn_interface;
    orion_mnn_interface = nullptr;

    for (i = 0; i < (int)(input_bufs.size()); i++)
    {
        if (nullptr != input_bufs[i])
        {
            delete[] input_bufs[i];
        }
        input_bufs[i] = nullptr;
    }

    for(i = 0 ; i < (int)(output_buf.size()) ; i ++)
    {
        if(nullptr != output_buf[i])
        {
            delete[] output_buf[i];
        }
        output_buf[i] = nullptr;
    }

    printf("end %d\r\n", res);

    return res;
}