/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file orion_mnn_impl.cpp
 * @brief the implementation for inference based on mnn
 * @author wuxiao@ainirobot.com
 * @date 2020-08-24
 */

#include "orion_mnn_impl.h"
#include "print_tensor.h"
#include "tensor_convert.h"
#include <chrono>

namespace vision{

static const std::string  kFeaturenName[OrionMNNImpl::FEATURE_IDX_SUM] = {
    "core_type",
    "thread_count",
    "precision",
    "power_mode",
    "print_tensor_shape",
    "model_cache"
};

static std::map<std::string, int>    kFeatureMapName2Idx;

OrionMNNImpl::OrionMNNImpl():forward_type_(MNN_FORWARD_CPU), 
                             precision_mode_(MNN::BackendConfig::Precision_High),
                             power_mode_(MNN::BackendConfig::Power_High), 
                             thread_count_(0),
                             print_tensor_shape_(false),
                             net_(nullptr),
                             session_(nullptr)
{
    if(kFeatureMapName2Idx.size() <= 0)
    {
        for(int i = 0 ; i < FEATURE_IDX_SUM ; i ++)
        {
            kFeatureMapName2Idx[kFeaturenName[i]] = i;
        }
    }
}

OrionMNNImpl::~OrionMNNImpl()
{
    release_resource();
}

int OrionMNNImpl::INF_load_model(const char *model_file, const int gpuid)
{
    release_resource();

    net_  = MNN::Interpreter::createFromFile(model_file);
    if(nullptr == net_)
        return -INVALID_MODEL_FILE;
	net_->setSessionMode(MNN::Interpreter::Session_Release);

    MNN::ScheduleConfig config;
    config.numThread        = thread_count_;
    config.type             = forward_type_;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = precision_mode_;
    backendConfig.power     = power_mode_;
    config.backendConfig    = &backendConfig;

	std::string  str_model_file = std::string(model_file);
	if ("" != model_cache_file_ && MNN_FORWARD_OPENCL == forward_type_)
	{
        int mnn_cache_last_dot_idx          = model_cache_file_.rfind(".mnn_cache");
        if(mnn_cache_last_dot_idx > 0)
        {
            int  last_slope_idx = str_model_file.rfind("/");
            if(last_slope_idx < 0)
                last_slope_idx = str_model_file.rfind("\\");
            std::string  model_name_without_slope = "";
            if(last_slope_idx >= 0)
            {
                model_name_without_slope = str_model_file.substr(0, last_slope_idx);
                model_name_without_slope = model_name_without_slope + "/" + model_cache_file_;
            }
            else
                model_name_without_slope = model_cache_file_;
            
            if ("" != model_name_without_slope)
            {
                FILE*   cache_file = fopen(model_name_without_slope.c_str(), "rb");
                if(NULL != cache_file)
                {
                    fflush(cache_file);
                    fclose(cache_file);
                    std::string  cache_file_name = model_name_without_slope;
                    net_->setCacheFile(cache_file_name.c_str());
                }
            }
        }
	}

    session_   = net_->createSession(config);
    if(nullptr == session_)
    {
        release_resource();
        return -MODEL_BUILD_FAILED;
    }

	bool resize_res = resize_session(net_, session_, input_usr_buf_map_);
	if (false == resize_res)
		return MODEL_RESIZE_FAILED;

	net_->releaseModel();

    std::map<std::string, TensorInfo>::iterator  iter_in_tensor = input_usr_buf_map_.begin();
    while(input_usr_buf_map_.end() != iter_in_tensor)
    {
        std::string const&  tensor_name    = iter_in_tensor->first;
        MNN::Tensor*        cur_input      = net_->getSessionInput(session_, tensor_name.c_str());
        MNN::Tensor*        input_tensor   = net_->CreateSessionIOTensor(session_, cur_input, MNN::Tensor::TENSOR_SESSION_INPUT, -1);
        input_tensor_map_[tensor_name]     = input_tensor;
        dev_input_tensor_map_[tensor_name] = cur_input;
        if(true == print_tensor_shape_)
            PrintTensor(input_tensor, tensor_name.c_str(), "");
        int  dims             = input_tensor->dimensions();
        dims                  = (dims <= 4 ? dims : 4);
        std::vector<int>  norminal_shape(4, 1);
        for(int d = 0 ; d < dims ; d ++)
            norminal_shape[4 - dims + d] = input_tensor->length(d);
        int  norminal_batch   = norminal_shape[0];
        int  norminal_channel = norminal_shape[1];
        int  norminal_height  = norminal_shape[2];
        int  norminal_width   = norminal_shape[3];
        int  norminal_size    = norminal_width * norminal_height * norminal_channel * norminal_batch;
        if((int)(iter_in_tensor->second.buf_.buf_size_) < norminal_size)
        {
            release_resource();
            return -INPUT_DATA_INVALID;
        }
        iter_in_tensor ++;
    }

    std::map<std::string, TensorInfo>::iterator  iter_out_tensor = output_usr_buf_map_.begin();
    while(output_usr_buf_map_.end() != iter_out_tensor)
    {
        std::string const&  tensor_name     = iter_out_tensor->first;
        MNN::Tensor*        cur_output      = net_->getSessionOutput(session_, tensor_name.c_str());
        int                 expect_host_dim = -1;
        if (MNN_FORWARD_OPENCL == forward_type_)
        {
            MNN::Tensor::InsideDescribe const* desc = MNN::TensorUtils::getDescribe(cur_output);
            if ((4 == cur_output->buffer().dimensions) && 
                (MNN::MNN_DATA_FORMAT_NC4HW4 == desc->dimensionFormat))
            {
                int o_c = cur_output->channel();
                int o_h = cur_output->height();
                int o_w = cur_output->width();
                if (o_c > 4 && o_h >= 32 && o_w >= 32)
                    expect_host_dim = MNN::Tensor::TENSORFLOW;
            }
        }
        MNN::Tensor*        output_tensor   = net_->CreateSessionIOTensor(session_, cur_output, MNN::Tensor::TENSOR_SESSION_OUTPUT, expect_host_dim);
        output_tensor_map_[tensor_name]     = output_tensor;
        dev_output_tensor_map_[tensor_name] = cur_output;
        if(true == print_tensor_shape_)
            PrintTensor(output_tensor, tensor_name.c_str(), "");
        int  dims             = output_tensor->dimensions();
        dims                  = (dims <= 4 ? dims : 4);
        std::vector<int>  norminal_shape(4, 1);
        for(int d = 0 ; d < dims ; d ++)
            norminal_shape[4 - dims + d] = output_tensor->length(d);
        int  norminal_batch   = norminal_shape[0];
        int  norminal_channel = norminal_shape[1];
        int  norminal_height  = norminal_shape[2];
        int  norminal_width   = norminal_shape[3];
        int  norminal_size    = norminal_width * norminal_height * norminal_channel * norminal_batch;
        if((int)(iter_out_tensor->second.buf_.buf_size_) < norminal_size)
        {
            release_resource();
            return -OUTPUT_DATA_INVALID;
        }
        iter_out_tensor ++;
    }

    return 0;
}

int OrionMNNImpl::INF_load_model_from_buffer(char* buffer, uint64_t bufferSize, const int gpuid)
{
    return -1;
}

int OrionMNNImpl::INF_set_data(const char*         tensor_name, 
                               const void*         data,
                               std::vector<int>&   size,
                               LayerInOut          type)
{
    std::string   target_name = std::string(tensor_name);
    unsigned int  buf_size    = 1;
    for(int i = 0 ; i < (int)(size.size()) ; i ++)
    {
        buf_size *= size[i];
    }
    
    if (INPUT == type)
    {
        input_usr_buf_map_[target_name]  = TensorInfo((const int*)(&size[0]), UsrBuffer((unsigned char*)data, buf_size));
    }
    else if (OUTPUT == type)
    {
        output_usr_buf_map_[target_name] = TensorInfo((const int*)(&size[0]), UsrBuffer((unsigned char*)data, buf_size));
    }
    else
    {
        return -1;
    }

    return 0;
}

int OrionMNNImpl::INF_forward(const int batch_size)
{
    if(nullptr == net_ || nullptr == session_)
        return -1;
    
    std::map<std::string, TensorInfo>::iterator  iter_in = input_usr_buf_map_.begin();
    while(input_usr_buf_map_.end() != iter_in)
    {
        std::string const&  tensor_name = iter_in->first;
        std::map<std::string, MNN::Tensor*>::iterator  iter_host = input_tensor_map_.find(tensor_name);
        if(input_tensor_map_.end() == iter_host)
        {
            printf("cannot find input_tensor: %s\n", tensor_name.c_str());
            return -INPUT_DATA_INVALID;
        }
        MNN::Tensor*    dev_in_tensor  = (dev_input_tensor_map_.find(tensor_name))->second;
        MNN::Tensor*    host_tensor    = iter_host->second;

        host_tensor->SyncIONStart(false);
        float*          host_buf         = host_tensor->host<float>();
        int             host_buf_size    = host_tensor->size();
        float*          usr_buf          = (float*)((iter_in->second).buf_.buf_);
        int             usr_buf_size     = sizeof(float) * ((iter_in->second).buf_.buf_size_);
        int             copy_size        = (host_buf_size < usr_buf_size ? host_buf_size : usr_buf_size);
		int             norminal_batch   = host_tensor->batch();
		int             norminal_channel = host_tensor->channel();
		int             norminal_height  = host_tensor->height();
		int             norminal_width   = host_tensor->width();
		norminal_batch    = (0 == norminal_batch ? 1 : norminal_batch);
		norminal_channel  = (0 == norminal_channel ? 1 : norminal_channel);
		norminal_height   = (0 == norminal_height ? 1 : norminal_height);
		norminal_width    = (0 == norminal_width ? 1 : norminal_width);
        MNN::Tensor::InsideDescribe* desc = MNN::TensorUtils::getDescribe(host_tensor);
        //printf("input format %d\n", desc->dimensionFormat);
        if(MNN::MNN_DATA_FORMAT_NC4HW4 != desc->dimensionFormat)
            memcpy(host_buf, usr_buf, copy_size);
        else
        {
            if(4 == norminal_channel)
                memcpy(host_buf, usr_buf, copy_size);
            else
            {
                int  norminal_count   = norminal_width * norminal_height * norminal_channel * norminal_batch;
                
                if(1 == norminal_channel)
                    nhwc_float_to_nc4hw4_1c(usr_buf, host_buf, norminal_count);
                else if(2 == norminal_channel)
                    nhwc_float_to_nc4hw4_2c(usr_buf, host_buf, norminal_count);
                else if(3 == norminal_channel)
                    nhwc_float_to_nc4hw4_3c(usr_buf, host_buf, norminal_count);
                else
                {
                    int  i = 0, j = 0, k = 0;
                    int  host_channel = (((norminal_channel + 3) >> 2) << 2);
                    int  src_pitch    = norminal_width * norminal_channel;
                    memset(host_buf, 0, copy_size);
                    for(i = 0 ; i < norminal_height ; i ++)
                    {
                        for(j = 0 ; j < norminal_width ; j ++)
                        {
                            for(k = 0 ; k < norminal_channel ; k ++)
                            {
                                int  src_pos      = i * src_pitch + j * norminal_channel + k;
                                int  dst_pos      = (k >> 2) * 4 * norminal_width * norminal_height + i * norminal_width * 4 + 4 * j + (k&3);
                                host_buf[dst_pos] = usr_buf[src_pos];
                            }
                        }
                    }
                }
            }
        }
        host_tensor->SyncIONEnd(false);
        dev_in_tensor->copyFromHostTensor(host_tensor);
        iter_in ++;
    }

    net_->runSession(session_);

	copy_out_tensor_from_dev(output_tensor_map_, dev_output_tensor_map_);
    std::map<std::string, MNN::Tensor*>::iterator  iter_out = dev_output_tensor_map_.begin();
    while(dev_output_tensor_map_.end() != iter_out)
    {
        std::string const&  tensor_name = iter_out->first;
        MNN::Tensor*    dev_out_tensor  = iter_out->second;
        MNN::Tensor*    host_out_tensor = (output_tensor_map_.find(tensor_name))->second;

        std::map<std::string, TensorInfo>::iterator  iter_usr = output_usr_buf_map_.find(tensor_name);
        if(output_usr_buf_map_.end() == iter_usr)
        {
            printf("cannot find output_tensor: %s\n", tensor_name.c_str());
            return -OUTPUT_DATA_INVALID;
        }

        int       host_buf_size    = host_out_tensor->size();
        float*    usr_buf          = (float*)((iter_usr->second).buf_.buf_);
        int       usr_buf_size     = sizeof(float) * ((iter_usr->second).buf_.buf_size_);
        int       copy_size        = (host_buf_size < usr_buf_size ? host_buf_size : usr_buf_size);
        int       usr_channel      = (iter_usr->second).dim_[3];
        int       buf_type         = host_out_tensor->getType().code;
        int       buf_bits         = host_out_tensor->getType().bits;
        int       norminal_batch   = host_out_tensor->batch();
        int       norminal_channel = host_out_tensor->channel();
        int       norminal_height  = host_out_tensor->height();
        int       norminal_width   = host_out_tensor->width();
        norminal_batch             = (0 == norminal_batch ? 1 : norminal_batch);
        norminal_channel           = (0 == norminal_channel ? 1 : norminal_channel);
        norminal_height            = (0 == norminal_height ? 1 : norminal_height);
        norminal_width             = (0 == norminal_width ? 1 : norminal_width);
        MNN::Tensor::InsideDescribe* desc = MNN::TensorUtils::getDescribe(host_out_tensor);
        //printf("output format %d\n", desc->dimensionFormat);
        if ((halide_type_int == buf_type || halide_type_uint == buf_type) && 32 != buf_bits)
        {
            printf("tensor_out %s, cannot support output buf_bits %d, if buf_type is int or uint\n", tensor_name.c_str(), buf_bits);
            return -OUTPUT_DATA_INVALID;
        }
        host_out_tensor->SyncIONStart(false);
        float*    host_buf         = host_out_tensor->host<float>();
        if(MNN::MNN_DATA_FORMAT_NC4HW4 != desc->dimensionFormat)
        {
            if ((halide_type_int == buf_type || halide_type_uint == buf_type))
                copy_int_buf_to_float_buf(host_buf, usr_buf, copy_size);
            else
                memcpy(usr_buf, host_buf, copy_size);
        }
        else
        {
            if(4 == norminal_channel)
            {
                if ((halide_type_int == buf_type || halide_type_uint == buf_type))
                    copy_int_buf_to_float_buf(host_buf, usr_buf, copy_size);
                else
                    memcpy(usr_buf, host_buf, copy_size);
            }
            else
            {
                int  norminal_count   = norminal_width * norminal_height * norminal_channel * norminal_batch;
                int  i = 0, j = 0, k = 0;
                if(1 == norminal_channel)
                {
                    if ((halide_type_int == buf_type || halide_type_uint == buf_type))
                        copy_4c_int_buf_to_1c_float_buf(host_buf, usr_buf, norminal_count);
                    else
                        copy_4c_float_buf_to_1c_float_buf(host_buf, usr_buf, norminal_count);
                }
                else if(2 == norminal_channel)
                {
                    if ((halide_type_int == buf_type || halide_type_uint == buf_type))
                        copy_4c_int_buf_to_2c_float_buf(host_buf, usr_buf, norminal_count);
                    else
                        copy_4c_float_buf_to_2c_float_buf(host_buf, usr_buf, norminal_count);
                }
                else if(3 == norminal_channel)
                {
                    if ((halide_type_int == buf_type || halide_type_uint == buf_type))
                        copy_4c_int_buf_to_3c_float_buf(host_buf, usr_buf, norminal_count);
                    else
                        copy_4c_float_buf_to_3c_float_buf(host_buf, usr_buf, norminal_count);
                }
                else
                {
                    if ((halide_type_int == buf_type || halide_type_uint == buf_type))
                    {
                        if(1 == norminal_batch && 1 == norminal_height && 1 == norminal_width)
                            nc4hw4_int_to_buf((int*)host_buf, usr_buf, norminal_count);
                        else
                            nc4hw4_int_to_nhwc((int*)host_buf, usr_buf, norminal_batch, norminal_channel, norminal_height, norminal_width);
                    } 
                    else
                    {
                        if(1 == norminal_batch && 1 == norminal_height && 1 == norminal_width)
                            nc4hw4_float_to_buf(host_buf, usr_buf, norminal_count);
                        else
                        {
                            if (1 == norminal_batch)
                                orion_nc4hw4_to_nhwc_float(host_buf, norminal_height, norminal_width, norminal_channel, usr_buf);
                            else
                                nc4hw4_float_to_nhwc(host_buf, usr_buf, norminal_batch, norminal_channel, norminal_height, norminal_width);
                        }
                    }
                }
            }
        }
        host_out_tensor->SyncIONEnd(false);
        iter_out ++;
    }

    return 0;
}

int OrionMNNImpl::INF_get_result(const char *tensor_name, void* data, int len)
{
    std::string   target_name = std::string(tensor_name);
    unsigned int  target_size = 0;
    
    std::map<std::string, TensorInfo>::iterator  iter_usr = output_usr_buf_map_.find(target_name);
    if(output_usr_buf_map_.end() == iter_usr)
        return 0;

    unsigned char*  tensor_buf = iter_usr->second.buf_.buf_;
    if(data == tensor_buf)
        return 0;
    
    int copy_size = sizeof(float) * iter_usr->second.buf_.buf_size_ ;
    copy_size = (len < copy_size ? len : copy_size);
    memcpy(data, tensor_buf, copy_size);

    return copy_size;
}

int OrionMNNImpl::INF_set_thread_num(int thread_num)
{
    return 0;
}

int OrionMNNImpl::INF_get_last_output(void* pData)
{
    return -1;
}

int OrionMNNImpl::INF_get_dims(const char *pcName, std::vector<int>& sz)
{
    return -1;
}

int OrionMNNImpl::INF_load_model_config(const char *model_file, const char* plugin_module, const int gpuid )
{
    return -1;
}

int OrionMNNImpl::INF_load_model_config_from_buffer(char*       buffer, 
                                                    uint64_t    buffer_size, 
                                                    char*       config_buffer, 
                                                    uint64_t    config_buffer_size, 
                                                    const int   gpuid)
{
    return -1;
}

int OrionMNNImpl::INF_enum_engine_feature(EngineFeature *engine_feature, int n)
{
    if(n < FEATURE_IDX_SUM)   return 0;
    
    memset(engine_feature, 0, FEATURE_IDX_SUM * sizeof(vision::EngineFeature));

    engine_feature[FEATURE_IDX_CORE_TYPE].valueType       = vision::INT;
    strcpy(engine_feature[FEATURE_IDX_CORE_TYPE].keyName, kFeaturenName[FEATURE_IDX_CORE_TYPE].c_str());
    engine_feature[FEATURE_IDX_CORE_TYPE].value.nValue    = (MNN_FORWARD_OPENCL == forward_type_ ? 1 : 0) ;

    engine_feature[FEATURE_IDX_THREAD_COUNT].valueType    = vision::INT;
    strcpy(engine_feature[FEATURE_IDX_THREAD_COUNT].keyName, kFeaturenName[FEATURE_IDX_THREAD_COUNT].c_str());
    engine_feature[FEATURE_IDX_THREAD_COUNT].value.nValue = thread_count_;

    engine_feature[FEATURE_IDX_PRECISION].valueType       = vision::INT;
    strcpy(engine_feature[FEATURE_IDX_PRECISION].keyName, kFeaturenName[FEATURE_IDX_PRECISION].c_str());
    engine_feature[FEATURE_IDX_PRECISION].value.nValue    = precision_mode_;

    engine_feature[FEATURE_IDX_POWER_MODE].valueType      = vision::INT;
    strcpy(engine_feature[FEATURE_IDX_POWER_MODE].keyName, kFeaturenName[FEATURE_IDX_POWER_MODE].c_str());
    engine_feature[FEATURE_IDX_POWER_MODE].value.nValue   = power_mode_;

    engine_feature[FEATURE_IDX_PRINT_TENSOR_SHAPE].valueType      = vision::INT;
    strcpy(engine_feature[FEATURE_IDX_PRINT_TENSOR_SHAPE].keyName, kFeaturenName[FEATURE_IDX_PRINT_TENSOR_SHAPE].c_str());
    engine_feature[FEATURE_IDX_PRINT_TENSOR_SHAPE].value.nValue   = print_tensor_shape_;
    

    engine_feature[FEATURE_IDX_MODEL_CACHE].valueType             = vision::BYTES;
    strcpy(engine_feature[FEATURE_IDX_MODEL_CACHE].keyName, kFeaturenName[FEATURE_IDX_MODEL_CACHE].c_str());

    return FEATURE_IDX_SUM;
}

int OrionMNNImpl::INF_set_engine_feature(const EngineFeature *engine_feature)
{
    int  res = 0;

    std::map<std::string, int>::iterator  iter = kFeatureMapName2Idx.find(std::string(engine_feature->keyName));

    if(kFeatureMapName2Idx.end() == iter)
        return -1;

    int  feature_idx = iter->second;

    switch(feature_idx)
    {
        case FEATURE_IDX_CORE_TYPE:
            if(0 == engine_feature->value.nValue)
                forward_type_ = MNN_FORWARD_CPU;
            else if(1 == engine_feature->value.nValue)
                forward_type_ = MNN_FORWARD_OPENCL;
            break;
        case FEATURE_IDX_THREAD_COUNT:
            thread_count_     = engine_feature->value.nValue;
            break;
        case FEATURE_IDX_PRECISION:
            switch(engine_feature->value.nValue)
            {
                case 0:
                    precision_mode_ = MNN::BackendConfig::Precision_Normal;
                    break;
                case 1:
                    precision_mode_ = MNN::BackendConfig::Precision_High;
                    break;
                case 2:
                    precision_mode_ = MNN::BackendConfig::Precision_Low;
                    break;
                default:
                    break;
            }
            break;
        case FEATURE_IDX_POWER_MODE:
            switch(engine_feature->value.nValue)
            {
                case 0:
                    power_mode_ = MNN::BackendConfig::Power_Normal;
                    break;
                case 1:
                    power_mode_ = MNN::BackendConfig::Power_High;
                    break;
                case 2:
                    power_mode_ = MNN::BackendConfig::Power_Low;
                    break;
                default:
                    break;
            }
            break;
        case FEATURE_IDX_PRINT_TENSOR_SHAPE:
            print_tensor_shape_ = (0 == engine_feature->value.nValue ? false : true);
            break;
        case FEATURE_IDX_MODEL_CACHE:
            model_cache_file_   = std::string((char*)(engine_feature->value.ucValue));
            break;
        default:
            break;
    }
    
    return res;
}

bool OrionMNNImpl::resize_session(MNN::Interpreter*                         net, 
	                              MNN::Session*                             session, 
	                              std::map<std::string, TensorInfo> const&  usr_inputs) noexcept
{
	std::map<std::string, TensorInfo>::const_iterator  iter_in_tensor = usr_inputs.begin();
	std::vector<std::pair<MNN::Tensor*, std::vector<int> > >  resize_tensor(0);
	while (usr_inputs.end() != iter_in_tensor)
	{
		std::string const&  tensor_name    = iter_in_tensor->first;
		TensorInfo const&   tensor_info    = iter_in_tensor->second;
		MNN::Tensor*        cur_input      = net->getSessionInput(session, tensor_name.c_str());
		int  norminal_batch   = cur_input->batch();
		int  norminal_channel = cur_input->channel();
		int  norminal_height  = cur_input->height();
		int  norminal_width   = cur_input->width();
		norminal_batch        = (0 == norminal_batch ? 1 : norminal_batch);
		norminal_channel      = (0 == norminal_channel ? 1 : norminal_channel);
		norminal_height       = (0 == norminal_height ? 1 : norminal_height);
		norminal_width        = (0 == norminal_width ? 1 : norminal_width);
		int  norminal_size    = norminal_width * norminal_height * norminal_channel * norminal_batch;
		int  usr_height       = 1;
		int  usr_width        = 1;
		MNN::Tensor::InsideDescribe* tensor_desc = MNN::TensorUtils::getDescribe(cur_input);
		if (MNN::MNN_DATA_FORMAT_NCHW == tensor_desc->dimensionFormat)
		{
			usr_height = tensor_info.dim_[2];
			usr_width  = tensor_info.dim_[3];
		}
		else if (MNN::MNN_DATA_FORMAT_NHWC == tensor_desc->dimensionFormat)
		{
			usr_height = tensor_info.dim_[1];
			usr_width  = tensor_info.dim_[2];
		}
		else if (MNN::MNN_DATA_FORMAT_NC4HW4 == tensor_desc->dimensionFormat)
		{
			usr_height = tensor_info.dim_[1];
			usr_width  = tensor_info.dim_[2];
		}

		if (usr_width != norminal_width || usr_height != norminal_height)
			resize_tensor.push_back(std::make_pair(cur_input, std::vector<int>({ usr_width, usr_height })));

		iter_in_tensor++;
	}

	int  resize_tensor_count = (int)(resize_tensor.size());
	if (resize_tensor_count > 0)
	{
		for (int i = 0; i < resize_tensor_count; i++)
		{
			MNN::Tensor*      cur_input = resize_tensor[i].first;
			int  dims = cur_input->dimensions();
			dims = (dims <= 4 ? dims : 4);
			std::vector<int>  raw_shape(4, 1);
			for (int d = 0; d < dims; d++)
				raw_shape[4 - dims + d] = cur_input->length(d);
			int  raw_batch   = raw_shape[0];
			int  raw_channel = raw_shape[1];
			int  raw_height  = raw_shape[2];
			int  raw_width   = raw_shape[3];
			int  usr_width   = (resize_tensor[i].second)[0];
			int  usr_height  = (resize_tensor[i].second)[1];
			net->resizeTensor(cur_input, raw_batch, raw_channel, usr_height, usr_width);
		}
		net->resizeSession(session);
	}

	return true;
}

bool OrionMNNImpl::copy_out_tensor_from_dev(std::map<std::string, MNN::Tensor*>& host_tensors, 
	                                        std::map<std::string, MNN::Tensor*>& dev_tensors) noexcept
{
	std::vector<MNN::Tensor*>  host_tensor_vec(dev_tensors.size(), nullptr);
	std::vector<MNN::Tensor*>  dev_tensor_vec(dev_tensors.size(), nullptr);
	std::map<std::string, MNN::Tensor*>::iterator  iter_out   = dev_tensors.begin();
	int                                            tensor_idx = 0;
	while (dev_tensors.end() != iter_out)
	{
		std::string const&  tensor_name     = iter_out->first;
		MNN::Tensor*        dev_out_tensor  = iter_out->second;
		MNN::Tensor*        host_out_tensor = (host_tensors.find(tensor_name))->second;
		dev_tensor_vec[tensor_idx]  = dev_out_tensor;
		host_tensor_vec[tensor_idx] = host_out_tensor;
		tensor_idx++;
		iter_out++;
	}

	MNN::Tensor::copy_to_host_batch(dev_tensor_vec, host_tensor_vec);

	return true;
}

void OrionMNNImpl::nhwc_float_to_nc4hw4_1c(float* src_buf, float* dst_buf, int norminal_count)
{
    int j = 0;
    for(int i = 0 ; i < norminal_count ; i += 1, j += 4)
    {
        dst_buf[j]     = src_buf[i];
        dst_buf[j + 1] = 0.0f;
        dst_buf[j + 2] = 0.0f;
        dst_buf[j + 3] = 0.0f;
    }
}

void OrionMNNImpl::nhwc_float_to_nc4hw4_2c(float* src_buf, float* dst_buf, int norminal_count)
{
    int j = 0;
    for(int i = 0 ; i < norminal_count ; i += 2, j += 4)
    {
        dst_buf[j]     = src_buf[i];
        dst_buf[j + 1] = src_buf[i + 1];
        dst_buf[j + 2] = 0.0f;
        dst_buf[j + 3] = 0.0f;
    }
}

void OrionMNNImpl::nhwc_float_to_nc4hw4_3c(float* src_buf, float* dst_buf, int norminal_count)
{
    int j = 0;
    for(int i = 0 ; i < norminal_count ; i += 3, j += 4)
    {
        dst_buf[j]     = src_buf[i];
        dst_buf[j + 1] = src_buf[i + 1];
        dst_buf[j + 2] = src_buf[i + 2];
        dst_buf[j + 3] = 0.0f;
    }
}

void OrionMNNImpl::copy_int_buf_to_float_buf(void* int_buf, void* dst_buf, int buf_size)
{
    int i = 0, j = 0;
    for(i = 0, j = 0; i < buf_size ; i += 4, j ++)
        ((float*)dst_buf)[j] = ((int*)int_buf)[j];
}

void OrionMNNImpl::copy_4c_float_buf_to_1c_float_buf(void* src_buf, void* dst_buf, int norminal_count)
{
    int i = 0, j = 0;
    for(i = 0 ; i < norminal_count ; i += 1, j += 4)
        ((float*)dst_buf)[i] = ((float*)src_buf)[j];
}

void OrionMNNImpl::copy_4c_float_buf_to_2c_float_buf(void* src_buf, void* dst_buf, int norminal_count)
{
    int i = 0, j = 0;
    for(i = 0 ; i < norminal_count ; i += 2, j += 4)
    {
        ((float*)dst_buf)[i]     = ((float*)src_buf)[j];
        ((float*)dst_buf)[i + 1] = ((float*)src_buf)[j + 1];
    }  
}

void OrionMNNImpl::copy_4c_float_buf_to_3c_float_buf(void* src_buf, void* dst_buf, int norminal_count)
{
    int i = 0, j = 0;
    for(i = 0 ; i < norminal_count ; i += 3, j += 4)
    {
        ((float*)dst_buf)[i]     = ((float*)src_buf)[j];
        ((float*)dst_buf)[i + 1] = ((float*)src_buf)[j + 1];
        ((float*)dst_buf)[i + 2] = ((float*)src_buf)[j + 2];
    }
}

void OrionMNNImpl::copy_4c_int_buf_to_1c_float_buf(void* int_buf, void* dst_buf, int norminal_count)
{
    int i = 0, j = 0;
    for(i = 0 ; i < norminal_count ; i += 1, j += 4)
        ((float*)dst_buf)[i] = ((int*)int_buf)[j];
}

void OrionMNNImpl::copy_4c_int_buf_to_2c_float_buf(void* int_buf, void* dst_buf, int norminal_count)
{
    int i = 0, j = 0;
    for(i = 0 ; i < norminal_count ; i += 2, j += 4)
    {
        ((float*)dst_buf)[i]     = ((int*)int_buf)[j];
        ((float*)dst_buf)[i + 1] = ((int*)int_buf)[j + 1];
    }  
}

void OrionMNNImpl::copy_4c_int_buf_to_3c_float_buf(void* int_buf, void* dst_buf, int norminal_count)
{
    int i = 0, j = 0;
    for(i = 0 ; i < norminal_count ; i += 3, j += 4)
    {
        ((float*)dst_buf)[i]     = ((int*)int_buf)[j];
        ((float*)dst_buf)[i + 1] = ((int*)int_buf)[j + 1];
        ((float*)dst_buf)[i + 2] = ((int*)int_buf)[j + 2];
    } 
}

void OrionMNNImpl::nc4hw4_int_to_nhwc(int* nc4hw4_int_buf,   float* dst_buf, int n, int c, int h, int w)
{
    int  host_channel_4    = c >> 2;
    int  host_channel_tail = c - host_channel_4;
    int  dst_pitch         = w * c;
    int  i = 0, j = 0, k = 0;
    for(k = 0 ; k < host_channel_4 ; k ++)
    {
        int src_offset = k * 4 * w * h;
        for(i = 0 ; i < h ; i ++)
        {
            for(j = 0 ; j < w ; j ++)
            {
                int  src_pos  = src_offset + i * w * 4 + 4 * j;
                int  dst_pos  = i * dst_pitch + j * c + 4 * k;
                dst_buf[dst_pos]     = nc4hw4_int_buf[src_pos];
                dst_buf[dst_pos + 1] = nc4hw4_int_buf[src_pos + 1];
                dst_buf[dst_pos + 2] = nc4hw4_int_buf[src_pos + 2];
                dst_buf[dst_pos + 3] = nc4hw4_int_buf[src_pos + 3];
            }
        }
    }

    if(host_channel_tail > 0)
    {
        int src_offset = host_channel_4 * 4 * w * h;
        for(i = 0 ; i < h ; i ++)
        {
            for(j = 0 ; j < w ; j ++)
            {
                for(k = 0 ; k < host_channel_tail ; k ++)
                {
                    int  src_pos     = src_offset + i * w * 4 + 4 * j + k;
                    int  dst_pos     = i * dst_pitch + j * c + 4 * host_channel_4 + k;
                    dst_buf[dst_pos] = nc4hw4_int_buf[src_pos];
                }
            }
        }
    }
}

void OrionMNNImpl::nc4hw4_float_to_nhwc(float* nc4hw4_float_buf, float* dst_buf, int n, int c, int h, int w)
{
    int  host_channel_4    = c >> 2;
    int  host_channel_tail = c - (host_channel_4 << 2);
    int  dst_pitch         = w * c;
    int  i = 0, j = 0, k = 0;
    for(k = 0 ; k < host_channel_4 ; k ++)
    {
        int src_offset = k * 4 * w * h;
        for(i = 0 ; i < h ; i ++)
        {
            for(j = 0 ; j < w ; j ++)
            {
                int  src_pos  = src_offset + i * w * 4 + 4 * j;
                int  dst_pos  = i * dst_pitch + j * c + 4 * k;
                dst_buf[dst_pos]     = nc4hw4_float_buf[src_pos];
                dst_buf[dst_pos + 1] = nc4hw4_float_buf[src_pos + 1];
                dst_buf[dst_pos + 2] = nc4hw4_float_buf[src_pos + 2];
                dst_buf[dst_pos + 3] = nc4hw4_float_buf[src_pos + 3];
            }
        }
    }

    if(host_channel_tail > 0)
    {
        int src_offset = host_channel_4 * 4 * w * h;
        for(i = 0 ; i < h ; i ++)
        {
            for(j = 0 ; j < w ; j ++)
            {
                for(k = 0 ; k < host_channel_tail ; k ++)
                {
                    int  src_pos     = src_offset + i * w * 4 + 4 * j + k;
                    int  dst_pos     = i * dst_pitch + j * c + 4 * host_channel_4 + k;
                    dst_buf[dst_pos] = nc4hw4_float_buf[src_pos];
                }
            }
        }
    }
}

void OrionMNNImpl::nc4hw4_int_to_buf(int* nc4hw4_int_buf, float* dst_buf, int norminal_count)
{
    for(int i = 0 ; i < norminal_count ; i ++)
        dst_buf[i] = nc4hw4_int_buf[i];
}

void OrionMNNImpl::nc4hw4_float_to_buf(float* nc4hw4_float_buf, float* dst_buf, int norminal_count)
{
    memcpy(dst_buf, nc4hw4_float_buf, norminal_count * sizeof(float));
}

}; //namespace vision

void* CreateInference() 
{
    vision::IInference* infer = new  vision::OrionMNNImpl();
    return (void*)(infer);
}

void  DestroyInference(void* inference_instance)
{
    if(0 == inference_instance) return;
    
    vision::IInference* infer = reinterpret_cast<vision::IInference*>(inference_instance);

    delete infer;
}