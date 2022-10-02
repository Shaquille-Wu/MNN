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
#include <math.h>
#if defined(_MSC_VER)
#include <Windows.h>
#include <direct.h>
#undef min
#undef max
#else
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif
#include "string_func.h"
#if !defined(_MSC_VER)
#include <getopt.h>
#endif

#include "filesystem/path.h"
#include "core/Backend.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "revertMNNModel.hpp"
#include "parse_json.h"

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

static int  GetOutputHostTensorDimType(MNN::Tensor const* dev_tensor, int forward_type)
{
    int    expect_host_dim_type = -1;
    if (MNN_FORWARD_OPENCL == forward_type)
    {
        MNN::Tensor::InsideDescribe const* desc = MNN::TensorUtils::getDescribe(dev_tensor);
        if ((4 == dev_tensor->buffer().dimensions) &&
            (MNN::MNN_DATA_FORMAT_NC4HW4 == desc->dimensionFormat))
        {
            int o_c = dev_tensor->channel();
            int o_h = dev_tensor->height();
            int o_w = dev_tensor->width();
            if (o_c > 4 && o_h >= 32 && o_w >= 32)
                expect_host_dim_type = MNN::Tensor::TENSORFLOW;
        }
    }

    return expect_host_dim_type;
}

static void nc4hw4_to_nchw(float const* src, int n, int c, int h, int w, float* dst)
{
    int  cq            = (c + 3) >> 2;
    int  src_line_size = 4 * w;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < h; k++)
            {
                for (int m = 0; m < w; m++)
                {
                    int dst_pos  = i * c * h * w + j * h * w + k * w + m;
                    int src_pos  = i * cq * h * src_line_size + (j >> 2) * h * src_line_size + k * src_line_size + 4 * m + (j & 3);
                    dst[dst_pos] = src[src_pos];
                }
            }
        }
    }
}

static std::vector<uint64_t>   tuning_model(MNN::ScheduleConfig const&            mnn_config,
                                            std::shared_ptr<MNN::Interpreter>     mnn_net,
                                            std::string const&                    mnn_cache_file,
                                            int                                   tuning_cnt,
                                            int                                   warmup,
                                            int                                   loop,
                                            std::vector<std::string> const&       tensor_out_names)
{
    printf("tuning start...\n");
    filesystem::path mnn_cache_path(mnn_cache_file);
    int                    best_idx   = 0;
    uint64_t               best_cost  = 0xFFFFFFFFFFFFFFFF;
    std::vector<uint64_t>  tuning_result(tuning_cnt, 0);
    mnn_cache_path.remove_file();

    for(int i = 0 ; i < tuning_cnt ; i ++)
    {
        printf("tuning %d start ...\n", i);
        mnn_cache_path.remove_file();

        mnn_net->setCacheFile(mnn_cache_file.c_str());
        MNN::Session*       session   = mnn_net->createSession(mnn_config);
        MNN::Tensor*        input     = mnn_net->getSessionInput(session, NULL);
        const MNN::Backend* inBackend = mnn_net->getBackend(session, input);
        std::shared_ptr<MNN::Tensor> givenTensor(mnn_net->CreateSessionIOTensor(session, input, MNN::Tensor::TENSOR_SESSION_INPUT, -1));

        std::vector<MNN::Tensor*>                    tensor_outs(0);
        std::vector<std::shared_ptr<MNN::Tensor> >   tensor_outs_host(0);
        if (tensor_out_names.size() > 0)
        {
            tensor_outs.resize(tensor_out_names.size());
            tensor_outs_host.resize(tensor_out_names.size());
            int tensor_idx = 0;
            for (tensor_idx = 0; tensor_idx < (int)(tensor_out_names.size()); tensor_idx++)
            {
                std::string const& tensor_name = tensor_out_names[tensor_idx];
                tensor_outs[tensor_idx] = mnn_net->getSessionOutput(session, tensor_name.c_str());
                int expect_host_dim_type     = GetOutputHostTensorDimType(tensor_outs[tensor_idx], MNN_FORWARD_OPENCL);
                tensor_outs_host[tensor_idx] = std::shared_ptr<MNN::Tensor>(mnn_net->CreateSessionIOTensor(session, tensor_outs[tensor_idx], MNN::Tensor::TENSOR_SESSION_OUTPUT, expect_host_dim_type));
            }
        }
        else
        {
            tensor_outs.push_back(mnn_net->getSessionOutput(session, NULL));
            int expect_host_dim_type     = GetOutputHostTensorDimType(tensor_outs[0], MNN_FORWARD_OPENCL);
            tensor_outs_host.push_back(std::shared_ptr<MNN::Tensor>(mnn_net->CreateSessionIOTensor(session, tensor_outs[0], MNN::Tensor::TENSOR_SESSION_OUTPUT, expect_host_dim_type)));
        }

        // Warming up...
        for (int i = 0; i < warmup; ++i) {
            input->copyFromHostTensor(givenTensor.get());
            mnn_net->runSession(session);
            for (int j = 0; j < (int)(tensor_outs.size()); j++)
                tensor_outs[j]->copyToHostTensor(tensor_outs_host[j].get());
        }

        auto tuning_begin = getTimeInUs();
        for (int round = 0; round < loop; round++) 
        {
            input->copyFromHostTensor(givenTensor.get());
            mnn_net->runSession(session);

            std::vector<MNN::Tensor*>   out_tensor_host(tensor_outs_host.size());
            for (int j = 0; j < (int)(tensor_outs_host.size()); j++)
                out_tensor_host[j] = tensor_outs_host[j].get();
            MNN::Tensor::copy_to_host_batch(tensor_outs, out_tensor_host);
        }
        auto tuning_end = getTimeInUs();
        uint64_t tuning_cost = tuning_end - tuning_begin;
        if(tuning_cost < best_cost)
        {
            best_cost = tuning_cost;
            best_idx  = i;
        }
        tuning_result[i] = tuning_cost;
        givenTensor.reset();
        for (int i = 0; i < (int)(tensor_outs_host.size()); i++)
            tensor_outs_host[i].reset();
        mnn_net->releaseSession(session);
        std::string   tmp_cache_file        = mnn_cache_file + std::string(".tmp_") + std::to_string(i);
        filesystem::rename_file(mnn_cache_file, tmp_cache_file);

        printf("tuning %d end ...\n", i);
    }
    printf("tuning end ..., and result as following:\n");
    for(int i = 0 ; i < tuning_cnt ; i ++)
        printf("%d, %luus\n", i, tuning_result[i]);
    printf("%d is the best select\n", best_idx);
    std::string   best_cache_file_name         = mnn_cache_file + std::string(".tmp_") + std::to_string(best_idx);
    filesystem::rename_file(best_cache_file_name, mnn_cache_file);
    for(int i = 0 ; i < tuning_cnt ; i ++)
    {
        filesystem::path tmp_cache_file(mnn_cache_file + std::string(".tmp_") + std::to_string(i));
        tmp_cache_file.remove_file();
    }

    return tuning_result;
}

std::vector<float> run_net(Model&                      model, 
                           int                         loop, 
                           uint64_t&                   total_time,
                           int                         warmup = 10, 
                           int                         tuning_cnt = 0,
                           int                         forward = MNN_FORWARD_CPU, 
                           bool                        only_inference = true,
                           int                         numberThread = 4, 
                           int                         precision = 2,
                           std::vector<std::string>    tensor_in_names = std::vector<std::string>(0),
                           std::vector<std::string>    tensor_out_names = std::vector<std::string>(0)) 
{
	auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.model_file.c_str()));
	net->setSessionMode(MNN::Interpreter::Session_Release);
    MNN::ScheduleConfig config;
    config.numThread = numberThread;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power     = MNN::BackendConfig::Power_High;
    config.backendConfig    = &backendConfig;

    char*          pwd          = new char[1024];
#ifndef _MSC_VER
    getcwd(pwd, 1023);
#else
	_getcwd(pwd, 1023);
#endif	
    std::string    pwd_str      = pwd;
    delete[] pwd;

    std::string  cache_file_name = "";
	int          last_dot_idx    = model.model_file.rfind(".mnn");
	if (last_dot_idx > 0)
	{
		int          str_len                = model.model_file.length();
		std::string  model_name_without_dot = model.model_file.substr(0, last_dot_idx);
		if ("" != model_name_without_dot)
		{
			//std::string  cur_time        = get_system_time();
			cache_file_name = model_name_without_dot/* + "_" + cur_time*/ + ".mnn_cache";
			//net->setCacheFile(cache_file_name.c_str());
		}
	}

    if(tuning_cnt > 0 && "" != cache_file_name && MNN_FORWARD_OPENCL == forward)
        tuning_model(config, net, cache_file_name, tuning_cnt, warmup, loop, tensor_out_names);
        
    if("" != cache_file_name && MNN_FORWARD_OPENCL == forward)
    {
        FILE* cache_file = fopen(cache_file_name.c_str(), "rb");
        if(NULL != cache_file)
        {
            fflush(cache_file);
            fclose(cache_file);
            net->setCacheFile(cache_file_name.c_str());
        }
    }
        
    std::vector<float> costs;
    MNN::Session*                session         = net->createSession(config);
	std::vector<MNN::Tensor*>    src_tensors(tensor_in_names.size(), nullptr);
	for(int i = 0 ; i < ((int)(tensor_in_names.size())) ; i ++)
		src_tensors[i] = net->getSessionInput(session, tensor_in_names[i].c_str());

    // if the model has not the input dimension, umcomment the below code to set the input dims
    // std::vector<int> dims{1, 3, 224, 224};
    // net->resizeTensor(input, dims);
    // net->resizeSession(session);

	net->releaseModel();
	std::vector<float*>                        buf_in_ptrs(tensor_in_names.size(), nullptr);
	std::vector<int>                           buf_in_sizes(tensor_in_names.size(), 0);
	std::vector<unsigned char*>                tensor_in_dummys(tensor_in_names.size(), nullptr);
	std::vector<std::shared_ptr<MNN::Tensor>>  givenTensors(tensor_in_names.size(), nullptr);
	for (int m = 0; m < ((int)(tensor_in_names.size())); m++)
	{
		const MNN::Backend* inBackend = net->getBackend(session, src_tensors[m]);
		givenTensors[m] = std::shared_ptr<MNN::Tensor>(net->CreateSessionIOTensor(session, src_tensors[m], MNN::Tensor::TENSOR_SESSION_INPUT, -1));
		givenTensors[m]->SyncIONStart(false);
		MNN::Tensor*   tensor_in       = givenTensors[m].get();
		auto      type                 = tensor_in->getType();
		buf_in_ptrs[m]                 = tensor_in->host<float>();
		buf_in_sizes[m]                = tensor_in->size();
		int       N_in                 = tensor_in->batch();
		int       C_in                 = tensor_in->channel();
		int       H_in                 = tensor_in->height();
		int       W_in                 = tensor_in->width();
		tensor_in_dummys[m]            = new unsigned char[buf_in_sizes[m]];
		//generate_data(buf_in_ptr, (buf_in_size >> 2));
		memset(buf_in_ptrs[m], 0, buf_in_sizes[m]);
		float*       cur_buf_in_ptr    = buf_in_ptrs[m];
		if (tensor_in_names.size() > 0)
		{
			std::string  tensor_in_file_name = pwd_str + "/" + tensor_in_names[m] + ".raw";
			FILE*        tensor_in_file      = fopen(tensor_in_file_name.c_str(), "rb");
			if (nullptr != tensor_in_file)
			{
				memset(buf_in_ptrs[m], 0, buf_in_sizes[m]);

				fseek(tensor_in_file, 0, SEEK_END);
				int tensor_in_file_size = (int)(ftell(tensor_in_file));
				fseek(tensor_in_file, 0, SEEK_SET);

				if (tensor_in_file_size <= buf_in_sizes[m])
				{
					MNN::Tensor::InsideDescribe* tensor_desc = MNN::TensorUtils::getDescribe(tensor_in);
					printf("tensor_in: %d\n", tensor_desc->dimensionFormat);
					if (MNN::MNN_DATA_FORMAT_NC4HW4 == tensor_desc->dimensionFormat)
					{
						float*    buf_in_raw     = new float[tensor_in_file_size >> 2];
						size_t    read_size      = fread(buf_in_raw, 1, tensor_in_file_size, tensor_in_file);
						(void)read_size;
						fflush(tensor_in_file);
						fclose(tensor_in_file);
						int   i       = 0;
						int   j       = 0;
						int   ele_cnt = (tensor_in_file_size >> 2);
						if (1 == C_in)
						{
							for (i = 0; i < ele_cnt; i++)
								cur_buf_in_ptr[4 * i] = buf_in_raw[i];
						}
						else if (2 == C_in)
						{
							for (i = 0; i < ele_cnt; i += 2, j += 4)
							{
								cur_buf_in_ptr[j]     = buf_in_raw[i];
								cur_buf_in_ptr[j + 1] = buf_in_raw[i + 1];
							}
						}
						else if (3 == C_in)
						{
							for (i = 0; i < ele_cnt; i += 3, j += 4)
							{
								cur_buf_in_ptr[j]     = buf_in_raw[i];
								cur_buf_in_ptr[j + 1] = buf_in_raw[i + 1];
								cur_buf_in_ptr[j + 2] = buf_in_raw[i + 2];
							}
						}
						else if (4 == C_in)
						{
							memcpy(cur_buf_in_ptr, buf_in_raw, buf_in_sizes[m]);
						}
						else
						{
							int channel_q = ((C_in + 3) >> 2);
							for (i = 0; i < H_in; i++)
							{
								for (j = 0; j < W_in; j++)
								{
									for (int c = 0; c < C_in; c++)
									{
										int  src_pos            = i * W_in * C_in + j * C_in + c;
										int  dst_pos            = (c >> 2) * W_in * H_in * 4 + i * W_in * 4 + 4 * j + (c & 3);
										cur_buf_in_ptr[dst_pos] = buf_in_raw[src_pos];
									}
								}
							}
						}
						delete[] buf_in_raw;
					}
					else
					{
						size_t read_size = fread(cur_buf_in_ptr, 1, tensor_in_file_size, tensor_in_file);
						(void)read_size;
						fflush(tensor_in_file);
						fclose(tensor_in_file);
					}
				}
				else
				{
					fflush(tensor_in_file);
					fclose(tensor_in_file);
					return std::vector<float>(0);
				}
			}
			else
			{
				;
			}
		}
		memcpy(tensor_in_dummys[m], cur_buf_in_ptr, buf_in_sizes[m]);
		givenTensors[m]->SyncIONEnd(false);
	}

	std::vector<MNN::Tensor*>                    tensor_outs(0);
	std::vector<std::shared_ptr<MNN::Tensor> >   tensor_outs_host(0);
	std::vector<unsigned char* >                 tensor_outs_dummy(0);
	if (tensor_out_names.size() > 0)
	{
		tensor_outs.resize(tensor_out_names.size());
		tensor_outs_host.resize(tensor_out_names.size());
		tensor_outs_dummy.resize(tensor_out_names.size());
		int tensor_idx = 0;
		for (tensor_idx = 0; tensor_idx < (int)(tensor_out_names.size()); tensor_idx++)
		{
			std::string const& tensor_name = tensor_out_names[tensor_idx];
			tensor_outs[tensor_idx]        = net->getSessionOutput(session, tensor_name.c_str());
            int expect_host_dim_type       = GetOutputHostTensorDimType(tensor_outs[tensor_idx], forward);
			tensor_outs_host[tensor_idx]   = std::shared_ptr<MNN::Tensor>(net->CreateSessionIOTensor(session, tensor_outs[tensor_idx], MNN::Tensor::TENSOR_SESSION_OUTPUT, -1));
			tensor_outs_dummy[tensor_idx]  = new unsigned char[tensor_outs_host[tensor_idx]->size()];
		}
	}
	else
	{
		tensor_outs.push_back(net->getSessionOutput(session, NULL));
        int expect_host_dim_type = GetOutputHostTensorDimType(tensor_outs[0], forward);
		tensor_outs_host.push_back(std::shared_ptr<MNN::Tensor>(net->CreateSessionIOTensor(session, tensor_outs[0], MNN::Tensor::TENSOR_SESSION_OUTPUT, -1)));
		tensor_outs_dummy.resize(1);
		tensor_outs_dummy[0] = new unsigned char[tensor_outs_host[0]->size()];
	}

	// Warming up...
    for (int i = 0; i < warmup; ++i) {
        for(int j = 0 ; j < ((int)(tensor_in_names.size())) ; j ++)
            src_tensors[j]->copyFromHostTensor(givenTensors[j].get());
        net->runSession(session);
        for (int j = 0; j < (int)(tensor_outs.size()); j++)
            tensor_outs[j]->copyToHostTensor(tensor_outs_host[j].get());
    }
	auto orion_begin = getTimeInUs();
	for (int round = 0; round < loop; round++) {
		auto timeBegin = getTimeInUs();
		//just dummy test, it can be ignore
        for(int i = 0 ; i < ((int)(tensor_in_names.size())) ; i ++)
        {
            givenTensors[i]->SyncIONStart(false);
            memcpy(buf_in_ptrs[i], tensor_in_dummys[i], buf_in_sizes[i]);
            givenTensors[i]->SyncIONEnd(false);
            src_tensors[i]->copyFromHostTensor(givenTensors[i].get());
        }

		net->runSession(session);

		std::vector<MNN::Tensor*>   out_tensor_host(tensor_outs_host.size());
		for (int j = 0; j < (int)(tensor_outs_host.size()); j++)
			out_tensor_host[j] = tensor_outs_host[j].get();
		MNN::Tensor::copy_to_host_batch(tensor_outs, out_tensor_host);
		for (int j = 0; j < (int)(tensor_outs.size()); j++)
		{
			//just dummy test, it can be ignore
			{
				tensor_outs[j]->SyncIONStart(false);
				memcpy(tensor_outs_dummy[j], tensor_outs_host[j]->buffer().host, tensor_outs_host[j]->size());
				tensor_outs[j]->SyncIONEnd(false);
			}
		}

		auto timeEnd = getTimeInUs();
		costs.push_back((timeEnd - timeBegin) / 1000.0);
	}
	auto orion_end = getTimeInUs();
	total_time = orion_end - orion_begin;

    if (tensor_out_names.size() > 0)
    {
        for (int i = 0; i < (int)(tensor_out_names.size()); i++)
        {
            MNN::Tensor*   tensor_out = tensor_outs_host[i].get();
            tensor_out->SyncIONStart(false);
            float*         buf_out_ptr_raw  = tensor_out->host<float>();
            int            buf_out_size     = tensor_out->size();
            int            N_out            = tensor_out->batch();
            int            C_out            = tensor_out->channel();
            int            H_out            = tensor_out->height();
            int            W_out            = tensor_out->width();
            int            buf_type         = tensor_out->getType().code;
            int            buf_bits         = tensor_out->getType().bits;
            int            ele_cnt          = buf_out_size / (buf_bits >> 3);
            if ((halide_type_int == buf_type || halide_type_uint == buf_type) && 32 != buf_bits)
            {
                printf("tensor_out %d, cannot support output buf_bits %d, if buf_type is int or uint\n", i, buf_bits);
                exit(-1);
            }
            float* buf_out_ptr = buf_out_ptr_raw;
#define NC4HW4_TO_NCHW
#ifdef NC4HW4_TO_NCHW
            if (MNN::MNN_DATA_FORMAT_NC4HW4 == MNN::TensorUtils::getDescribe(tensor_out)->dimensionFormat)
            {
                ele_cnt     = (0 == N_out ? 1 : N_out)  * (0 == C_out ? 1 : C_out) * (0 == H_out ? 1 : H_out) * (0 == W_out ? 1 : W_out);
                buf_out_ptr = new float[ele_cnt];
                memset(buf_out_ptr, 0, ele_cnt * sizeof(float));
                nc4hw4_to_nchw(buf_out_ptr_raw, (0 == N_out ? 1 : N_out), (0 == C_out ? 1 : C_out), (0 == H_out ? 1 : H_out), (0 == W_out ? 1 : W_out), buf_out_ptr);
            }
#endif
            std::string    cur_tensor_out_name  = tensor_out_names[i];
			while(-1 != cur_tensor_out_name.find('/'))
				cur_tensor_out_name = cur_tensor_out_name.replace(cur_tensor_out_name.find("/"), 1, "_");
			while(-1 != cur_tensor_out_name.find(':'))
				cur_tensor_out_name = cur_tensor_out_name.replace(cur_tensor_out_name.find(":"), 1, "_");
			std::string    tensor_out_file_name = pwd_str + "/" + cur_tensor_out_name + ".raw";
			FILE*  file_out_tensor = fopen(tensor_out_file_name.c_str(), "wb");
			if (nullptr != file_out_tensor)
			{
                MNN::Tensor::InsideDescribe* tensor_desc = MNN::TensorUtils::getDescribe(tensor_out);
                std::cout << "tensor_out: " << tensor_out_names[i] << ", buf_len: " << buf_out_size << ", format: " << tensor_desc->dimensionFormat << std::endl;

                if (ele_cnt <= 32)
                {
                    int out_cnt = ele_cnt >= 8 ? 8 : ele_cnt;
                    if (halide_type_float == buf_type)
                    {
                        for (int j = 0; j < out_cnt; j++)
                            printf("%.6f, ", ((float*)(buf_out_ptr))[j]);
                    }
                    else
                    {
                        for (int j = 0; j < out_cnt; j++)
                            printf("%d, ", ((int*)(buf_out_ptr))[j]);
                    }
                    printf("\n");
                }
                else
                {
                    if (halide_type_float == buf_type)
                    {
                        printf("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
                            ((float*)(buf_out_ptr))[0], ((float*)(buf_out_ptr))[1],
                            ((float*)(buf_out_ptr))[2], ((float*)(buf_out_ptr))[3],
                            ((float*)(buf_out_ptr))[4], ((float*)(buf_out_ptr))[5],
                            ((float*)(buf_out_ptr))[6], ((float*)(buf_out_ptr))[7]);
                        printf("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
                            ((float*)(buf_out_ptr))[ele_cnt / 2 + 0], ((float*)(buf_out_ptr))[ele_cnt / 2 + 1],
                            ((float*)(buf_out_ptr))[ele_cnt / 2 + 2], ((float*)(buf_out_ptr))[ele_cnt / 2 + 3],
                            ((float*)(buf_out_ptr))[ele_cnt / 2 + 4], ((float*)(buf_out_ptr))[ele_cnt / 2 + 5],
                            ((float*)(buf_out_ptr))[ele_cnt / 2 + 6], ((float*)(buf_out_ptr))[ele_cnt / 2 + 7]);
                        printf("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
                            ((float*)(buf_out_ptr))[ele_cnt - 8],     ((float*)(buf_out_ptr))[ele_cnt - 8 + 1],
                            ((float*)(buf_out_ptr))[ele_cnt - 8 + 2], ((float*)(buf_out_ptr))[ele_cnt - 8 + 3],
                            ((float*)(buf_out_ptr))[ele_cnt - 8 + 4], ((float*)(buf_out_ptr))[ele_cnt - 8 + 5],
                            ((float*)(buf_out_ptr))[ele_cnt - 8 + 6], ((float*)(buf_out_ptr))[ele_cnt - 8 + 7]);
                    }
                    else
                    {
                        printf("%d, %d, %d, %d, %d, %d, %d, %d\n",
                            ((int*)(buf_out_ptr))[0], ((int*)(buf_out_ptr))[1],
                            ((int*)(buf_out_ptr))[2], ((int*)(buf_out_ptr))[3],
                            ((int*)(buf_out_ptr))[4], ((int*)(buf_out_ptr))[5],
                            ((int*)(buf_out_ptr))[6], ((int*)(buf_out_ptr))[7]);
                        printf("%d, %d, %d, %d, %d, %d, %d, %d\n",
                            ((int*)(buf_out_ptr))[ele_cnt / 2 + 0], ((int*)(buf_out_ptr))[ele_cnt / 2 + 1],
                            ((int*)(buf_out_ptr))[ele_cnt / 2 + 2], ((int*)(buf_out_ptr))[ele_cnt / 2 + 3],
                            ((int*)(buf_out_ptr))[ele_cnt / 2 + 4], ((int*)(buf_out_ptr))[ele_cnt / 2 + 5],
                            ((int*)(buf_out_ptr))[ele_cnt / 2 + 6], ((int*)(buf_out_ptr))[ele_cnt / 2 + 7]);
                        printf("%d, %d, %d, %d, %d, %d, %d, %d\n",
                            ((int*)(buf_out_ptr))[ele_cnt - 8],     ((int*)(buf_out_ptr))[ele_cnt - 8 + 1],
                            ((int*)(buf_out_ptr))[ele_cnt - 8 + 2], ((int*)(buf_out_ptr))[ele_cnt - 8 + 3],
                            ((int*)(buf_out_ptr))[ele_cnt - 8 + 4], ((int*)(buf_out_ptr))[ele_cnt - 8 + 5],
                            ((int*)(buf_out_ptr))[ele_cnt - 8 + 6], ((int*)(buf_out_ptr))[ele_cnt - 8 + 7]);
                    }
				}
#ifdef NC4HW4_TO_NCHW
                if (MNN::MNN_DATA_FORMAT_NC4HW4 == MNN::TensorUtils::getDescribe(tensor_out)->dimensionFormat)
                    delete buf_out_ptr;
#endif
                fwrite(buf_out_ptr_raw, 1, buf_out_size, file_out_tensor);
                fflush(file_out_tensor);
                fclose(file_out_tensor);
                tensor_out->SyncIONEnd(false);
			}
		}
	}
    for (int i = 0; i < ((int)(tensor_in_names.size())); i++)
    {
        if (nullptr != tensor_in_dummys[i])
            delete tensor_in_dummys[i];
        givenTensors[i].reset();
    }

    for (int i = 0; i < (int)(tensor_outs_host.size()); i++)
    {
        tensor_outs_host[i].reset();
        delete tensor_outs_dummy[i];
    }

	return costs;
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

void displayStats(MNNForwardType forward_type, int precision_type, const std::string& name, const std::vector<float>& costs, uint64_t total_time) {
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs) {
        max = fmax(max, v);
        min = fmin(min, v);
        sum += v;
        //printf("[ - ] cost：%f ms\n", v);
    }
    avg = costs.size() > 0 ? (float)(total_time / costs.size())*(0.001f) : 0;
    printf("[ - ] %-24s    max = %8.3fms  min = %8.3fms  avg = %8.3fms\n", name.c_str(), max, avg == 0 ? 0 : min, avg);

    char output_log[256] = { 0 };
    int  str_len = sprintf(output_log, "%s_%s: total time: %lu\n",
                           forwardType(forward_type).c_str(),
                           0 == precision_type ? "normal" : (1 == precision_type ? "fp32" : "fp16"),
                           total_time);
    
    FILE* profile_result = fopen("profile_result.txt", "wb");
    if(nullptr != profile_result)
    {
        fwrite(output_log, str_len, 1, profile_result);
        fflush(profile_result);
        fclose(profile_result);
        profile_result = nullptr;
    }
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
    std::string  tuning_cnt_str    = "";
    std::string  forward_type_str  = "";
    std::string  thread_num_str    = "";
    std::string  precision_str     = "";
    std::string  in_tensors_str    = "";
    std::string  out_tensors_str   = "";
    std::string  json_file_str     = "";

    int loop               = 10;
    int warmup             = 10;
    int tuning_cnt         = 0;
    MNNForwardType forward = MNN_FORWARD_CPU;
    int numberThread       = 4;
#if !defined(_MSC_VER)
	int opt                = 0;
    while ((opt = getopt(argc, argv, "hm:l:w:n:f:t:p:i:o:j:x")) != -1)
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
                         << "  -n  find best tuning result in n-turns, if n is > 0\n"
                         << "  -f  forward_type, 0 for cpu, 3 for opencl, other is invalid\n"
                         << "  -t  thread num, it useful for cpu.\n"
                         << "  -p  precision mode, o for normal, 1 for high, 2 for low, default is 2.\n"
                         << "  -i  input tensor name, sperator is \":\"\n"
                         << "  -o  output tensor name, sperator is \":\".\n"
                         << "  -j  config file(dlcv style), \"m:f:t:p:i:o\" will be ignored, if this option is selected.\n"
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
            case 'n':
                tuning_cnt_str   = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
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
            case 'j':
                json_file_str    = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;
            default:
                std::cout  << "Invalid parameter specified." << std::endl;
                std::exit(-1);
        }
    }
#else
    model_file_str   = trim_string(argv[1]);
    warm_up_str      = trim_string(argv[2]);
    loop_count_str   = trim_string(argv[3]);
    forward_type_str = trim_string(argv[4]);
    precision_str    = trim_string(argv[5]);
    in_tensors_str   = trim_string((0 != argv[6] ? trim_string(argv[6]) : ""));
    out_tensors_str  = trim_string((0 != argv[7] ? trim_string(argv[7]) : ""));
    tuning_cnt_str   = (0 != argv[8] ? trim_string(argv[8]) : "");
#endif
    if ("" != loop_count_str) 
        loop = atoi(loop_count_str.c_str());

    if ("" != warm_up_str) 
        warmup = atoi(warm_up_str.c_str());

    if ("" != tuning_cnt_str) 
        tuning_cnt = atoi(tuning_cnt_str.c_str());

    if ("" != forward_type_str) 
        forward = static_cast<MNNForwardType>(atoi(forward_type_str.c_str()));

    if ("" != thread_num_str) 
        numberThread = atoi(thread_num_str.c_str());

    int precision = 2;
    if ("" != precision_str) 
        precision = atoi(precision_str.c_str());

    std::vector<std::string>    tensor_in_names  = split_string(in_tensors_str, ':');
    std::vector<std::string>    tensor_out_names = split_string(out_tensors_str, ':');

    if("" != json_file_str)
    {
        InputOption  json_cfg(json_file_str);
        if(true == json_cfg.parse_ok_)
        {
            printf("json file:   %s\n", json_file_str.c_str());
            model_file_str  = InputOption::GetModelPathFromJsonPath(json_file_str, json_cfg.model_file_);
            forward         = (0 == json_cfg.core_type_ ? MNN_FORWARD_CPU : MNN_FORWARD_OPENCL);
            numberThread    = json_cfg.thread_count_;
            precision       = json_cfg.precision_;
            tensor_in_names.resize(json_cfg.in_tensor_.size());
            tensor_out_names.resize(json_cfg.out_tensor_.size());
            for(int i = 0 ; i < (int)(tensor_in_names.size()) ; i ++)
                tensor_in_names[i] = std::get<0>(json_cfg.in_tensor_[i]);
            for(int i = 0 ; i < (int)(tensor_out_names.size()) ; i ++)
                tensor_out_names[i] = std::get<0>(json_cfg.out_tensor_[i]);
        }
    }
    printf("model file:  %s\n", model_file_str.c_str());
    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numberThread << "** precision=" <<precision << std::endl;
    Model model = getModelFile(model_file_str.c_str());
    std::cout << "--------> Benchmarking... tuning = " << tuning_cnt << ", loop = " << loop << ", warmup = " << warmup << std::endl;

    /* not called yet */
    // set_cpu_affinity();
    uint64_t total_time = 0;
    std::vector<float> costs = run_net(model, 
                                       loop, 
                                       total_time,
                                       warmup, 
                                       tuning_cnt,
                                       forward, 
                                       false, 
                                       numberThread, 
                                       precision,
                                       tensor_in_names,
                                       tensor_out_names);
    displayStats(forward, precision, model.name, costs, total_time);
}
