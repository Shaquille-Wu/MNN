/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file orion_mnn_impl.h
 * @brief This header file defines OrionSnpeImpl struct
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-08-24
 */

#ifndef __ORION_MNN_IMPL_H__
#define __ORION_MNN_IMPL_H__

#include "IInference.h"
#include "usr_buffer.h"
#include <string.h>
#include "MNN/Interpreter.hpp"

namespace vision{

#define     ORION_DELETE(p)       { if(nullptr != p) delete p; p = nullptr; }

/**
 * @brief OrionMNNImpl 
 * 
 */
class OrionMNNImpl : public IInference
{
public:
    /**
     * @brief default constructor for OrionMNNImpl
     */
    OrionMNNImpl();

    /**
     * @brief destructor
     */    
    ~OrionMNNImpl();

    typedef enum tag_orion_error_code{
        NONE = 0,
        INVALID_MODEL_FILE,
        MODEL_BUILD_FAILED,
		MODEL_RESIZE_FAILED,
        INPUT_DATA_INVALID,
        OUTPUT_DATA_INVALID,
        UNKNOWN,
        SUM
    }ORION_ERROR_CODE;

    typedef enum tag_feature_idx{
        FEATURE_IDX_CORE_TYPE = 0,
        FEATURE_IDX_THREAD_COUNT,
        FEATURE_IDX_PRECISION,
        FEATURE_IDX_POWER_MODE,
        FEATURE_IDX_PRINT_TENSOR_SHAPE,
        FEATURE_IDX_MODEL_CACHE,
        FEATURE_IDX_SUM
    }FEATURE_IDX;

public:
    /*
     * @brief load AI model from file.
     *
     * @param pcModelFile model's name.
     * 
     * @param gpuid specify gpu's id if the target platform has more than one gpu
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_load_model(const char *model_file, const int gpuid=0) override;

    /*
     * @brief load AI model from buffer.
     *
     * @param buffer model's buffer.
     * 
     * @param bufferSize size of model's buffer.
     * 
     * @param gpuid specify gpu's id if the target platform has more than one gpu
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_load_model_from_buffer(char* buffer, uint64_t bufferSize, const int gpuid = 0) override;

    // setup input data dimension
    virtual int INF_set_data(const char         *tensor_name, 
                             const void         *data,
                             std::vector<int>&   size,
                             LayerInOut          type) override;
    
     /*
     * @brief execute the computation of forward
     *
     * @param batch_size batch count for computation.
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */   
    virtual int INF_forward(const int batch_size) override;

     /*
     * @brief get the result of inference
     *
     * @param tensor_name name of out tensor.
     * 
     * @param data the buffer used to save the result.
     * 
     * @param len the size of buffer(pData).
     *
     * @return the size of copied to user buffer(pData)
     *
     */
    virtual int INF_get_result(const char *tensor_name, void* data, int len) override;
    
     /*
     * @brief set the number of thread, if the target platform support OpenMP
     *
     * @param thread_num the number of thread.
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_set_thread_num(int thread_num) override;

     /*
     * @brief get the result of last inference
     *
     * @pData
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_get_last_output(void* pData) override;

     /*
     * @brief get the dimensions of tensor through its name
     *
     * @tensor_name the name of tensor
     * 
     * @sz the dimensions of tensor
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_get_dims(const char *tensor_name, std::vector<int>& sz) override;

    /*
     * @brief load the plugin of AI model from file.
     *
     * @param model_file model's name.
     * 
     * @plugin_module
     * 
     * @param gpuid specify gpu's id if the target platform has more than one gpu
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_load_model_config(const char *model_file, const char* plugin_module, const int gpuid = 0) override;

    /*
     * @brief load the plugin of AI model from buffer.
     *
     * @param buffer model's buffer.
     * 
     * @param buffer_size size of model's buffer.
     * 
     * @param config_buffer buf of model's config.
     * 
     * @config_buffer_size size of config_buffer.
     * 
     * @param gpuid specify gpu's id if the target platform has more than one gpu
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_load_model_config_from_buffer(char*            buffer, 
                                                  uint64_t         buffer_size, 
                                                  char*            config_buffer, 
                                                  uint64_t         config_buffer_size, 
                                                  const int        gpuid = 0) override;

    /*
     * @brief enum various of features from library.
     *
     * @param engine_feature feature's buffer which save real data.
     * 
     * @param n number of "EngineFeature".
     * 
     * @return the number of feature which is passed to user
     *
     */
    virtual int INF_enum_engine_feature(EngineFeature *engine_feature, int n) override;


    /*
     * @brief set feature.
     *
     * @param engine_feature engine's feature.
     * 
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_set_engine_feature(const EngineFeature *engine_feature) override;    

    /*
     * @brief get the version for this engine.
     *
     * @return a string, like: major.minor.tiny
     *
     */
    virtual const char* INF_version() override
    {
        return "1.1.3";
    };

    class TensorInfo
    {
        public:
            TensorInfo() noexcept
            {
                memset(dim_, 0, 4 * sizeof(int));
            };

            ~TensorInfo() noexcept {};

            /**
             * @brief Construct from parameter
             *
             * @param dim A dim array.
             * 
             * @param buf A UsrBuffer object.
             * 
             * @exceptsafe No throw.
             */
            TensorInfo(const int* dim, const UsrBuffer& buf) noexcept
            {
                memcpy(dim_, dim, 4 * sizeof(int));
                buf_ = buf;
            };

            /**
             * @brief Copy constructor
             *
             * @param other The other TensorInfo object.
             * 
             * @exceptsafe No throw.
             */
            TensorInfo(const TensorInfo& other) noexcept
            {
                memcpy(dim_, other.dim_, 4 * sizeof(int));
                buf_ = other.buf_;
            };

            /**
             * @brief assignment constructor
             * 
             * @param other The other TensorInfo object.
             * 
             * @exceptsafe No throw.
             */
            TensorInfo& operator=(const TensorInfo&  other) noexcept
            {
                if(this == &other)
                    return *this;

                memcpy(dim_, other.dim_, 4 * sizeof(int));
                buf_ = other.buf_;

                return *this;
            };

        public:
            int            dim_[4];
            UsrBuffer      buf_;
    };

private:
	static bool                           resize_session(MNN::Interpreter* net, MNN::Session* session, std::map<std::string, TensorInfo> const& usr_inputs) noexcept;
	static bool                           copy_out_tensor_from_dev(std::map<std::string, MNN::Tensor*>& host_tensors, 
		                                                           std::map<std::string, MNN::Tensor*>& dev_tensors) noexcept;
    void                                  release_tensor_in() noexcept
    {
        std::map<std::string, MNN::Tensor*>::iterator  iter_in_tensor = input_tensor_map_.begin();
        while(input_tensor_map_.end() != iter_in_tensor)
        {
            std::string const&  tensor_name = iter_in_tensor->first;
            MNN::Tensor*        tensor      = iter_in_tensor->second;
            ORION_DELETE(tensor);
            iter_in_tensor ++;
        }
        input_tensor_map_.clear();
        dev_input_tensor_map_.clear();
    };

    void                                  release_tensor_out() noexcept
    {
        std::map<std::string, MNN::Tensor*>::iterator  iter_out_tensor = output_tensor_map_.begin();
        while(output_tensor_map_.end() != iter_out_tensor)
        {
            std::string const&  tensor_name = iter_out_tensor->first;
            MNN::Tensor*        tensor      = iter_out_tensor->second;
            ORION_DELETE(tensor);
            iter_out_tensor ++;
        }
        output_tensor_map_.clear();
        dev_output_tensor_map_.clear();
    };

    void                                  release_resource()
    {
        release_tensor_in();
        release_tensor_out();
        if(nullptr != net_)
        {
            if(nullptr != session_)
                net_->releaseSession(session_);
            session_ = nullptr;
            net_->releaseModel();
        }
        ORION_DELETE(net_);
    }

    static void                           nhwc_float_to_nc4hw4_1c(float* src_buf, float* dst_buf, int norminal_count);
    static void                           nhwc_float_to_nc4hw4_2c(float* src_buf, float* dst_buf, int norminal_count);
    static void                           nhwc_float_to_nc4hw4_3c(float* src_buf, float* dst_buf, int norminal_count);

    static void                           copy_int_buf_to_float_buf(void* int_buf, void* dst_buf, int buf_size);
    static void                           copy_4c_float_buf_to_1c_float_buf(void* src_buf, void* dst_buf, int norminal_count);
    static void                           copy_4c_float_buf_to_2c_float_buf(void* src_buf, void* dst_buf, int norminal_count);
    static void                           copy_4c_float_buf_to_3c_float_buf(void* src_buf, void* dst_buf, int norminal_count);

    static void                           copy_4c_int_buf_to_1c_float_buf(void* int_buf, void* dst_buf, int norminal_count);
    static void                           copy_4c_int_buf_to_2c_float_buf(void* int_buf, void* dst_buf, int norminal_count);
    static void                           copy_4c_int_buf_to_3c_float_buf(void* int_buf, void* dst_buf, int norminal_count);

    static void                           nc4hw4_int_to_nhwc(int* nc4hw4_int_buf,   float* dst_buf, int n, int c, int h, int w);
    static void                           nc4hw4_float_to_nhwc(float* nc4hw4_float_buf, float* dst_buf, int n, int c, int h, int w);

    static void                           nc4hw4_int_to_buf(int* nc4hw4_int_buf,   float* dst_buf, int norminal_count);
    static void                           nc4hw4_float_to_buf(float* nc4hw4_float_buf,   float* dst_buf, int norminal_count);

private:
    MNNForwardType                        forward_type_;
    MNN::BackendConfig::PrecisionMode     precision_mode_;
    MNN::BackendConfig::PowerMode         power_mode_;
    int                                   thread_count_;
    bool                                  print_tensor_shape_;
    std::string                           model_cache_file_;

    MNN::Interpreter*                     net_;
    MNN::Session*                         session_;

    std::map<std::string, MNN::Tensor*>   input_tensor_map_;
    std::map<std::string, MNN::Tensor*>   dev_input_tensor_map_;
    std::map<std::string, MNN::Tensor*>   output_tensor_map_;
    std::map<std::string, MNN::Tensor*>   dev_output_tensor_map_;

    std::map<std::string, TensorInfo>     input_usr_buf_map_;
    std::map<std::string, TensorInfo>     output_usr_buf_map_;
}; //class OrionMNNImpl

} //namespace vision

#ifdef _WIN32
#define   INFERENCE_SDK_EXPORT __declspec(dllexport)
#else
#define   INFERENCE_SDK_EXPORT
#endif

/*
 * @brief Create instance for inference.
 *
 *
 * @return the pointer of instance for inferenece, nul is failed, other is ok
 *
 */
extern "C" INFERENCE_SDK_EXPORT void* CreateInference();

/*
 * @brief Destroy instance for inference.
 *
 *
 * @return none
 *
 */
extern "C" INFERENCE_SDK_EXPORT void  DestroyInference(void* inference_instance);

#endif