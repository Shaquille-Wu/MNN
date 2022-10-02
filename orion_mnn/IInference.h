/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file IInference.h
 * @brief This header file defines IInference struct, it is a interface
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-03-12
 */

#ifndef __H_IInference__
#define __H_IInference__

#include <vector>
#include <memory>
#include <map>
#include <iostream>

namespace vision{
/**
 * @addtogroup vision
 * @{
 *
 */


enum LayerInOut{
    INPUT = 0,
    OUTPUT
};

enum ValueType {
    PVOID = 1,
    FLOAT = 2,
    INT   = 3,
    BYTES = 4
};

struct MemAddress {
    void*   pValue;
    int     nSize;
};

union FeatureValue
{
    MemAddress     addr;
    float          fValue;
    int            nValue;
    unsigned char  ucValue[256];
};

struct EngineFeature {
    ValueType      valueType;
    char           keyName[64];
    FeatureValue   value;
};

/**
 * @brief Inference Interface 
 * 
 */
struct IInference {
    virtual ~IInference() {}

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
    virtual int INF_load_model(const char *pcModelFile, const int gpuid=0) = 0;

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
    virtual int INF_load_model_from_buffer(char* buffer, uint64_t bufferSize, const int gpuid = 0) = 0;

    // setup input data dimension
    virtual int INF_set_data(const char         *pcName, 
                             const void         *pData,
                             std::vector<int>&   size,
                             LayerInOut          type) = 0;
    
     /*
     * @brief execute the computation of forward
     *
     * @param iBatchSize batch count for computation.
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */   
    virtual int INF_forward(const int iBatchSize) = 0;

     /*
     * @brief get the result of inference
     *
     * @param pcName name of out tensor.
     * 
     * @param pData the buffer used to save the result.
     * 
     * @param iLen the size of buffer(pData).
     *
     * @return the size of copied to user buffer(pData)
     *
     */
    virtual int INF_get_result(const char *pcName, void* pData, int iLen) = 0;
    
     /*
     * @brief set the number of thread, if the target platform support OpenMP
     *
     * @param iNum the number of thread.
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_set_thread_num(int iNum) = 0;

     /*
     * @brief get the result of last inference
     *
     * @pData
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_get_last_output(void* pData) = 0;

     /*
     * @brief get the dimensions of tensor through its name
     *
     * @pcName the name of tensor
     * 
     * @sz the dimensions of tensor
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_get_dims(const char *pcName, std::vector<int>& sz) = 0;

    /*
     * @brief load the config of AI model from file.
     *
     * @param pcModelFile model's name.
     * 
     * @plugin_module
     * 
     * @param gpuid specify gpu's id if the target platform has more than one gpu
     *
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_load_model_config(const char *pcModelFile, const char* plugin_module, const int gpuid = 0) = 0;

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
                                                  const int        gpuid = 0) = 0;

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
    virtual int INF_enum_engine_feature(EngineFeature *engine_feature, int n) = 0;


    /*
     * @brief set feature.
     *
     * @param engine_feature engine's feature.
     * 
     * @return the code of status, 0 is ok, other is failed
     *
     */
    virtual int INF_set_engine_feature(const EngineFeature *engine_feature) = 0;    

    /*
     * @brief get the version for this engine.
     *
     * @return a string, like: major.minor.tiny
     *
     */
    virtual const char* INF_version() = 0;
};// IInference

typedef void*  (*CreateInferenceFunc)();
typedef void   (*DestroyInferenceFunc)(void* inference_instance);

}// namespace vision

#endif
