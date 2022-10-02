//
//  quantizeWeight.cpp
//  MNN
//
//  Created by MNN on 2019/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "quantizeWeight.hpp"
#include <math.h>
#include <algorithm>
#include <cmath>
#include "logkit.h"
#include <MNN/MNNDefine.h>

void InitAlpha(const float* weight, const int weightNum, const int kernelNum, float* alpha, const float weightClampValue) {
    const int kernelDim = weightNum / kernelNum;

    for (int i = 0; i < kernelNum; i++) {
        float avg = 0;
        float max = 0;
        float absVal;

//Shaquille, Added 20210131 Start
        float y   = 0.0f;
        float c   = 0.0f;
        float t   = 0.0f;
//Shaquille, Added 20210131 End

        for (int j = 0; j < kernelDim; j++) {
            absVal = std::fabs(weight[i * kernelDim + j]);
//Shaquille, Modified 20210131 Start
#if 0
            avg += absVal;
            if (absVal > max) {
                max = absVal;
            }
#else
            y    = absVal - c;
            t    = avg + y;
            c    = (t - avg) - y;
            avg  = t;
            max  = std::max(absVal, max);
#endif
//Shaquille, Modified 20210131 End
        }
        avg = avg / float(kernelDim);

        if (weightClampValue > 1) {
            alpha[i] = max / (weightClampValue * 1.25f);
        }
        else {
            alpha[i] = avg;
        }
    }
}

void UpdateQuantizedWeights(const float* weight, const int weightNum, const int kernelNum, float* alpha,
        const float weightClampValue, int8_t* quantizedWeight) {
    const int kernelDim = weightNum / kernelNum;
    const float eps = 1e-9f;
    float weightQuan;
    CHECK((int)weightClampValue >= 7) << "quantization bits less than 4 not supported yet.";

    for (int i = 0; i < weightNum; i++) {
//Shaquille, Modified 20210131 Start
#if 0
        weightQuan = weight[i] / (alpha[i / kernelDim]+ eps);
#else
        int   alpha_idx = i / kernelDim;
        float den       = alpha[alpha_idx];
        den             = std::max(den, eps);
        weightQuan      = weight[i] / den;
#endif
//Shaquille, Modified 20210131 End
        quantizedWeight[i] = std::min(weightClampValue, std::max(-weightClampValue, std::roundf(weightQuan)));
    }
}

void UpdateAlpha(const float* weight, const int weightNum, const int kernelNum, float* alpha, int8_t* quantizedWeight) {
    const int kernelDim = weightNum / kernelNum;
    const float eps = 1e-9f;

//Shaquille, Modified 20210131 Start
#if 0
    for (int i = 0; i < kernelNum; i++) {
        const int offset = i * kernelDim;
        float sum1 = 0;
        float sum2 = 0;

        for (int j = 0; j < kernelDim; j++) {
            sum1 += weight[offset + j] * quantizedWeight[offset + j];
            sum2 += quantizedWeight[offset + j] * quantizedWeight[offset + j];
        }
        alpha[i] = sum1 / (sum2+eps);
    }
#else
    for (int i = 0; i < kernelNum; i++) {
        const int offset = i * kernelDim;
        float sum1 = 0;
        float sum2 = 0;

        float y0   = 0.0f;
        float y1   = 0.0f;
        float c0   = 0.0f;
        float c1   = 0.0f;
        float t0   = 0.0f;
        float t1   = 0.0f;
        for (int j = 0; j < kernelDim; j++) {
            float a0 = weight[offset + j] * quantizedWeight[offset + j];
            float a1 = quantizedWeight[offset + j] * quantizedWeight[offset + j];

            y0     = a0 - c0;
            t0     = sum1 + y0;
            c0     = (t0 - sum1) - y0;
            sum1   = t0;

            y1     = a1 - c1;
            t1     = sum2 + y1;
            c1     = (t1 - sum2) - y1;
            sum2   = t1;
        }
        sum2     = std::max(sum2, eps);
        alpha[i] = sum1 / sum2;
    }
#endif
//Shaquille, Modified 20210131 End
}

// weight format is [co, ci, kh, kw]
int QuantizeWeightADMM(const float* weight, const int weightNum, int8_t* quantizedWeight, float* alpha,
                            const int kernelNum, const float weightClampValue) {
    // channels: co
    DCHECK((weightNum % kernelNum) == 0) << "weight size error!";
    const int kernelDim     = weightNum / kernelNum; // ci * kh * kw

    InitAlpha(weight, weightNum, kernelNum, alpha, weightClampValue);

    int iter = 0;
    float diffRate = 1;
    float preSum = 0;
    float curSum = 0;
    const int maxIter = 1000;

//Shaquille, Modified 20210131 Start    
#if 0
    for (int i = 0; i < weightNum; i++){
        preSum += std::fabs(weight[i]);
    }
#else
    float y   = 0.0f;
    float c   = 0.0f;
    float t   = 0.0f;
    for (int i = 0; i < weightNum; i++){
        y       = std::fabs(weight[i]) - c;
        t       = preSum + y;
        c       = (t - preSum) - y;
        preSum  = t;
    }
#endif    
//Shaquille, Modified 20210131 End
    // update weights quan
    while(iter < maxIter) {
        UpdateQuantizedWeights(weight, weightNum, kernelNum, alpha, weightClampValue, quantizedWeight);
        UpdateAlpha(weight, weightNum, kernelNum, alpha, quantizedWeight);
        iter++;
    }

//Shaquille, Modified 20210131 Start
#if 0
    for (int i = 0; i < weightNum; i++){
        curSum += std::fabs(quantizedWeight[i]*alpha[i/kernelDim]);
    }
    DLOG(INFO) << "iter: " << iter << " with diff "<< preSum-curSum;
#else
    y   = 0.0f;
    c   = 0.0f;
    t   = 0.0f;
    for (int i = 0; i < weightNum; i++){
        y       = std::fabs(quantizedWeight[i]*alpha[i/kernelDim]) - c;
        t       = curSum + y;
        c       = (t - curSum) - y;
        curSum  = t;
    }
    DLOG(INFO) << "iter: " << iter << " " << preSum << ", " << curSum << ", with diff "<< preSum-curSum;
#endif
//Shaquille, Modified 20210131 End
    
    return 0;
}

// weight format is [co, ci, kh, kw]
int SymmetricQuantizeWeight(const float* weight, const int size, int8_t* quantizedWeight, float* scale,
                            const int channels, float weightClampValue) {
    DCHECK((size % channels) == 0) << "weight size error!";
    const int channelStride     = size / channels;
    const int quantizedMaxValue = weightClampValue;

    for (int c = 0; c < channels; ++c) {
        const auto weightChannelStart    = weight + c * channelStride;
        auto quantizedWeightChannelStart = quantizedWeight + c * channelStride;
        auto minmaxValue                 = std::minmax_element(weightChannelStart, weightChannelStart + channelStride);
        const float dataAbsMax           = std::fmax(std::fabs(*minmaxValue.first), std::fabs(*minmaxValue.second));

        float scaleDataToInt8 = 1.0f;
        if (dataAbsMax == 0) {
            scale[c] = 0.0f;
        } else {
            scale[c]        = dataAbsMax / quantizedMaxValue;
            scaleDataToInt8 = quantizedMaxValue / dataAbsMax;
        }

        for (int i = 0; i < channelStride; ++i) {
            const int32_t quantizedInt8Value = static_cast<int32_t>(roundf(weightChannelStart[i] * scaleDataToInt8));
            quantizedWeightChannelStart[i] =
                std::min(quantizedMaxValue, std::max(-quantizedMaxValue, quantizedInt8Value));
        }
    }

    return 0;
}

//Shaquille, Added 20210204 Start
int FakeQuantizeConvPerChannelWeightBiasScale(const float*   weight, 
                                              const int      size, 
                                              const float*   bias,
                                              float*         fake_quantize_weight,
                                              float*         fake_quantize_bias,
                                              float*         scale, 
                                              int            input_channels,
                                              int            output_channels,
                                              std::string    method, 
                                              float          weightClampValue, 
                                              bool           mergeChannel)
{
    const int icXoc          = input_channels * output_channels;
    DCHECK(size % icXoc == 0) << "Input Data Size Error!";

    std::vector<int8_t>  quantizedWeight(size, 0);
    if (mergeChannel) {
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weight, size, quantizedWeight.data(), scale, output_channels, weightClampValue);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weight, size, quantizedWeight.data(), scale, output_channels, weightClampValue);
        }
    } else {
        const int kernelSize = size / icXoc;
        const int ocStride   = size / output_channels;

        std::vector<float> weightMultiByInputScale(size);
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int ic = 0; ic < input_channels; ++ic) {
                for (int i = 0; i < kernelSize; ++i) {
                    const int index                = oc * ocStride + ic * kernelSize + i;
                    weightMultiByInputScale[index] = weight[index];
                }
            }
        }
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weightMultiByInputScale.data(), size, quantizedWeight.data(), scale, output_channels, weightClampValue);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weightMultiByInputScale.data(), size, quantizedWeight.data(), scale, output_channels, weightClampValue);
        }
    }

    int    clamp_value           = (int)(std::roundf(weightClampValue));
    int    internel_kernel_size  = size / output_channels;
    for(int i = 0 ; i < output_channels ; i ++)
    {
        float   quant_scale   = scale[i];
        for(int j = 0 ; j < internel_kernel_size ; j ++)
            fake_quantize_weight[i * internel_kernel_size + j] = quantizedWeight[i * internel_kernel_size + j] * quant_scale;
    }

    if (bias)
    {
        for(int i = 0 ; i < output_channels ; i ++)
        {
            float   quant_scale   = scale[i];
            if(quant_scale < 1e-7f)
                fake_quantize_bias[i] = 0.0f;
            else
            {
                float   quant_scale   = scale[i];
                fake_quantize_bias[i] = std::roundf(bias[i] / quant_scale);
                fake_quantize_bias[i] = fake_quantize_bias[i] * quant_scale;
            }   
        }
    }

    return 0;
}

int FakeQuantizeDepthwiseConvWeightBiasScale(const float*  weight, 
                                             const int     size, 
                                             const float*  bias,
                                             float*        fake_quantize_weight,
                                             float*        fake_quantize_bias,
                                             float*        scale, 
                                             int           input_channels,
                                             int           output_channels,
                                             std::string   method, 
                                             float         weightClampValue,
                                             bool          mergeChannel)
{
    DCHECK(input_channels == output_channels) << "Input Data Size Error!";

    std::vector<int8_t>   quantizedWeight(size, 0);
    if (method == "MAX_ABS") {
        SymmetricQuantizeWeight(weight, size, quantizedWeight.data(), scale, input_channels, weightClampValue);
    }
    else if (method == "ADMM") {
        QuantizeWeightADMM(weight, size, quantizedWeight.data(), scale, input_channels, weightClampValue);
    }

    int    clamp_value           = (int)(std::roundf(weightClampValue));
    int    internel_kernel_size  = size / output_channels;
    for(int i = 0 ; i < output_channels ; i ++)
    {
        float   quant_scale   = scale[i];
        for(int j = 0 ; j < internel_kernel_size ; j ++)
            fake_quantize_weight[i * internel_kernel_size + j] = quantizedWeight[i * internel_kernel_size + j] * quant_scale;
    }

    if (bias)
    {
        for(int i = 0 ; i < output_channels ; i ++)
        {
            float   quant_scale   = scale[i];
            if(quant_scale < 1e-7f)
                fake_quantize_bias[i] = 0.0f;
            else
            {
                float   quant_scale   = scale[i];
                fake_quantize_bias[i] = std::roundf(bias[i] / quant_scale);
                fake_quantize_bias[i] = fake_quantize_bias[i] * quant_scale;
            }   
        }
    }

    return 0;
}
//Shaquille, Added 20210204 End

int QuantizeConvPerChannel(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                           int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                           const std::vector<float>& outputScale, std::string method, float weightClampValue, bool mergeChannel) {
    const int inputChannels  = inputScale.size();
    const int outputChannels = outputScale.size();
    const int icXoc          = inputChannels * outputChannels;
    DCHECK(size % icXoc == 0) << "Input Data Size Error!";

    std::vector<float> quantizedWeightScale(outputChannels);

    float inputScalexWeight = 1.0f;
    if (mergeChannel) {
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weight, size, quantizedWeight, quantizedWeightScale.data(), outputChannels, weightClampValue);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weight, size, quantizedWeight, quantizedWeightScale.data(), outputChannels, weightClampValue);
        }
        inputScalexWeight = inputScale[0];
    } else {
        const int kernelSize = size / icXoc;
        const int ocStride   = size / outputChannels;

        std::vector<float> weightMultiByInputScale(size);
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int ic = 0; ic < inputChannels; ++ic) {
                for (int i = 0; i < kernelSize; ++i) {
                    const int index                = oc * ocStride + ic * kernelSize + i;
                    weightMultiByInputScale[index] = inputScale[ic] * weight[index];
                }
            }
        }
        if (method == "MAX_ABS"){
            SymmetricQuantizeWeight(weightMultiByInputScale.data(), size, quantizedWeight, quantizedWeightScale.data(), outputChannels, weightClampValue);
        }
        else if (method == "ADMM") {
            QuantizeWeightADMM(weightMultiByInputScale.data(), size, quantizedWeight, quantizedWeightScale.data(), outputChannels, weightClampValue);
        }
    }

    for (int i = 0; i < outputChannels; ++i) {
        if (fabs(outputScale[i]) <= 1e-6) {
            scale[i] = 0.0f;
        } else {
            scale[i] = inputScalexWeight * quantizedWeightScale[i] / outputScale[0];
        }
    }

    if (bias) {
        for (int i = 0; i < outputChannels; ++i) {
            if (fabs(inputScalexWeight) <= 1e-6 || fabs(quantizedWeightScale[i]) <= 1e-6) {
                quantizedBias[i] = 0;
            } else {
//Shaquille, Modified 20210201 Start
#if 0
                quantizedBias[i] = static_cast<int32_t>(bias[i] / (inputScalexWeight * quantizedWeightScale[i]));
#else
                float   bias_quant = bias[i] / (inputScalexWeight * quantizedWeightScale[i]);
                quantizedBias[i]   = static_cast<int32_t>(std::roundf(bias_quant));
#endif
//Shaquille, Modified 20210201 End
            }
        }
    }

    return 0;
}

int QuantizeDepthwiseConv(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                          int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                          const std::vector<float>& outputScale, std::string method, float weightClampValue) {
    const int inputChannels  = inputScale.size();
    const int outputChannels = outputScale.size();
    DCHECK(inputChannels == outputChannels) << "Input Data Size Error!";

    std::vector<float> quantizedWeightScale(inputChannels);
    if (method == "MAX_ABS") {
        SymmetricQuantizeWeight(weight, size, quantizedWeight, quantizedWeightScale.data(), inputChannels, weightClampValue);
    }
    else if (method == "ADMM") {
        QuantizeWeightADMM(weight, size, quantizedWeight, quantizedWeightScale.data(), inputChannels, weightClampValue);
    }

    for (int c = 0; c < inputChannels; ++c) {
        const int index = c;
        if (fabs(outputScale[c]) <= 1e-6) {
            scale[index] = 0.0f;
        } else {
            scale[index] = inputScale[c] * quantizedWeightScale[c] / outputScale[c];
        }
    }

    if (bias) {
        for (int i = 0; i < outputChannels; ++i) {
            if (fabs(inputScale[i]) <= 1e-6 || fabs(quantizedWeightScale[i]) <= 1e-6) {
                quantizedBias[i] = 0;
            } else {
//Shaquille, Modified 20210201 Start
#if 0
                quantizedBias[i] = static_cast<int32_t>(bias[i] / (inputScale[i] * quantizedWeightScale[i]));
#else
                float   bias_quant = bias[i] / (inputScale[i] * quantizedWeightScale[i]);
                quantizedBias[i]   = static_cast<int32_t>(std::roundf(bias_quant));
#endif
//Shaquille, Modified 20210201 End
            }
        }
    }

    return 0;
}


//Shaquille, Added 20210201 Start
int FakeQuantizeConvWeight(std::vector<float> const&    inputScale,
                           std::vector<float> const&    outputScale,
                           std::vector<float> const&    quantize_scale,
                           float*                       weight,
                           int                          weight_cnt,
                           float*                       bias,
                           int                          clamp_value)
{
    int    channels              = (int)(outputScale.size());
    int    internel_kernel_size  = weight_cnt / channels;
    for(int i = 0 ; i < channels ; i ++)
    {
        float   quant_scale_w = quantize_scale[i] * outputScale[0] / inputScale[0];
        float   quant_scale_b = quantize_scale[i] * outputScale[0];
        for(int j = 0 ; j < internel_kernel_size ; j ++)
        {
            int   cur_w   = static_cast<int>(std::roundf(weight[i * internel_kernel_size + j] / quant_scale_w));
            cur_w         = std::max(-clamp_value, std::min(cur_w, clamp_value));
            weight[i * internel_kernel_size + j] = cur_w;
        }
        for(int j = 0 ; j < internel_kernel_size ; j ++)
            weight[i * internel_kernel_size + j] = weight[i * internel_kernel_size + j] * quant_scale_w;

        if(bias)
        {
            bias[i] = std::roundf(bias[i] / quant_scale_b);
            bias[i] = bias[i] * quant_scale_b;
        }
    }
}

int FakeQuantizeDepthwiseConvWeight(std::vector<float> const&    inputScale,
                                    std::vector<float> const&    outputScale,
                                    std::vector<float> const&    quantize_scale,
                                    float*                       weight,
                                    int                          weight_cnt,
                                    float*                       bias,
                                    int                          clamp_value)
{
    int    channels              = (int)(outputScale.size());
    int    internel_kernel_size  = weight_cnt / channels;
    for(int i = 0 ; i < channels ; i ++)
    {
        float   quant_scale_w = quantize_scale[i] * outputScale[i] / inputScale[i];
        float   quant_scale_b = quantize_scale[i] * outputScale[i];
        for(int j = 0 ; j < internel_kernel_size ; j ++)
        {
            int cur_w   = static_cast<int>(std::roundf(weight[i * internel_kernel_size + j] / quant_scale_w));
            cur_w       = std::max(-clamp_value, std::min(cur_w, clamp_value));
            weight[i * internel_kernel_size + j] = cur_w;
        }
        
        for(int j = 0 ; j < internel_kernel_size ; j ++)
            weight[i * internel_kernel_size + j] = weight[i * internel_kernel_size + j] * quant_scale_w;

        if(bias)
        {
            bias[i] = std::roundf(bias[i] / quant_scale_b);
            bias[i] = bias[i] * quant_scale_b;
        }
    }
}
//Shaquille, Added 20210201 End