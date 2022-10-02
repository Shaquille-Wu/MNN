//
//  quantizeWeight.hpp
//  MNN
//
//  Created by MNN on 2019/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef QUANTIZEWEIGHT_HPP
#define QUANTIZEWEIGHT_HPP
#include <stdint.h>
#include <vector>
#include <string>

// default: quantize weight every channel
int SymmetricQuantizeWeight(const float* weight, const int size, int8_t* quantizedWeight, float* scale,
                            const int channels, float weightClampValue);

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
                                              bool           mergeChannel = true);

int FakeQuantizeDepthwiseConvWeightBiasScale(const float*   weight, 
                                             const int      size, 
                                             const float*   bias,
                                             float*         fake_quantize_weight,
                                             float*         fake_quantize_bias,
                                             float*         scale, 
                                             int            input_channels,
                                             int            output_channels,
                                             std::string    method, 
                                             float          weightClampValue, 
                                             bool           mergeChannel = true);
//Shaquille, Added 20210204 End

// quantize convolution weight per channle
// firstly, multiply float weight by input_scale, then quantize the result to get input_sacle*weight_scale
// secondly, divide input_sacle*weight_scale by output_scale
int QuantizeConvPerChannel(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                           int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                           const std::vector<float>& outputScale, std::string method, float weightClampValue, bool mergeChannel = true);

int QuantizeDepthwiseConv(const float* weight, const int size, const float* bias, int8_t* quantizedWeight,
                          int32_t* quantizedBias, float* scale, const std::vector<float>& inputScale,
                          const std::vector<float>& outputScale, std::string method, float weightClampValue);

//Shaquille, Added 20210201 Start
int FakeQuantizeConvWeight(std::vector<float> const&    inputScale,
                           std::vector<float> const&    outputScale,
                           std::vector<float> const&    quantize_scale,
                           float*                       weight,
                           int                          weight_cnt,
                           float*                       bias,
                           int                          clamp_value);

int FakeQuantizeDepthwiseConvWeight(std::vector<float> const&    inputScale,
                                    std::vector<float> const&    outputScale,
                                    std::vector<float> const&    quantize_scale,
                                    float*                       weight,
                                    int                          weight_cnt,
                                    float*                       bias,
                                    int                          clamp_value);
//Shaquille, Added 20210201 End

#endif // QUANTIZEWEIGHT_HPP
