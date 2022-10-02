//
//  calibration.cpp
//  MNN
//
//  Created by MNN on 2019/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "calibration_orion.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <algorithm>
#include <MNN/ImageProcess.hpp>
#include "flatbuffers/util.h"
#include "logkit.h"
#include "quantizeWeight.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "Helper.hpp"
#include "core/TensorUtils.hpp"
#include <chrono>

using namespace MNN::CV;

static const int    kTuningFineCnt                               = 20;
static std::string  kQuantizeStrategy[Calibration::QUANTIZE_SUM] = {
    "Normal",
    "Layer_By_Layer",
    "Refine",
};

static int          GetQuantizeStrategy(std::string const& quantize_strategy)
{
    for(int i = 0 ; i < Calibration::QUANTIZE_SUM ; i ++)
    {
        if(quantize_strategy == kQuantizeStrategy[i])
            return i;
    }
    return Calibration::QUANTIZE_NORMAL;
}

Calibration::Calibration(MNN::NetT* model, const uint8_t* modelBuffer,  const int bufferSize,  const std::string& configPath)
    : _originaleModel(model) {
    // when the format of input image is RGB/BGR, channels equal to 3, GRAY is 1
    int channles = 3;

    rapidjson::Document document;
    {
        std::ifstream fileNames(configPath.c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return;
        }
    }
    auto picObj = document.GetObject();
    ImageProcess::Config config;
    config.filterType = BILINEAR;
    config.destFormat = BGR;
    {
        if (picObj.HasMember("format")) {
            auto format = picObj["format"].GetString();
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}, {"RGBA", RGBA}, {"BGRA", BGRA}};
            if (formatMap.find(format) != formatMap.end()) {
                config.destFormat = formatMap.find(format)->second;
            }
        }
    }

    switch (config.destFormat) {
        case GRAY:
            channles = 1;
            break;
        case RGB:
        case BGR:
            channles = 3;
            break;
        case RGBA:
        case BGRA:
            channles = 4;
            break;
        default:
            break;
    }

    config.sourceFormat = RGBA;
    std::string imagePath;
    _imageNum = 0;
    {
        if (picObj.HasMember("mean")) {
            auto mean = picObj["mean"].GetArray();
            int cur   = 0;
            for (auto iter = mean.begin(); iter != mean.end(); iter++) {
                config.mean[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("normal")) {
            auto normal = picObj["normal"].GetArray();
            int cur     = 0;
            for (auto iter = normal.begin(); iter != normal.end(); iter++) {
                config.normal[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("width")) {
            _width = picObj["width"].GetInt();
        }
        if (picObj.HasMember("height")) {
            _height = picObj["height"].GetInt();
        }
        if (picObj.HasMember("path")) {
            imagePath = picObj["path"].GetString();
        }
        if (picObj.HasMember("used_image_num")) {
            _imageNum = picObj["used_image_num"].GetInt();
        }
        if (picObj.HasMember("input_raw")) {
            input_raw_      = picObj["input_raw"].GetBool();
        }
        if (picObj.HasMember("min_quantize_threshold")) {
            min_quantize_threshold_ = picObj["min_quantize_threshold"].GetInt();
        }
        if (picObj.HasMember("quantize_strategy")) {
            std::string quantize_strategy = picObj["quantize_strategy"].GetString();
            quantize_strategy_ = GetQuantizeStrategy(quantize_strategy);
        }
        if (picObj.HasMember("feature_quantize_method")) {
            std::string method = picObj["feature_quantize_method"].GetString();
            if (Helper::featureQuantizeMethod.find(method) != Helper::featureQuantizeMethod.end()) {
                _featureQuantizeMethod = method;
            } else {
                MNN_ERROR("not supported feature quantization method: %s\n", method.c_str());
                return;
            }
        }
        if (picObj.HasMember("weight_quantize_method")) {
            std::string method = picObj["weight_quantize_method"].GetString();
            if (Helper::weightQuantizeMethod.find(method) != Helper::weightQuantizeMethod.end()) {
                _weightQuantizeMethod = method;
            } else {
                MNN_ERROR("not supported weight quantization method: %s\n", method.c_str());
                return;
            }
        }
        DLOG(INFO) << "Use feature quantization method: " << _featureQuantizeMethod;
        DLOG(INFO) << "Use weight quantization method: " << _weightQuantizeMethod;
        if (picObj.HasMember("feature_clamp_value")) {
            float value = (int)picObj["feature_clamp_value"].GetFloat();
            if (value < 0.0f || value > 127.0f) {
                MNN_ERROR("feature_clamp_value should be in (0, 127], got: %f\n", value);
                return;
            }
            _featureClampValue = value;
        }
        if (picObj.HasMember("weight_clamp_value")) {
            float value = (int)picObj["weight_clamp_value"].GetFloat();
            if (value < 0.0f || value > 127.0f) {
                MNN_ERROR("weight_clamp_value should be in (0, 127], got: %f\n", value);
                return;
            }
            _weightClampValue = value;
        }
        DLOG(INFO) << "feature_clamp_value: " << _featureClampValue;
        DLOG(INFO) << "weight_clamp_value: " << _weightClampValue;
        if (picObj.HasMember("skip_quant_op_names")) {
            auto skip_quant_op_names = picObj["skip_quant_op_names"].GetArray();
            for (auto iter = skip_quant_op_names.begin(); iter != skip_quant_op_names.end(); iter++) {
                std::string skip_quant_op_name = iter->GetString();
                _skip_quant_ops.emplace_back(skip_quant_op_name);
                DLOG(INFO) << "skip quant op name: " << skip_quant_op_name;
            }
        }
        if (picObj.HasMember("debug")) {
            _debug = picObj["debug"].GetBool();
        }
    }
    std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
    _process = process;

    // read images file names
    int image_num = 0;
    std::vector<std::string> raw_img_list;
    Helper::readImages(raw_img_list, imagePath.c_str(), &image_num);
    if(0 == _imageNum)
    {
        _imgaes   = raw_img_list;
        _imageNum = image_num;
    }
    else
    {
        if(_imageNum < image_num)
            _imgaes = select_rand_image(raw_img_list, _imageNum);
        else
        {
            _imageNum = image_num;
            _imgaes   = raw_img_list;
        }
    }

    _initMNNSession(modelBuffer, bufferSize, channles);
    _initMaps();
}

void Calibration::_initMNNSession(const uint8_t* modelBuffer, const int bufferSize, const int channels) {
    _interpreterOrigin.reset(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    MNN::ScheduleConfig config;
    _sessionOrigin     = _interpreterOrigin->createSession(config);
    _inputTensorOrigin = _interpreterOrigin->getSessionInput(_sessionOrigin, NULL);

    fake_quantize_model_.reset();
    fake_quantize_model_ = MNN::UnPackNet(modelBuffer);

    //_fake_quant_weights();

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, _originaleModel);
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto buffer = builder.GetBufferPointer();

    _interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    _session     = _interpreter->createSession(config);
    _inputTensor = _interpreter->getSessionInput(_session, NULL);

    _inputTensorDims.resize(4);
    auto inputTensorDataFormat = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
    if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        _inputTensorDims[0] = 1;
        _inputTensorDims[1] = _height;
        _inputTensorDims[2] = _width;
        _inputTensorDims[3] = channels;
    } else {
        _inputTensorDims[0] = 1;
        _inputTensorDims[1] = channels;
        _inputTensorDims[2] = _height;
        _inputTensorDims[3] = _width;
    }
    if (_featureQuantizeMethod == "KL") {
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);  
    } else if (_featureQuantizeMethod == "ADMM") {
        DCHECK((_imageNum * 4 * _height * _width) < (INT_MAX / 4)) << "Use Little Number of Images When Use ADMM";
        _inputTensorDims[0] = _imageNum;
        _interpreter->resizeTensor(_inputTensor, _inputTensorDims);
        _interpreter->resizeSession(_session);
        _interpreterOrigin->resizeTensor(_inputTensorOrigin, _inputTensorDims);
        _interpreterOrigin->resizeSession(_sessionOrigin);
    }
}

void Calibration::_initMaps() {
    _featureInfo.clear();
    _featureInfoOrigin.clear();
    _featureInfoQuant.clear();
    fake_quantize_feature_info_.clear();
    _opInfo.clear();
    _tensorMap.clear();
    // run mnn once, initialize featureMap, opInfo map
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        _opInfo[opName].first = nTensors;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                    _featureInfo[t]->set_min_threshold(min_quantize_threshold_);
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo after = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        _opInfo[opName].second = nTensors;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfo.find(t) == _featureInfo.end()) {
                    _featureInfo[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                    _featureInfo[t]->set_min_threshold(min_quantize_threshold_);
                }
                i++;
            }
        }
        return true;
    };
    _interpreter->runSessionWithCallBackInfo(_session, before, after);


    MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                    _featureInfoOrigin[t]->set_min_threshold(min_quantize_threshold_);
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo afterOrigin = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) == _featureInfoOrigin.end()) {
                    _featureInfoOrigin[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                    _featureInfoOrigin[t]->set_min_threshold(min_quantize_threshold_);
                }
                i++;
            }
        }
        return true;
    };
    _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

    for (auto& op : _originaleModel->oplists) {
        if (_opInfo.find(op->name) == _opInfo.end()) {
            continue;
        }
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            _tensorMap[op->inputIndexes[i]] = _opInfo[op->name].first[i];
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            _tensorMap[op->outputIndexes[i]] = _opInfo[op->name].second[i];
        }
    }

    if (_featureQuantizeMethod == "KL") {
        // set the tensor-statistic method of input tensor as THRESHOLD_MAX
        auto inputTensorStatistic = _featureInfo.find(_inputTensor);
        if (inputTensorStatistic != _featureInfo.end()) {
            inputTensorStatistic->second->setThresholdMethod(THRESHOLD_MAX);
        }
    }
}

void Calibration::_computeFeatureMapsRange() {
    // feed input data according to input images
    int count = 0;
    for (const auto& img : _imgaes) {
        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedRangeFlags();
        }
        count++;
        load_image(_process.get(), _width, _height, img, _inputTensor, input_raw_);

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _featureInfo[t]->updateRange();
                    }
                }
            }
            return true;
        };

        _interpreter->runSessionWithCallBackInfo(_session, before, after);
        MNN_PRINT("\rComputeFeatureRange: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_collectFeatureMapsDistribution() {
    for (auto& iter : _featureInfo) {
        iter.second->resetDistribution();
    }
    // feed input data according to input images
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->updateDistribution();
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if (_featureInfo.find(t) != _featureInfo.end()) {
                if (_featureInfo[t]->visited() == false) {
                    _featureInfo[t]->updateDistribution();
                }
            }
        }
        return true;
    };
    int count = 0;
    for (const auto& img : _imgaes) {
        count++;

        for (auto& iter : _featureInfo) {
            iter.second->setVisited(false);
        }

        for (auto& iter : _featureInfo) {
            iter.second->resetUpdatedDistributionFlag();
        }
        load_image(_process.get(), _width, _height, img, _inputTensor, input_raw_);
        _interpreter->runSessionWithCallBackInfo(_session, before, after);

        MNN_PRINT("\rCollectFeatureDistribution: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
}

void Calibration::_computeFeatureScaleKL() {
    _computeFeatureMapsRange();
    _collectFeatureMapsDistribution();

    _scales.clear();
    for (auto& iter : _featureInfo) {
        AUTOTIME;
        _scales[iter.first] = iter.second->finishAndCompute();
    }
    //_featureInfo.clear();//No need now
}

void Calibration::_computeFeatureScaleADMM() {
    // feed input data according to input images
    int count                           = 0;
    std::vector<int> oneImageTensorDims = _inputTensorDims;
    oneImageTensorDims[0]               = 1;
    auto inputTensorDataFormat          = MNN::TensorUtils::getDescribe(_inputTensor)->dimensionFormat;
    auto dimType                        = MNN::Tensor::CAFFE_C4;
    if (inputTensorDataFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        dimType = MNN::Tensor::TENSORFLOW;
    }

    for (const auto& img : _imgaes) {
        auto curPtr = _inputTensor->host<float>() + count * _inputTensor->stride(0);
        std::shared_ptr<MNN::Tensor> tensorWarp(
            MNN::Tensor::create(oneImageTensorDims, _inputTensor->getType(), curPtr, dimType));
        load_image(_process.get(), _width, _height, img, tensorWarp.get(), input_raw_);

        count++;
        MNN_PRINT("\rProcessImage: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n");
    _scales.clear();

    const int totalLayers = _featureInfo.size();
    count                 = 0;

    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            for (auto t : nTensors) {
                if (_featureInfo.find(t) != _featureInfo.end()) {
                    if (_featureInfo[t]->visited() == false) {
                        _scales[t] = _featureInfo[t]->computeScaleADMM();
                        count++;
                        MNN_PRINT("\rComputeADMM: %.2lf %%", (float)count * 100.0f / (float)totalLayers);
                        fflush(stdout);
                    }
                }
            }
        }
        return true;
    };

    _interpreter->runSessionWithCallBackInfo(_session, before, after);
    MNN_PRINT("\n");
}

void Calibration::_updateScale() {
    int  op_idx = 0;
    for (const auto& op : _originaleModel->oplists) {
        op_idx ++;
        int cur_op_idx = op_idx - 1;
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
            opType != MNN::OpType_Eltwise) {
            continue;
        }
        auto tensorsPair = _opInfo.find(op->name);
        if (tensorsPair == _opInfo.end()) {
            MNN_ERROR("Can't find tensors for %s\n", op->name.c_str());
        }

        if (opType == MNN::OpType_Eltwise) {
            auto param = op->main.AsEltwise();
            // Now only support AddInt8
            if (param->type != MNN::EltwiseType_SUM) {
                continue;
            }
            const auto& inputScale0   = _scales[tensorsPair->second.first[0]];
            const auto& inputScale1   = _scales[tensorsPair->second.first[1]];
            const auto& outputScale   = _scales[tensorsPair->second.second[0]];
            const int outputScaleSize = outputScale.size();
            std::vector<float> outputInvertScale(outputScaleSize);
            Helper::invertData(outputInvertScale.data(), outputScale.data(), outputScaleSize);
            op->type = MNN::OpType_EltwiseInt8;
            op->main.Reset();
            op->main.type = MNN::OpParameter_EltwiseInt8;

            auto eltwiseInt8Param         = new MNN::EltwiseInt8T;
            auto input0ScaleParam         = new MNN::QuantizedFloatParamT;
            auto input1ScaleParam         = new MNN::QuantizedFloatParamT;
            auto outputScaleParam         = new MNN::QuantizedFloatParamT;
            input0ScaleParam->tensorScale = inputScale0;
            input1ScaleParam->tensorScale = inputScale1;
            outputScaleParam->tensorScale = outputInvertScale;
            eltwiseInt8Param->inputQuan0  = std::unique_ptr<MNN::QuantizedFloatParamT>(input0ScaleParam);
            eltwiseInt8Param->inputQuan1  = std::unique_ptr<MNN::QuantizedFloatParamT>(input1ScaleParam);
            eltwiseInt8Param->outputQuan  = std::unique_ptr<MNN::QuantizedFloatParamT>(outputScaleParam);
            op->main.value                = eltwiseInt8Param;
            continue;
        }
        /*
        else if(opType == MNN::OpType_Pooling)
        {
            op->type = MNN::OpType_PoolInt8;
            continue;
        }
        */
        // below is Conv/DepthwiseConv
        const auto& inputScale  = _scales[tensorsPair->second.first[0]];
        const auto& outputScale = _scales[tensorsPair->second.second[0]];

        auto param                = op->main.AsConvolution2D();
        const int channles        = param->common->outputCount;
        const int weightSize      = param->weight.size();
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        auto& quantizedParam = param->symmetricQuan;
        quantizedParam->scale.resize(channles);
        quantizedParam->weight.resize(weightSize);
        quantizedParam->bias.resize(channles);

        auto    cur_quant_op    = (fake_quantize_model_->oplists)[cur_op_idx].get();
        auto    quant_param     = cur_quant_op->main.AsConvolution2D();
        if (opType == MNN::OpType_Convolution) {
            QuantizeConvPerChannel(param->weight.data(), param->weight.size(), param->bias.data(),
                                   quantizedParam->weight.data(), quantizedParam->bias.data(),
                                   quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod, _weightClampValue);
            FakeQuantizeConvWeight(inputScale, 
                                   outputScale, 
                                   quantizedParam->scale, 
                                   quant_param->weight.data(), 
                                   quant_param->weight.size(), 
                                   quant_param->bias.data(),
                                   (int)(_weightClampValue + 0.5f));
            op->type = MNN::OpType_ConvInt8;
        } else if (opType == MNN::OpType_ConvolutionDepthwise) {
            QuantizeDepthwiseConv(param->weight.data(), param->weight.size(), param->bias.data(),
                                  quantizedParam->weight.data(), quantizedParam->bias.data(),
                                  quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod, _weightClampValue);
            FakeQuantizeDepthwiseConvWeight(inputScale, 
                                            outputScale, 
                                            quantizedParam->scale, 
                                            quant_param->weight.data(), 
                                            quant_param->weight.size(), 
                                            quant_param->bias.data(),
                                            (int)(_weightClampValue + 0.5f));
            op->type = MNN::OpType_DepthwiseConvInt8;
        }
        if (param->common->relu6) {
            param->common->relu  = true;
            param->common->relu6 = false;
        }

        param->weight.clear();
        param->bias.clear();
    }
}

void Calibration::compute_conv_weight_scale()
{
    int  op_cnt      = (int)(_originaleModel->oplists.size());
    int  op_idx      = 0;
    int  conv_op_cnt = 0;
    fake_quantize_weight_scale_.clear();
    fake_quantize_conv_op_idx_.clear();
    for(op_idx = 0 ; op_idx < op_cnt ; op_idx ++)
    {
        const auto& op     = _originaleModel->oplists[op_idx];
        const auto  opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise)
            continue;
        conv_op_cnt ++;
    }
    fake_quantize_weight_scale_.resize(conv_op_cnt);
    fake_quantize_conv_op_idx_.resize(conv_op_cnt);
    conv_op_cnt = 0;
    for(op_idx = 0 ; op_idx < op_cnt ; op_idx ++)
    {
        const auto& op     = _originaleModel->oplists[op_idx];
        const auto  opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise)
            continue;
        fake_quantize_conv_op_idx_[conv_op_cnt] = op_idx;
        conv_op_cnt ++;
    }

    for(int conv_op_idx = 0 ; conv_op_idx < conv_op_cnt ; conv_op_idx ++)
    {
        int         op_idx        = fake_quantize_conv_op_idx_[conv_op_idx];
        const auto& op_src        = _originaleModel->oplists[op_idx];
        const auto& op_quant      = fake_quantize_model_->oplists[op_idx];
        const auto  opType        = op_src->type;
        auto        param_src     = op_src->main.AsConvolution2D();
        const int   channles      = param_src->common->outputCount;
        const int   weightSize    = param_src->weight.size();
        auto        param_quant   = op_quant->main.AsConvolution2D();
        fake_quantize_weight_scale_[conv_op_idx].resize(channles);

        if (opType == MNN::OpType_Convolution) {
            FakeQuantizeConvPerChannelWeightBiasScale(param_src->weight.data(), 
                                                      weightSize, 
                                                      param_src->bias.data(),
                                                      param_quant->weight.data(),
                                                      param_quant->bias.data(),
                                                      fake_quantize_weight_scale_[conv_op_idx].data(),
                                                      param_src->common->inputCount,
                                                      channles,
                                                      _weightQuantizeMethod,
                                                      _weightClampValue);
        } else if (opType == MNN::OpType_ConvolutionDepthwise) {
            FakeQuantizeDepthwiseConvWeightBiasScale(param_src->weight.data(), 
                                                     weightSize, 
                                                     param_src->bias.data(),
                                                     param_quant->weight.data(),
                                                     param_quant->bias.data(),
                                                     fake_quantize_weight_scale_[conv_op_idx].data(),
                                                     param_src->common->inputCount,
                                                     channles,
                                                     _weightQuantizeMethod,
                                                     _weightClampValue);
        }
    }
}

void Calibration::compute_feature_scale()
{
    fake_quantize_feature_info_.clear();
    fake_quantize_feature_name_.clear();
    fake_quantize_feature_name_idx_map_.clear();
    fake_quantize_session_op_name_.clear();
    _scales.clear();
    _tensorMap.clear();
    _opInfo.clear();
    create_fake_quantize_session();

    int  op_cnt          = (int)(fake_quantize_model_->oplists.size());
    int  op_idx          = 0;
    int  quantize_op_cnt = 0;
    int  i               = 0;
    std::vector<int> quantize_op(op_cnt, 0);

    init_quantize_feature_name(fake_quantize_model_.get(),
                               fake_quantize_interpret_.get(), 
                               fake_quantize_session_, 
                               fake_quantize_input_tensor_,
                               _skip_quant_ops,
                               fake_quantize_feature_name_,
                               fake_quantize_feature_name_idx_map_, 
                               fake_quantize_feature_info_,
                               _tensorMap,
                               _opInfo,
                               _featureQuantizeMethod,
                               _featureClampValue,
                               min_quantize_threshold_);

    init_quantize_oplist(fake_quantize_model_.get(),
                         fake_quantize_interpret_.get(), 
                         fake_quantize_session_, 
                         fake_quantize_session_op_name_);

    std::map<MNN::Tensor*, int> quantize_feature_flag;
    std::map<MNN::Tensor*, std::shared_ptr<TensorStatistic>>::const_iterator feature_iter = fake_quantize_feature_info_.begin();
    while(fake_quantize_feature_info_.end() != feature_iter)
    {
        MNN::Tensor* cur_tensor     = feature_iter->first;
        quantize_feature_flag[cur_tensor] = 0;
        feature_iter ++;
    }
    
    quantize_op_cnt = (int)(fake_quantize_session_op_name_.size());
    for(i = 0 ; i < quantize_op_cnt ; i ++)
    {
        int  stop_op_idx = std::get<1>(fake_quantize_session_op_name_[i]);
        std::map<MNN::Tensor*, TensorStatistic*> none_stop_out_feature;
        std::map<MNN::Tensor*, TensorStatistic*> stop_op_out_feature;
        retrieve_quantize_feature(stop_op_idx, quantize_feature_flag, none_stop_out_feature, stop_op_out_feature);
        tunning_quantize_network(stop_op_idx, i, quantize_op_cnt, none_stop_out_feature, stop_op_out_feature, _scales);
        for(auto& none_out_iter:none_stop_out_feature)
        {
            MNN::Tensor* cur_tensor           = none_out_iter.first;
            quantize_feature_flag[cur_tensor] = 1;
        }
        for(auto& out_iter:stop_op_out_feature)
        {
            MNN::Tensor* cur_tensor           = out_iter.first;
            quantize_feature_flag[cur_tensor] = 1;
        }
    }
}

void Calibration::init_quantize_feature_name(MNN::NetT*                                                  model,
                                             MNN::Interpreter*                                           interpreter, 
                                             MNN::Session*                                               session,
                                             MNN::Tensor*                                                input_tensor,
                                             std::vector<std::string>&                                   skip_quant_ops,
                                             std::vector<std::string>&                                   tensor_name_list,
                                             std::map<std::string, int>&                                 tensor_name_idx_map,
                                             std::map<MNN::Tensor*, std::shared_ptr<TensorStatistic>>&   feature_info,
                                             std::map<int, const MNN::Tensor*>&                          idx_tensor_map,
                                             OpToTensorMap&                                              quantize_op_info,
                                             std::string const&                                          feature_quantize_method,
                                             float                                                       feature_clamp_value,
                                             int                                                         min_quantize_threshold)
{
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(skip_quant_ops.begin(), skip_quant_ops.end(), opName);
        if (iter != skip_quant_ops.end()) {
            return false;
        }
        quantize_op_info[opName].first = nTensors;
        int i = 0;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) 
        {
            for (auto t : nTensors) {
                if (feature_info.find(t) == feature_info.end()) {
                    int tensor_idx                    = tensor_name_idx_map.size();
                    std::string tensor_name           = opName + " input_tensor_" + flatbuffers::NumToString(i);
                    tensor_name_idx_map[tensor_name]  = tensor_idx;
                    feature_info[t]                   = std::shared_ptr<TensorStatistic>(new TensorStatistic(t, 
                                                                                                             feature_quantize_method, 
                                                                                                             tensor_name, 
                                                                                                             feature_clamp_value));
                    feature_info[t]->set_min_threshold(min_quantize_threshold);
                }
                i++;
            }
        }

        return false;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(skip_quant_ops.begin(), skip_quant_ops.end(), opName);
        if (iter != skip_quant_ops.end()) {
            return true;
        }
        quantize_op_info[opName].second = nTensors;
        int i = 0;
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) 
        {
            for (auto t : nTensors) {
                if (feature_info.find(t) == feature_info.end()) {
                    int tensor_idx                    = tensor_name_idx_map.size();
                    std::string tensor_name           = opName + " output_tensor_" + flatbuffers::NumToString(i);
                    tensor_name_idx_map[tensor_name]  = tensor_idx;
                    feature_info[t]                   = std::shared_ptr<TensorStatistic>(new TensorStatistic(t, 
                                                                                                             feature_quantize_method, 
                                                                                                             tensor_name, 
                                                                                                             feature_clamp_value));
                    feature_info[t]->set_min_threshold(min_quantize_threshold);
                }
                i++;
            }
        }

        return true;
    };
    interpreter->runSessionWithCallBackInfo(session, before, after);

    int  tensor_cnt = (int)(tensor_name_idx_map.size());
    tensor_name_list.resize(tensor_cnt);
    std::map<std::string, int>::const_iterator  iter = tensor_name_idx_map.begin();
    while(tensor_name_idx_map.end() != iter)
    {
        std::string const&  tensor_name = iter->first;
        int                 tensor_idx  = iter->second;
        tensor_name_list[tensor_idx]    = tensor_name;
        iter ++;
    }

    for (auto& op : model->oplists) 
    {
        if (quantize_op_info.find(op->name) == quantize_op_info.end())
            continue;
        for (int i = 0; i < op->inputIndexes.size(); ++i)
            idx_tensor_map[op->inputIndexes[i]]  = quantize_op_info[op->name].first[i];
        for (int i = 0; i < op->outputIndexes.size(); ++i)
            idx_tensor_map[op->outputIndexes[i]] = quantize_op_info[op->name].second[i];
    }

    if ("KL" == feature_quantize_method) {
        // set the tensor-statistic method of input tensor as THRESHOLD_MAX
        auto input_tensor_statistic = feature_info.find(input_tensor);
        if (input_tensor_statistic != feature_info.end())
            input_tensor_statistic->second->setThresholdMethod(THRESHOLD_MAX);
    }
}

void Calibration::init_quantize_oplist(MNN::NetT*                                               model,
                                       MNN::Interpreter*                                        interpreter, 
                                       MNN::Session*                                            session,
                                       std::vector<std::tuple<std::string, int, MNN::OpType>>&  op_list)
{
    std::vector<std::pair<std::string, std::string>>  oplist(2048);
    int                                               op_cnt = 0;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        oplist[op_cnt]     = std::make_pair(opName, info->type());
        op_cnt ++;
        return false;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        return true;
    };
    interpreter->runSessionWithCallBackInfo(session, before, after);

    std::vector<std::tuple<std::string, int, MNN::OpType>>  tmp_oplist(op_cnt);
    int   op_idx = 0; 
    int   i      = 0;
    int   j      = 0;
    for(i = 0 ; i < op_cnt ; i ++)
    {
        std::string  op_name = oplist[i].first;
        std::string  op_type = oplist[i].second;
        if (Helper::gNeedFeatureOp.find(op_type) != Helper::gNeedFeatureOp.end()) 
        {
            MNN::OpType  cur_op_type = MNN::OpType_Convolution;
            if("Convolution" == op_type)
                cur_op_type = MNN::OpType_Convolution;
            else if("ConvolutionDepthwise" == op_type)
                cur_op_type = MNN::OpType_ConvolutionDepthwise;
            else if("Eltwise" == op_type)
                cur_op_type = MNN::OpType_Eltwise;
            
            bool  valid = true;
            bool  found = false;
            if(MNN::OpType_Eltwise == cur_op_type)
            {
                int model_op_cnt = (int)(model->oplists.size());
                for(j = 0 ; j < model_op_cnt; j ++)
                {
                    const auto&         src_op      = model->oplists[j];
                    const auto          src_opType  = src_op->type;
                    std::string const&  src_op_name = src_op->name;
                    if(src_op_name == op_name)
                    {
                        if (src_opType != MNN::OpType_Eltwise)
                        {
                            DLOG(INFO) << "quantize error, error op_type";
                            exit(0);
                        }
                        found      = true;
                        auto param = src_op->main.AsEltwise();
                        if (param->type != MNN::EltwiseType_SUM)
                            valid = false;
                        break;
                    }
                }
                if(false == found)
                {
                    DLOG(INFO) << "quantize error, cannot find op in oplist";
                    exit(0);
                }
            }
            if(true == valid)
            {
                tmp_oplist[op_idx] = std::make_tuple(op_name, i, cur_op_type);
                op_idx ++;
            }
        }
    }

    op_list.resize(op_idx);
    for(i = 0 ; i < op_idx ; i ++)
        op_list[i] = tmp_oplist[i];
}

void Calibration::create_fake_quantize_session()
{
    std::vector<int> inputShape = {1, _inputTensorDims[1], _inputTensorDims[2], _inputTensorDims[3]};
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset   = MNN::Net::Pack(builder, fake_quantize_model_.get());
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto buffer   = builder.GetBufferPointer();

    MNN::ScheduleConfig config;
    if(nullptr != fake_quantize_session_)
        fake_quantize_interpret_->releaseSession(fake_quantize_session_);
    fake_quantize_session_      = nullptr;
    fake_quantize_interpret_.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    fake_quantize_session_      = fake_quantize_interpret_->createSession(config);
    fake_quantize_input_tensor_ = fake_quantize_interpret_->getSessionInput(fake_quantize_session_, NULL);
    fake_quantize_interpret_->resizeTensor(fake_quantize_input_tensor_, inputShape);
    fake_quantize_interpret_->resizeSession(fake_quantize_session_);
}

void Calibration::compute_feature_range(int                                         stop_op_idx, 
                                        int                                         cur_quantize_idx,
                                        int                                         total_quantize_op_cnt,
                                        int                                         in_out,
                                        std::map<MNN::Tensor*, TensorStatistic*>&   none_stop_out_feature, 
                                        std::map<MNN::Tensor*, TensorStatistic*>&   stop_op_out_feature)
{
    int   count          = 0;
    int   proc_op_cnt0   = 0;
    int   proc_op_cnt1   = 0;
    std::chrono::time_point<std::chrono::system_clock> fwd_start;
    std::chrono::time_point<std::chrono::system_clock> fwd_end;
    std::chrono::microseconds                          forward_cost;
    fwd_start = std::chrono::system_clock::now();
    for (const auto& img : _imgaes) 
    {
        count++;
        proc_op_cnt0 = 0;
        proc_op_cnt1 = 0;

        load_image(_process.get(), _width, _height, img, fake_quantize_input_tensor_, input_raw_);

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if(none_stop_out_feature.find(t) != none_stop_out_feature.end())
                {
                    if (none_stop_out_feature[t]->visited() == false)
                    {
                        if(0 == in_out)
                            none_stop_out_feature[t]->updateRange();
                        else
                            none_stop_out_feature[t]->fakeQuantFeature(false, false, true);
                    }
                }
                else
                {
                    if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end() &&
                       stop_op_out_feature.find(t) == stop_op_out_feature.end())
                    {
                        if (fake_quantize_feature_info_[t]->visited() == false)
                            fake_quantize_feature_info_[t]->fakeQuantFeature(false, false, true);
                    }
                }
            }
            if(0 == in_out)
            {
                if(proc_op_cnt0 >= stop_op_idx)
                    return false;
            }
            proc_op_cnt0 ++;
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            if(0 == in_out)
            {
                if(proc_op_cnt1 >= stop_op_idx)
                {
                    proc_op_cnt1 ++;
                    return false;
                }
            }
            for (auto t : nTensors) {
                std::map<MNN::Tensor*, TensorStatistic*>::iterator  none_out_iter = none_stop_out_feature.find(t);
                std::map<MNN::Tensor*, TensorStatistic*>::iterator  stop_out_iter = stop_op_out_feature.find(t);
                if(none_out_iter != none_stop_out_feature.end() ||
                   stop_out_iter != stop_op_out_feature.end())
                {
                    TensorStatistic*  feature_info = nullptr;
                    if(none_out_iter != none_stop_out_feature.end())
                        feature_info = none_out_iter->second;
                    else
                        feature_info = stop_out_iter->second;
                    if (feature_info->visited() == false) {
                        feature_info->updateRange();
                        if(1 == in_out && none_out_iter != none_stop_out_feature.end())
                            feature_info->fakeQuantFeature(false, false, true);
                    }
                }
                else
                {
                    if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end())
                    {
                        if (fake_quantize_feature_info_[t]->visited() == false)
                            fake_quantize_feature_info_[t]->fakeQuantFeature(false, false, true);
                    }
                }
            }
            if(proc_op_cnt1 >= stop_op_idx)
                return false;
            proc_op_cnt1 ++;
            return true;
        };

        for (auto& iter : fake_quantize_feature_info_)
        {
            MNN::Tensor*      cur_tensor   = iter.first;
            TensorStatistic*  feature_info = iter.second.get();
            feature_info->resetUpdatedRangeFlags();
            feature_info->setVisited(false);
        }
        fake_quantize_interpret_->runSessionWithCallBackInfo(fake_quantize_session_, before, after);

        float elapsed_ratio       = (float)count * 100.0f / (float)_imageNum;
        MNN_PRINT("\r(%d/%d), %s, caculate %s feature range: %.4lf %%", 
                    cur_quantize_idx, total_quantize_op_cnt, 
                    (std::get<0>(fake_quantize_session_op_name_[cur_quantize_idx])).c_str(), 
                    0 == in_out ? "pre_input" : "output",
                    elapsed_ratio);
        fflush(stdout);
    }
    fwd_end      = std::chrono::system_clock::now();
    forward_cost = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
    MNN_PRINT("\r\n");
    MNN_PRINT("(%d/%d), %s, %s range_computation elapsed %.3f ms\r\n", 
               cur_quantize_idx, 
               total_quantize_op_cnt, 
               (std::get<0>(fake_quantize_session_op_name_[cur_quantize_idx])).c_str(), 
               0 == in_out ? "pre_input" : "output",
               (double)(forward_cost.count()) / 1000.0);
}

void Calibration::compute_feature_distribution(int                                         stop_op_idx, 
                                               int                                         cur_quantize_idx,
                                               int                                         total_quantize_op_cnt,
                                               int                                         in_out,
                                               std::map<MNN::Tensor*, TensorStatistic*>&   none_stop_out_feature, 
                                               std::map<MNN::Tensor*, TensorStatistic*>&   stop_op_out_feature)
{
    int   count          = 0;
    int   proc_op_cnt0   = 0;
    int   proc_op_cnt1   = 0;
    std::chrono::time_point<std::chrono::system_clock> fwd_start;
    std::chrono::time_point<std::chrono::system_clock> fwd_end;
    std::chrono::microseconds                          forward_cost;
    fwd_start      = std::chrono::system_clock::now();
    for (auto& iter : none_stop_out_feature)
        iter.second->resetDistribution();
    for (auto& iter : stop_op_out_feature)
        iter.second->resetDistribution();
    for (const auto& img : _imgaes)
    {
        count++;
        proc_op_cnt0 = 0;
        proc_op_cnt1 = 0;

        load_image(_process.get(), _width, _height, img, fake_quantize_input_tensor_, input_raw_);
        std::map<std::string, std::vector<float>> fakeQuantedFeatures;
        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            if(1 == in_out)
            {
                proc_op_cnt0 ++;
                return true;
            } 
            for (auto t : nTensors) {
                if(none_stop_out_feature.find(t) != none_stop_out_feature.end())
                {
                    if (none_stop_out_feature[t]->visited() == false)
                    {
                        if(0 == in_out)
                            none_stop_out_feature[t]->updateDistribution();
                        else
                            none_stop_out_feature[t]->fakeQuantFeature(false, false, true);
                    }
                }
                else
                {
                    if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end() &&
                       stop_op_out_feature.find(t) == stop_op_out_feature.end())
                    {
                        if (fake_quantize_feature_info_[t]->visited() == false)
                            fake_quantize_feature_info_[t]->fakeQuantFeature(false, false, true);
                    }
                }
            }
            if(0 == in_out)
            {
                if(proc_op_cnt0 >= stop_op_idx)
                    return false;
            }
            proc_op_cnt0 ++;
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            if(0 == in_out)
            {
                if(proc_op_cnt1 >= stop_op_idx)
                {
                    proc_op_cnt1 ++;
                    return false;
                }
            }
            for (auto t : nTensors) {
                std::map<MNN::Tensor*, TensorStatistic*>::iterator  none_out_iter = none_stop_out_feature.find(t);
                std::map<MNN::Tensor*, TensorStatistic*>::iterator  stop_out_iter = stop_op_out_feature.find(t);
                if(none_out_iter != none_stop_out_feature.end() ||
                   stop_out_iter != stop_op_out_feature.end())
                {
                    TensorStatistic*  feature_info = nullptr;
                    if(none_out_iter != none_stop_out_feature.end())
                        feature_info = none_out_iter->second;
                    else
                        feature_info = stop_out_iter->second;
                    if (feature_info->visited() == false) {
                        feature_info->updateDistribution();
                        if(1 == in_out && none_out_iter != none_stop_out_feature.end())
                            feature_info->fakeQuantFeature(false, false, true);
                    }
                }
                else
                {
                    if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end())
                    {
                        if (fake_quantize_feature_info_[t]->visited() == false)
                            fake_quantize_feature_info_[t]->fakeQuantFeature(false, false, true);
                    }
                }
            }
            if(proc_op_cnt1 >= stop_op_idx)
                return false;
            proc_op_cnt1 ++;
            return true;
        };

        for (auto& iter : fake_quantize_feature_info_)
        {
            iter.second->resetUpdatedDistributionFlag();
            iter.second->setVisited(false);
        }
        fake_quantize_interpret_->runSessionWithCallBackInfo(fake_quantize_session_, before, after);

        float elapsed_ratio       = (float)count * 100.0f / (float)_imageNum;
        MNN_PRINT("\r(%d/%d), %s, caculate %s feature distribution: %.4lf %%", 
                    cur_quantize_idx, total_quantize_op_cnt, 
                    (std::get<0>(fake_quantize_session_op_name_[cur_quantize_idx])).c_str(), 
                    0 == in_out ? "pre_input" : "output",
                    elapsed_ratio);
        fflush(stdout);
    }
    fwd_end      = std::chrono::system_clock::now();
    forward_cost = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
    MNN_PRINT("\r\n");
    MNN_PRINT("(%d/%d), %s, %s distribution_computation elapsed %.3f ms\r\n", 
               cur_quantize_idx, 
               total_quantize_op_cnt, 
               (std::get<0>(fake_quantize_session_op_name_[cur_quantize_idx])).c_str(), 
               0 == in_out ? "pre_input" : "output",
               (double)(forward_cost.count()) / 1000.0);
}

void Calibration::retrieve_quantize_feature(int                                        stop_op_idx, 
                                            std::map<MNN::Tensor*, int>                quantize_feature_flag,
                                            std::map<MNN::Tensor*, TensorStatistic*>&  none_stop_out_feature, 
                                            std::map<MNN::Tensor*, TensorStatistic*>&  stop_op_out_feature)
{
    int  proc_op_cnt = 0;
    MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end())
            {
                if(0 == quantize_feature_flag.at(t))
                    none_stop_out_feature[t] = fake_quantize_feature_info_[t].get();
            }
        }
        return true;
    };
    MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                            const MNN::OperatorInfo* info) {
        for (auto t : nTensors) {
            if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end())
            {
                if(0 == quantize_feature_flag.at(t))
                {
                    if(stop_op_idx != proc_op_cnt)
                        none_stop_out_feature[t] = fake_quantize_feature_info_[t].get();
                    else
                        stop_op_out_feature[t] = fake_quantize_feature_info_[t].get();
                }
            }
        }
        if(proc_op_cnt >= stop_op_idx)
            return false;
        proc_op_cnt ++;
        return true;
    };

    fake_quantize_interpret_->runSessionWithCallBackInfo(fake_quantize_session_, before, after);
}

void Calibration::tunning_quantize_network(int                                               stop_op_idx, 
                                           int                                               cur_quantize_idx,
                                           int                                               total_quantize_op_cnt,
                                           std::map<MNN::Tensor*, TensorStatistic*>&         none_stop_out_feature, 
                                           std::map<MNN::Tensor*, TensorStatistic*>&         stop_op_out_feature,
                                           std::map<const MNN::Tensor*, std::vector<float>>& tensor_scale)
{
    std::chrono::time_point<std::chrono::system_clock> fwd_start;
    std::chrono::time_point<std::chrono::system_clock> fwd_end;
    std::chrono::microseconds                          forward_cost;
    fwd_start = std::chrono::system_clock::now();
    std::vector<std::vector<double> > std_distribute_hist;
    if(none_stop_out_feature.size() > 0)
    {
        compute_feature_range(stop_op_idx, cur_quantize_idx, total_quantize_op_cnt, 0, none_stop_out_feature, stop_op_out_feature);
        compute_feature_distribution(stop_op_idx, cur_quantize_idx, total_quantize_op_cnt, 0, none_stop_out_feature, stop_op_out_feature);
        for(auto& new_iter:none_stop_out_feature)
        {
            MNN::Tensor*                    cur_tensor    = new_iter.first;
            TensorStatistic*                feature_info  = new_iter.second;
            tensor_scale[cur_tensor]                      = feature_info->finishAndCompute();
            std::string                     tensor_name   = feature_info->name();
            int                             tensor_idx    = fake_quantize_feature_name_idx_map_[tensor_name];
            int                             tensor_cnt    = (int)(fake_quantize_feature_name_idx_map_.size());
            std::pair<float, float> const&  data_range    = feature_info->range_per_channel(0);
            MNN_PRINT("(%d/%d), %s, coarse quantized, range (%.6f, %.6f), threshold %d, scale %.6f, max %.6f\n", 
                      tensor_idx, 
                      tensor_cnt,
                      tensor_name.c_str(), 
                      data_range.first,
                      data_range.second,
                      feature_info->threshold(),
                      tensor_scale[cur_tensor][0],
                      127.0f * tensor_scale[cur_tensor][0]);
        }
    }

    compute_feature_range(stop_op_idx, cur_quantize_idx, total_quantize_op_cnt, 1, none_stop_out_feature, stop_op_out_feature);
    compute_feature_distribution(stop_op_idx, cur_quantize_idx, total_quantize_op_cnt, 1, none_stop_out_feature, stop_op_out_feature);
    std::vector<QUANTIZE_RESULT>                            fine_quantize_res(stop_op_out_feature.size());
    std::vector<std::pair<MNN::Tensor*, TensorStatistic*> > quantize_feature_tensor(stop_op_out_feature.size());
    int                                                     feature_idx = 0;
    for(auto& new_iter:stop_op_out_feature)
    {
        MNN::Tensor*                    cur_tensor    = new_iter.first;
        TensorStatistic*                feature_info  = new_iter.second;
        tensor_scale[cur_tensor]                      = feature_info->finishAndCompute();
        std::string                     tensor_name   = feature_info->name();
        int                             tensor_idx    = fake_quantize_feature_name_idx_map_[tensor_name];
        int                             tensor_cnt    = (int)(fake_quantize_feature_name_idx_map_.size());
        std::pair<float, float> const&  data_range    = feature_info->range_per_channel(0);
        quantize_feature_tensor[feature_idx].first    = cur_tensor;
        quantize_feature_tensor[feature_idx].second   = feature_info;
        MNN_PRINT("(%d/%d), %s, coarse quantized, range (%.6f, %.6f), threshold %d, scale %.6f, max %.6f\n", 
                    tensor_idx, 
                    tensor_cnt,
                    tensor_name.c_str(), 
                    data_range.first,
                    data_range.second,
                    feature_info->threshold(),
                    tensor_scale[cur_tensor][0],
                    127.0f * tensor_scale[cur_tensor][0]);
        feature_idx ++;
    }
    if(QUANTIZE_REFINE == quantize_strategy_)
    {
        tuning_scale_from_candidate(stop_op_idx, cur_quantize_idx, total_quantize_op_cnt, quantize_feature_tensor, fine_quantize_res);
        feature_idx = 0;
        for(auto& quant_pair:quantize_feature_tensor)
        {
            MNN::Tensor*      quant_tensor = quant_pair.first;
            TensorStatistic*  feature_info = quant_pair.second;
            tensor_scale[quant_tensor][0]  = fine_quantize_res[feature_idx].scale;
            feature_info->set_refine_scale(tensor_scale[quant_tensor]);
            feature_idx ++;
        }
    }
    fwd_end      = std::chrono::system_clock::now();
    forward_cost = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
    MNN_PRINT("(%d/%d), %s, op total elapsed %.3f ms\r\n", 
               cur_quantize_idx, 
               total_quantize_op_cnt, 
               (std::get<0>(fake_quantize_session_op_name_[cur_quantize_idx])).c_str(), 
               (double)(forward_cost.count()) / 1000.0);
}

static int find_tuning_feature_idx(std::vector<std::pair<MNN::Tensor*, TensorStatistic*> > const& tuning_feature, MNN::Tensor* raw_tensor)
{
    for(int i = 0 ; i < (int)(tuning_feature.size()); i ++)
    {
        if(tuning_feature[i].first == raw_tensor)
            return i;
    }
    return 0;
}

void Calibration::tuning_scale_from_candidate(int                                                             op_idx,
                                              int                                                             cur_quantize_op_idx,
                                              int                                                             total_quantize_op_cnt,
                                              std::vector<std::pair<MNN::Tensor*, TensorStatistic*> > const&  quantize_feature_tensor,
                                              std::vector<QUANTIZE_RESULT>&                                   fine_quantize_info)
{
    int     img_cnt       = (int)(_imgaes.size());
    int     i             = 0;
    int     j             = 0;
    int     tensor_cnt    = (int)(quantize_feature_tensor.size());
    std::vector<std::vector<float> >  candidate_scale(tensor_cnt);
    std::vector<int>                  raw_candidate_idx(tensor_cnt);
    std::vector<std::vector<std::vector<float> > >  feature_cos(tensor_cnt);
    int                               tuning_feature_cnt = 0;

    for(i = 0; i < tensor_cnt ; i ++)
    {
        MNN::Tensor const*              cur_tensor    = quantize_feature_tensor[i].first;
        TensorStatistic*                feature_info  = quantize_feature_tensor[i].second;
        std::pair<float, float> const&  data_range    = feature_info->range_per_channel(0);
        int                             kl_threshold  = feature_info->threshold();
        fine_quantize_info[i].min_value    = data_range.first;
        fine_quantize_info[i].max_value    = data_range.second;
        fine_quantize_info[i].kl_threshold = kl_threshold;
        fine_quantize_info[i].scale        = feature_info->caculate_scale((float)kl_threshold, 0);
        int                             candidate_t_max = (int)(((float)(kl_threshold * 1.1)) + 0.5f);
        int                             candidate_t_min = (int)(((float)(kl_threshold * 0.9)) + 0.5f);
        if(candidate_t_max > 2047)
            candidate_t_max = 2047;
        if(candidate_t_min < 127)
            candidate_t_min = 127;
        float                           candidate_scale_max = feature_info->caculate_scale((float)candidate_t_max, 0);
        float                           candidate_scale_min = feature_info->caculate_scale((float)candidate_t_min, 0);
        float                           min_max_delta       = (candidate_scale_max - candidate_scale_min);
        static const int                kTuningFineIdxMax   = (2 * kTuningFineCnt);
        float                           raw_scale_idx       = kTuningFineIdxMax;
        for(j = 0 ; j < kTuningFineIdxMax ; j ++)
        {
            float cur_scale      = (((float)j)/((float)kTuningFineIdxMax)) * min_max_delta + candidate_scale_min;
            if(fine_quantize_info[i].scale <= cur_scale && kTuningFineIdxMax == raw_scale_idx)
            {
                candidate_scale[i].emplace_back(fine_quantize_info[i].scale);
                raw_scale_idx = j;
            }

            if(fine_quantize_info[i].scale != cur_scale)
                candidate_scale[i].emplace_back(cur_scale);
        }
        if(kTuningFineIdxMax == raw_scale_idx)
            candidate_scale[i].emplace_back(fine_quantize_info[i].scale);
        raw_candidate_idx[i]               = raw_scale_idx;
        feature_cos[i].resize(img_cnt);
        for(j = 0 ; j < (int)img_cnt ; j ++)
            feature_cos[i][j].resize(candidate_scale[i].size(), 0.0f);

        if(THRESHOLD_KL == feature_info->threshold_method())
            tuning_feature_cnt ++;
    }
    if(0 == tuning_feature_cnt)
        return;

    std::vector<MNN::Tensor*>         quantize_out_tensor(quantize_feature_tensor.size(), nullptr);
    std::vector<MNN::Tensor*>         origin_out_tensor(quantize_feature_tensor.size(), nullptr);
    int                               img_idx = 0;
    std::chrono::time_point<std::chrono::system_clock> fwd_start;
    std::chrono::time_point<std::chrono::system_clock> fwd_end;
    std::chrono::microseconds                          forward_cost;

    fwd_start = std::chrono::system_clock::now();
    for (const auto& img : _imgaes) {
        load_image(_process.get(), _width, _height, img, fake_quantize_input_tensor_, input_raw_);
        load_image(_process.get(), _width, _height, img, _inputTensorOrigin, input_raw_);

        int proc_op_cnt0 = 0;
        int proc_op_cnt1 = 0;
        MNN::TensorCallBackWithInfo quantize_before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                          const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end())
                {
                    if (fake_quantize_feature_info_[t]->visited() == false)
                        fake_quantize_feature_info_[t]->fakeQuantFeature(false, false, true);
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo quantize_after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                         const MNN::OperatorInfo* info) {
            if(proc_op_cnt0 == op_idx)
            {
                for(i = 0 ; i < (int)(nTensors.size()) ; i ++)
                    quantize_out_tensor[i] = nTensors[i];
                return false;
            }
            for (auto t : nTensors) {
                if(fake_quantize_feature_info_.find(t) != fake_quantize_feature_info_.end())
                {
                    if (fake_quantize_feature_info_[t]->visited() == false)
                        fake_quantize_feature_info_[t]->fakeQuantFeature(false, false, true);
                }
            }
            proc_op_cnt0 += 1;
            return true;
        };

        MNN::TensorCallBackWithInfo origin_before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                          const MNN::OperatorInfo* info) {
            return true;
        };
        MNN::TensorCallBackWithInfo origin_after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                         const MNN::OperatorInfo* info) {
            if(proc_op_cnt1 == op_idx)
            {
                for(i = 0 ; i < (int)(nTensors.size()) ; i ++)
                    origin_out_tensor[i] = nTensors[i];
                return false;
            }
            proc_op_cnt1 += 1;
            return true;
        };

        for (auto& iter : fake_quantize_feature_info_)
            iter.second->setVisited(false);
        
        fake_quantize_interpret_->runSessionWithCallBackInfo(fake_quantize_session_, quantize_before, quantize_after);
        _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, origin_before, origin_after);

        if(quantize_out_tensor.size() != origin_out_tensor.size())
        {
            MNN_PRINT("quantize_out_tensor.size() != origin_out_tensor.size(), %d != %d\n", 
                      (int)quantize_out_tensor.size(), (int)origin_out_tensor.size());
            exit(0);
        }

        std::vector<std::vector<float> >  candidate_data(quantize_out_tensor.size());
        for(i = 0 ; i < (int)(quantize_out_tensor.size()) ; i ++)
        {
            MNN::Tensor*      cur_tensor            = quantize_out_tensor[i];
            MNN::Tensor*      cur_std_tensor        = origin_out_tensor[i];
            int               tuning_feature_idx    = find_tuning_feature_idx(quantize_feature_tensor, cur_tensor);
            TensorStatistic*  cur_quant_feature     = quantize_feature_tensor[tuning_feature_idx].second;
            if(THRESHOLD_KL != cur_quant_feature->threshold_method())
                continue;
            int          data_ele_cnt          = cur_tensor->elementSize();
            candidate_data[i].resize(data_ele_cnt);
            for(j = 0 ; j < (int)(candidate_scale[tuning_feature_idx].size()); j ++)
            {
                float cur_scale = candidate_scale[tuning_feature_idx][j];
                memcpy(candidate_data[i].data(), cur_tensor->host<float>(), data_ele_cnt * sizeof(float));
                fake_quantize_feature_info_[cur_tensor]->fakeQuantData(candidate_data[i].data(), data_ele_cnt, cur_scale);
                float cur_cos   = TensorStatistic::compute_data_distance(cur_std_tensor->host<float>(), candidate_data[i].data(), data_ele_cnt);
                feature_cos[tuning_feature_idx][img_idx][j] = cur_cos;
            }
        }
        img_idx ++;
        float elapsed_ratio       = (float)img_idx * 100.0f / (float)_imageNum;
        MNN_PRINT("\r(%d/%d), %s, tuning %s scale: %.4lf %%", 
                    cur_quantize_op_idx, total_quantize_op_cnt, 
                    (std::get<0>(fake_quantize_session_op_name_[cur_quantize_op_idx])).c_str(), 
                    "output",
                    elapsed_ratio);
        fflush(stdout);
    }

    MNN_PRINT("\r\n");
    std::vector<std::vector<float> >  feature_cos_sum(tensor_cnt);
    for(i = 0 ; i < tensor_cnt ; i ++)
    {
        TensorStatistic*  cur_quant_feature = quantize_feature_tensor[i].second;
        std::string       tensor_name       = cur_quant_feature->name();
        if(THRESHOLD_KL != cur_quant_feature->threshold_method())
            continue;
        feature_cos_sum[i].resize(candidate_scale[i].size(), 0.0f);
        for(j = 0 ; j < (int)(candidate_scale[i].size()) ; j ++)
        {
            for(int k = 0 ; k < img_cnt ; k ++)
                feature_cos_sum[i][j] += feature_cos[i][k][j];
        }
        int    best_scale_idx = raw_candidate_idx[i];
        float  best_scale     = candidate_scale[i][best_scale_idx];
        float  best_cos       = feature_cos_sum[i][best_scale_idx];
        for(j = 0 ; j < (int)(feature_cos_sum[i].size()) ; j ++)
        {
            if(best_cos < feature_cos_sum[i][j])
            {
                best_cos       = feature_cos_sum[i][j];
                best_scale     = candidate_scale[i][j];
                best_scale_idx = j;
            }
        }

        float  kl_fine_threshold           = 127.0f * best_scale;
        float  max_abs                     = std::fmax(fabsf(fine_quantize_info[i].min_value), fabsf(fine_quantize_info[i].max_value));
        kl_fine_threshold                  = 2048.0f * kl_fine_threshold / max_abs;
        int    kl_thd_int                  = kl_fine_threshold + 0.5f;
        if(kl_thd_int < 127)
            kl_thd_int = 127;
        if(kl_thd_int > 2048)
            kl_thd_int = 2048;
        fine_quantize_info[i].kl_threshold = kl_thd_int;
        fine_quantize_info[i].scale        = best_scale;
        MNN_PRINT("(%d/%d), %s, tuning tensor %s, range (%.6f, %.6f), threshold %.6f, scale %.6f, max %.6f\r\n", 
               cur_quantize_op_idx, 
               total_quantize_op_cnt, 
               (std::get<0>(fake_quantize_session_op_name_[cur_quantize_op_idx])).c_str(), 
               tensor_name.c_str(), 
               fine_quantize_info[i].min_value,
               fine_quantize_info[i].max_value,
               kl_fine_threshold,
               best_scale,
               127.0f * best_scale);
    }
    fwd_end      = std::chrono::system_clock::now();
    forward_cost = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
    MNN_PRINT("(%d/%d), %s, op tuning elapsed %.3f ms\r\n", 
               cur_quantize_op_idx, 
               total_quantize_op_cnt, 
               (std::get<0>(fake_quantize_session_op_name_[cur_quantize_op_idx])).c_str(), 
               (double)(forward_cost.count()) / 1000.0);
}   

void Calibration::update_quantize_op_scale()
{
    int  op_idx = 0;
    for (const auto& op : _originaleModel->oplists) {
        op_idx ++;
        int cur_op_idx = op_idx - 1;
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise &&
            opType != MNN::OpType_Eltwise) {
            continue;
        }
        auto tensorsPair = _opInfo.find(op->name);
        if (tensorsPair == _opInfo.end()) {
            MNN_ERROR("Can't find tensors for %s\n", op->name.c_str());
        }

        if (opType == MNN::OpType_Eltwise) {
            auto param = op->main.AsEltwise();
            // Now only support AddInt8
            if (param->type != MNN::EltwiseType_SUM) {
                continue;
            }

            const auto&        inputScale0   = _scales[tensorsPair->second.first[0]];
            std::vector<float> inputScale1   = _scales[tensorsPair->second.first[1]];
            const auto&        outputScale   = _scales[tensorsPair->second.second[0]];
            const int          outputScaleSize = outputScale.size();
            std::vector<float> outputInvertScale(outputScaleSize);
            Helper::invertData(outputInvertScale.data(), outputScale.data(), outputScaleSize);
            op->type = MNN::OpType_EltwiseInt8;
            op->main.Reset();
            op->main.type = MNN::OpParameter_EltwiseInt8;

            auto eltwiseInt8Param         = new MNN::EltwiseInt8T;
            auto input0ScaleParam         = new MNN::QuantizedFloatParamT;
            auto input1ScaleParam         = new MNN::QuantizedFloatParamT;
            auto outputScaleParam         = new MNN::QuantizedFloatParamT;
            input0ScaleParam->tensorScale = inputScale0;
            input1ScaleParam->tensorScale = inputScale1;
            outputScaleParam->tensorScale = outputInvertScale;
            eltwiseInt8Param->inputQuan0  = std::unique_ptr<MNN::QuantizedFloatParamT>(input0ScaleParam);
            eltwiseInt8Param->inputQuan1  = std::unique_ptr<MNN::QuantizedFloatParamT>(input1ScaleParam);
            eltwiseInt8Param->outputQuan  = std::unique_ptr<MNN::QuantizedFloatParamT>(outputScaleParam);
            op->main.value                = eltwiseInt8Param;
            continue;
        }
        /*
        else if(opType == MNN::OpType_Pooling)
        {
            op->type = MNN::OpType_PoolInt8;
            continue;
        }
        */
        // below is Conv/DepthwiseConv
        const auto& inputScale  = _scales[tensorsPair->second.first[0]];
        const auto& outputScale = _scales[tensorsPair->second.second[0]];

        auto param                = op->main.AsConvolution2D();
        const int channles        = param->common->outputCount;
        const int weightSize      = param->weight.size();
        param->symmetricQuan.reset(new MNN::QuantizedFloatParamT);
        auto& quantizedParam = param->symmetricQuan;
        quantizedParam->scale.resize(channles);
        quantizedParam->weight.resize(weightSize);
        quantizedParam->bias.resize(channles);

        auto    cur_quant_op    = (fake_quantize_model_->oplists)[cur_op_idx].get();
        auto    quant_param     = cur_quant_op->main.AsConvolution2D();
        if (opType == MNN::OpType_Convolution) {
            QuantizeConvPerChannel(param->weight.data(), param->weight.size(), param->bias.data(),
                                   quantizedParam->weight.data(), quantizedParam->bias.data(),
                                   quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod, _weightClampValue);
            op->type = MNN::OpType_ConvInt8;
        } else if (opType == MNN::OpType_ConvolutionDepthwise) {
            QuantizeDepthwiseConv(param->weight.data(), param->weight.size(), param->bias.data(),
                                  quantizedParam->weight.data(), quantizedParam->bias.data(),
                                  quantizedParam->scale.data(), inputScale, outputScale, _weightQuantizeMethod, _weightClampValue);
            op->type = MNN::OpType_DepthwiseConvInt8;
        }
        if (param->common->relu6) {
            param->common->relu  = true;
            param->common->relu6 = false;
        }

        param->weight.clear();
        param->bias.clear();
    }
}

void Calibration::_insertDequantize(MNN::NetT*                                         model,
                                    std::map<int, const MNN::Tensor*>&                 idx_tensor_map, 
                                    std::map<const MNN::Tensor*, std::vector<float>>&  tensor_scale_map) 
{
    // Search All Int Tensors
    std::set<int> int8Tensors;
    std::set<int> int8Outputs;
    for (auto& op : model->oplists) {
        if (Helper::INT8SUPPORTED_OPS.count(op->type) > 0) {
            for (auto index : op->inputIndexes) {
                int8Tensors.insert(index);
            }
            for (auto index : op->outputIndexes) {
                int8Tensors.insert(index);
                int8Outputs.insert(index);
            }
        }
    }
    for (auto& op : model->oplists) {
        for (auto index : op->inputIndexes) {
            auto iter = int8Outputs.find(index);
            if (iter != int8Outputs.end()) {
                int8Outputs.erase(iter);
            }
        }
    }

    // Insert Convert For Not Support Int8 Ops
    for (auto iter = model->oplists.begin(); iter != model->oplists.end();) {
        auto op           = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;
        // check whether is output op
        // if Yes, insert dequantization op after this op
        if (Helper::INT8SUPPORTED_OPS.find(opType) != Helper::INT8SUPPORTED_OPS.end()) {
            // this is quantized op
            iter++;
            continue;
        }

        auto& inputIndexes  = op->inputIndexes;
        const int inputSize = inputIndexes.size();

        // insert dequantization op before this op
        for (int i = 0; i < inputSize; ++i) {
            const auto curInputIndex = inputIndexes[i];
            if (int8Tensors.find(curInputIndex) == int8Tensors.end()) {
                continue;
            }
            auto input        = idx_tensor_map[curInputIndex];
            auto inputOpScale = tensor_scale_map[input];

            // construct new op
            auto dequantizationOp       = new MNN::OpT;
            dequantizationOp->main.type = MNN::OpParameter_QuantizedFloatParam;
            dequantizationOp->name      = "___Int8ToFloat___For_" + name + flatbuffers::NumToString(i);

            dequantizationOp->type           = MNN::OpType_Int8ToFloat;
            auto dequantizationParam         = new MNN::QuantizedFloatParamT;
            dequantizationOp->main.value     = dequantizationParam;
            dequantizationParam->tensorScale = inputOpScale;

            dequantizationOp->inputIndexes.push_back(curInputIndex);
            dequantizationOp->outputIndexes.push_back(model->tensorName.size());
            model->tensorName.push_back(dequantizationOp->name);

            // reset current op's input index at i
            inputIndexes[i] = dequantizationOp->outputIndexes[0];

            iter = model->oplists.insert(iter, std::unique_ptr<MNN::OpT>(dequantizationOp));
            iter++;
        }

        iter++;
        // LOG(INFO) << "insert quantization op after this op if neccessary";
        // insert quantization op after this op if neccessary
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            const auto outputIndex = op->outputIndexes[i];
            if (int8Tensors.find(outputIndex) == int8Tensors.end()) {
                continue;
            }
            auto output   = idx_tensor_map[outputIndex];
            auto curScale = tensor_scale_map[output];
            // construct one quantization op(FloatToInt8)
            auto quantizationOp        = new MNN::OpT;
            quantizationOp->main.type  = MNN::OpParameter_QuantizedFloatParam;
            quantizationOp->name       = name + "___FloatToInt8___" + flatbuffers::NumToString(i);
            quantizationOp->type       = MNN::OpType_FloatToInt8;
            auto quantizationParam     = new MNN::QuantizedFloatParamT;
            quantizationOp->main.value = quantizationParam;

            const int channels = curScale.size();
            std::vector<float> quantizationScale(channels);
            Helper::invertData(quantizationScale.data(), curScale.data(), channels);
            quantizationParam->tensorScale = quantizationScale;

            quantizationOp->inputIndexes.push_back(model->tensorName.size());
            quantizationOp->outputIndexes.push_back(outputIndex);
            model->tensorName.push_back(model->tensorName[outputIndex]);
            model->tensorName[outputIndex] = quantizationOp->name;
            op->outputIndexes[i]           = quantizationOp->inputIndexes[0];

            iter = model->oplists.insert(iter, std::unique_ptr<MNN::OpT>(quantizationOp));
            iter++;
        }
    }

    // Insert Turn float Op for output
    for (auto index : int8Outputs) {
        // construct new op
        auto dequantizationOp       = new MNN::OpT;
        dequantizationOp->main.type = MNN::OpParameter_QuantizedFloatParam;
        dequantizationOp->name      = "___Int8ToFloat___For_" + flatbuffers::NumToString(index);

        dequantizationOp->type           = MNN::OpType_Int8ToFloat;
        auto dequantizationParam         = new MNN::QuantizedFloatParamT;
        dequantizationOp->main.value     = dequantizationParam;
        dequantizationParam->tensorScale = tensor_scale_map[idx_tensor_map[index]];

        dequantizationOp->inputIndexes.push_back(index);
        dequantizationOp->outputIndexes.push_back(model->tensorName.size());
        auto originTensorName    = model->tensorName[index];
        model->tensorName[index] = dequantizationOp->name;
        model->tensorName.emplace_back(originTensorName);

        model->oplists.insert(model->oplists.end(), std::unique_ptr<MNN::OpT>(dequantizationOp));
    }


    // check PoolInt8, Int8ToFloat's scale should be equal to PoolInt8's input_scale
    for (auto iter = model->oplists.begin(); iter != model->oplists.end();iter ++) 
    {
        auto       op      = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;
        // check whether is output op
        // if Yes, insert dequantization op after this op
        if (MNN::OpType_PoolInt8 == opType) 
        {
            int       in_idx      = op->inputIndexes[0];
            const int output_size = op->outputIndexes.size();
            for (auto out_idx : op->outputIndexes) 
            {
                for(auto inner_iter = model->oplists.begin(); inner_iter != model->oplists.end(); inner_iter ++)
                {
                    auto                         next_op     = inner_iter->get();
                    const std::vector<int32_t>&  input_idxes = next_op->inputIndexes;
                    bool                         is_next     = false;
                    for(int j = 0 ; j < (int)(input_idxes.size()); j ++)
                    {
                        if(out_idx == input_idxes[j])
                        {
                            is_next = true;
                            break;
                        }
                    }
                    const auto                   next_opType = next_op->type;
                    if(next_opType == MNN::OpType_Int8ToFloat && true == is_next)
                    {
                        auto                       pool_int8_input       = idx_tensor_map[in_idx];
                        auto                       pool_int8_in_scale    = tensor_scale_map[pool_int8_input];
                        MNN::QuantizedFloatParamT* param                 = next_op->main.AsQuantizedFloatParam();
                        param->tensorScale                              = pool_int8_in_scale;
                    }
                }
            }
        }
    }
}

void Calibration::_fake_quant_weights() {
    auto findAbsMax = [&] (const float* weights, const int size) {
        float absMax = 0;
        for (int i = 0; i < size; i++) {
            if (std::fabs(weights[i]) > absMax) {
                absMax = std::fabs(weights[i]);
            }
        }

        return absMax;
    };

    for (const auto& op : _originaleModel->oplists) {
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), op->name);
        if (iter != _skip_quant_ops.end()) {
            continue;
        }

        const auto opType = op->type;
        if (opType != MNN::OpType_Convolution && opType != MNN::OpType_ConvolutionDepthwise) {
            continue;
        }

        auto param = op->main.AsConvolution2D();
        const int kernelNum = param->common->outputCount;
        std::vector<float> weights = param->weight;
        const int weightSize = weights.size();
        const int kernelSize = weightSize / kernelNum;

        for (int i = 0; i < kernelNum; i++) {
            const int offset = i * kernelSize;
            float absMax = findAbsMax(weights.data() + offset, kernelSize);
            float scale = absMax / _weightClampValue;
            if (absMax < 1e-6f) {
                scale = absMax;
            }

            for (int j = 0; j < kernelSize; j++) {
                float value = weights[offset + j];
                float quantValue = std::round(value / scale);
                float clampedValue = std::max(std::min(quantValue, _weightClampValue), -_weightClampValue);
                float dequantValue = scale * clampedValue;
                param->weight[offset + j] = dequantValue;
            }
        }
    }
}

void Calibration::_computeQuantError()
{
    int count = 0;
    std::map<std::string, std::vector<float>> overflowRatiosMap;
    std::map<std::string, std::vector<float>> tensorCosDistanceMap;

    std::vector<int> inputShape = {1, _inputTensorDims[1], _inputTensorDims[2], _inputTensorDims[3]};
    _interpreter->resizeTensor(_inputTensor, inputShape);
    _interpreter->resizeSession(_session);

    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset   = MNN::Net::Pack(builder, fake_quantize_model_.get());
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto buffer   = builder.GetBufferPointer();

    MNN::ScheduleConfig config;
    fake_quantize_interpret_.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    fake_quantize_session_      = fake_quantize_interpret_->createSession(config);
    fake_quantize_input_tensor_ = fake_quantize_interpret_->getSessionInput(fake_quantize_session_, NULL);
    fake_quantize_interpret_->resizeTensor(fake_quantize_input_tensor_, inputShape);
    fake_quantize_interpret_->resizeSession(fake_quantize_session_);


    MNN::TensorCallBackWithInfo init_before = [&](const std::vector<MNN::Tensor*>& nTensors, const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return false;
        }
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoQuant.find(t) == _featureInfoQuant.end()) {
                    _featureInfoQuant[t] = std::shared_ptr<TensorStatistic>(
                        new TensorStatistic(t, _featureQuantizeMethod, opName + " input_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                    _featureInfoQuant[t]->set_min_threshold(min_quantize_threshold_);
                }
                i++;
            }
        }
        return false;
    };
    MNN::TensorCallBackWithInfo init_after = [this](const std::vector<MNN::Tensor*>& nTensors,
                                               const MNN::OperatorInfo* info) {
        std::string opName = info->name();
        std::vector<std::string>::iterator iter = std::find(_skip_quant_ops.begin(), _skip_quant_ops.end(), opName);
        if (iter != _skip_quant_ops.end()) {
            return true;
        }
        if (Helper::gNeedFeatureOp.find(info->type()) != Helper::gNeedFeatureOp.end()) {
            int i = 0;
            for (auto t : nTensors) {
                if (_featureInfoQuant.find(t) == _featureInfoQuant.end()) {
                    _featureInfoQuant[t] =
                        std::shared_ptr<TensorStatistic>(new TensorStatistic(t, _featureQuantizeMethod, opName + " output_tensor_" + flatbuffers::NumToString(i), _featureClampValue));
                    _featureInfoQuant[t]->set_min_threshold(min_quantize_threshold_);
                }
                i++;
            }
        }
        return true;
    };

    fake_quantize_interpret_->runSessionWithCallBackInfo(fake_quantize_session_, init_before, init_after);
    {
        for(auto& cur_feature: _featureInfoQuant)
        {
            std::shared_ptr<TensorStatistic>& cur_tensor_statistic = cur_feature.second;
            std::string  cur_feature_name = cur_tensor_statistic->name();
            for(auto& std_feature: _featureInfo)
            {
                std::shared_ptr<TensorStatistic> const& std_tensor_statistic = std_feature.second;
                std::string  std_feature_name = std_tensor_statistic->name();
                if(cur_feature_name == std_feature_name)
                {
                    cur_tensor_statistic->copy_param_from(*(std_tensor_statistic.get()));
                    break;
                }
                    
            }
        }
    }

    std::vector<std::string> feature_name_vec;
    feature_name_vec.resize(_featureInfo.size());
    int  feature_name_idx = 0;
    for (const auto& img : _imgaes) {
        count++;
        load_image(_process.get(), _width, _height, img, fake_quantize_input_tensor_, input_raw_);

        std::map<std::string, std::vector<float>> fakeQuantedFeatures;

        MNN::TensorCallBackWithInfo before = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoQuant.find(t) != _featureInfoQuant.end()) {
                    if (_featureInfoQuant[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio  = _featureInfoQuant[t]->fakeQuantFeature();
                        std::string  feature_name            = _featureInfoQuant[t]->name();
                        fakeQuantedFeatures[feature_name]    = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[feature_name].emplace_back(dequantFeatureAndOverflowRatio.second);
                        if(feature_name_idx < (int)(feature_name_vec.size()))
                        {
                            feature_name_vec[feature_name_idx]   = _featureInfoQuant[t]->name();
                            feature_name_idx ++;
                        }
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo after = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoQuant.find(t) != _featureInfoQuant.end()) {
                    if (_featureInfoQuant[t]->visited() == false) {
                        auto dequantFeatureAndOverflowRatio   = _featureInfoQuant[t]->fakeQuantFeature();
                        std::string  feature_name             = _featureInfoQuant[t]->name();
                        fakeQuantedFeatures[feature_name]     = dequantFeatureAndOverflowRatio.first;
                        overflowRatiosMap[feature_name].emplace_back(dequantFeatureAndOverflowRatio.second);
                        if(feature_name_idx < (int)(feature_name_vec.size()))
                        {
                            feature_name_vec[feature_name_idx]   = _featureInfoQuant[t]->name();
                            feature_name_idx ++;
                        }
                    }
                }
            }
            return true;
        };

        for (auto& iter : _featureInfoQuant) 
            iter.second->setVisited(false);

        fake_quantize_interpret_->runSessionWithCallBackInfo(fake_quantize_session_, before, after);

        load_image(_process.get(), _width, _height, img, _inputTensorOrigin, input_raw_);

        MNN::TensorCallBackWithInfo beforeOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                 const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto name = _featureInfoOrigin[t]->name();
                        float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                        tensorCosDistanceMap[name].emplace_back(cosDis);
                    }
                }
            }
            return true;
        };
        MNN::TensorCallBackWithInfo afterOrigin = [&](const std::vector<MNN::Tensor*>& nTensors,
                                                const MNN::OperatorInfo* info) {
            for (auto t : nTensors) {
                if (_featureInfoOrigin.find(t) != _featureInfoOrigin.end()) {
                    if (_featureInfoOrigin[t]->visited() == false) {
                        auto name = _featureInfoOrigin[t]->name();
                        if("fc1 output_tensor_0" == name)
                        {
                            MNN_PRINT("\n%s\n", img.c_str());
                            std::vector<float> const&  feature_val_quant = fakeQuantedFeatures[name];
                            int                        feature_cnt       = (int)(feature_val_quant.size());
                            for(int i = 0 ; i < feature_cnt ; i ++)
                                MNN_PRINT("%.6f, ", feature_val_quant[i]);
                            MNN_PRINT("\n");
                            float*                     data              = t->host<float>();
                            for(int i = 0 ; i < feature_cnt ; i ++)
                                MNN_PRINT("%.6f, ", data[i]);
                            MNN_PRINT("\n");
                        }
                        float cosDis = _featureInfoOrigin[t]->computeDistance(fakeQuantedFeatures[name]);
                        tensorCosDistanceMap[name].emplace_back(cosDis);
                    }
                }
            }
            return true;
        };

        for (auto& iter : _featureInfoOrigin)
            iter.second->setVisited(false);

        _interpreterOrigin->runSessionWithCallBackInfo(_sessionOrigin, beforeOrigin, afterOrigin);

        MNN_PRINT("\r2nd, computeDistance: %.2lf %%", (float)count * 100.0f / (float)_imageNum);
        fflush(stdout);
    }
    MNN_PRINT("\n\nDebug info:\n\n");

    for(int n = 0 ; n < (int)(feature_name_vec.size()) ; n ++)
    {
        std::string const& name = feature_name_vec[n];
        std::map<std::string, std::vector<float>>::const_iterator iter = tensorCosDistanceMap.find(name);
        float sumCos = 0.0f, sumOverflow = 0.0f;
        for (int i = 0; i < iter->second.size(); i++) {
            sumCos      += iter->second[i];
            sumOverflow += overflowRatiosMap[name][i];
        }
        float avgCosDistance   = sumCos / _imgaes.size();
        float avgOverflowRatio = sumOverflow / _imgaes.size();

        MNN_PRINT("%s:  2nd, cos distance: %f, overflow ratio: %f\n", name.c_str(), avgCosDistance, avgOverflowRatio);
    }
}

void Calibration::runQuantizeModel() {
    std::chrono::time_point<std::chrono::system_clock> fwd_start;
    std::chrono::time_point<std::chrono::system_clock> fwd_end;
    std::chrono::microseconds                          forward_cost;
    fwd_start = std::chrono::system_clock::now();
    if(QUANTIZE_NORMAL == quantize_strategy_)
    {
        if (_featureQuantizeMethod == "KL") {
            _computeFeatureScaleKL();
        } else if (_featureQuantizeMethod == "ADMM") {
            _computeFeatureScaleADMM();
        }
        _updateScale();
        /*
        if (_debug)
            _computeQuantError();
        */
    }
    else
    {
        compute_conv_weight_scale();
        compute_feature_scale();
        update_quantize_op_scale();
    }

    _insertDequantize(_originaleModel, _tensorMap, _scales);
    fwd_end = std::chrono::system_clock::now();
    forward_cost = std::chrono::duration_cast<std::chrono::microseconds>(fwd_end - fwd_start);
    MNN_PRINT("Quantize elapsed %.3f ms\r\n", (double)(forward_cost.count()) / 1000.0);
}

void Calibration::validate_quantize_model(const unsigned char* quantized_model, int buf_size)
{
    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Interpreter>  valid_interpreter;
    valid_interpreter.reset(MNN::Interpreter::createFromBuffer(quantized_model, buf_size));
    MNN::Session*                   valid_session      = valid_interpreter->createSession(config);
    MNN::Tensor*                    valid_input_tensor = valid_interpreter->getSessionInput(valid_session, NULL);
    std::vector<int>                input_shape        = {1, _inputTensorDims[1], _inputTensorDims[2], _inputTensorDims[3]};
    valid_interpreter->resizeTensor(valid_input_tensor, input_shape);
    valid_interpreter->resizeSession(valid_session);

    const std::map<std::string, MNN::Tensor*>&  valid_output_tensor  = valid_interpreter->getSessionOutputAll(valid_session);
    const std::map<std::string, MNN::Tensor*>&  origin_output_tensor = _interpreterOrigin->getSessionOutputAll(_sessionOrigin);

    for (const auto& img : _imgaes) {
        load_image(_process.get(), _width, _height, img, valid_input_tensor, input_raw_);
        valid_interpreter->runSession(valid_session);

        load_image(_process.get(), _width, _height, img, _inputTensorOrigin, input_raw_);
        _interpreterOrigin->runSession(_sessionOrigin);

        std::vector<std::vector<float> >  valid_tensor_value(valid_output_tensor.size());
        std::vector<std::vector<float> >  origin_tensor_value(origin_output_tensor.size());

        int  tensor_idx = 0;
        MNN_PRINT("%s:\n", img.c_str());
        for(auto& iter : valid_output_tensor)
        {
            MNN::Tensor*  tensor   = iter.second;
            float*        data     = tensor->host<float>();
            int           data_cnt = tensor->elementSize();
            valid_tensor_value[tensor_idx].resize(data_cnt);
            memcpy(valid_tensor_value[tensor_idx].data(), data, data_cnt * sizeof(float));

            if(data_cnt < 16)
            {
                int data_max = std::min(data_cnt, 8);
                for(int n = 0 ; n < data_max ; n ++)
                    MNN_PRINT("%.6f, ", data[n]);
                MNN_PRINT("\n");
            }
            else if(data_cnt < 32)
            {
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[n]);
                MNN_PRINT("\n");
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[data_cnt - 8 + n]);
                MNN_PRINT("\n");
            }
            else
            {
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[n]);
                MNN_PRINT("\n");
                int start_pos = (data_cnt / 2);
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[start_pos + n]);
                MNN_PRINT("\n");
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[data_cnt - 8 + n]);
                MNN_PRINT("\n");
            }

            tensor_idx ++;
        }
        tensor_idx = 0;
        for(auto& iter : origin_output_tensor)
        {
            MNN::Tensor*  tensor   = iter.second;
            float*        data     = tensor->host<float>();
            int           data_cnt = tensor->elementSize();
            origin_tensor_value[tensor_idx].resize(data_cnt);
            memcpy(origin_tensor_value[tensor_idx].data(), data, data_cnt * sizeof(float));

            if(data_cnt < 16)
            {
                int data_max = std::min(data_cnt, 8);
                for(int n = 0 ; n < data_max ; n ++)
                    MNN_PRINT("%.6f, ", data[n]);
                MNN_PRINT("\n");
            }
            else if(data_cnt < 32)
            {
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[n]);
                MNN_PRINT("\n");
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[data_cnt - 8 + n]);
                MNN_PRINT("\n");
            }
            else
            {
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[n]);
                MNN_PRINT("\n");
                int start_pos = (data_cnt / 2);
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[start_pos + n]);
                MNN_PRINT("\n");
                for(int n = 0 ; n < 8 ; n ++)
                    MNN_PRINT("%.6f, ", data[data_cnt - 8 + n]);
                MNN_PRINT("\n");
            }
            tensor_idx ++;
        }

        float  cos_dis_sum = 0.0f;
        if(valid_tensor_value.size() == origin_tensor_value.size())
        {
            for(int n = 0 ; n < (int)(valid_tensor_value.size()) ; n ++)
            {
                if(valid_tensor_value[n].size() == origin_tensor_value[n].size())
                {
                    float axbSum = 0.0f;
                    float a2Sum  = 0.0f;
                    float b2Sum  = 0.0f;
                    for(int m = 0 ; m < (int)(valid_tensor_value[n].size()) ; m ++)
                    {
                        axbSum += (origin_tensor_value[n][m] * valid_tensor_value[n][m]);
                        a2Sum  += (origin_tensor_value[n][m] * origin_tensor_value[n][m]);
                        b2Sum  += (valid_tensor_value[n][m] * valid_tensor_value[n][m]);
                    }
                    a2Sum = std::sqrt(a2Sum);
                    b2Sum = std::sqrt(b2Sum);
                    float cosDis = axbSum / (a2Sum * b2Sum);
                    cos_dis_sum += cosDis;
                }
            }
            cos_dis_sum = cos_dis_sum / (float)(valid_tensor_value.size());
            MNN_PRINT("cos_dis: %.6f\n", cos_dis_sum);
        }
    }
}

std::vector<std::string> Calibration::select_rand_image(std::vector<std::string> const& src, int max_num)
{
    int src_num  = (int)(src.size());
    if(max_num >= src_num || src_num <= 0)
        return src;

    std::vector<int> hash_tab(src_num, 0);
    std::vector<int> out_idx(src_num, 0);
    std::vector<int> sel_idx(max_num, 0);
    rand_uniq_array(src_num, hash_tab.data(), out_idx.data());
    memcpy(sel_idx.data(), out_idx.data(), max_num * sizeof(int));
    std::sort(sel_idx.begin(), sel_idx.begin() + max_num, std::less<int>());

    std::vector<std::string>  res(max_num);
    for(int i = 0 ; i < max_num ; i ++)
        res[i] = src[sel_idx[i]];
    return res;
}

int Calibration::rand_uniq_array(int iCnt, int* pHashTableArray, int* pOutIdxArray)
{
    int i          = 0 ;
    int iIdx       = 0 ;
    int iRemainCnt = iCnt ;
    int iLastValue = 0 ;
    int iTemp      = 0 ;

    if (iCnt <= 0)
        return 0 ;

    srand(iCnt);

    if (1 == iCnt)
        pOutIdxArray[0] = (rand() % iCnt) ;

    for (i = 0; i < iCnt ; i++)
        pHashTableArray[i] = i ;

    for (i = 0; i < iCnt ; i++)
    {
        if (iRemainCnt > 1)
        {
            iIdx                             = rand() % (iRemainCnt - 1);
            iLastValue                       = pHashTableArray[iRemainCnt - 1] ;
            iTemp                            = pHashTableArray[iIdx] ;
            pHashTableArray[iIdx]            = iLastValue ;
            pHashTableArray[iRemainCnt - 1]  = iTemp ;
            pOutIdxArray[i]                  = iTemp ;
        }
        else
            pOutIdxArray[i] = pHashTableArray[0] ;

        iRemainCnt-- ;
    }

    return 0 ;
}

void Calibration::dumpTensorScales(const std::string& modelFile) {
    rapidjson::StringBuffer sb;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);

    writer.StartArray();

    for (auto iter = _originaleModel->oplists.begin(); iter != _originaleModel->oplists.end(); iter++) {
        auto op           = iter->get();
        const auto opType = op->type;
        const auto name   = op->name;
        
        if (opType == MNN::OpType_Raster) {
            continue;
        }

        writer.StartObject();

        writer.Key("name");
        writer.String(rapidjson::StringRef(name.c_str(), name.size()));

        auto& inputIndexes  = op->inputIndexes;
        const int inputSize = inputIndexes.size();

        if (inputSize > 0) {
            writer.Key("inputs");
            writer.StartArray();
            for (int i = 0; i < inputSize; ++i) {
                const auto curInputIndex = inputIndexes[i];
                
                auto input        = _tensorMap[curInputIndex];
                auto inputOpScale = _scales[input];
                
                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curInputIndex);

                writer.Key("scales");
                writer.StartArray();
                for(auto scale : inputOpScale) {
                    writer.Double(scale);
                }
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }
 
        auto& outputIndexes  = op->outputIndexes;
        const int outputSize = outputIndexes.size();

        if (outputSize > 0) {
            writer.Key("outputs");
            writer.StartArray();
            for (int i = 0; i < outputSize; ++i) {
                const auto curOutputIndex = outputIndexes[i];
                
                auto output        = _tensorMap[curOutputIndex];
                auto outputOpScale = _scales[output];
                
                writer.StartObject();
                writer.Key("tensorIndex");
                writer.Int(curOutputIndex);

                writer.Key("scales");
                writer.StartArray();
                for(auto scale : outputOpScale) {
                    writer.Double(scale);
                }
                writer.EndArray();

                writer.EndObject();
            }
            writer.EndArray();
        }

        writer.EndObject();
    }
    writer.EndArray();

    std::string scaleFile = modelFile + ".json";
    std::ofstream os(scaleFile);
    if (os.is_open()) {
        os << sb.GetString() << std::endl;
        os.close();
    } else {
        std::cerr << "open scale file " << scaleFile << " fail. error code:" << os.failbit << std::endl;
    }
}

void Calibration::load_image(MNN::CV::ImageProcess* pretreat, 
                             int                    targetWidth, 
                             int                    targetHeight,
                             const std::string&     inputImageFileName, 
                             MNN::Tensor*           input,
                             bool                   input_raw)
{
    if(false == input_raw)
        Helper::preprocessInput(pretreat, targetWidth, targetHeight, inputImageFileName, input);
    else
    {
        FILE*  img_file = fopen(inputImageFileName.c_str(), "rb");
        if(nullptr != img_file)
        {
            fseek(img_file, 0, SEEK_END);
            int img_file_size = (int)(ftell(img_file));
            int img_data_cnt  = (img_file_size >> 2);
            fseek(img_file, 0, SEEK_SET);

            float*  tensor_buf_ptr = input->host<float>();

            MNN::Tensor::InsideDescribe const* desc = MNN::TensorUtils::getDescribe(input);
            if (MNN::MNN_DATA_FORMAT_NC4HW4 == desc->dimensionFormat)
            {
                int  c       = input->channel();
                int  h       = input->height();
                int  w       = input->width();
                bool img_err = false;
                if(1 == c)
                {
                    if(img_file_size != (h*w) << 2)
                    {
                        MNN_PRINT("load image size error, %s, %d\n", inputImageFileName.c_str(), img_file_size);
                        img_err = true;
                    }
                        
                }
                else if(3 == c)
                {
                    if(img_file_size != 3 * ((h*w) << 2))
                    {
                        MNN_PRINT("load image size error, %s, %d\n", inputImageFileName.c_str(), img_file_size);
                        img_err = true;
                    }
                }
                else
                {
                    MNN_PRINT("input tensor channel error, %s, %d\n", inputImageFileName.c_str(), c);
                    img_err = true;
                }
                if(false == img_err)
                {
                    float*  raw_data = new float[img_data_cnt];
                    size_t  read_len = fread(raw_data, img_data_cnt * sizeof(float), 1, img_file);
                    (void)read_len;
                    nchw_to_nc4hw4(raw_data, c, h, w, tensor_buf_ptr);
                    delete raw_data;
                }
            }
            else
            {
                if(img_data_cnt != input->elementSize())
                    MNN_PRINT("load image error, %s\n", inputImageFileName.c_str());
                else
                {
                    size_t  read_len = fread(tensor_buf_ptr, img_file_size, 1, img_file);
                    (void)read_len;
                }
            }
            fflush(img_file);
            fclose(img_file);
        }
    }
}

void Calibration::nchw_to_nc4hw4(float const* src, int c, int h, int w, float* dst)
{
    if(1 == c)
        nchw_to_nc4hw4_channel_1(src, h, w, dst);
    else if(3 == c)
        nchw_to_nc4hw4_channel_3(src, h, w, dst);
}

void Calibration::nchw_to_nc4hw4_channel_1(float const* src, int h, int w, float* dst)
{
    memset(dst, 0, 4 * h * w * sizeof(float));
    int line_size_dst = 4 * w;
    for(int i = 0 ; i < h ; i ++)
    {
        for(int j = 0; j < w ; j ++)
            dst[i * line_size_dst + (j << 2)] = src[i * w + j];
    }
}

void Calibration::nchw_to_nc4hw4_channel_3(float const* src, int h, int w, float* dst)
{
    memset(dst, 0, 4 * h * w * sizeof(float));
    int line_size_src = 3 * w;
    int line_size_dst = 4 * w;
    for(int i = 0 ; i < h ; i ++)
    {
        for(int j = 0; j < w ; j ++)
        {
            dst[i * line_size_dst + (j << 2)]     = src[i * line_size_src + 3 * j];
            dst[i * line_size_dst + (j << 2) + 1] = src[i * line_size_src + 3 * j + 1];
            dst[i * line_size_dst + (j << 2) + 2] = src[i * line_size_src + 3 * j + 2];
        }  
    }
}