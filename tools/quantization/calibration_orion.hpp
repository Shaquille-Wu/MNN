//
//  calibration.hpp
//  MNN
//
//  Created by MNN on 2019/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CALIBRATION_HPP
#define CALIBRATION_HPP

#include <map>

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include "TensorStatistic.hpp"
#include "MNN_generated.h"

// Calibration find the optimal threshold according to KL-divergence
// process: the below process is applied on the whole Conv|DepthwiseConv layers
// 1. run the model on the batch samples, update the max(abs(feature_maps)) when the op is Convolution|Depthwise
// 2. cut the max(abs(feature_maps)) into 2048 slices
// 3. run the model on the batch samples again, update the distribution of feature maps every Conv|DepthwiseConv layer
// 4. apply Calibration on every distribution to get the optimal thereshold
// 5. compute the (input_scale * weight_scale) / output_scale, update the scale of symmetricQuan in Convolution Paramter
class Calibration {
public:
    Calibration(MNN::NetT* model, const uint8_t* modelBuffer, const int bufferSize, const std::string& configPath);

    void runQuantizeModel();
    
    void dumpTensorScales(const std::string& modelFile);
//Shaquille, Added 20210202 Start
    void validate_quantize_model(const unsigned char* quantized_model, int buf_size);
//Shaquille, Added 20210202 End
    typedef enum tag_quantize_strategy
    {
        QUANTIZE_NORMAL = 0,
        QUANTIZE_LAYER_BY_LAYER,
        QUANTIZE_REFINE,
        QUANTIZE_SUM
    }QUANTIZE_STRATEGY;

private:
    Calibration();
    MNN::NetT* _originaleModel;
    std::shared_ptr<MNN::CV::ImageProcess> _process;
    const int _binNums = 2048;
    int _imageNum      = 0;
    int _width;
    int _height;
    std::vector<std::string> _imgaes;

    // Tensor and Info
    std::map<const MNN::Tensor*, std::shared_ptr<TensorStatistic>> _featureInfo;
    std::map<const MNN::Tensor*, std::shared_ptr<TensorStatistic>> _featureInfoOrigin;
    std::map<const MNN::Tensor*, std::shared_ptr<TensorStatistic>> _featureInfoQuant;

    std::map<int, const MNN::Tensor*> _tensorMap;

    using OpToTensorMap = std::map<std::string, std::pair<std::vector<MNN::Tensor*>, std::vector<MNN::Tensor*>>>;
    // Op's name, Inputs, Outputs
    OpToTensorMap                                    _opInfo;

    // The scale results
    std::map<const MNN::Tensor*, std::vector<float>> _scales;

    std::shared_ptr<MNN::Interpreter> _interpreter;
    // keep mnn forward information
    MNN::Session* _session;
    MNN::Tensor* _inputTensor;
    std::vector<int> _inputTensorDims;

    std::shared_ptr<MNN::Interpreter> _interpreterOrigin;
    MNN::Session* _sessionOrigin;
    MNN::Tensor* _inputTensorOrigin;

    std::unique_ptr<MNN::NetT>                                  fake_quantize_model_;
    std::shared_ptr<MNN::Interpreter>                           fake_quantize_interpret_;
    MNN::Session*                                               fake_quantize_session_      = nullptr;
    MNN::Tensor*                                                fake_quantize_input_tensor_ = nullptr;
    std::vector<std::vector<float> >                            fake_quantize_weight_scale_;
    std::vector<int>                                            fake_quantize_conv_op_idx_;
    std::vector<std::string>                                    fake_quantize_feature_name_;
    std::map<std::string, int>                                  fake_quantize_feature_name_idx_map_;
    std::vector<std::tuple<std::string, int, MNN::OpType>>      fake_quantize_session_op_name_;
    std::map<MNN::Tensor*, std::shared_ptr<TensorStatistic>>    fake_quantize_feature_info_;

    std::string _featureQuantizeMethod = "KL";
    std::string _weightQuantizeMethod  = "MAX_ABS";

    float _featureClampValue = 127.0f;
    float _weightClampValue  = 127.0f;
    std::vector<std::string> _skip_quant_ops;
    bool _debug = false;
    bool  input_raw_              = false;
    int   min_quantize_threshold_ = 768;
    int   quantize_strategy_      = Calibration::QUANTIZE_NORMAL;

    void _initMNNSession(const uint8_t* modelBuffer, const int bufferSize, const int channels);
    void _initMaps();

    void _computeFeatureMapsRange();
    void _collectFeatureMapsDistribution();
    void _computeFeatureScaleKL();
    void _computeFeatureScaleADMM();
    void _computeFeatureScaleMoving();
    void _updateScale();
    void _fake_quant_weights();
    void _computeQuantError();
    // insert the dequantization op before the not supported op(int8), and insert dequantization op
    // after the output op, so that get original float data conveniently
    static void _insertDequantize(MNN::NetT*                                         model,
                                  std::map<int, const MNN::Tensor*>&                 idx_tensor_map, 
                                  std::map<const MNN::Tensor*, std::vector<float>>&  tensor_scale_map);

    static std::vector<std::string>  select_rand_image(std::vector<std::string> const& src, int max_num);
    static int                       rand_uniq_array(int iCnt, int* pHashTableArray, int* pOutIdxArray);
    
    void            compute_conv_weight_scale();
    void            compute_feature_scale();

    static void     init_quantize_feature_name(MNN::NetT*                                                  model,
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
                                               int                                                         min_quantize_threshold);

    static void     init_quantize_oplist(MNN::NetT*                                               model,
                                         MNN::Interpreter*                                        interpreter, 
                                         MNN::Session*                                            session,
                                         std::vector<std::tuple<std::string, int, MNN::OpType>>&  op_list);
    void            tunning_quantize_network(int                                               stop_op_idx, 
                                             int                                               cur_quantize_idx,
                                             int                                               total_quantize_op_cnt,
                                             std::map<MNN::Tensor*, TensorStatistic*>&         none_stop_out_feature, 
                                             std::map<MNN::Tensor*, TensorStatistic*>&         stop_op_out_feature,
                                             std::map<const MNN::Tensor*, std::vector<float>>& tensor_scale);
    void            compute_feature_range(int                                           stop_op_idx, 
                                          int                                           cur_quantize_idx,
                                          int                                           total_quantize_op_cnt,
                                          int                                           in_out,
                                          std::map<MNN::Tensor*, TensorStatistic*>&     none_stop_out_feature, 
                                          std::map<MNN::Tensor*, TensorStatistic*>&     stop_op_out_feature);
    void            compute_feature_distribution(int                                           stop_op_idx, 
                                                 int                                           cur_quantize_idx,
                                                 int                                           total_quantize_op_cnt,
                                                 int                                           in_out,
                                                 std::map<MNN::Tensor*, TensorStatistic*>&     none_stop_out_feature, 
                                                 std::map<MNN::Tensor*, TensorStatistic*>&     stop_op_out_feature);
    void            create_fake_quantize_session();
    void            retrieve_quantize_feature(int                                        stop_op_idx, 
                                              std::map<MNN::Tensor*, int>                quantize_feature_flag,
                                              std::map<MNN::Tensor*, TensorStatistic*>&  none_stop_out_feature, 
                                              std::map<MNN::Tensor*, TensorStatistic*>&  stop_op_out_feature);
    typedef struct tag_quantize_result{
        float       min_value;
        float       max_value;
        int         kl_threshold;
        float       scale;
    }QUANTIZE_RESULT, *PQUANTIZE_RESULT;
    void            tuning_scale_from_candidate(int                                                             op_idx,
                                                int                                                             cur_quantize_op_idx,
                                                int                                                             total_quantize_op_cnt,
                                                std::vector<std::pair<MNN::Tensor*, TensorStatistic*> > const&  quantize_feature_tensor,
                                                std::vector<QUANTIZE_RESULT>&                                   fine_quantize_info);
    
    void            update_quantize_op_scale();
    static void     load_image(MNN::CV::ImageProcess* pretreat, 
                               int                    targetWidth, 
                               int                    targetHeight,
                               const std::string&     inputImageFileName, 
                               MNN::Tensor*           input,
                               bool                   input_raw);
    static void     nchw_to_nc4hw4(float const* src, int c, int h, int w, float* dst);
    static void     nchw_to_nc4hw4_channel_1(float const* src, int h, int w, float* dst);
    static void     nchw_to_nc4hw4_channel_3(float const* src, int h, int w, float* dst);
};

#endif // CALIBRATION_HPP
