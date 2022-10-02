/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file MergeQuantRedundantOp.cpp
 * @brief the implementation for MergeQuantRedundantOp
 * @author wuxiao@ainirobot.com
 * @date 2020-12-17
 */

#include "MergeQuantRedundantOp.hpp"

namespace MNN {
namespace Train {

int MergeQuantRedundantOp::reference_count(MNN::NetT* model, int tensor_idx)
{
    int                result = 0;
    int                i      = 0;
    int                j      = 0;
    int                op_cnt = (int)(model->oplists.size());
    for(i = 0 ; i < op_cnt ; i ++)
    {
        std::unique_ptr<MNN::OpT> const& op = model->oplists[i];
        auto& inputIndexes   = op->inputIndexes;
        int   input_cnt      = (int)(inputIndexes.size());
        for(j = 0 ; j < input_cnt ; j ++)
        {
            if(tensor_idx == inputIndexes[j])
                result ++;
        }
    }
    return result;
}

std::vector<int> MergeQuantRedundantOp::find_op_by_output_idx(MNN::NetT* model, int tensor_idx)
{
    std::vector<int>   result(0);
    int                i = 0;
    int                j = 0;
    int                op_cnt = (int)(model->oplists.size());
    for(i = 0 ; i < op_cnt ; i ++)
    {
        std::unique_ptr<MNN::OpT> const& op = model->oplists[i];
        auto& outputIndexes   = op->outputIndexes;
        int   output_cnt      = (int)(outputIndexes.size());
        for(j = 0 ; j < output_cnt ; j ++)
        {
            if(tensor_idx == outputIndexes[j])
            {
                result.push_back(i);
                break;
            }
        }
    }
    return result;
}

std::vector<int> MergeQuantRedundantOp::find_op_by_input_idx(MNN::NetT* model, int tensor_idx)
{
    std::vector<int>   result(0);
    int                i = 0;
    int                j = 0;
    int                op_cnt = (int)(model->oplists.size());
    for(i = 0 ; i < op_cnt ; i ++)
    {
        std::unique_ptr<MNN::OpT> const& op = model->oplists[i];
        auto& inputIndexes   = op->inputIndexes;
        int   input_cnt      = (int)(inputIndexes.size());
        for(j = 0 ; j < input_cnt ; j ++)
        {
            if(tensor_idx == inputIndexes[j])
            {
                result.push_back(i);
                break;
            }
        }
    }
    return result;
}

void MergeQuantRedundantOp::RemoveOpAndTensor(MNN::NetT*                model, 
                                              std::vector<int> const&   erase_op_idx, 
                                              std::vector<int> const&   erase_tensor_idx)
{
    int     i                = 0;
    int     j                = 0;
    int     op_cnt           = (int)(model->oplists.size());
    int     erase_op_idx_cnt = (int)(erase_op_idx.size());
    int     new_op_cnt       = op_cnt - erase_op_idx_cnt;
    int     new_op_idx       = 0;
    std::vector<std::unique_ptr<MNN::OpT> >  new_op_list(new_op_cnt);
    for(i = 0 ; i < op_cnt ; i ++)
    {
        for(j = 0 ; j < erase_op_idx_cnt ; j ++)
        {
            if(erase_op_idx[j] == i)
                break;
        }
        if(j >= erase_op_idx_cnt)
        {
            new_op_list[new_op_idx] = std::move(model->oplists[i]);
            new_op_idx ++;
        }
            
    }
    model->oplists.clear();
    model->oplists = std::move(new_op_list);

    int                       tensor_cnt        = model->tensorName.size();
    int                       erase_tensor_cnt  = (int)(erase_tensor_idx.size());
    int                       new_tensor_idx    = 0;
    std::vector<std::string>  new_tensor_name(tensor_cnt - erase_tensor_cnt);
    std::vector<int>          full_tensor_idx(tensor_cnt);
    for(i = 0 ; i < tensor_cnt ; i ++)
        full_tensor_idx[i] = i;
    for(i = 0 ; i < tensor_cnt ; i ++)
    {
        for(j = 0 ; j < erase_tensor_cnt ; j ++)
        {
            if(i == erase_tensor_idx[j])
            {
                full_tensor_idx[i] = -1;
            }
        }
        if(-1 != full_tensor_idx[i])
        {
            full_tensor_idx[i]              = new_tensor_idx;
            new_tensor_name[new_tensor_idx] = model->tensorName[i];
            new_tensor_idx ++;
        }
    }

    model->tensorName = new_tensor_name;
    op_cnt            = (int)(model->oplists.size());
    for(i = 0 ; i < op_cnt ; i ++)
    {
        auto op         = model->oplists[i].get();
        int  input_cnt  = (int)(op->inputIndexes.size());
        int  output_cnt = (int)(op->outputIndexes.size()); 
        for(j = 0 ; j < input_cnt ; j ++)
        {
            int input_idx       = op->inputIndexes[j];
            if(-1 == full_tensor_idx[input_idx])
            {
                MNN_PRINT("found exception input tensor idx\n");
                exit(0);
            }
            op->inputIndexes[j] = full_tensor_idx[input_idx];
        }

        for(j = 0 ; j < output_cnt ; j ++)
        {
            int output_idx       = op->outputIndexes[j];
            if(-1 == full_tensor_idx[output_idx])
            {
                MNN_PRINT("found exception output tensor idx\n");
                exit(0);
            }
            op->outputIndexes[j] = full_tensor_idx[output_idx];
        }
    }
}

void MergeQuantRedundantOp::ReplaceInputIdx(MNN::NetT* model, int src_output_idx, int dst_output_idx)
{
    int              j                 = 0;
    std::vector<int> reference_op_idx  = find_op_by_input_idx(model, src_output_idx);
    for(j = 0 ; j < (int)(reference_op_idx.size()) ; j ++)
    {
        std::unique_ptr<MNN::OpT>& next_op = model->oplists[reference_op_idx[j]];
        for(int m = 0 ; m < (int)(next_op->inputIndexes.size()) ; m ++)
        {
            if(next_op->inputIndexes[m] == src_output_idx)
                next_op->inputIndexes[m] = dst_output_idx;
        }
    }
}

std::vector<float> MergeQuantRedundantOp::GetScaleInv(std::vector<float> const& src)
{
    int                 j = 0;
    std::vector<float>  inv_output_scale(src.size());
    for(j = 0 ; j < (int)(src.size()) ; j ++)
    {
        if(src[j] < 1e-6 && src[j] >= 0.0f)
            inv_output_scale[j] = 1.0f/1e-6;
        else if(src[j] > -1e-6 && src[j] < 0.0f)
            inv_output_scale[j] = -1.0f/1e-6;
        else
            inv_output_scale[j] = 1.0f/src[j];
    }

    return inv_output_scale;
}

void MergeQuantRedundantOp::Merge(MNN::NetT* model)
{
    MergeEltwiseOne(model);
    MergeEltwiseTwo(model);
    MergeEltwiseThree(model);
}

void MergeQuantRedundantOp::MergeEltwiseOne(MNN::NetT* model)
{
    int               i               = 0 ;
    int               j               = 0;
    int               op_cnt          = (int)(model->oplists.size());
    std::vector<int>  erase_tensor_idx;
    std::vector<int>  erase_op_idx;
    for(i = 0 ; i < op_cnt ; i ++)
    {
        auto op           = model->oplists[i].get();
        const auto opType = op->type;
        const auto name   = op->name;
        // check whether is output op
        // if Yes, insert dequantization op after this op
        if (opType != OpType_Eltwise)
            continue;
        auto param = op->main.AsEltwise();
        // Now only support AddInt8
        if (param->type != MNN::EltwiseType_SUM)
            continue;

        std::vector<int32_t> inputIndexes    = op->inputIndexes;
        const int inputSize                  = inputIndexes.size();
        std::vector<int32_t> outputIndexes   = op->outputIndexes;
        const int outputSize                 = outputIndexes.size();

        if(2 != inputSize && 1 != outputSize)
            continue;

        int    input0_ref_cnt = reference_count(model, inputIndexes[0]);
        int    input1_ref_cnt = reference_count(model, inputIndexes[1]);
        int    output_ref_cnt = reference_count(model, outputIndexes[0]);

        if(1 != input0_ref_cnt || 1 != input1_ref_cnt || 1 != output_ref_cnt)
            continue;

        std::vector<int> input0_op_idx  = find_op_by_output_idx(model, inputIndexes[0]);
        std::vector<int> input1_op_idx  = find_op_by_output_idx(model, inputIndexes[1]);
        std::vector<int> output_op_idx  = find_op_by_input_idx(model, outputIndexes[0]);
        if(1 != input0_op_idx.size() || 1 != input1_op_idx.size() || 1 != output_op_idx.size())
            continue;

        std::unique_ptr<MNN::OpT> const& input0_op = model->oplists[input0_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& input1_op = model->oplists[input1_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& output_op = model->oplists[output_op_idx[0]];

        if(1 != input0_op->inputIndexes.size() ||
           1 != input0_op->outputIndexes.size() ||
           1 != input1_op->inputIndexes.size() ||
           1 != input1_op->outputIndexes.size() ||
           1 != output_op->inputIndexes.size() ||
           1 != output_op->outputIndexes.size())
           continue;
        
        if(OpType_Int8ToFloat != input0_op->type ||
           OpType_Int8ToFloat != input1_op->type ||
           OpType_FloatToInt8 != output_op->type)
           continue;
        
        int next_op_out_idx = output_op->outputIndexes[0];
        std::vector<int> next_op_idx         = find_op_by_input_idx(model, next_op_out_idx);

        op->inputIndexes[0]      = input0_op->inputIndexes[0];
        op->inputIndexes[1]      = input1_op->inputIndexes[0];
        for(j = 0 ; j < (int)(next_op_idx.size()) ; j ++)
        {
            std::unique_ptr<MNN::OpT>& next_op = model->oplists[next_op_idx[j]];
            for(int k = 0 ; k < (int)(next_op->inputIndexes.size()) ; k ++)
            {
                if(next_op->inputIndexes[k] == next_op_out_idx)
                    next_op->inputIndexes[k] = outputIndexes[0];
            }
        }

        auto int8tofloat_param0   = input0_op->main.AsQuantizedFloatParam();
        auto int8tofloat_param1   = input1_op->main.AsQuantizedFloatParam();
        auto input_scale0         = int8tofloat_param0->tensorScale;
        auto input_scale1         = int8tofloat_param1->tensorScale;
        auto floattoint8_param    = output_op->main.AsQuantizedFloatParam();
        auto output_scale         = floattoint8_param->tensorScale;
        op->type = MNN::OpType_EltwiseInt8;
        op->main.Reset();
        op->main.type = MNN::OpParameter_EltwiseInt8;

        auto eltwiseInt8Param         = new MNN::EltwiseInt8T;
        auto input0ScaleParam         = new MNN::QuantizedFloatParamT;
        auto input1ScaleParam         = new MNN::QuantizedFloatParamT;
        auto outputScaleParam         = new MNN::QuantizedFloatParamT;
        input0ScaleParam->tensorScale = input_scale0;
        input1ScaleParam->tensorScale = input_scale1;
        outputScaleParam->tensorScale = output_scale;
        eltwiseInt8Param->type        = MNN::EltwiseType_SUM;
        eltwiseInt8Param->inputQuan0  = std::unique_ptr<MNN::QuantizedFloatParamT>(input0ScaleParam);
        eltwiseInt8Param->inputQuan1  = std::unique_ptr<MNN::QuantizedFloatParamT>(input1ScaleParam);
        eltwiseInt8Param->outputQuan  = std::unique_ptr<MNN::QuantizedFloatParamT>(outputScaleParam);
        op->main.value                = eltwiseInt8Param;

        erase_tensor_idx.push_back(inputIndexes[0]);
        erase_tensor_idx.push_back(inputIndexes[1]);
        erase_tensor_idx.push_back(next_op_out_idx);
        erase_op_idx.push_back(input0_op_idx[0]);
        erase_op_idx.push_back(input1_op_idx[0]);
        erase_op_idx.push_back(output_op_idx[0]);
    }

    RemoveOpAndTensor(model, erase_op_idx, erase_tensor_idx);
}

void MergeQuantRedundantOp::MergeEltwiseTwo(MNN::NetT* model)
{
    int               i               = 0 ;
    int               j               = 0;
    int               op_cnt          = (int)(model->oplists.size());
    std::vector<int>  erase_tensor_idx;
    std::vector<int>  erase_op_idx;
    for(i = 0 ; i < op_cnt ; i ++)
    {
        auto op           = model->oplists[i].get();
        const auto opType = op->type;
        const auto name   = op->name;
        // check whether is output op
        // if Yes, insert dequantization op after this op
        if (opType != OpType_Eltwise)
            continue;
        auto param = op->main.AsEltwise();
        // Now only support AddInt8
        if (param->type != MNN::EltwiseType_SUM)
            continue;

        std::vector<int32_t> inputIndexes    = op->inputIndexes;
        const int            inputSize       = inputIndexes.size();
        std::vector<int32_t> outputIndexes   = op->outputIndexes;
        const int            outputSize      = outputIndexes.size();

        if(2 != inputSize && 1 != outputSize)
            continue;

        int    input0_ref_cnt = reference_count(model, inputIndexes[0]);
        int    input1_ref_cnt = reference_count(model, inputIndexes[1]);
        int    output_ref_cnt = reference_count(model, outputIndexes[0]);

        if(1 != input0_ref_cnt || 1 != input1_ref_cnt || 2 != output_ref_cnt)
            continue;

        std::vector<int> input0_op_idx  = find_op_by_output_idx(model, inputIndexes[0]);
        std::vector<int> input1_op_idx  = find_op_by_output_idx(model, inputIndexes[1]);
        std::vector<int> output_op_idx  = find_op_by_input_idx(model, outputIndexes[0]);
        if(1 != input0_op_idx.size() || 1 != input1_op_idx.size() || 2 != output_op_idx.size())
            continue;

        std::unique_ptr<MNN::OpT> const& input0_op  = model->oplists[input0_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& input1_op  = model->oplists[input1_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& output0_op = model->oplists[output_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& output1_op = model->oplists[output_op_idx[1]];

        if(1 != input0_op->inputIndexes.size() ||
           1 != input0_op->outputIndexes.size() ||
           1 != input1_op->inputIndexes.size() ||
           1 != input1_op->outputIndexes.size())
           continue;
        
        if(OpType_Int8ToFloat != input0_op->type ||
           OpType_Int8ToFloat != input1_op->type)
           continue;
        if(((OpType_FloatToInt8 != output0_op->type &&
             OpType_Eltwise != output1_op->type) ||
            (OpType_Eltwise != output0_op->type &&
             OpType_FloatToInt8 != output1_op->type)) == false)
           continue;
        
        MNN::OpT*  floattoint8_op                = nullptr;
        MNN::OpT*  another_int8tofloat_op        = nullptr;
        MNN::OpT*  eltwise_op                    = nullptr;
        MNN::OpT*  eltwise_output_op             = nullptr;
        int        erase_next_op_idx[3]          = { 0, 0, 0 };
        int        erase_next_tensor_idx[3]      = { 0, 0, 0 };
        int        eltwise_direct_link_input_idx = -1;
        if(OpType_FloatToInt8 == output0_op->type)
        {
            erase_next_op_idx[0]              = output_op_idx[0];
            int              next0_op_out_idx = output0_op->outputIndexes[0];
            erase_next_tensor_idx[0]          = next0_op_out_idx;
            std::vector<int> next0_op_idx     = find_op_by_input_idx(model, next0_op_out_idx);
            floattoint8_op                    = output0_op.get();
            eltwise_op                        = output1_op.get();

            if(i == eltwise_op->inputIndexes[0])
                eltwise_direct_link_input_idx = 0;
            else if(i == eltwise_op->inputIndexes[1])
                eltwise_direct_link_input_idx = 1;

            if(-1 != eltwise_direct_link_input_idx)
            {
                int              eltwise_int8tofloat_input_tensor_idx = eltwise_op->inputIndexes[1 - eltwise_direct_link_input_idx];
                std::vector<int> eltwise_int8tofloat_input_op_idx     = find_op_by_output_idx(model, eltwise_int8tofloat_input_tensor_idx);
                if(1 != eltwise_int8tofloat_input_op_idx.size())
                    continue;
                another_int8tofloat_op = model->oplists[eltwise_int8tofloat_input_op_idx[0]].get();
                if(OpType_Int8ToFloat != another_int8tofloat_op->type || 1 != another_int8tofloat_op->outputIndexes.size())
                    continue;
                int              eltwise_output_tensor_idx = eltwise_op->outputIndexes[0];
                std::vector<int> eltwise_output_op_idx     = find_op_by_input_idx(model, eltwise_output_tensor_idx);
                if(1 != eltwise_output_op_idx.size())
                    continue;
                eltwise_output_op = model->oplists[eltwise_output_op_idx[0]].get();
                if(OpType_FloatToInt8 != eltwise_output_op->type || 1 != eltwise_output_op->outputIndexes.size())
                    continue;

                erase_next_op_idx[1]     = eltwise_int8tofloat_input_op_idx[0];
                erase_next_op_idx[2]     = eltwise_output_op_idx[0];
                erase_next_tensor_idx[1] = another_int8tofloat_op->outputIndexes[0];
                erase_next_tensor_idx[2] = eltwise_output_op->outputIndexes[0];

                for(int k = 0 ; k < (int)(next0_op_idx.size()) ; k ++)
                {
                    std::unique_ptr<MNN::OpT>& next_op = model->oplists[next0_op_idx[k]];
                    for(int m = 0 ; m < (int)(next_op->inputIndexes.size()) ; m ++)
                    {
                        if(next_op->inputIndexes[m] == erase_next_op_idx[0])
                            next_op->inputIndexes[m] = outputIndexes[0];
                    }
                }
                eltwise_op->inputIndexes[1 - eltwise_direct_link_input_idx] = another_int8tofloat_op->inputIndexes[0];
                int              eltwise_floattoint8_op_out_idx             = eltwise_output_op->outputIndexes[0];
                std::vector<int> eltwise_floattoint8_op_next_op_idx         = find_op_by_input_idx(model, eltwise_floattoint8_op_out_idx);
                for(int k = 0 ; k < (int)(eltwise_floattoint8_op_next_op_idx.size()) ; k ++)
                {
                    std::unique_ptr<MNN::OpT>& next_op = model->oplists[eltwise_floattoint8_op_next_op_idx[k]];
                    for(int m = 0 ; m < (int)(next_op->inputIndexes.size()) ; m ++)
                    {
                        if(next_op->inputIndexes[m] == erase_next_tensor_idx[2])
                            next_op->inputIndexes[m] = eltwise_op->outputIndexes[0];
                    }
                }
            }
            else
                continue;
        }
        else
        {
            erase_next_op_idx[0]              = output_op_idx[1];
            int              next1_op_out_idx = output1_op->outputIndexes[0];
            erase_next_tensor_idx[0]          = next1_op_out_idx;
            std::vector<int> next1_op_idx     = find_op_by_input_idx(model, next1_op_out_idx);
            eltwise_op                        = output0_op.get();
            floattoint8_op                    = output1_op.get();

            if(i == eltwise_op->inputIndexes[0])
                eltwise_direct_link_input_idx = 0;
            else if(i == eltwise_op->inputIndexes[1])
                eltwise_direct_link_input_idx = 1;

            if(-1 != eltwise_direct_link_input_idx)
            {
                int              eltwise_int8tofloat_input_tensor_idx = eltwise_op->inputIndexes[1 - eltwise_direct_link_input_idx];
                std::vector<int> eltwise_int8tofloat_input_op_idx     = find_op_by_output_idx(model, eltwise_int8tofloat_input_tensor_idx);
                if(1 != eltwise_int8tofloat_input_op_idx.size())
                    continue;
                another_int8tofloat_op = model->oplists[eltwise_int8tofloat_input_op_idx[0]].get();
                if(OpType_Int8ToFloat != another_int8tofloat_op->type || 1 != another_int8tofloat_op->outputIndexes.size())
                    continue;
                int              eltwise_output_tensor_idx = eltwise_op->outputIndexes[0];
                std::vector<int> eltwise_output_op_idx     = find_op_by_input_idx(model, eltwise_output_tensor_idx);
                if(1 != eltwise_output_op_idx.size())
                    continue;
                eltwise_output_op = model->oplists[eltwise_output_op_idx[0]].get();
                if(OpType_FloatToInt8 != eltwise_output_op->type || 1 != eltwise_output_op->outputIndexes.size())
                    continue;

                erase_next_op_idx[1]     = eltwise_int8tofloat_input_op_idx[0];
                erase_next_op_idx[2]     = eltwise_output_op_idx[0];
                erase_next_tensor_idx[1] = another_int8tofloat_op->outputIndexes[0];
                erase_next_tensor_idx[2] = eltwise_output_op->outputIndexes[0];

                for(int k = 0 ; k < (int)(next1_op_idx.size()) ; k ++)
                {
                    std::unique_ptr<MNN::OpT>& next_op = model->oplists[next1_op_idx[k]];
                    for(int m = 0 ; m < (int)(next_op->inputIndexes.size()) ; m ++)
                    {
                        if(next_op->inputIndexes[m] == erase_next_op_idx[0])
                            next_op->inputIndexes[m] = outputIndexes[0];
                    }
                }
                eltwise_op->inputIndexes[1 - eltwise_direct_link_input_idx] = another_int8tofloat_op->inputIndexes[0];
                int              eltwise_floattoint8_op_out_idx             = eltwise_output_op->outputIndexes[0];
                std::vector<int> eltwise_floattoint8_op_next_op_idx         = find_op_by_input_idx(model, eltwise_floattoint8_op_out_idx);
                for(int k = 0 ; k < (int)(eltwise_floattoint8_op_next_op_idx.size()) ; k ++)
                {
                    std::unique_ptr<MNN::OpT>& next_op = model->oplists[eltwise_floattoint8_op_next_op_idx[k]];
                    for(int m = 0 ; m < (int)(next_op->inputIndexes.size()) ; m ++)
                    {
                        if(next_op->inputIndexes[m] == erase_next_tensor_idx[2])
                            next_op->inputIndexes[m] = eltwise_op->outputIndexes[0];
                    }
                }
            }
            else
                continue;
        }

        op->inputIndexes[0]           = input0_op->inputIndexes[0];
        op->inputIndexes[1]           = input1_op->inputIndexes[0];

        auto int8tofloat_param0       = input0_op->main.AsQuantizedFloatParam();
        auto int8tofloat_param1       = input1_op->main.AsQuantizedFloatParam();
        auto input_scale0             = int8tofloat_param0->tensorScale;
        auto input_scale1             = int8tofloat_param1->tensorScale;
        auto floattoint8_param        = floattoint8_op->main.AsQuantizedFloatParam();
        auto output_scale             = floattoint8_param->tensorScale;
        op->type = MNN::OpType_EltwiseInt8;
        op->main.Reset();
        op->main.type = MNN::OpParameter_EltwiseInt8;
        auto eltwiseInt8Param         = new MNN::EltwiseInt8T;
        auto input0ScaleParam         = new MNN::QuantizedFloatParamT;
        auto input1ScaleParam         = new MNN::QuantizedFloatParamT;
        auto outputScaleParam         = new MNN::QuantizedFloatParamT;
        input0ScaleParam->tensorScale = input_scale0;
        input1ScaleParam->tensorScale = input_scale1;
        outputScaleParam->tensorScale = output_scale;
        eltwiseInt8Param->type        = MNN::EltwiseType_SUM;
        eltwiseInt8Param->inputQuan0  = std::unique_ptr<MNN::QuantizedFloatParamT>(input0ScaleParam);
        eltwiseInt8Param->inputQuan1  = std::unique_ptr<MNN::QuantizedFloatParamT>(input1ScaleParam);
        eltwiseInt8Param->outputQuan  = std::unique_ptr<MNN::QuantizedFloatParamT>(outputScaleParam);
        op->main.value                = eltwiseInt8Param;


        eltwise_op->type                      = MNN::OpType_EltwiseInt8;
        eltwise_op->main.Reset();
        eltwise_op->main.type                 = MNN::OpParameter_EltwiseInt8;
        auto eltwise_eltwiseInt8Param         = new MNN::EltwiseInt8T;
        auto eltwise_input0ScaleParam         = new MNN::QuantizedFloatParamT;
        auto eltwise_input1ScaleParam         = new MNN::QuantizedFloatParamT;
        auto eltwise_outputScaleParam         = new MNN::QuantizedFloatParamT;
        auto eltwise_floattoint8_param        = eltwise_output_op->main.AsQuantizedFloatParam();
        auto eltwise_output_scale             = eltwise_floattoint8_param->tensorScale;
        eltwise_eltwiseInt8Param->type        = MNN::EltwiseType_SUM;
        std::vector<float>  inv_output_scale  = GetScaleInv(output_scale);
        if(0 == eltwise_direct_link_input_idx)
        {
            eltwise_input0ScaleParam->tensorScale = inv_output_scale;
            auto eltwise_right_floattoint8_param  = another_int8tofloat_op->main.AsQuantizedFloatParam();
            eltwise_input1ScaleParam->tensorScale = eltwise_right_floattoint8_param->tensorScale;
        }
        else
        {
            auto eltwise_left_floattoint8_param   = another_int8tofloat_op->main.AsQuantizedFloatParam();
            eltwise_input0ScaleParam->tensorScale = eltwise_left_floattoint8_param->tensorScale;
            eltwise_input1ScaleParam->tensorScale = inv_output_scale;
        }
        eltwise_outputScaleParam->tensorScale  = eltwise_output_scale;
        eltwise_eltwiseInt8Param->inputQuan0   = std::unique_ptr<MNN::QuantizedFloatParamT>(eltwise_input0ScaleParam);
        eltwise_eltwiseInt8Param->inputQuan1   = std::unique_ptr<MNN::QuantizedFloatParamT>(eltwise_input1ScaleParam);
        eltwise_eltwiseInt8Param->outputQuan   = std::unique_ptr<MNN::QuantizedFloatParamT>(eltwise_outputScaleParam);
        eltwise_op->main.value                 = eltwise_eltwiseInt8Param;

        erase_tensor_idx.push_back(inputIndexes[0]);
        erase_tensor_idx.push_back(inputIndexes[1]);
        erase_tensor_idx.push_back(erase_next_tensor_idx[0]);
        erase_tensor_idx.push_back(erase_next_tensor_idx[1]);
        erase_tensor_idx.push_back(erase_next_tensor_idx[2]);
        erase_op_idx.push_back(input0_op_idx[0]);
        erase_op_idx.push_back(input1_op_idx[0]);
        erase_op_idx.push_back(erase_next_op_idx[0]);
        erase_op_idx.push_back(erase_next_op_idx[1]);
        erase_op_idx.push_back(erase_next_op_idx[2]);
    }

    RemoveOpAndTensor(model, erase_op_idx, erase_tensor_idx);
}

void MergeQuantRedundantOp::MergeEltwiseThree(MNN::NetT* model)
{
    int               i                         = 0 ;
    int               j                         = 0;
    int               op_cnt                    = (int)(model->oplists.size());
    int               first_eltwise             = 0;
    int               second_eltwise            = 0;
    int               third_eltwise             = 0;
    int               first_floattoint8         = 0;
    int               first_int8tofloat         = 0;
    int               first_floatotint8_output  = 0;
    int               first_int8tofloat_output  = 0;
    int               second_floattoint8        = 0;
    int               second_int8tofloat        = 0;
    int               second_floatotint8_output = 0;
    int               second_int8tofloat_output = 0;
    int               third_floattoint8         = 0;
    int               third_floattoint8_output  = 0;
    std::vector<int>  erase_tensor_idx;
    std::vector<int>  erase_op_idx;

    for(i = 0 ; i < op_cnt ; i ++)
    {
        auto op           = model->oplists[i].get();
        const auto opType = op->type;
        const auto name   = op->name;
        // check whether is output op
        // if Yes, insert dequantization op after this op
        if (opType != OpType_Eltwise)
            continue;
        auto param = op->main.AsEltwise();
        // Now only support AddInt8
        if (param->type != MNN::EltwiseType_SUM)
            continue;

        std::vector<int32_t> inputIndexes   = op->inputIndexes;
        const int            inputSize      = inputIndexes.size();
        std::vector<int32_t> outputIndexes  = op->outputIndexes;
        const int            outputSize     = outputIndexes.size();

        if(2 != inputSize && 1 != outputSize)
            continue;

        int    input0_ref_cnt = reference_count(model, inputIndexes[0]);
        int    input1_ref_cnt = reference_count(model, inputIndexes[1]);
        int    output_ref_cnt = reference_count(model, outputIndexes[0]);

        if(1 != input0_ref_cnt || 1 != input1_ref_cnt || 2 != output_ref_cnt)
            continue;

        std::vector<int> input0_op_idx  = find_op_by_output_idx(model, inputIndexes[0]);
        std::vector<int> input1_op_idx  = find_op_by_output_idx(model, inputIndexes[1]);
        std::vector<int> output_op_idx  = find_op_by_input_idx(model, outputIndexes[0]);
        if(1 != input0_op_idx.size() || 1 != input1_op_idx.size() || 2 != output_op_idx.size())
            continue;

        std::unique_ptr<MNN::OpT> const& input0_op  = model->oplists[input0_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& input1_op  = model->oplists[input1_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& output0_op = model->oplists[output_op_idx[0]];
        std::unique_ptr<MNN::OpT> const& output1_op = model->oplists[output_op_idx[1]];

        if(1 != input0_op->inputIndexes.size() ||
           1 != input0_op->outputIndexes.size() ||
           1 != input1_op->inputIndexes.size() ||
           1 != input1_op->outputIndexes.size())
           continue;
        
        if(OpType_Int8ToFloat != input0_op->type ||
           OpType_Int8ToFloat != input1_op->type)
           continue;

        if(((OpType_FloatToInt8 == output0_op->type && OpType_Eltwise == output1_op->type) ||
            (OpType_Eltwise == output0_op->type && OpType_FloatToInt8 == output1_op->type)) == false)
            continue;

        first_eltwise      = i;
        MNN::OpT*   second_eltwise_op = nullptr;
        if(OpType_FloatToInt8 == output0_op->type)
        {
            auto eltwise_param = output1_op->main.AsEltwise();
            // Now only support AddInt8
            if (eltwise_param->type != MNN::EltwiseType_SUM)
                continue;

            second_eltwise_op        = output1_op.get();
            second_eltwise           = output_op_idx[1];
            first_floattoint8        = output_op_idx[0];
            first_floatotint8_output = output0_op->outputIndexes[0];
        }
        else
        {
            auto eltwise_param = output0_op->main.AsEltwise();
            // Now only support AddInt8
            if (eltwise_param->type != MNN::EltwiseType_SUM)
                continue;

            second_eltwise_op        = output0_op.get();
            second_eltwise           = output_op_idx[0];
            first_floattoint8        = output_op_idx[1];
            first_floatotint8_output = output1_op->outputIndexes[0];
        }
        int         second_eltwise_direct_link_slot_idx  = -1;
        if(first_eltwise == second_eltwise_op->inputIndexes[0])
            second_eltwise_direct_link_slot_idx = 0;
        else if(first_eltwise == second_eltwise_op->inputIndexes[1])
            second_eltwise_direct_link_slot_idx = 1;
        if(-1 == second_eltwise_direct_link_slot_idx)
            continue;
        int                 second_eltwise_op_another_input        = second_eltwise_op->inputIndexes[1 - second_eltwise_direct_link_slot_idx];
        std::vector<int>    second_eltwise_op_another_input_op_idx = find_op_by_output_idx(model, second_eltwise_op_another_input);
        if(1 != second_eltwise_op_another_input_op_idx.size())
            continue;
        MNN::OpT*           second_eltwise_another_input_op        = model->oplists[second_eltwise_op_another_input_op_idx[0]].get();
        if(OpType_Int8ToFloat != second_eltwise_another_input_op->type)
            continue;
        first_int8tofloat        = second_eltwise_op_another_input_op_idx[0];
        first_int8tofloat_output = second_eltwise_op_another_input;
        std::vector<int> first_int8tofloat_output_op = find_op_by_input_idx(model, first_int8tofloat_output);
        if(1 != first_int8tofloat_output_op.size())
            continue;
        if(1 != second_eltwise_op->outputIndexes.size())
            continue;
        std::vector<int> second_eltwise_output_op_idx  = find_op_by_input_idx(model, second_eltwise_op->outputIndexes[0]);
        if(2 != second_eltwise_output_op_idx.size())
            continue;
        
        MNN::OpT* second_eltwise_output0_op = model->oplists[second_eltwise_output_op_idx[0]].get();
        MNN::OpT* second_eltwise_output1_op = model->oplists[second_eltwise_output_op_idx[1]].get();
        if(((OpType_FloatToInt8 == second_eltwise_output0_op->type && OpType_Eltwise == second_eltwise_output1_op->type) ||
            (OpType_Eltwise == second_eltwise_output0_op->type && OpType_FloatToInt8 == second_eltwise_output1_op->type)) == false)
            continue;

        MNN::OpT*   third_eltwise_op = nullptr;
        if(OpType_FloatToInt8 == second_eltwise_output0_op->type)
        {
            auto eltwise_param = second_eltwise_output1_op->main.AsEltwise();
            // Now only support AddInt8
            if (eltwise_param->type != MNN::EltwiseType_SUM)
                continue;

            third_eltwise_op          = second_eltwise_output1_op;
            third_eltwise             = second_eltwise_output_op_idx[1];
            second_floattoint8        = second_eltwise_output_op_idx[0];
            second_floatotint8_output = second_eltwise_output0_op->outputIndexes[0];
        }
        else
        {
            auto eltwise_param = second_eltwise_output0_op->main.AsEltwise();
            // Now only support AddInt8
            if (eltwise_param->type != MNN::EltwiseType_SUM)
                continue;

            third_eltwise_op          = second_eltwise_output0_op;
            third_eltwise             = second_eltwise_output_op_idx[0];
            second_floattoint8        = second_eltwise_output_op_idx[1];
            second_floatotint8_output = second_eltwise_output1_op->outputIndexes[0];
        }

        int         third_eltwise_direct_link_slot_idx  = -1;
        if(second_eltwise == third_eltwise_op->inputIndexes[0])
            third_eltwise_direct_link_slot_idx = 0;
        else if(second_eltwise == third_eltwise_op->inputIndexes[1])
            third_eltwise_direct_link_slot_idx = 1;
        if(-1 == third_eltwise_direct_link_slot_idx)
            continue;
        int                 third_eltwise_op_another_input        = third_eltwise_op->inputIndexes[1 - third_eltwise_direct_link_slot_idx];
        std::vector<int>    third_eltwise_op_another_input_op_idx = find_op_by_output_idx(model, third_eltwise_op_another_input);
        if(1 != third_eltwise_op_another_input_op_idx.size())
            continue;
        MNN::OpT*           third_eltwise_another_input_op        = model->oplists[third_eltwise_op_another_input_op_idx[0]].get();
        if(OpType_Int8ToFloat != third_eltwise_another_input_op->type)
            continue;
        second_int8tofloat        = third_eltwise_op_another_input_op_idx[0];
        second_int8tofloat_output = third_eltwise_op_another_input;
        std::vector<int> second_int8tofloat_output_op = find_op_by_input_idx(model, second_int8tofloat_output);
        if(1 != first_int8tofloat_output_op.size())
            continue;
        if(1 != third_eltwise_op->outputIndexes.size())
            continue;

        std::vector<int> third_eltwise_output_op_idx  = find_op_by_input_idx(model, third_eltwise_op->outputIndexes[0]);
        if(1 != third_eltwise_output_op_idx.size())
            continue;

        MNN::OpT* third_eltwise_output_op = model->oplists[third_eltwise_output_op_idx[0]].get();
        if(OpType_FloatToInt8 != third_eltwise_output_op->type)
            continue;
        third_floattoint8        = third_eltwise_output_op_idx[0];
        third_floattoint8_output = third_eltwise_output_op->outputIndexes[0];


        op->inputIndexes[0]                = input0_op->inputIndexes[0];
        op->inputIndexes[1]                = input1_op->inputIndexes[0];
        second_eltwise_op->inputIndexes[1 - second_eltwise_direct_link_slot_idx] = second_eltwise_another_input_op->inputIndexes[0];
        third_eltwise_op->inputIndexes[1 - third_eltwise_direct_link_slot_idx]   = third_eltwise_another_input_op->inputIndexes[0];
        ReplaceInputIdx(model, first_floatotint8_output, outputIndexes[0]);
        ReplaceInputIdx(model, second_floatotint8_output, second_eltwise_op->outputIndexes[0]);
        ReplaceInputIdx(model, third_floattoint8_output, third_eltwise_op->outputIndexes[0]);


        auto first_int8tofloat_param0 = input0_op->main.AsQuantizedFloatParam();
        auto first_int8tofloat_param1 = input1_op->main.AsQuantizedFloatParam();
        auto first_input_scale0       = first_int8tofloat_param0->tensorScale;
        auto first_input_scale1       = first_int8tofloat_param1->tensorScale;
        auto first_floattoint8_param  = model->oplists[first_floattoint8]->main.AsQuantizedFloatParam();
        auto first_output_scale       = first_floattoint8_param->tensorScale;
        op->type = MNN::OpType_EltwiseInt8;
        op->main.Reset();
        op->main.type = MNN::OpParameter_EltwiseInt8;
        auto first_eltwiseInt8Param         = new MNN::EltwiseInt8T;
        auto first_input0ScaleParam         = new MNN::QuantizedFloatParamT;
        auto first_input1ScaleParam         = new MNN::QuantizedFloatParamT;
        auto first_outputScaleParam         = new MNN::QuantizedFloatParamT;
        first_input0ScaleParam->tensorScale = first_input_scale0;
        first_input1ScaleParam->tensorScale = first_input_scale1;
        first_outputScaleParam->tensorScale = first_output_scale;
        first_eltwiseInt8Param->type        = MNN::EltwiseType_SUM;
        first_eltwiseInt8Param->inputQuan0  = std::unique_ptr<MNN::QuantizedFloatParamT>(first_input0ScaleParam);
        first_eltwiseInt8Param->inputQuan1  = std::unique_ptr<MNN::QuantizedFloatParamT>(first_input1ScaleParam);
        first_eltwiseInt8Param->outputQuan  = std::unique_ptr<MNN::QuantizedFloatParamT>(first_outputScaleParam);
        op->main.value                      = first_eltwiseInt8Param;


        second_eltwise_op->type              = MNN::OpType_EltwiseInt8;
        second_eltwise_op->main.Reset();
        second_eltwise_op->main.type         = MNN::OpParameter_EltwiseInt8;
        auto second_eltwiseInt8Param         = new MNN::EltwiseInt8T;
        auto second_input0ScaleParam         = new MNN::QuantizedFloatParamT;
        auto second_input1ScaleParam         = new MNN::QuantizedFloatParamT;
        auto second_outputScaleParam         = new MNN::QuantizedFloatParamT;
        auto second_floattoint8_param        = model->oplists[second_floattoint8]->main.AsQuantizedFloatParam();
        auto second_output_scale             = second_floattoint8_param->tensorScale;
        second_eltwiseInt8Param->type        = MNN::EltwiseType_SUM;
        std::vector<float>  inv_output_scale  = GetScaleInv(first_output_scale);
        if(0 == second_eltwise_direct_link_slot_idx)
        {
            second_input0ScaleParam->tensorScale = inv_output_scale;
            auto eltwise_right_floattoint8_param = second_eltwise_another_input_op->main.AsQuantizedFloatParam();
            second_input1ScaleParam->tensorScale = eltwise_right_floattoint8_param->tensorScale;
        }
        else
        {
            auto eltwise_left_floattoint8_param   = second_eltwise_another_input_op->main.AsQuantizedFloatParam();
            second_input0ScaleParam->tensorScale  = eltwise_left_floattoint8_param->tensorScale;
            second_input1ScaleParam->tensorScale  = inv_output_scale;
        }
        second_outputScaleParam->tensorScale  = second_output_scale;
        second_eltwiseInt8Param->inputQuan0   = std::unique_ptr<MNN::QuantizedFloatParamT>(second_input0ScaleParam);
        second_eltwiseInt8Param->inputQuan1   = std::unique_ptr<MNN::QuantizedFloatParamT>(second_input1ScaleParam);
        second_eltwiseInt8Param->outputQuan   = std::unique_ptr<MNN::QuantizedFloatParamT>(second_outputScaleParam);
        second_eltwise_op->main.value         = second_eltwiseInt8Param;


        third_eltwise_op->type              = MNN::OpType_EltwiseInt8;
        third_eltwise_op->main.Reset();
        third_eltwise_op->main.type         = MNN::OpParameter_EltwiseInt8;
        auto third_eltwiseInt8Param         = new MNN::EltwiseInt8T;
        auto third_input0ScaleParam         = new MNN::QuantizedFloatParamT;
        auto third_input1ScaleParam         = new MNN::QuantizedFloatParamT;
        auto third_outputScaleParam         = new MNN::QuantizedFloatParamT;
        auto third_floattoint8_param        = model->oplists[third_floattoint8]->main.AsQuantizedFloatParam();
        auto third_output_scale             = third_floattoint8_param->tensorScale;
        third_eltwiseInt8Param->type        = MNN::EltwiseType_SUM;
        inv_output_scale  = GetScaleInv(second_output_scale);
        if(0 == third_eltwise_direct_link_slot_idx)
        {
            third_input0ScaleParam->tensorScale  = inv_output_scale;
            auto eltwise_right_floattoint8_param = third_eltwise_another_input_op->main.AsQuantizedFloatParam();
            third_input1ScaleParam->tensorScale  = eltwise_right_floattoint8_param->tensorScale;
        }
        else
        {
            auto eltwise_left_floattoint8_param  = third_eltwise_another_input_op->main.AsQuantizedFloatParam();
            third_input0ScaleParam->tensorScale  = eltwise_left_floattoint8_param->tensorScale;
            third_input1ScaleParam->tensorScale  = inv_output_scale;
        }
        third_outputScaleParam->tensorScale  = third_output_scale;
        third_eltwiseInt8Param->inputQuan0   = std::unique_ptr<MNN::QuantizedFloatParamT>(third_input0ScaleParam);
        third_eltwiseInt8Param->inputQuan1   = std::unique_ptr<MNN::QuantizedFloatParamT>(third_input1ScaleParam);
        third_eltwiseInt8Param->outputQuan   = std::unique_ptr<MNN::QuantizedFloatParamT>(third_outputScaleParam);
        third_eltwise_op->main.value         = third_eltwiseInt8Param;

        erase_tensor_idx.push_back(inputIndexes[0]);
        erase_tensor_idx.push_back(inputIndexes[1]);
        erase_tensor_idx.push_back(first_floatotint8_output);
        erase_tensor_idx.push_back(first_int8tofloat_output);
        erase_tensor_idx.push_back(second_floatotint8_output);
        erase_tensor_idx.push_back(second_int8tofloat_output);
        erase_tensor_idx.push_back(third_floattoint8_output);
        erase_op_idx.push_back(input0_op_idx[0]);
        erase_op_idx.push_back(input1_op_idx[0]);
        erase_op_idx.push_back(first_floattoint8);
        erase_op_idx.push_back(first_int8tofloat);
        erase_op_idx.push_back(second_floattoint8);
        erase_op_idx.push_back(second_int8tofloat);
        erase_op_idx.push_back(third_floattoint8);
    }

    RemoveOpAndTensor(model, erase_op_idx, erase_tensor_idx);
}

} // namespace Train
} // namespace MNN