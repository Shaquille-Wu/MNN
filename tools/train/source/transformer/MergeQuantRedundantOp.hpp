/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file MergeQuantRedundantOp.hpp
 * @brief Merge Redundant Op, such as Int8ToFloat->Eltwise->FloatToInt8
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-12-17
 */

#ifndef MergeQuantRedundantOp_hpp
#define MergeQuantRedundantOp_hpp

#include <MNN/Interpreter.hpp>
#include "MNN_generated.h"

namespace MNN {
namespace Train {

class MNN_PUBLIC MergeQuantRedundantOp {
public:
    MergeQuantRedundantOp()           {;};
    virtual ~MergeQuantRedundantOp()  {;};
    static void Merge(MNN::NetT* model);

public:
    static int               reference_count(MNN::NetT* model, int tensor_idx);
    static std::vector<int>  find_op_by_output_idx(MNN::NetT* model, int tensor_idx);
    static std::vector<int>  find_op_by_input_idx(MNN::NetT* model, int tensor_idx);
    static void              MergeEltwiseOne(MNN::NetT* model);
    static void              MergeEltwiseTwo(MNN::NetT* model);
    static void              MergeEltwiseThree(MNN::NetT* model);
    static void              RemoveOpAndTensor(MNN::NetT*                model, 
                                               std::vector<int> const&   erase_op_idx, 
                                               std::vector<int> const&   erase_tensor_idx);
    static void                ReplaceInputIdx(MNN::NetT* model, int src_output_idx, int dst_output_idx);
    static std::vector<float>  GetScaleInv(std::vector<float> const& src);
};

} // namespace Train
} // namespace MNN

#endif