//
//  GridSample.cpp
//  MNNConverter
//
//  Created by Shauille on 2021/02/27.
//  Copyright © 2021, OrionStar
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(GridSampleOnnx);

MNN::OpType GridSampleOnnx::opType() {
    return MNN::OpType_GridSample;
}

MNN::OpParameter GridSampleOnnx::type() {
    return MNN::OpParameter_GridSample;
}

void GridSampleOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         std::vector<const onnx::TensorProto*> initializers) {
    int32_t    mode          = 0;
    int32_t    padding_mode  = 0;
    bool       align_corners = 0;
    const auto attrSize      = onnxNode->attribute_size();
    for (int i = 0; i < attrSize; ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if ("mode" == attributeName)
        {
            if("bilinear" == attributeProto.s())
                mode = 0;
            else if("nearest" == attributeProto.s())
                mode = 1;
        }
        else if("padding_mode" == attributeName)
        {
            if("zeros" == attributeProto.s())
                padding_mode = 0;
            else if("border" == attributeProto.s())
                padding_mode = 1;
            else if("reflection" == attributeProto.s())
                padding_mode = 2;
        }
        else if("align_corners" == attributeName)
        {
            int32_t  flag = attributeProto.i();
            align_corners = (0 != flag ? true : false);
        }
    }
    auto grid_sample           = new MNN::GridSampleT;
    grid_sample->mode          = mode;
    grid_sample->padding_mode  = padding_mode;
    grid_sample->align_corners = align_corners;
    dstOp->main.value          = grid_sample;
}

REGISTER_CONVERTER(GridSampleOnnx, GridSample);
