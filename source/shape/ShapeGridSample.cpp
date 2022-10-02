//
//  ShapeResize.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
// Size Computer
class GridSampleComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(4 == inputs[0]->dimensions());
        MNN_ASSERT(4 == inputs[1]->dimensions());
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(4 == outputs[0]->dimensions());


        int  out_b   = inputs[0]->buffer().dim[0].extent;
        int  out_h   = 0;
        int  out_w   = 0;
        int  out_c   = 0;
        auto in_fmt0 = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto in_fmt1 = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
        if (MNN_DATA_FORMAT_NHWC == in_fmt1)
        {
            out_h = inputs[1]->buffer().dim[1].extent;
            out_w = inputs[1]->buffer().dim[2].extent;
        }
        else
        {
            out_h = inputs[1]->buffer().dim[2].extent;
            out_w = inputs[1]->buffer().dim[3].extent;
        }

        if (MNN_DATA_FORMAT_NHWC == in_fmt0)
            out_c = inputs[0]->buffer().dim[3].extent;
        else
            out_c = inputs[0]->buffer().dim[1].extent;

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = inputs[0]->buffer().dimensions;
        outputBuffer.type          = inputs[0]->getType();
        outputBuffer.dim[0].extent = out_b;
        if (MNN_DATA_FORMAT_NHWC == in_fmt0)
        {
            outputBuffer.dim[1].extent = out_h;
            outputBuffer.dim[2].extent = out_w;
            outputBuffer.dim[3].extent = out_c;
        }
        else
        {
            outputBuffer.dim[1].extent = out_c;
            outputBuffer.dim[2].extent = out_h;
            outputBuffer.dim[3].extent = out_w;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
    virtual float onComputeFlops(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) const override {
        return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f * 4;
    }
};

REGISTER_SHAPE(GridSampleComputer, OpType_GridSample);
} // namespace MNN
