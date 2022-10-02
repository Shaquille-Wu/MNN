//
//  ShapeConvolution3D.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {
class Convolution3DSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        
        auto layer        = op->main_as_Convolution3D()->common();
        auto input = inputs[0];
//Shaquille, Modified 20210215 Start
#if 0
        if (input->buffer().dimensions != 5) {
            return false;
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        outputBuffer.dim[1].extent = layer->outputCount();
        
        for (int i = 0; i < 3; ++i) {
            const int inputLength = input->length(i + 2), stride = (*layer->strides())[i];
            if (inputLength <= 0) {
                return false;
            }
            int outputLength;
            if (layer->padMode() == PadMode_SAME) {
                outputLength = UP_DIV(inputLength, stride);
            } else {
                const int pad = (*layer->pads())[i], kernel = (*layer->kernels())[i], dialate = (*layer->dilates())[i];
                const int dialatedKernel = (kernel - 1) * dialate + 1;
                outputLength = (inputLength + 2 * pad - dialatedKernel) / stride + 1;
            }
            outputBuffer.dim[i + 2].extent = outputLength;
        }
#else
        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        outputBuffer.dim[1].extent = layer->outputCount();
        if(4 == input->buffer().dimensions)
        {
            int raw_dialate_z        = (*layer->dilates())[0];
            int raw_dialate_y        = (*layer->dilates())[1];
            int raw_dialate_x        = (*layer->dilates())[2];
            int input_depth          = (raw_dialate_z & 0xFFFF0000) >> 16;
			int in_batch_as_depth    = (raw_dialate_y & 0xFFFF0000) >> 16;
            int out_depth_as_channel = (raw_dialate_x & 0xFFFF0000) >> 16;
            int dialate_z            = (raw_dialate_z & 0x0000FFFF);
            int dialate_y            = (raw_dialate_y & 0x0000FFFF);
            int dialate_x            = (raw_dialate_x & 0x0000FFFF);
			int input_b              = input->buffer().dim[0].extent;
            int input_h              = input->buffer().dim[2].extent;
            int input_w              = input->buffer().dim[3].extent;
            int input_real_h         = input_h / input_depth;
            if (0 != in_batch_as_depth && input_depth != input_b)
                return false;
            if (input_depth <= 0)
                return false;
            if (0 == in_batch_as_depth)
            {
                if (0 != (input_h % input_depth))
                    return false;
            }
            else
                input_real_h = input_h;

            if(input_real_h <= 0 || input_w <= 0)
                return false;

            int stride_z         = (*layer->strides())[0];
            int stride_y         = (*layer->strides())[1];
            int stride_x         = (*layer->strides())[2];
            stride_z             = (stride_z & 0x0000FFFF);
            int output_depth     = 0;
            int output_height    = 0;
            int output_width     = 0;
            if (layer->padMode() == PadMode_SAME) 
            {
                output_depth     = (input_depth + stride_z - 1) / stride_z;
                output_height    = (input_real_h + stride_y - 1) / stride_y;
                output_width     = (input_w + stride_x - 1) / stride_x;
            }
            else
            {
                const int pad_z             = (*layer->pads())[0];
                const int pad_y             = (*layer->pads())[1];
                const int pad_x             = (*layer->pads())[2];
                const int kernel_z          = (*layer->kernels())[0];
                const int kernel_y          = (*layer->kernels())[1];
                const int kernel_x          = (*layer->kernels())[2];
                const int dialated_kernel_z = (kernel_z - 1) * dialate_z + 1;
                const int dialated_kernel_y = (kernel_y - 1) * dialate_y + 1;
                const int dialated_kernel_x = (kernel_x - 1) * dialate_x + 1;

                output_depth                = (input_depth + 2 * pad_z - dialated_kernel_z) / stride_z + 1;
                output_height               = (input_real_h + 2 * pad_y - dialated_kernel_y) / stride_y + 1;
                output_width                = (input_w + 2 * pad_x - dialated_kernel_x) / stride_x + 1;
            }

            if(0 == out_depth_as_channel)
            {
                if (0 != in_batch_as_depth)
                    outputBuffer.dim[0].extent = 1;
                outputBuffer.dim[2].extent = output_depth * output_height;
                outputBuffer.dim[3].extent = output_width;
            }
            else
            {
                outputBuffer.dim[1].extent = output_depth;
                outputBuffer.dim[2].extent = output_height;
                outputBuffer.dim[3].extent = output_width;
            }
        }
        else if(5 == input->buffer().dimensions)
        {
            for (int i = 0; i < 3; ++i) {
                int32_t   cur_stride  = (*layer->strides())[i];
                cur_stride            = (cur_stride & 0x0000FFFF);
                const int inputLength = input->length(i + 2), stride = cur_stride;
                if (inputLength <= 0) {
                    return false;
                }
                int outputLength;
                if (layer->padMode() == PadMode_SAME) {
                    outputLength = UP_DIV(inputLength, stride);
                } else {
                    const int pad = (*layer->pads())[i], kernel = (*layer->kernels())[i], dialate = (*layer->dilates())[i];
                    const int dialatedKernel = (kernel - 1) * dialate + 1;
                    outputLength = (inputLength + 2 * pad - dialatedKernel) / stride + 1;
                }
                outputBuffer.dim[i + 2].extent = outputLength;
            }
        }
        else
            return false;
#endif
//Shaquille, Modified 20210215 End
        outputBuffer.type = input->getType();

        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_Convolution3D()->common();
        int oSize = outputs[0]->length(1);
        float flopsPerElement = inputs[0]->length(1);
        for (int i = 0; i < 3; ++i) {
            flopsPerElement *= (*layer->kernels())[i];
            oSize *= outputs[0]->length(i + 2);
        }
        float flops = oSize * flopsPerElement / FLOPS_M;

        return flops;
    }
};

REGISTER_SHAPE(Convolution3DSizeComputer, OpType_Convolution3D);
} // namespace MNN
