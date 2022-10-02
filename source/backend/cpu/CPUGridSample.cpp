//
//  CPUGridSample.cpp
//  OrionStar
//
//  Created by Shaquille.Wu on 2021/02/27.
//  Copyright © 2021, OrionStar
//

#include "backend/cpu/CPUGridSample.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUGridSample::CPUGridSample(Backend *backend, int mode, int padding_mode, bool align_corners)
    : Execution(backend),
      mode_(mode),
      padding_mode_(padding_mode),
      align_corners_(align_corners){
    // nothing to do
}

CPUGridSample::~CPUGridSample() {
}

static void sample_grid_bilinear_zero_pad_nchw(float const* src,
                                               int          n,
                                               int          c,
                                               int          ih,
                                               int          iw,
                                               float const* grid,
                                               int          oh,
                                               int          ow,
                                               float*       dst)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < oh; k++)
            {
                for (int m = 0; m < ow; m++)
                {
                    int   grid_pos_x  = i * 2 * oh * ow + 0 * oh * ow + k * ow + m;
                    int   grid_pos_y  = i * 2 * oh * ow + 1 * oh * ow + k * ow + m;
                    float grid_data_x = grid[grid_pos_x];
                    float grid_data_y = grid[grid_pos_y];
                    float grid_raw_data_x = (grid_data_x + 1.0f) * (0.5f * ((float)iw)) - 0.5f;
                    float grid_raw_data_y = (grid_data_y + 1.0f) * (0.5f * ((float)ih)) - 0.5f;

                    int   tmp_x_int = (int)(grid_raw_data_x);
                    int   tmp_y_int = (int)(grid_raw_data_y);

                    int   grid_raw_data_x_int = tmp_x_int - (((float)tmp_x_int) > grid_raw_data_x);
                    int   grid_raw_data_y_int = tmp_y_int - (((float)tmp_y_int) > grid_raw_data_y);
                    int   grid_raw_data_x_int_plus = grid_raw_data_x_int + 1;
                    int   grid_raw_data_y_int_plus = grid_raw_data_y_int + 1;
                    float data00 = 0.0f, data01 = 0.0f, data10 = 0.0f, data11 = 0.0f;
                    int   input_pos00 = i * c * ih * iw + j * ih * iw + grid_raw_data_y_int * iw + grid_raw_data_x_int;
                    int   input_pos01 = input_pos00 + 1;
                    int   input_pos10 = input_pos00 + iw;
                    int   input_pos11 = input_pos00 + iw + 1;
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data00 = src[input_pos00];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data01 = src[input_pos01];
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data10 = src[input_pos10];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data11 = src[input_pos11];

                    float u       = (grid_raw_data_x - grid_raw_data_x_int);
                    float v       = (grid_raw_data_y - grid_raw_data_y_int);
                    float res     = data00 * (1.0f - u) * (1.0f - v) + data01 * u * (1.0f - v) + data10 * (1.0f - u) * v + data11 * u * v;
                    int   dst_pos = i * c * oh * ow + j * oh * ow + k * ow + m;
                    dst[dst_pos]  = res;
                }
            }
        }
    }
}

static void sample_grid_bilinear_zero_pad_nc4hw4(float const*   src,
                                                 int            n,
                                                 int            c,
                                                 int            ih,
                                                 int            iw,
                                                 float const*   grid,
                                                 int            oh,
                                                 int            ow,
                                                 float*         dst)
{
    int  cq               = (c + 3) >> 2;
    int  input_line_size  = cq * 4 * iw;
    int  grid_line_size   = 4 * iw;
    int  output_line_size = cq * 4 * ow;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < oh; k++)
            {
                for (int m = 0; m < ow; m++)
                {
                    int   grid_pos_x      = i * oh * grid_line_size + k * grid_line_size + 4 * m;
                    int   grid_pos_y      = grid_pos_x + 1;
                    float grid_data_x     = grid[grid_pos_x];
                    float grid_data_y     = grid[grid_pos_y];
                    float grid_raw_data_x = (grid_data_x + 1.0f) * (0.5f * ((float)iw)) - 0.5f;
                    float grid_raw_data_y = (grid_data_y + 1.0f) * (0.5f * ((float)ih)) - 0.5f;

                    int   tmp_x_int = (int)(grid_raw_data_x);
                    int   tmp_y_int = (int)(grid_raw_data_y);

                    int   grid_raw_data_x_int      = tmp_x_int - (((float)tmp_x_int) > grid_raw_data_x);
                    int   grid_raw_data_y_int      = tmp_y_int - (((float)tmp_y_int) > grid_raw_data_y);
                    int   grid_raw_data_x_int_plus = grid_raw_data_x_int + 1;
                    int   grid_raw_data_y_int_plus = grid_raw_data_y_int + 1;
                    float data00 = 0.0f, data01 = 0.0f, data10 = 0.0f, data11 = 0.0f;
                    int   input_pos00 = i * ih * input_line_size + grid_raw_data_y_int      * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int      + (j & 3);
                    int   input_pos01 = i * ih * input_line_size + grid_raw_data_y_int      * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int_plus + (j & 3);
                    int   input_pos10 = i * ih * input_line_size + grid_raw_data_y_int_plus * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int      + (j & 3);
                    int   input_pos11 = i * ih * input_line_size + grid_raw_data_y_int_plus * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int_plus + (j & 3);
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data00 = src[input_pos00];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data01 = src[input_pos01];
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data10 = src[input_pos10];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data11 = src[input_pos11];

                    float u       = (grid_raw_data_x - grid_raw_data_x_int);
                    float v       = (grid_raw_data_y - grid_raw_data_y_int);
                    float res     = data00 * (1.0f - u) * (1.0f - v) + data01 * u * (1.0f - v) + data10 * (1.0f - u) * v + data11 * u * v;
                    int   dst_pos = i * oh * output_line_size + k * output_line_size + (j >> 2) * ow * 4 + 4 * m + (j&3);
                    dst[dst_pos]  = res;
                }
            }
        }
    }
}

static void sample_grid_bilinear_zero_pad_align_corner_nchw(float const*  src,
                                                            int           n,
                                                            int           c,
                                                            int           ih,
                                                            int           iw,
                                                            float const*  grid,
                                                            int           oh,
                                                            int           ow,
                                                            float*        dst)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < oh; k++)
            {
                for (int m = 0; m < ow; m++)
                {
                    int   grid_pos_x      = i * 2 * oh * ow + 0 * oh * ow + k * ow + m;
                    int   grid_pos_y      = i * 2 * oh * ow + 1 * oh * ow + k * ow + m;
                    float grid_data_x     = grid[grid_pos_x];
                    float grid_data_y     = grid[grid_pos_y];
                    float grid_raw_data_x = (grid_data_x + 1.0f) * (0.5f * ((float)(iw - 1)));
                    float grid_raw_data_y = (grid_data_y + 1.0f) * (0.5f * ((float)(ih - 1)));

                    int   tmp_x_int       = (int)(grid_raw_data_x);
                    int   tmp_y_int       = (int)(grid_raw_data_y);

                    int   grid_raw_data_x_int = tmp_x_int - (((float)tmp_x_int) > grid_raw_data_x);
                    int   grid_raw_data_y_int = tmp_y_int - (((float)tmp_y_int) > grid_raw_data_y);
                    int   grid_raw_data_x_int_plus = grid_raw_data_x_int + 1;
                    int   grid_raw_data_y_int_plus = grid_raw_data_y_int + 1;
                    float data00 = 0.0f, data01 = 0.0f, data10 = 0.0f, data11 = 0.0f;
                    int   input_pos00 = i * c * ih * iw + j * ih * iw + grid_raw_data_y_int * iw + grid_raw_data_x_int;
                    int   input_pos01 = input_pos00 + 1;
                    int   input_pos10 = input_pos00 + iw;
                    int   input_pos11 = input_pos00 + iw + 1;
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data00 = src[input_pos00];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data01 = src[input_pos01];
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data10 = src[input_pos10];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data11 = src[input_pos11];

                    float u = (grid_raw_data_x - grid_raw_data_x_int);
                    float v = (grid_raw_data_y - grid_raw_data_y_int);
                    float res = data00 * (1.0f - u) * (1.0f - v) + data01 * u * (1.0f - v) + data10 * (1.0f - u) * v + data11 * u * v;
                    int   dst_pos = i * c * oh * ow + j * oh * ow + k * ow + m;
                    dst[dst_pos] = res;
                }
            }
        }
    }
}

static void sample_grid_bilinear_zero_pad_align_corner_nc4hw4(float const*   src,
                                                              int            n,
                                                              int            c,
                                                              int            ih,
                                                              int            iw,
                                                              float const*   grid,
                                                              int            oh,
                                                              int            ow,
                                                              float*         dst)
{
    int  cq = (c + 3) >> 2;
    int  input_line_size = cq * 4 * iw;
    int  grid_line_size = 4 * iw;
    int  output_line_size = cq * 4 * ow;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < oh; k++)
            {
                for (int m = 0; m < ow; m++)
                {
                    int   grid_pos_x      = i * oh * grid_line_size + k * grid_line_size + 4 * m;
                    int   grid_pos_y      = grid_pos_x + 1;
                    float grid_data_x     = grid[grid_pos_x];
                    float grid_data_y     = grid[grid_pos_y];
                    float grid_raw_data_x = (grid_data_x + 1.0f) * (0.5f * ((float)(iw - 1)));
                    float grid_raw_data_y = (grid_data_y + 1.0f) * (0.5f * ((float)(ih - 1)));

                    int   tmp_x_int = (int)(grid_raw_data_x);
                    int   tmp_y_int = (int)(grid_raw_data_y);

                    int   grid_raw_data_x_int = tmp_x_int - (((float)tmp_x_int) > grid_raw_data_x);
                    int   grid_raw_data_y_int = tmp_y_int - (((float)tmp_y_int) > grid_raw_data_y);
                    int   grid_raw_data_x_int_plus = grid_raw_data_x_int + 1;
                    int   grid_raw_data_y_int_plus = grid_raw_data_y_int + 1;
                    float data00 = 0.0f, data01 = 0.0f, data10 = 0.0f, data11 = 0.0f;
                    int   input_pos00 = i * ih * input_line_size + grid_raw_data_y_int * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int + (j & 3);
                    int   input_pos01 = i * ih * input_line_size + grid_raw_data_y_int * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int_plus + (j & 3);
                    int   input_pos10 = i * ih * input_line_size + grid_raw_data_y_int_plus * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int + (j & 3);
                    int   input_pos11 = i * ih * input_line_size + grid_raw_data_y_int_plus * input_line_size + (j >> 2) * iw * 4 + 4 * grid_raw_data_x_int_plus + (j & 3);
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data00 = src[input_pos00];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int >= 0 && grid_raw_data_y_int < ih)
                        data01 = src[input_pos01];
                    if (grid_raw_data_x_int >= 0 && grid_raw_data_x_int < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data10 = src[input_pos10];
                    if (grid_raw_data_x_int_plus >= 0 && grid_raw_data_x_int_plus < iw && grid_raw_data_y_int_plus >= 0 && grid_raw_data_y_int_plus < ih)
                        data11 = src[input_pos11];

                    float u = (grid_raw_data_x - grid_raw_data_x_int);
                    float v = (grid_raw_data_y - grid_raw_data_y_int);
                    float res = data00 * (1.0f - u) * (1.0f - v) + data01 * u * (1.0f - v) + data10 * (1.0f - u) * v + data11 * u * v;
                    int   dst_pos = i * oh * output_line_size + k * output_line_size + (j >> 2) * ow * 4 + 4 * m + (j & 3);
                    dst[dst_pos] = res;
                }
            }
        }
    }
}

ErrorCode CPUGridSample::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &input  = inputs[0]->buffer();
    auto &output = outputs[0]->buffer();

    auto in_fmt  = MNN::TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto out_fmt = MNN::TensorUtils::getDescribe(outputs[0])->dimensionFormat;

    float const*  src_ptr       = inputs[0]->host<float>();
    float const*  grid_ptr      = inputs[1]->host<float>();
    float*        dst_ptr       = outputs[0]->host<float>();
    int           input_batch   = inputs[0]->batch();
    int           input_height  = inputs[0]->height();
    int           input_width   = inputs[0]->width();
    int           input_channel = inputs[0]->channel();
    int           output_height = inputs[1]->height();
    int           output_width  = inputs[1]->width();

    if (MNN::MNN_DATA_FORMAT_NCHW == in_fmt)
    {
        if (0 == mode_)
        {
            if (0 == padding_mode_)
            {
                if (false == align_corners_)
                {
                    sample_grid_bilinear_zero_pad_nchw(src_ptr, 
                                                        input_batch, 
                                                        input_channel, 
                                                        input_height, 
                                                        input_width, 
                                                        grid_ptr, 
                                                        output_height, 
                                                        output_width, 
                                                        dst_ptr);
                }
                else
                {
                    sample_grid_bilinear_zero_pad_align_corner_nchw(src_ptr,
                                                                    input_batch,
                                                                    input_channel,
                                                                    input_height,
                                                                    input_width,
                                                                    grid_ptr,
                                                                    output_height,
                                                                    output_width,
                                                                    dst_ptr);
                }
            }
        }
    }
    else if (MNN::MNN_DATA_FORMAT_NC4HW4 == in_fmt)
    {
        if (0 == mode_)
        {
            if (0 == padding_mode_)
            {
                if (false == align_corners_)
                {
                    sample_grid_bilinear_zero_pad_nc4hw4(src_ptr,
                                                            input_batch,
                                                            input_channel,
                                                            input_height,
                                                            input_width,
                                                            grid_ptr,
                                                            output_height,
                                                            output_width,
                                                            dst_ptr);
                }
                else
                {
                    sample_grid_bilinear_zero_pad_align_corner_nchw(src_ptr,
                                                                    input_batch,
                                                                    input_channel,
                                                                    input_height,
                                                                    input_width,
                                                                    grid_ptr,
                                                                    output_height,
                                                                    output_width,
                                                                    dst_ptr);
                }
            }
        }
    }

    return NO_ERROR;
}

ErrorCode CPUGridSample::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) 
{
    return NO_ERROR;
}

class CPUGridSampleCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto grid_sample = op->main_as_GridSample();

		auto in0_fmt = MNN::TensorUtils::getDescribe(inputs[0])->dimensionFormat;
		auto in1_fmt = MNN::TensorUtils::getDescribe(inputs[1])->dimensionFormat;
		auto out_fmt = MNN::TensorUtils::getDescribe(outputs[0])->dimensionFormat;

		if (in0_fmt != in1_fmt || in0_fmt != out_fmt)
			return nullptr;

		if (MNN::MNN_DATA_FORMAT_NCHW != in0_fmt && MNN::MNN_DATA_FORMAT_NC4HW4 != in0_fmt)
			return nullptr;

        return new CPUGridSample(backend, 
                                 grid_sample->mode(),
                                 grid_sample->padding_mode(), 
                                 grid_sample->align_corners());
    }
};
REGISTER_CPU_OP_CREATOR(CPUGridSampleCreator, OpType_GridSample);

} // namespace MNN
