#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define  ACTIVE_RELU(res0, res1, res2, res3, res4, res5, res6, res7)           \
    res0 = fmax(res0, (FLOAT4)0);                                              \
    res1 = fmax(res1, (FLOAT4)0);                                              \
    res2 = fmax(res2, (FLOAT4)0);                                              \
    res3 = fmax(res3, (FLOAT4)0);                                              \
    res4 = fmax(res4, (FLOAT4)0);                                              \
    res5 = fmax(res5, (FLOAT4)0);                                              \
    res6 = fmax(res6, (FLOAT4)0);                                              \
    res7 = fmax(res7, (FLOAT4)0);

#define  ACTIVE_RELU6(res0, res1, res2, res3, res4, res5, res6, res7)           \
    res0 = clamp(res0, (FLOAT4)0, (FLOAT4)6);                                   \
    res1 = clamp(res1, (FLOAT4)0, (FLOAT4)6);                                   \
    res2 = clamp(res2, (FLOAT4)0, (FLOAT4)6);                                   \
    res3 = clamp(res3, (FLOAT4)0, (FLOAT4)6);                                   \
    res4 = clamp(res4, (FLOAT4)0, (FLOAT4)6);                                   \
    res5 = clamp(res5, (FLOAT4)0, (FLOAT4)6);                                   \
    res6 = clamp(res6, (FLOAT4)0, (FLOAT4)6);                                   \
    res7 = clamp(res7, (FLOAT4)0, (FLOAT4)6);

#define  ACTIVE_LEAKY_RELU(res0, res1, res2, res3, res4, res5, res6, res7)           \
    res0 = select(((FLOAT)(LEAKY_RELU))*res0, res0, res0 >= (FLOAT4)0);              \
    res1 = select(((FLOAT)(LEAKY_RELU))*res1, res1, res1 >= (FLOAT4)0);              \
    res2 = select(((FLOAT)(LEAKY_RELU))*res2, res2, res2 >= (FLOAT4)0);              \
    res3 = select(((FLOAT)(LEAKY_RELU))*res3, res3, res3 >= (FLOAT4)0);              \
    res4 = select(((FLOAT)(LEAKY_RELU))*res4, res4, res4 >= (FLOAT4)0);              \
    res5 = select(((FLOAT)(LEAKY_RELU))*res5, res5, res5 >= (FLOAT4)0);              \
    res6 = select(((FLOAT)(LEAKY_RELU))*res6, res6, res6 >= (FLOAT4)0);              \
    res7 = select(((FLOAT)(LEAKY_RELU))*res7, res7, res7 >= (FLOAT4)0);

#define CAL_BN_AFTER_RELU(res0, res1, res2, res3, bias_img, output_channel_q, output_channel_q_idx)  \
    { \
        FLOAT4  scale_data = RI_F(bias_img, SAMPLER, (int2)(output_channel_q + output_channel_q_idx, 0)); \
		FLOAT4  bias_data  = RI_F(bias_img, SAMPLER, (int2)(mad24(output_channel_q, 2, output_channel_q_idx), 0)); \
		res0               = mad(res0, scale_data, bias_data); \
		res1               = mad(res1, scale_data, bias_data); \
		res2               = mad(res2, scale_data, bias_data); \
		res3               = mad(res3, scale_data, bias_data); \
    }

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_s1x1_dilation_4row(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
	                            __read_only image2d_t  weights,
#ifdef BIAS
	                            __read_only image2d_t  bias,
#endif
	                            __write_only image2d_t output,
	                            __private const int2   input_shape,
	                            __private const int    in_channel_block_length,
	                            __private const int2   output_shape,
	                            __private const int2   weights_shape,
	                            __private const int2   padding_shape,
	                            __private const int2   dilation_shape,
	                            __private const int    out_channel_q)
{
	const int output_channel_width_idx   = get_global_id(0);
	const int output_batch_height_idx    = get_global_id(1);
	DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

	const int  out_height_q              = ((output_shape.x + 3) >> 2);
	const int  out_channel_blk_idx       = (output_channel_width_idx / output_shape.y);
	const int  output_channel_block_idx0 = out_channel_blk_idx << 1;
	const int  output_channel_block_idx1 = select(output_channel_block_idx0 + 1, -1, output_channel_block_idx0 + 1 >= out_channel_q);
	const int  out_x_pos                 = output_channel_width_idx - mul24(out_channel_blk_idx, output_shape.y);
	const int  out_batch_idx             = output_batch_height_idx / out_height_q;
	int        out_batch_base            = mul24(out_batch_idx, out_height_q);
	const int  out_y_pos                 = (output_batch_height_idx - out_batch_base) << 2;
	out_batch_base                       = mul24(out_batch_idx, output_shape.x);
#ifdef BIAS
	FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx0, 0));
	FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx1, 0));
#else
	FLOAT4 out0 = (FLOAT4)0;
	FLOAT4 out4 = (FLOAT4)0;
#endif
	FLOAT4 out1 = out0;
	FLOAT4 out2 = out0;
	FLOAT4 out3 = out0;
	FLOAT4 out5 = out4;
	FLOAT4 out6 = out4;
	FLOAT4 out7 = out4;

	const int in_height_idx    = out_y_pos - padding_shape.x;
	const int weigth_size      = mul24(weights_shape.y, weights_shape.x);
	const int weights_y_base0  = mul24(output_channel_block_idx0, weigth_size);
	const int weights_y_base1  = mul24(output_channel_block_idx1, weigth_size);
	int4      in_y_ofs         = (int4)(in_height_idx, in_height_idx + 1, in_height_idx + 2, in_height_idx + 3);
	FLOAT4 in0, in1, in2, in3;
	FLOAT4 weights0, weights1, weights2, weights3;
	int in_channel_block_idx = 0;
#ifdef IN_CHANNEL_LOOP
	for (in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
#endif
		int       in_width_idx  = out_x_pos - padding_shape.y;
		const int in_x_base     = mul24(in_channel_block_idx, input_shape.y);
		int       weights_x_idx = in_channel_block_idx << 2;
		for (int w = 0; w < weights_shape.y; w++)
		{
			int cur_x_pos  = select(in_x_base + in_width_idx, -1, (in_width_idx < 0 || in_width_idx >= input_shape.y));
			in_width_idx  += dilation_shape.y;
			for (int h = 0; h < weights_shape.x; h++)
			{
				int4 cur_in_y_pos = mad24(dilation_shape.x, h, in_y_ofs);
				cur_in_y_pos      = select(((int4)out_batch_base) + cur_in_y_pos, (int4)(-1), (cur_in_y_pos < (int4)0 || cur_in_y_pos >= (int4)(input_shape.x)));
				in0 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos.x));
				in1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos.y));
				in2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos.z));
				in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos.w));

				int weight_y_idx0 = weights_y_base0 + mad24(h, weights_shape.y, w);
				int weight_y_idx1 = select(weights_y_base1 + mad24(h, weights_shape.y, w), -1, output_channel_block_idx1 < 0);
				weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weight_y_idx0));
				weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weight_y_idx0));
				weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weight_y_idx0));
				weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weight_y_idx0));

				out0 = mad(in0.x, weights0, out0);
#ifdef INPUT_CHANNEL_2 
				out0 = mad(in0.y, weights1, out0);
#endif
#ifdef INPUT_CHANNEL_3 
				out0 = mad(in0.z, weights2, out0);
#endif
#ifdef INPUT_CHANNEL_4
				out0 = mad(in0.w, weights3, out0);
#endif

				out1 = mad(in1.x, weights0, out1);
#ifdef INPUT_CHANNEL_2
				out1 = mad(in1.y, weights1, out1);
#endif        
#ifdef INPUT_CHANNEL_3
				out1 = mad(in1.z, weights2, out1);
#endif
#ifdef INPUT_CHANNEL_4     
				out1 = mad(in1.w, weights3, out1);
#endif

				out2 = mad(in2.x, weights0, out2);
#ifdef INPUT_CHANNEL_2
				out2 = mad(in2.y, weights1, out2);
#endif
#ifdef INPUT_CHANNEL_3
				out2 = mad(in2.z, weights2, out2);
#endif
#ifdef INPUT_CHANNEL_4
				out2 = mad(in2.w, weights3, out2);
#endif

				out3 = mad(in3.x, weights0, out3);
#ifdef INPUT_CHANNEL_2
				out3 = mad(in3.y, weights1, out3);
#endif
#ifdef INPUT_CHANNEL_3
				out3 = mad(in3.z, weights2, out3);
#endif
#ifdef INPUT_CHANNEL_4
				out3 = mad(in3.w, weights3, out3);
#endif

				weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weight_y_idx1));
				weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weight_y_idx1));
				weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weight_y_idx1));
				weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weight_y_idx1));

				out4 = mad(in0.x, weights0, out4);
#ifdef INPUT_CHANNEL_2 
				out4 = mad(in0.y, weights1, out4);
#endif
#ifdef INPUT_CHANNEL_3 
				out4 = mad(in0.z, weights2, out4);
#endif
#ifdef INPUT_CHANNEL_4
				out4 = mad(in0.w, weights3, out4);
#endif

				out5 = mad(in1.x, weights0, out5);
#ifdef INPUT_CHANNEL_2
				out5 = mad(in1.y, weights1, out5);
#endif        
#ifdef INPUT_CHANNEL_3
				out5 = mad(in1.z, weights2, out5);
#endif
#ifdef INPUT_CHANNEL_4     
				out5 = mad(in1.w, weights3, out5);
#endif

				out6 = mad(in2.x, weights0, out6);
#ifdef INPUT_CHANNEL_2
				out6 = mad(in2.y, weights1, out6);
#endif
#ifdef INPUT_CHANNEL_3
				out6 = mad(in2.z, weights2, out6);
#endif
#ifdef INPUT_CHANNEL_4
				out6 = mad(in2.w, weights3, out6);
#endif

				out7 = mad(in3.x, weights0, out7);
#ifdef INPUT_CHANNEL_2
				out7 = mad(in3.y, weights1, out7);
#endif
#ifdef INPUT_CHANNEL_3
				out7 = mad(in3.z, weights2, out7);
#endif
#ifdef INPUT_CHANNEL_4
				out7 = mad(in3.w, weights3, out7);
#endif
			}
		}
#ifdef IN_CHANNEL_LOOP
	}
#endif

#ifdef RELU
    ACTIVE_RELU(out0, out1, out2, out3, out4, out5, out6, out7);
#endif

#ifdef RELU6
	ACTIVE_RELU6(out0, out1, out2, out3, out4, out5, out6, out7);
#endif

#ifdef LEAKY_RELU
	ACTIVE_LEAKY_RELU(out0, out1, out2, out3, out4, out5, out6, out7);
#endif

#ifdef BN_AFTER_RELU
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, out_channel_q, output_channel_block_idx0);
	if (output_channel_block_idx1 >= 0)
	{
		CAL_BN_AFTER_RELU(out4, out5, out6, out7, bias, out_channel_q, output_channel_block_idx1);
	}
#endif

	const int remain      = output_shape.x - out_y_pos;
	const int output_y    = out_batch_base + out_y_pos;
	const int out_x_base0 = mul24(output_channel_block_idx0, output_shape.y);
	const int out_x_base1 = select(mul24(output_channel_block_idx1, output_shape.y), -1, output_channel_block_idx1 < 0);
	const int output_idx0 = out_x_base0 + out_x_pos;
	const int output_idx1 = out_x_base1 + out_x_pos;
	if (remain >= 4) {
		WI_F(output, (int2)(output_idx0, output_y),     out0);
		WI_F(output, (int2)(output_idx0, output_y + 1), out1);
		WI_F(output, (int2)(output_idx0, output_y + 2), out2);
		WI_F(output, (int2)(output_idx0, output_y + 3), out3);
		if (out_x_base1 >= 0)
		{
			WI_F(output, (int2)(output_idx1, output_y),     out4);
			WI_F(output, (int2)(output_idx1, output_y + 1), out5);
			WI_F(output, (int2)(output_idx1, output_y + 2), out6);
			WI_F(output, (int2)(output_idx1, output_y + 3), out7);
		}
	}
	else if (remain == 3) {
		WI_F(output, (int2)(output_idx0, output_y),     out0);
		WI_F(output, (int2)(output_idx0, output_y + 1), out1);
		WI_F(output, (int2)(output_idx0, output_y + 2), out2);
		if (out_x_base1 >= 0)
		{
			WI_F(output, (int2)(output_idx1, output_y),     out4);
			WI_F(output, (int2)(output_idx1, output_y + 1), out5);
			WI_F(output, (int2)(output_idx1, output_y + 2), out6);
		}
	}
	else if (remain == 2) {
		WI_F(output, (int2)(output_idx0, output_y),     out0);
		WI_F(output, (int2)(output_idx0, output_y + 1), out1);
		if (out_x_base1 >= 0)
		{
			WI_F(output, (int2)(output_idx1, output_y),     out4);
			WI_F(output, (int2)(output_idx1, output_y + 1), out5);
		}
	}
	else if (remain == 1) {
		WI_F(output, (int2)(output_idx0, output_y), out0);
		if (out_x_base1 >= 0)
		{
			WI_F(output, (int2)(output_idx1, output_y), out4);
		}
	}
}