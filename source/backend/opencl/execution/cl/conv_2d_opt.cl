#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);    

#define CALCULATE_OUTPUT_IO(i_idx, o_idx)    \
    out##o_idx = mad(in##i_idx.x, weights0, out##o_idx); \
    out##o_idx = mad(in##i_idx.y, weights1, out##o_idx); \
    out##o_idx = mad(in##i_idx.z, weights2, out##o_idx); \
    out##o_idx = mad(in##i_idx.w, weights3, out##o_idx); 

#define OUTPUT_CONV1x1(res0, res1, res2, res3, col_idx , channel_q_idx, input_width) \
	const int pix_pos  = (col_idx << 2);                                             \
	const int pix_x[4] = {                                                           \
		pix_pos       % input_width,                                                 \
		(pix_pos + 1) % input_width,                                                 \
		(pix_pos + 2) % input_width,                                                 \
		(pix_pos + 3) % input_width                                                  \
	};                                                                               \
	const int pix_y[4] = {                                                           \
		pix_pos       / input_width,                                                 \
		(pix_pos + 1) / input_width,                                                 \
		(pix_pos + 2) / input_width,                                                 \
		(pix_pos + 3) / input_width                                                  \
	};                                                                               \
	const int output_start_x = channel_q_idx * input_width;                          \
	WI_F(output, (int2)(output_start_x + pix_x[0], pix_y[0]), res0);                 \
	WI_F(output, (int2)(output_start_x + pix_x[1], pix_y[1]), res1);                 \
	WI_F(output, (int2)(output_start_x + pix_x[2], pix_y[2]), res2);                 \
	WI_F(output, (int2)(output_start_x + pix_x[3], pix_y[3]), res3);

#define OUTPUT_CONV1x1_CHECK_IMAGE_BORDER(res0, res1, res2, res3, col_idx , channel_q_idx, input_width, image_size)    \
	const int pix_pos  = (col_idx << 2);                                                                               \
	const int pix_x[4] = {                                                                                             \
		pix_pos       % input_width,                                                                                   \
		(pix_pos + 1) % input_width,                                                                                   \
		(pix_pos + 2) % input_width,                                                                                   \
		(pix_pos + 3) % input_width                                                                                    \
	};                                                                                                                 \
	const int pix_y[4] = {                                                                                             \
		pix_pos       / input_width,                                                                                   \
		(pix_pos + 1) / input_width,                                                                                   \
		(pix_pos + 2) / input_width,                                                                                   \
		(pix_pos + 3) / input_width                                                                                    \
	};                                                                                                                 \
	const int output_start_x = channel_q_idx * input_width;                                                            \
    if(pix_pos + 3 < image_size)                                                                                       \
    {                                                                                                                  \
        WI_F(output, (int2)(output_start_x + pix_x[0], pix_y[0]), res0);                                               \
        WI_F(output, (int2)(output_start_x + pix_x[1], pix_y[1]), res1);                                               \
        WI_F(output, (int2)(output_start_x + pix_x[2], pix_y[2]), res2);                                               \
        WI_F(output, (int2)(output_start_x + pix_x[3], pix_y[3]), res3);                                               \
    }                                                                                                                  \
    else if(pix_pos + 2 < image_size)                                                                                  \
    {                                                                                                                  \
        WI_F(output, (int2)(output_start_x + pix_x[0], pix_y[0]), res0);                                               \
        WI_F(output, (int2)(output_start_x + pix_x[1], pix_y[1]), res1);                                               \
        WI_F(output, (int2)(output_start_x + pix_x[2], pix_y[2]), res2);                                               \
    }                                                                                                                  \
    else if(pix_pos + 1 < image_size)                                                                                  \
    {                                                                                                                  \
        WI_F(output, (int2)(output_start_x + pix_x[0], pix_y[0]), res0);                                               \
        WI_F(output, (int2)(output_start_x + pix_x[1], pix_y[1]), res1);                                               \
    }                                                                                                                  \
    else if(pix_pos < image_size)                                                                                      \
    {                                                                                                                  \
        WI_F(output, (int2)(output_start_x + pix_x[0], pix_y[0]), res0);                                               \
    }

#define READ_INPUT_IMAGE(i, base)                                                                         \
    int in_width_value##i = in_width##i + base;                                                           \
    in_width_value##i =                                                                                   \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_shape.y)); \
    in##i = RI_F(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));

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

#define TILE_DIM8 8

#define CAL_BN_AFTER_RELU(res0, res1, res2, res3, bias_img, output_channel_q, output_channel_q_idx)  \
    { \
        FLOAT4  scale_data = RI_F(bias_img, SAMPLER, (int2)(output_channel_q + output_channel_q_idx,          0)); \
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
void conv_2d_k1_s1_dual_output(GLOBAL_SIZE_2_DIMS 
                               __read_only image2d_t input, 
							   __read_only image2d_t weights,
#ifdef BIAS						   
                               __read_only image2d_t bias,
#endif							   
                               __write_only image2d_t output,
                               __private const int2 input_shape,
                               __private const int  in_channel_q, 
							   __private const int2 output_shape,
							   __private const int  out_channel_q,
                               __private const int  output_width_q) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int  output_channel_block_idx  = (output_channel_width_idx / output_width_q) << 1;
    const int  output_channel_block_idx2 = select(output_channel_block_idx + 1, -1, output_channel_block_idx + 1 >= out_channel_q);
	const int  output_width_block_idx    = output_channel_width_idx % output_width_q;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx, 0));
	FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx2, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;
    FLOAT4 out5 = out4;
    FLOAT4 out6 = out4;
    FLOAT4 out7 = out4;
#else
	FLOAT4 out0 = (FLOAT4)(0.0f);
    FLOAT4 out1 = (FLOAT4)(0.0f);
    FLOAT4 out2 = (FLOAT4)(0.0f);
    FLOAT4 out3 = (FLOAT4)(0.0f);
	FLOAT4 out4 = (FLOAT4)(0.0f);
    FLOAT4 out5 = (FLOAT4)(0.0f);
    FLOAT4 out6 = (FLOAT4)(0.0f);
    FLOAT4 out7 = (FLOAT4)(0.0f);
#endif

    int intput_width_idx0 = (output_width_block_idx << 2);
    int intput_width_idx1 = intput_width_idx0 + 1;
    int intput_width_idx2 = intput_width_idx0 + 2;
    int intput_width_idx3 = intput_width_idx0 + 3;

	int4 input_width_pos       = (int4)(intput_width_idx0, intput_width_idx1, intput_width_idx2, intput_width_idx3);
    input_width_pos            = select(input_width_pos, (int4)(INT_MIN), input_width_pos >= (int4)(input_shape.y));

    int batch_index            = output_batch_height_idx / output_shape.x;
    int input_height_block_idx = mul24((output_batch_height_idx % output_shape.x), 1) + batch_index * input_shape.x;
	
	__private FLOAT4 in0;
	__private FLOAT4 in1;
	__private FLOAT4 in2;
	__private FLOAT4 in3;
	__private FLOAT4 weights0;
	__private FLOAT4 weights1;
	__private FLOAT4 weights2;
	__private FLOAT4 weights3;
	int4  weights_width_pos    = (int4)(0, 1, 2, 3);
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_q; ++in_channel_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_width_pos.x, input_height_block_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_width_pos.y, input_height_block_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_width_pos.z, input_height_block_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_width_pos.w, input_height_block_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.x, output_channel_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.y, output_channel_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.z, output_channel_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.w, output_channel_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
		
        weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.x, output_channel_block_idx2));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.y, output_channel_block_idx2));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.z, output_channel_block_idx2));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.w, output_channel_block_idx2));
		
		CALCULATE_OUTPUT_IO(0, 4);
		CALCULATE_OUTPUT_IO(1, 5);
		CALCULATE_OUTPUT_IO(2, 6);
		CALCULATE_OUTPUT_IO(3, 7);
		
		weights_width_pos += (int4)(4);
		input_width_pos   += (int4)(input_shape.y);
    }

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
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, out_channel_q, output_channel_block_idx);
	if (output_channel_block_idx2 >= 0)
	{
		CAL_BN_AFTER_RELU(out4, out5, out6, out7, bias, out_channel_q, output_channel_block_idx2);
	}
#endif

    const int out_x_base  = mul24(output_channel_block_idx, output_shape.y);
	const int out_x_base2 = select(mul24(output_channel_block_idx2, output_shape.y), -1, output_channel_block_idx2 < 0);
    int out_x_idx         = output_width_block_idx << 2;

    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base  + out_x_idx;
	int output_idx2  = out_x_base2 + out_x_idx;
    if (remain >= 4)
	{
        WI_F(output, (int2)(output_idx,     output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out3);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
			WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
			WI_F(output, (int2)(output_idx2 + 3, output_batch_height_idx), out7);
		}
    } 
	else if (remain == 3) {
        WI_F(output, (int2)(output_idx,     output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
			WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
		}
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx,     output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
		}
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
		}
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_k1_s1_2_in_row_2_col_4_dual_out(GLOBAL_SIZE_2_DIMS 
                                             __read_only image2d_t input, 
                                             __read_only image2d_t weights,
#ifdef BIAS							   
                                             __read_only image2d_t bias,
#endif
                                             __write_only image2d_t output,
                                             __private const int2 input_shape,
                                             __private const int  in_channel_q, 
                                             __private const int2 output_shape,
                                             __private const int  out_channel_q,
                                             __private const int  out_width_q) 
{
    const int output_channel_width_idx = get_global_id(0);
    const int output_row_idx           = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_row_idx);

    const int  output_channel_block_idx  = output_channel_width_idx / out_width_q;
	const int  output_width_block_idx    = output_channel_width_idx % out_width_q;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;
	FLOAT4 out4 = out0;
    FLOAT4 out5 = out0;
    FLOAT4 out6 = out0;
    FLOAT4 out7 = out0;
#else
	FLOAT4 out0 = (FLOAT4)(0.0f);
    FLOAT4 out1 = (FLOAT4)(0.0f);
    FLOAT4 out2 = (FLOAT4)(0.0f);
    FLOAT4 out3 = (FLOAT4)(0.0f);
	FLOAT4 out4 = (FLOAT4)(0.0f);
    FLOAT4 out5 = (FLOAT4)(0.0f);
    FLOAT4 out6 = (FLOAT4)(0.0f);
    FLOAT4 out7 = (FLOAT4)(0.0f);
#endif

    int  input_col_idx         = output_width_block_idx << 2;
	int4 input_col_idx4        = (int4)(input_col_idx, input_col_idx + 1, input_col_idx + 2, input_col_idx + 3);
	int  input_row_idx_0       = (output_row_idx << 1);
	int2 input_row_idx         = (int2)(input_row_idx_0, input_row_idx_0 + 1);
	input_col_idx4             = select(input_col_idx4, (int4)(INT_MIN), input_col_idx4 >= (int4)(input_shape.y));
	input_row_idx              = select(input_row_idx,  (int2)(INT_MIN), input_row_idx  >= (int2)(input_shape.x));
	
	int4  weights_width_pos    = (int4)(0, 1, 2, 3);
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_q;  ++in_channel_block_idx) 
	{
		__private const FLOAT4 weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.x, output_channel_block_idx));
		__private const FLOAT4 weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.y, output_channel_block_idx));
		__private const FLOAT4 weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.z, output_channel_block_idx));
		__private const FLOAT4 weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.w, output_channel_block_idx));
		
		__private const FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(input_col_idx4.x, input_row_idx.x));
		__private const FLOAT4 in1      = RI_F(input, SAMPLER, (int2)(input_col_idx4.y, input_row_idx.x));
		__private const FLOAT4 in2      = RI_F(input, SAMPLER, (int2)(input_col_idx4.z, input_row_idx.x));
		__private const FLOAT4 in3      = RI_F(input, SAMPLER, (int2)(input_col_idx4.w, input_row_idx.x));
		
        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

		__private const FLOAT4 in4 = RI_F(input, SAMPLER, (int2)(input_col_idx4.x, input_row_idx.y));
		__private const FLOAT4 in5 = RI_F(input, SAMPLER, (int2)(input_col_idx4.y, input_row_idx.y));
		__private const FLOAT4 in6 = RI_F(input, SAMPLER, (int2)(input_col_idx4.z, input_row_idx.y));
		__private const FLOAT4 in7 = RI_F(input, SAMPLER, (int2)(input_col_idx4.w, input_row_idx.y));

        CALCULATE_OUTPUT(4);
        CALCULATE_OUTPUT(5);
        CALCULATE_OUTPUT(6);
        CALCULATE_OUTPUT(7);
		
		weights_width_pos += (int4)(4);
		input_col_idx4    += input_shape.y;
    }

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
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, out_channel_q, output_channel_block_idx);
	CAL_BN_AFTER_RELU(out4, out5, out6, out7, bias, out_channel_q, output_channel_block_idx);
#endif

    const int out_x_base  = mul24(output_channel_block_idx, output_shape.y);
    const int remain      = output_shape.y - input_col_idx;
    int output_idx        = out_x_base + input_col_idx;
    if (remain >= 4)
	{
        WI_F(output, (int2)(output_idx,     input_row_idx.x), out0);
        WI_F(output, (int2)(output_idx + 1, input_row_idx.x), out1);
        WI_F(output, (int2)(output_idx + 2, input_row_idx.x), out2);
        WI_F(output, (int2)(output_idx + 3, input_row_idx.x), out3);
		if(input_row_idx.y >= 0)
		{
			WI_F(output, (int2)(output_idx,     input_row_idx.y), out4);
			WI_F(output, (int2)(output_idx + 1, input_row_idx.y), out5);
			WI_F(output, (int2)(output_idx + 2, input_row_idx.y), out6);
			WI_F(output, (int2)(output_idx + 3, input_row_idx.y), out7);
		}
    } 
	else if (remain == 3) {
        WI_F(output, (int2)(output_idx,     input_row_idx.x), out0);
        WI_F(output, (int2)(output_idx + 1, input_row_idx.x), out1);
        WI_F(output, (int2)(output_idx + 2, input_row_idx.x), out2);
		if(input_row_idx.y >= 0)
		{
			WI_F(output, (int2)(output_idx,     input_row_idx.y), out4);
			WI_F(output, (int2)(output_idx + 1, input_row_idx.y), out5);
			WI_F(output, (int2)(output_idx + 2, input_row_idx.y), out6);
		}
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx,     input_row_idx.x), out0);
        WI_F(output, (int2)(output_idx + 1, input_row_idx.x), out1);
		if(input_row_idx.y >= 0)
		{
			WI_F(output, (int2)(output_idx,     input_row_idx.y), out4);
			WI_F(output, (int2)(output_idx + 1, input_row_idx.y), out5);
		}
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx,     input_row_idx.x), out0);
		if(input_row_idx.y >= 0)
		{
			WI_F(output, (int2)(output_idx,     input_row_idx.y), out4);
		}
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_k1_s1_4row_dual_output(GLOBAL_SIZE_2_DIMS 
                                    __read_only image2d_t input, 
                                    __read_only image2d_t weights,
#ifdef BIAS							   
                                    __read_only image2d_t bias,
#endif
                                    __write_only image2d_t output,
                                    __private const int2 input_shape,
                                    __private const int  in_channel_q, 
                                    __private const int2 output_shape,
                                    __private const int  out_channel_q,
                                    __private const int  output_height_q) 
{
    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

	const int  out_height_q              = (input_shape.x + 3) >> 2;
	const int  batch_idx                 = output_batch_height_idx / out_height_q;
	int        batch_base                = mul24(batch_idx, out_height_q);
	const int  out_height_blk            = (output_batch_height_idx - batch_base) << 2;
	const int  blk_idx                   = output_channel_width_idx / input_shape.y;
	const int  output_channel_block_idx0 = blk_idx << 1;
	const int  output_channel_block_idx1 = select(output_channel_block_idx0 + 1, -1, (output_channel_block_idx0 + 1) >= out_channel_q);
	const int  output_width_idx          = output_channel_width_idx - mul24(blk_idx, input_shape.y);

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx0, 0));
	FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx1, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;
    FLOAT4 out5 = out4;
    FLOAT4 out6 = out4;
    FLOAT4 out7 = out4;
#else
	FLOAT4 out0 = (FLOAT4)(0.0f);
    FLOAT4 out1 = (FLOAT4)(0.0f);
    FLOAT4 out2 = (FLOAT4)(0.0f);
    FLOAT4 out3 = (FLOAT4)(0.0f);
	FLOAT4 out4 = (FLOAT4)(0.0f);
    FLOAT4 out5 = (FLOAT4)(0.0f);
    FLOAT4 out6 = (FLOAT4)(0.0f);
    FLOAT4 out7 = (FLOAT4)(0.0f);
#endif

    int intput_height_idx0 = (output_batch_height_idx << 2);
    int intput_height_idx1 = intput_height_idx0 + 1;
    int intput_height_idx2 = intput_height_idx0 + 2;
    int intput_height_idx3 = intput_height_idx0 + 3;

	int  input_width_pos        = output_width_idx;
	int4 input_height_pos       = (int4)(intput_height_idx0, intput_height_idx1, intput_height_idx2, intput_height_idx3);
    input_height_pos            = select(input_height_pos, (int4)(INT_MIN), input_height_pos >= (int4)(input_shape.x));

	int4  weights_width_pos     = (int4)(0, 1, 2, 3);
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_q; ++in_channel_block_idx) {

		__private FLOAT4 weights0;
		__private FLOAT4 weights1;
		__private FLOAT4 weights2;
		__private FLOAT4 weights3;
	
        __private const FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(input_width_pos, input_height_pos.x));
        __private const FLOAT4 in1 = RI_F(input, SAMPLER, (int2)(input_width_pos, input_height_pos.y));
        __private const FLOAT4 in2 = RI_F(input, SAMPLER, (int2)(input_width_pos, input_height_pos.z));
        __private const FLOAT4 in3 = RI_F(input, SAMPLER, (int2)(input_width_pos, input_height_pos.w));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.x, output_channel_block_idx0));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.y, output_channel_block_idx0));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.z, output_channel_block_idx0));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.w, output_channel_block_idx0));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
		
        weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.x, output_channel_block_idx1));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.y, output_channel_block_idx1));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.z, output_channel_block_idx1));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_pos.w, output_channel_block_idx1));
		
		CALCULATE_OUTPUT_IO(0, 4);
		CALCULATE_OUTPUT_IO(1, 5);
		CALCULATE_OUTPUT_IO(2, 6);
		CALCULATE_OUTPUT_IO(3, 7);
		
		weights_width_pos += (int4)(4);
		input_width_pos   += input_shape.y;
    }

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

    const int out_x_base0 = mul24(output_channel_block_idx0, input_shape.y);
    const int out_x_base1 = select(mul24(output_channel_block_idx1, input_shape.y), -1, output_channel_block_idx1 < 0);

    const int remain       = input_shape.x - out_height_blk;
    int       output_idx0  = out_x_base0 + output_width_idx;
    int       output_idx1  = out_x_base1 + output_width_idx;
    int       output_y_idx = batch_base + intput_height_idx0;
    if (remain >= 4)
    {
        WI_F(output, (int2)(output_idx0, output_y_idx),    out0);
        WI_F(output, (int2)(output_idx0, output_y_idx + 1), out1);
        WI_F(output, (int2)(output_idx0, output_y_idx + 2), out2);
        WI_F(output, (int2)(output_idx0, output_y_idx + 3), out3);
        if (out_x_base1 >= 0)
        {
            WI_F(output, (int2)(output_idx1, output_y_idx),     out4);
            WI_F(output, (int2)(output_idx1, output_y_idx + 1), out5);
            WI_F(output, (int2)(output_idx1, output_y_idx + 2), out6);
            WI_F(output, (int2)(output_idx1, output_y_idx + 3), out7);
        }
    }
    else if (remain == 3) {
        WI_F(output, (int2)(output_idx0, output_y_idx),     out0);
        WI_F(output, (int2)(output_idx0, output_y_idx + 1), out1);
        WI_F(output, (int2)(output_idx0, output_y_idx + 2), out2);
        if (out_x_base1 >= 0)
        {
            WI_F(output, (int2)(output_idx1, output_y_idx),     out4);
            WI_F(output, (int2)(output_idx1, output_y_idx + 1), out5);
            WI_F(output, (int2)(output_idx1, output_y_idx + 2), out6);
        }
    }
    else if (remain == 2) {
        WI_F(output, (int2)(output_idx0, output_y_idx),     out0);
        WI_F(output, (int2)(output_idx0, output_y_idx + 1), out1);
        if (out_x_base1 >= 0)
        {
            WI_F(output, (int2)(output_idx1, output_y_idx),     out4);
            WI_F(output, (int2)(output_idx1, output_y_idx + 1), out5);
        }
	}
    else if (remain == 1) {
        WI_F(output, (int2)(output_idx0, output_y_idx), out0);
        if (out_x_base1 >= 0)
        {
            WI_F(output, (int2)(output_idx1, output_y_idx), out4);
        }
    }
}

#define TILE_DIM8 8

__kernel 
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(TILE_DIM8, TILE_DIM8, 1)))
#endif
void conv_2d_kernel1x1_stride1x1_opt(__read_only image2d_t   input,
	                                 __read_only image2d_t   filter,
#ifdef BIAS
	                                 __read_only image2d_t   bias,
#endif
	                                 __write_only image2d_t  output,
	                                 __private const int     input_width,
	                                 __private const int     input_height,
#ifndef CHECK_OUT_CHANNEL
	                                 __private const int     input_channel
#else
                                     __private const int     input_channel,
									 __private const int     output_channel_q
#endif
									 )
{
	__local FLOAT4 src_shared[TILE_DIM8 * TILE_DIM8];
	const int  localx         = get_local_id(0), localy = get_local_id(1);
	const int  globalx        = get_global_id(0);
	const int  grpx           = get_group_id(0), grpy = get_group_id(1);
	const int  pix_start_pos  = (grpx << 5); 
	int        cur_filter_y   = (grpy << 4) + localy;
	int        cur_filter_y2  = cur_filter_y + TILE_DIM8;
	const int  src_shared_idx = mad24(localx, TILE_DIM8, localy);
#ifdef CHECK_IMG_BORDER
    const int  image_size     = input_width * input_height;	
#endif

	FLOAT4 acc0 = (FLOAT4)(0.0f);
	FLOAT4 acc1 = (FLOAT4)(0.0f);
	FLOAT4 acc2 = (FLOAT4)(0.0f);
	FLOAT4 acc3 = (FLOAT4)(0.0f);
	FLOAT4 acc4 = (FLOAT4)(0.0f);
	FLOAT4 acc5 = (FLOAT4)(0.0f);
	FLOAT4 acc6 = (FLOAT4)(0.0f);
	FLOAT4 acc7 = (FLOAT4)(0.0f);
	
    int    image_pos = (pix_start_pos + (localy << 2) + (localx & 3));
	int    src_x     = (image_pos % input_width) + (localx >> 2) * input_width ;
	int    src_y     = (image_pos / input_width);
#ifdef CHECK_IMG_BORDER
    src_x            = select(src_x, -1, image_pos >= image_size);
#endif
	FLOAT4 fetch_src = RI_F(input,    SAMPLER, (int2)(src_x, src_y));
	for (int i = TILE_DIM8; i < input_channel; i += TILE_DIM8)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		src_shared[src_shared_idx]    = fetch_src;
		barrier(CLK_LOCAL_MEM_FENCE);

#ifdef CHECK_IMG_BORDER
        src_x      = select(src_x + (input_width << 1), -1, image_pos >= image_size);
#else	
		src_x     += (input_width << 1);
#endif

#ifdef CHECK_OUT_CHANNEL
		cur_filter_y     = select(cur_filter_y,  -1, cur_filter_y  >= output_channel_q);
		cur_filter_y2    = select(cur_filter_y2, -1, cur_filter_y2 >= output_channel_q);
#endif
		fetch_src  = RI_F(input,    SAMPLER, (int2)(src_x, src_y));
		for (int k = 0; k < TILE_DIM8; k += 4)
		{
			int    cur_filter_x = (i - TILE_DIM8) + k;
			int    src_idx      = mad24(k, TILE_DIM8, localx);
			FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
			FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
			FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
			FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
			
			FLOAT4 src0_data    = src_shared[src_idx];
			src_idx += TILE_DIM8;
			FLOAT4 src1_data    = src_shared[src_idx];
			src_idx += TILE_DIM8;
			FLOAT4 src2_data    = src_shared[src_idx];
			src_idx += TILE_DIM8;
			FLOAT4 src3_data    = src_shared[src_idx];
			
			acc0 = mad(filter0_data, src0_data.x, acc0);
			acc0 = mad(filter1_data, src0_data.y, acc0);
			acc0 = mad(filter2_data, src0_data.z, acc0);
			acc0 = mad(filter3_data, src0_data.w, acc0);
			
			acc1 = mad(filter0_data, src1_data.x, acc1);
			acc1 = mad(filter1_data, src1_data.y, acc1);
			acc1 = mad(filter2_data, src1_data.z, acc1);
			acc1 = mad(filter3_data, src1_data.w, acc1);
			
			acc2 = mad(filter0_data, src2_data.x, acc2);
			acc2 = mad(filter1_data, src2_data.y, acc2);
			acc2 = mad(filter2_data, src2_data.z, acc2);
			acc2 = mad(filter3_data, src2_data.w, acc2);
			
			acc3 = mad(filter0_data, src3_data.x, acc3);
			acc3 = mad(filter1_data, src3_data.y, acc3);
			acc3 = mad(filter2_data, src3_data.z, acc3);
			acc3 = mad(filter3_data, src3_data.w, acc3);
			
			filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y2));
			filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y2));
			filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y2));
			filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y2));
			
			acc4 = mad(filter0_data, src0_data.x, acc4);
			acc4 = mad(filter1_data, src0_data.y, acc4);
			acc4 = mad(filter2_data, src0_data.z, acc4);
			acc4 = mad(filter3_data, src0_data.w, acc4);
			
			acc5 = mad(filter0_data, src1_data.x, acc5);
			acc5 = mad(filter1_data, src1_data.y, acc5);
			acc5 = mad(filter2_data, src1_data.z, acc5);
			acc5 = mad(filter3_data, src1_data.w, acc5);
			
			acc6 = mad(filter0_data, src2_data.x, acc6);
			acc6 = mad(filter1_data, src2_data.y, acc6);
			acc6 = mad(filter2_data, src2_data.z, acc6);
			acc6 = mad(filter3_data, src2_data.w, acc6);
			
			acc7 = mad(filter0_data, src3_data.x, acc7);
			acc7 = mad(filter1_data, src3_data.y, acc7);
			acc7 = mad(filter2_data, src3_data.z, acc7);
			acc7 = mad(filter3_data, src3_data.w, acc7);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	src_shared[src_shared_idx] = fetch_src;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = 0; k < TILE_DIM8; k += 4)
	{
		int    cur_filter_x = (input_channel - TILE_DIM8) + k;
		int    src_idx      = mad24(k, TILE_DIM8, localx);
		FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
		FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
		FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
		FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
		
		FLOAT4 src0_data    = src_shared[src_idx];
		src_idx += TILE_DIM8;
		FLOAT4 src1_data    = src_shared[src_idx];
		src_idx += TILE_DIM8;
		FLOAT4 src2_data    = src_shared[src_idx];
		src_idx += TILE_DIM8;
		FLOAT4 src3_data    = src_shared[src_idx];
		
		acc0 = mad(filter0_data, src0_data.x, acc0);
		acc0 = mad(filter1_data, src0_data.y, acc0);
		acc0 = mad(filter2_data, src0_data.z, acc0);
		acc0 = mad(filter3_data, src0_data.w, acc0);
		
		acc1 = mad(filter0_data, src1_data.x, acc1);
		acc1 = mad(filter1_data, src1_data.y, acc1);
		acc1 = mad(filter2_data, src1_data.z, acc1);
		acc1 = mad(filter3_data, src1_data.w, acc1);
		
		acc2 = mad(filter0_data, src2_data.x, acc2);
		acc2 = mad(filter1_data, src2_data.y, acc2);
		acc2 = mad(filter2_data, src2_data.z, acc2);
		acc2 = mad(filter3_data, src2_data.w, acc2);
		
		acc3 = mad(filter0_data, src3_data.x, acc3);
		acc3 = mad(filter1_data, src3_data.y, acc3);
		acc3 = mad(filter2_data, src3_data.z, acc3);
		acc3 = mad(filter3_data, src3_data.w, acc3);
		
		filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y2));
		filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y2));
		filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y2));
		filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y2));
		
		acc4 = mad(filter0_data, src0_data.x, acc4);
		acc4 = mad(filter1_data, src0_data.y, acc4);
		acc4 = mad(filter2_data, src0_data.z, acc4);
		acc4 = mad(filter3_data, src0_data.w, acc4);
		
		acc5 = mad(filter0_data, src1_data.x, acc5);
		acc5 = mad(filter1_data, src1_data.y, acc5);
		acc5 = mad(filter2_data, src1_data.z, acc5);
		acc5 = mad(filter3_data, src1_data.w, acc5);
		
		acc6 = mad(filter0_data, src2_data.x, acc6);
		acc6 = mad(filter1_data, src2_data.y, acc6);
		acc6 = mad(filter2_data, src2_data.z, acc6);
		acc6 = mad(filter3_data, src2_data.w, acc6);
		
		acc7 = mad(filter0_data, src3_data.x, acc7);
		acc7 = mad(filter1_data, src3_data.y, acc7);
		acc7 = mad(filter2_data, src3_data.z, acc7);
		acc7 = mad(filter3_data, src3_data.w, acc7);
	}

#ifdef BIAS
	FLOAT4 bias_data_0 = RI_F(bias, SAMPLER, (int2)(cur_filter_y,  0));
	FLOAT4 bias_data_1 = RI_F(bias, SAMPLER, (int2)(cur_filter_y2, 0));
	acc0 += bias_data_0;
	acc1 += bias_data_0;
	acc2 += bias_data_0;
	acc3 += bias_data_0;
	acc4 += bias_data_1;
	acc5 += bias_data_1;
	acc6 += bias_data_1;
	acc7 += bias_data_1;
#endif

#ifdef RELU
	ACTIVE_RELU(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
#endif

#ifdef RELU6
	ACTIVE_RELU6(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
#endif

#ifdef LEAKY_RELU
	ACTIVE_LEAKY_RELU(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
#endif

#ifdef CHECK_OUT_CHANNEL
	if(cur_filter_y2 >= 0)
	{
#ifndef CHECK_IMG_BORDER
		{
			OUTPUT_CONV1x1(acc0, acc1, acc2, acc3, globalx , cur_filter_y,  input_width);
		}
		{
			OUTPUT_CONV1x1(acc4, acc5, acc6, acc7, globalx , cur_filter_y2, input_width);
		}
#else
		{
			OUTPUT_CONV1x1_CHECK_IMAGE_BORDER(acc0, acc1, acc2, acc3, globalx , cur_filter_y, input_width, image_size)
		}
		{
			OUTPUT_CONV1x1_CHECK_IMAGE_BORDER(acc4, acc5, acc6, acc7, globalx , cur_filter_y2, input_width, image_size)
		}
#endif
	}
	else if(cur_filter_y >= 0)
	{
#ifndef CHECK_IMG_BORDER
		{
			OUTPUT_CONV1x1(acc0, acc1, acc2, acc3, globalx , cur_filter_y,  input_width);
		}
#else
		{
			OUTPUT_CONV1x1_CHECK_IMAGE_BORDER(acc0, acc1, acc2, acc3, globalx , cur_filter_y, input_width, image_size)
		}
#endif
	}
#else
#ifndef CHECK_IMG_BORDER
	{
		OUTPUT_CONV1x1(acc0, acc1, acc2, acc3, globalx , cur_filter_y,  input_width);
	}
	{
		OUTPUT_CONV1x1(acc4, acc5, acc6, acc7, globalx , cur_filter_y2, input_width);
	}
#else
	{
		OUTPUT_CONV1x1_CHECK_IMAGE_BORDER(acc0, acc1, acc2, acc3, globalx , cur_filter_y, input_width, image_size)
	}
	{
		OUTPUT_CONV1x1_CHECK_IMAGE_BORDER(acc4, acc5, acc6, acc7, globalx , cur_filter_y2, input_width, image_size)
	}
#endif
#endif
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_s1x1_4row(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                       __read_only image2d_t weights,
#ifdef BIAS
                       __read_only image2d_t bias,
#endif
                       __write_only image2d_t output,
                       __private const int2   input_shape,
                       __private const int    in_channel_block_length,
                       __private const int2   output_shape,
                       __private const int2   weights_shape,
                       __private const int2   padding_shape,
                       __private const int    out_channel_q)
{
    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx = get_global_id(1);
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

    const int in_height_idx   = out_y_pos - padding_shape.x;
    const int weigth_size     = mul24(weights_shape.y, weights_shape.x);
    const int weights_y_base0 = mul24(output_channel_block_idx0, weigth_size);
    const int weights_y_base1 = mul24(output_channel_block_idx1, weigth_size);
    int4      in_y_ofs        = (int4)(in_height_idx, in_height_idx + 1, in_height_idx + 2, in_height_idx + 3);
    int4      in_y_pos        = select(out_batch_base + in_y_ofs, -1, (in_y_ofs < ((int4)(0)) || in_y_ofs >= (int4)(input_shape.x)));
    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    int in_channel_block_idx = 0;
#ifdef IN_CHANNEL_LOOP
    for (in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
#endif
        int       in_width_idx = out_x_pos - padding_shape.y;
        const int in_x_base = mul24(in_channel_block_idx, input_shape.y);
        int weights_x_idx = in_channel_block_idx << 2;
        for (int w = 0; w < weights_shape.y; w++)
        {
            int cur_x_pos = select(in_x_base + in_width_idx, -1, (in_width_idx < 0 || in_width_idx >= input_shape.y));
            in_width_idx++;

            in1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.x));
            in2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.y));
            in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.z));
            for (int h = 0; h < weights_shape.x; h++)
            {
                int weight_y_idx0 = weights_y_base0 + mad24(h, weights_shape.y, w);
                int weight_y_idx1 = select(weights_y_base1 + mad24(h, weights_shape.y, w), -1, output_channel_block_idx1 < 0);
                in0 = in1;
                in1 = in2;
                in2 = in3;
                int cur_in_y_pos = in_y_ofs.w + h;
                cur_in_y_pos = select(out_batch_base + cur_in_y_pos, -1, (cur_in_y_pos < 0 || cur_in_y_pos >= input_shape.x));
                in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos));

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
            WI_F(output, (int2)(output_idx1, output_y),    out4);
            WI_F(output, (int2)(output_idx1, output_y + 1), out5);
            WI_F(output, (int2)(output_idx1, output_y + 2), out6);
            WI_F(output, (int2)(output_idx1, output_y + 3), out7);
        }
    }
    else if (remain == 3) {
        WI_F(output, (int2)(output_idx0, output_y),    out0);
        WI_F(output, (int2)(output_idx0, output_y + 1), out1);
        WI_F(output, (int2)(output_idx0, output_y + 2), out2);
        if (out_x_base1 >= 0)
        {
            WI_F(output, (int2)(output_idx1, output_y),    out4);
            WI_F(output, (int2)(output_idx1, output_y + 1), out5);
            WI_F(output, (int2)(output_idx1, output_y + 2), out6);
        }
    }
    else if (remain == 2) {
        WI_F(output, (int2)(output_idx0, output_y),    out0);
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

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_kN_s2x2(__read_only image2d_t  input, 
					 __read_only image2d_t  weights,
#ifdef BIAS
					 __read_only image2d_t  bias,
#endif
					 __write_only image2d_t output,
					 __private const int2   input_shape,
					 __private const int    in_channel_block_length,
					 __private const int2   output_shape,
					 __private const int    output_channel_q,
					 __private const int    global_size1) 
{
#ifndef KERNEL_SHAPE_SIZE
#define KERNEL_SHAPE_SIZE 3
#endif

#ifndef KERNEL_PADDING_SIZE
#define KERNEL_PADDING_SIZE 1
#endif
    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    const int output_width_q           = (output_shape.y + 3) >> 2;
	if((output_channel_width_idx >= (output_width_q * ((output_channel_q + 1) >> 1))) ||
	   (output_batch_height_idx >= global_size1))
	   return ;

    const int  out_channel_block_idx  = (output_channel_width_idx / output_width_q) << 1;
    const int  out_channel_block_idx2 = select(out_channel_block_idx + 1, -1, out_channel_block_idx + 1 >= output_channel_q);
    const int  out_height_block_idx   = output_channel_width_idx % output_width_q;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
	FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx2, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;
	FLOAT4 out5 = out4;
	FLOAT4 out6 = out4;
	FLOAT4 out7 = out4;
#else
    FLOAT4 out0 = (FLOAT4)(0.0f);
    FLOAT4 out1 = (FLOAT4)(0.0f);
    FLOAT4 out2 = (FLOAT4)(0.0f);
    FLOAT4 out3 = (FLOAT4)(0.0f);
	FLOAT4 out4 = (FLOAT4)(0.0f);
	FLOAT4 out5 = (FLOAT4)(0.0f);
	FLOAT4 out6 = (FLOAT4)(0.0f);
	FLOAT4 out7 = (FLOAT4)(0.0f);
#endif


    int in_width0            = mad24(out_height_block_idx, 8, -KERNEL_PADDING_SIZE);
    int in_width1            = in_width0 + 2;
    int in_width2            = in_width0 + 4;
    int in_width3            = in_width0 + 6;
    
	const int height_start   = mad24((output_batch_height_idx % output_shape.x), 2, -KERNEL_PADDING_SIZE);
	int in_height_start      = mad24(select(0, -height_start, height_start < 0), 1, height_start);
	int in_height_end        = min(KERNEL_SHAPE_SIZE + height_start, input_shape.x);

	const int batch_idx      = mul24((output_batch_height_idx / output_shape.x), input_shape.x);
	const int weights_h_idx  = mul24(out_channel_block_idx, (KERNEL_SHAPE_SIZE*KERNEL_SHAPE_SIZE))  + mul24(select(0, -height_start, height_start < 0), KERNEL_SHAPE_SIZE);
    const int weights_h_idx2 = select(mul24(out_channel_block_idx2, (KERNEL_SHAPE_SIZE*KERNEL_SHAPE_SIZE)) + mul24(select(0, -height_start, height_start < 0), KERNEL_SHAPE_SIZE), -1, out_channel_block_idx2 < 0);

	int in_channel_block_idx = 0;
#ifdef IN_CHANNEL_LOOP
    for (in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
#endif
        const __private int in_idx     = mul24(in_channel_block_idx, input_shape.y);
        __private int4  weights_x_idx  = (int4)(in_channel_block_idx << 2);
		weights_x_idx                 += (int4)(0, 1, 2, 3);
        __private int2  weights_y_idx  = (int2)(weights_h_idx, weights_h_idx2);

#ifndef INPUT_CHANNEL_4
		int4 in_width_value        = (int4)(in_width0,     in_width1,     in_width2,     in_width3);
		const int4 in_width_value2 = in_width_value + (int4)(1);
		const int4 cur_src         = select((int4)(in_idx) + in_width_value,  (int4)(-1), (in_width_value  < (int4)(0)) || (in_width_value  >= (int4)(input_shape.y)));
		const int4 cur_src2        = select((int4)(in_idx) + in_width_value2, (int4)(-1), (in_width_value2 < (int4)(0)) || (in_width_value2 >= (int4)(input_shape.y)));
#endif
        for (int iy = in_height_start; iy < in_height_end; ++iy) 
		{
            __private int in_hb_value = iy + batch_idx;
#ifndef INPUT_CHANNEL_4
			__private FLOAT4 cur_in0, cur_in1, cur_in2, cur_in3, cur_in4, cur_in5, cur_in6, cur_in7;
			cur_in1                    = RI_F(input, SAMPLER, (int2)(cur_src.x,  in_hb_value));
			cur_in5                    = RI_F(input, SAMPLER, (int2)(cur_src2.x, in_hb_value));
			cur_in2                    = RI_F(input, SAMPLER, (int2)(cur_src.y,  in_hb_value));
			cur_in6                    = RI_F(input, SAMPLER, (int2)(cur_src2.y, in_hb_value));
			cur_in3                    = RI_F(input, SAMPLER, (int2)(cur_src.z,  in_hb_value));
			cur_in7                    = RI_F(input, SAMPLER, (int2)(cur_src2.z, in_hb_value));
#endif
			for (int w = 0; w < KERNEL_SHAPE_SIZE; ++w)
			{
				__private FLOAT4 weights0;
#ifdef INPUT_CHANNEL_2 
				__private FLOAT4 weights1;
#endif
#ifdef INPUT_CHANNEL_3
				__private FLOAT4 weights2;
#endif
#ifdef INPUT_CHANNEL_4
				__private FLOAT4 weights3;
#endif

				__private FLOAT4 in0, in1, in2, in3;
#ifdef INPUT_CHANNEL_4
                READ_INPUT_IMAGE(0, w);
                READ_INPUT_IMAGE(1, w);
                READ_INPUT_IMAGE(2, w);
                READ_INPUT_IMAGE(3, w);
#else
				int cur_src_x   = in_width_value.w + w;
				cur_src_x       = select(in_idx + cur_src_x, -1, (cur_src_x < 0) || (cur_src_x >= input_shape.y));
				if(0 == (w&1))
				{
					cur_in0 = cur_in1;
					cur_in1 = cur_in2;
					cur_in2 = cur_in3;
					cur_in3 = RI_F(input, SAMPLER, (int2)(cur_src_x, in_hb_value));
					in0 	= cur_in0;
					in1 	= cur_in1;
					in2 	= cur_in2;
					in3 	= cur_in3;
				}
				else
				{
					cur_in4 = cur_in5;
					cur_in5 = cur_in6;
					cur_in6 = cur_in7;
					cur_in7 = RI_F(input, SAMPLER, (int2)(cur_src_x, in_hb_value));
					in0     = cur_in4;
					in1     = cur_in5;
					in2     = cur_in6;
					in3     = cur_in7;
				}
#endif
				weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.x)); 
#ifdef INPUT_CHANNEL_2 
				weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.x)); 
#endif
#ifdef INPUT_CHANNEL_3
				weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.x)); 
#endif
#ifdef INPUT_CHANNEL_4
				weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.x));
#endif

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

				weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.y));
#ifdef INPUT_CHANNEL_2 
				weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.y));
#endif
#ifdef INPUT_CHANNEL_3
				weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.y));
#endif
#ifdef INPUT_CHANNEL_4
				weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.y));
#endif

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
				weights_y_idx += (int2)(1);
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
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, output_channel_q, out_channel_block_idx);
	if (out_channel_block_idx2 >= 0)
	{
		CAL_BN_AFTER_RELU(out4, out5, out6, out7, bias, output_channel_q, out_channel_block_idx2);
	}
#endif

    const int out_x_base  = mul24(out_channel_block_idx, output_shape.y);
	const int out_x_base2 = select(mul24(out_channel_block_idx2, output_shape.y), -1, out_channel_block_idx2 < 0);
    int out_x_idx        = out_height_block_idx << 2;
    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base + out_x_idx;
	int output_idx2  = out_x_base2 + out_x_idx;
    if (remain >= 4) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2,  output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3,  output_batch_height_idx), out3);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
			WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
			WI_F(output, (int2)(output_idx2 + 3, output_batch_height_idx), out7);
		}
    } 
	else if (remain == 3) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2,  output_batch_height_idx), out2);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
			WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
		}
    } 
	else if (remain == 2) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
		}
    } 
	else if (remain == 1) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
		if(out_x_base2 >= 0)
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_k3_s2x2(__read_only image2d_t  input, 
					 __read_only image2d_t  weights,
#ifdef BIAS
					 __read_only image2d_t  bias,
#endif
					 __write_only image2d_t output,
					 __private const int2   input_shape,
					 __private const int    in_channel_block_length,
					 __private const int2   output_shape,
					 __private const int    output_channel_q,
					 __private const int    global_size1)
{
    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    const int output_width_q           = (output_shape.y + 3) >> 2;
	if((output_channel_width_idx >= (output_width_q * ((output_channel_q + 1) >> 1))) ||
	   (output_batch_height_idx >= global_size1))
	   return ;

    const int  out_channel_block_idx  = (output_channel_width_idx / output_width_q) << 1;
    const int  out_channel_block_idx2 = select(out_channel_block_idx + 1, -1, out_channel_block_idx + 1 >= output_channel_q);
    const int  out_height_block_idx   = output_channel_width_idx % output_width_q;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
	FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx2, 0));
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

    int in_width0            = mad24(out_height_block_idx, 8, -1);
    int in_width1            = in_width0 + 2;
    int in_width2            = in_width0 + 4;
    int in_width3            = in_width0 + 6;
	
	const int height_start   = mad24((output_batch_height_idx % output_shape.x), 2, -1);
	int in_height_start      = mad24(select(0, -height_start, height_start < 0), 1, height_start);
	int in_height_end        = min(3 + height_start, input_shape.x);

	const int batch_idx      = mul24((output_batch_height_idx / output_shape.x), input_shape.x);
	const int weights_h_idx  = mul24(out_channel_block_idx, (3*3))   + mul24(select(0, -height_start, height_start < 0), 3);
	int       weights_h_idx2 = mul24(out_channel_block_idx2, (3*3))  + mul24(select(0, -height_start, height_start < 0), 3);
    weights_h_idx2           = select(weights_h_idx2, -1, out_channel_block_idx2 < 0);
	
	int in_channel_block_idx = 0;
#ifdef IN_CHANNEL_LOOP
    for (in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) 
	{
#endif
        const __private int in_idx     = mul24(in_channel_block_idx, input_shape.y);
        __private int4  weights_x_idx  = (int4)(in_channel_block_idx << 2);
		weights_x_idx                 += (int4)(0, 1, 2, 3);
        __private int2  weights_y_idx  = (int2)(weights_h_idx, weights_h_idx2);
        for (int iy = in_height_start; iy < in_height_end; ++iy) 
		{	
            __private int in_hb_value = iy + batch_idx;
			__private FLOAT4 in0, in1, in2, in3;
			__private FLOAT4 weights0;
			__private FLOAT4 weights1;
			__private FLOAT4 weights2;
			__private FLOAT4 weights3;
			__private FLOAT4 in4 = (FLOAT4)(0.0f), in5 = (FLOAT4)(0.0f), in6 = (FLOAT4)(0.0f);
			for (int w = 0; w < 2; w++)
			{
				READ_INPUT_IMAGE(0, w);
				READ_INPUT_IMAGE(1, w);
				READ_INPUT_IMAGE(2, w);
				READ_INPUT_IMAGE(3, w);

				weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.x)); 
#ifdef INPUT_CHANNEL_2 
				weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.x)); 
#endif
#ifdef INPUT_CHANNEL_3
				weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.x)); 
#endif
#ifdef INPUT_CHANNEL_4
				weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.x));
#endif

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

				weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.y));
#ifdef INPUT_CHANNEL_2 
				weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.y));
#endif
#ifdef INPUT_CHANNEL_3
				weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.y));
#endif
#ifdef INPUT_CHANNEL_4
				weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.y));
#endif

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
				
				weights_y_idx += (int2)(1);
				
				if(0 == w)
				{
					in4 = in1;
					in5 = in2;
					in6 = in3;
				}
				//in4 = select(in4, in1, ((int4)(0) == (int4)(w)));
				//in5 = select(in5, in2, ((int4)(0) == (int4)(w)));
				//in6 = select(in6, in3, ((int4)(0) == (int4)(w)));
			}		

			in0 = in4;
			in1 = in5;
			in2 = in6;
			READ_INPUT_IMAGE(3, 2);
			
			weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.x)); 
#ifdef INPUT_CHANNEL_2 
			weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.x)); 
#endif
#ifdef INPUT_CHANNEL_3
			weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.x)); 
#endif
#ifdef INPUT_CHANNEL_4
			weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.x));
#endif

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

			weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.y));
#ifdef INPUT_CHANNEL_2 
			weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.y));
#endif
#ifdef INPUT_CHANNEL_3
			weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.y));
#endif
#ifdef INPUT_CHANNEL_4
			weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.y));
#endif

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
			
			weights_y_idx += (int2)(1);
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
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, output_channel_q, out_channel_block_idx);
	if (out_channel_block_idx2 >= 0)
	{
		CAL_BN_AFTER_RELU(out4, out5, out6, out7, bias, output_channel_q, out_channel_block_idx2);
	}
#endif

    const int out_x_base  = mul24(out_channel_block_idx, output_shape.y);
	const int out_x_base2 = select(mul24(out_channel_block_idx2, output_shape.y), -1, out_channel_block_idx2 < 0);
    int out_x_idx         = out_height_block_idx << 2;
    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base  + out_x_idx;
	int output_idx2  = out_x_base2 + out_x_idx;
    if (remain >= 4) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2,  output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3,  output_batch_height_idx), out3);

		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
			WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
			WI_F(output, (int2)(output_idx2 + 3, output_batch_height_idx), out7);
		}
    } 
	else if (remain == 3) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2,  output_batch_height_idx), out2);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
			WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
		}
    } 
	else if (remain == 2) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
		if(out_x_base2 >= 0)
		{
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
			WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
		}
    } 
	else if (remain == 1) 
	{
        WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
		if(out_x_base2 >= 0)
			WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
    }
}