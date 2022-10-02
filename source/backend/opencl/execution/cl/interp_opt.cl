#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void interp_nearest_half_opt(__read_only  image2d_t input,
                                      __write_only image2d_t output,
                                      __private    int       input_width,
                                      __private    int       input_height) //input_height is batch * input_height
{
    int    gx          = get_global_id(0);
	int    gy          = get_global_id(1);
	int    channel_idx = gx / input_width;
	int    x_pos       = gx % input_width;
	if(x_pos >= input_width || gy >= input_height)
		return;

	int    x_offset       = mul24(channel_idx, input_width);
	int    src_x_pos      = x_pos;
	int    src_y_pos      = gy;
	__private const FLOAT4 in00  = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos,      src_y_pos));

	const int out_x = gx << 1;
	const int out_y = gy << 1;
    WI_F(output, (int2)(out_x,     out_y),     in00);
	WI_F(output, (int2)(out_x + 1, out_y),     in00);
	WI_F(output, (int2)(out_x,     out_y + 1), in00);
	WI_F(output, (int2)(out_x + 1, out_y + 1), in00);
}

__kernel void interp_nearest_opt(__read_only  image2d_t input,
                                 __write_only image2d_t output,
                                 __private    int       input_width,
                                 __private    int       input_height) //input_height is batch * input_height
{
#ifdef NEAREST_4X
#define NEAREST_SHIFT_BITS  2
#define NEAREST_LOOP_CNT    4
#define NEAREST_DELTA       0.25f
#else
#define NEAREST_SHIFT_BITS  3
#define NEAREST_LOOP_CNT    8
#define NEAREST_DELTA       0.125f
#endif

    int    gx          = get_global_id(0);
	int    gy          = get_global_id(1);
	int    channel_idx = gx / input_width;
	int    x_pos       = gx % input_width;
	if(x_pos >= input_width || gy >= input_height)
		return;

	int    x_offset       = mul24(channel_idx, input_width);
	int    src_x_pos      = x_pos;
	int    src_y_pos      = gy;
	__private const FLOAT4 in00     = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos,      src_y_pos));
	const int out_x = gx << NEAREST_SHIFT_BITS;
	const int out_y = gy << NEAREST_SHIFT_BITS;
	for(int i = 0 ; i < NEAREST_LOOP_CNT ; i ++)
	{
		for(int j = 0 ; j < NEAREST_LOOP_CNT ; j ++)
		{
			WI_F(output, (int2)(out_x + j, out_y + i), in00);
		}
	}
}

__kernel void interp_bilinear_half_opt(__read_only  image2d_t input,
                                       __write_only image2d_t output,
                                       __private    int       input_width,
                                       __private    int       input_height) //input_height is batch * input_height
{
    int    gx          = get_global_id(0);
	int    gy          = get_global_id(1);
	int    channel_idx = gx / input_width;
	int    x_pos       = gx % input_width;
	if(x_pos >= input_width || gy >= input_height)
		return;

	int    x_offset       = mul24(channel_idx, input_width);
	int    src_x_pos      = x_pos;
	int    src_y_pos      = gy;
	int    src_x_pos_plus = src_x_pos + 1;
	int    src_y_pos_plus = src_y_pos + 1;
	src_x_pos_plus        = select(src_x_pos_plus, input_width  - 1, src_x_pos_plus >= input_width);
	src_y_pos_plus        = select(src_y_pos_plus, input_height - 1, src_y_pos_plus >= input_height);

	__private const FLOAT4 in00  = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos,      src_y_pos));
	__private const FLOAT4 in01  = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos_plus, src_y_pos));
	__private const FLOAT4 in10  = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos,      src_y_pos_plus));
	__private const FLOAT4 in11  = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos_plus, src_y_pos_plus));
	
	FLOAT4 add_res        = in00 + in01;
	FLOAT4 out_value0     = add_res                 * (FLOAT4)(0.5f);
	FLOAT4 out_value1     = (in00 + in10)           * (FLOAT4)(0.5f);
	FLOAT4 out_value2     = (add_res + in10 + in11) * (FLOAT4)(0.25f);

	const int out_x = gx << 1;
	const int out_y = gy << 1;
    WI_F(output, (int2)(out_x,     out_y),     in00);
	WI_F(output, (int2)(out_x + 1, out_y),     out_value0);
	WI_F(output, (int2)(out_x,     out_y + 1), out_value1);
	WI_F(output, (int2)(out_x + 1, out_y + 1), out_value2);
}

__kernel void interp_bilinear_opt(__read_only  image2d_t input,
                                  __write_only image2d_t output,
                                  __private    int       input_width,
                                  __private    int       input_height)  //input_height is batch * input_height
{
#ifdef BILINEAR_4X
#define BILINEAR_SHIFT_BITS  2
#define BILINEAR_LOOP_CNT    4
#define BILINEAR_DELTA       0.25f
#else
#define BILINEAR_SHIFT_BITS  3
#define BILINEAR_LOOP_CNT    8
#define BILINEAR_DELTA       0.125f
#endif

    int    gx          = get_global_id(0);
	int    gy          = get_global_id(1);
	int    channel_idx = gx / input_width;
	int    x_pos       = gx % input_width;
	if(x_pos >= input_width || gy >= input_height)
		return;

	int    x_offset       = mul24(channel_idx, input_width);
	int    src_x_pos      = x_pos;
	int    src_y_pos      = gy;
	int    src_x_pos_plus = src_x_pos + 1;
	int    src_y_pos_plus = src_y_pos + 1;
	src_x_pos_plus        = select(src_x_pos_plus, input_width  - 1, src_x_pos_plus >= input_width);
	src_y_pos_plus        = select(src_y_pos_plus, input_height - 1, src_y_pos_plus >= input_height);

	__private const FLOAT4 in00     = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos,      src_y_pos));
	__private const FLOAT4 in01     = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos_plus, src_y_pos));
	__private const FLOAT4 in10     = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos,      src_y_pos_plus));
	__private const FLOAT4 in11     = RI_F(input, SAMPLER, (int2)(x_offset + src_x_pos_plus, src_y_pos_plus));
	__private FLOAT        u0       = 1.0f;
	__private FLOAT        u1       = 0.0f;
	__private FLOAT        v0       = 1.0f;
	__private FLOAT        v1       = 0.0f;
	const int out_x = gx << BILINEAR_SHIFT_BITS;
	const int out_y = gy << BILINEAR_SHIFT_BITS;
	for(int i = 0 ; i < BILINEAR_LOOP_CNT ; i ++)
	{
		u0       = 1.0f;
		u1       = 0.0f;
		for(int j = 0 ; j < BILINEAR_LOOP_CNT ; j ++)
		{
			FLOAT4 out_value = in00 * (u0 * v0) + 
			                   in01 * (u1 * v0) +
						       in10 * (u0 * v1) +
						       in11 * (u1 * v1);
			WI_F(output, (int2)(out_x + j, out_y + i), out_value);
			u0 -= BILINEAR_DELTA;
			u1 += BILINEAR_DELTA;
		}
		v0 -= BILINEAR_DELTA;
		v1 += BILINEAR_DELTA;
	}
}
