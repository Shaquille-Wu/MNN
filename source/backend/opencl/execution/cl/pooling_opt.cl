#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#if 0
__kernel void pooling_2x2_opt(__read_only image2d_t    input,
	                          __write_only image2d_t   output)
{
	const int  gx         = get_global_id(0);
	const int  gy         = get_global_id(1);
	const int  pos_x_src  = (gx << 1);
	const int  pos_y      = (gy << 1);

	FLOAT4  d0  = RI_F(input, SAMPLER, (int2)(pos_x_src,     pos_y));
	FLOAT4  d1  = RI_F(input, SAMPLER, (int2)(pos_x_src + 1, pos_y));
	FLOAT4  d2  = RI_F(input, SAMPLER, (int2)(pos_x_src,     pos_y + 1));
	FLOAT4  d3  = RI_F(input, SAMPLER, (int2)(pos_x_src + 1, pos_y + 1));

#ifdef POOL_AVG
	FLOAT4  output_0 = d0 + d1 + d2 + d3;
	output_0         = (FLOAT)(0.25f) * output_0;
#else
	FLOAT4  output_0 = fmax(d0, d1);
	FLOAT4  output_1 = fmax(d2, d3);
	output_0         = fmax(output_0, output_1);
#endif
	WI_F(output, (int2)(gx, gy), output_0);
}
#else
__kernel void pooling_2x2_opt(__read_only image2d_t    input,
	                          __write_only image2d_t   output,
							  __private int            output_channel_width,
							  __private int            output_height)
{
	const int  gx         = get_global_id(0);
	const int  gy         = get_global_id(1);

#ifdef CHECK_POOLING_BORDER	
	if(gx >= output_channel_width || gy >= output_height)
		return;
#endif
/*
	const int  pos_x_src         = (gx << 1);
	const int  pos_y_src         = (gy << 1);
	const int  src_channel_width = (output_channel_width << 1);
	const int  src_height        = (output_height << 1);
	const int  pos_x             = select(pos_x_src,     -1, pos_x_src       >= src_channel_width);
	const int  pos_x_1           = select(pos_x_src + 1, -1, (pos_x_src + 1) >= src_channel_width);
	const int  pos_y             = select(pos_y_src,     -1, pos_y_src       >= src_height);
	const int  pos_y_1           = select(pos_y_src + 1, -1, (pos_y_src + 1) >= src_height);
*/
	const int  pos_x_src         = (gx << 1);
	const int  pos_y_src         = (gy << 1);
	const int  pos_x             = pos_x_src;
	const int  pos_x_1           = pos_x_src + 1;
	const int  pos_y             = pos_y_src;
	const int  pos_y_1           = pos_y_src + 1;

	FLOAT4  d0  = RI_F(input, SAMPLER, (int2)(pos_x,   pos_y));
	FLOAT4  d1  = RI_F(input, SAMPLER, (int2)(pos_x_1, pos_y));
	FLOAT4  d2  = RI_F(input, SAMPLER, (int2)(pos_x,   pos_y_1));
	FLOAT4  d3  = RI_F(input, SAMPLER, (int2)(pos_x_1, pos_y_1));

#ifdef POOL_AVG
	FLOAT4  output_0 = d0 + d1 + d2 + d3;
	output_0         = (FLOAT)(0.25f) * output_0;
#else
	FLOAT4  output_0 = fmax(d0, d1);
	FLOAT4  output_1 = fmax(d2, d3);
	output_0         = fmax(output_0, output_1);
#endif
	WI_F(output, (int2)(gx, gy), output_0);
}
#endif

#define TILE_DIM       8
#define TILE_DIM_PRE   7
#define TILE_DIM_BITS  3
__kernel void pooling_global_avg(__read_only image2d_t    input,
	                             __write_only image2d_t   output,
	                             __private const int      input_width,
								 __private const int      input_height)
{
	int   lx               = get_local_id(0);
	int   ly               = get_local_id(1);
	int   gz               = get_global_id(2);
	int   input_width_new  = ((input_width + TILE_DIM_PRE) >> TILE_DIM_BITS) << TILE_DIM_BITS;
	int   input_height_new = ((input_height + TILE_DIM_PRE) >> TILE_DIM_BITS) << TILE_DIM_BITS;
	int   shared_pos       = (ly << TILE_DIM_BITS) + lx;
	__local FLOAT4     src_data[TILE_DIM*TILE_DIM];
	
	__private const int   x_offset  = mul24(gz, input_width);
	__private int         cur_x     = lx;
	__private int         cur_y     = ly;
	__private FLOAT4      sum       = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
	for(int i = 0 ; i < input_height_new ; i += TILE_DIM)
	{
		__private int     pos_y     = select(cur_y, -1, cur_y >= input_height);
		cur_x = lx;
		for(int j = 0; j < input_width_new ; j += TILE_DIM)
		{
			__private int          pos_x     = select(x_offset + cur_x, -1, cur_x >= input_width);
			__private const FLOAT4 cur_value = RI_F(input, SAMPLER, (int2)(pos_x, pos_y));
			sum   += cur_value;
			cur_x += TILE_DIM;
		}
		cur_y += TILE_DIM;
	}


	src_data[shared_pos] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	if(0 == (lx & 1))
	{
		sum                  += src_data[shared_pos + 1];
		src_data[shared_pos]  = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(0 == (lx & 3))
	{
		sum                  += src_data[shared_pos + 2];
		src_data[shared_pos]  = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(0 == lx)
	{
		sum                  += src_data[shared_pos + 4];
		src_data[shared_pos]  = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if((0 == lx) && (0 == (ly & 1)))
	{
		sum                  += src_data[shared_pos + 8];
		src_data[shared_pos]  = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);	
	if((0 == lx) && (0 == (ly & 3)))
	{
		sum                  += src_data[shared_pos + 16];
		src_data[shared_pos]  = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if((0 == lx) && (0 == ly))
	{
		sum += src_data[shared_pos + 32];
		sum  = sum / ((FLOAT)(input_width * input_height));
		WI_F(output, (int2)(gz, 0), sum);
	}
}