#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define TILE_DIM16 16
__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void gemm_local_mem_16x(__read_only  image2d_t  src,
	                    __read_only  image2d_t  filter,
	                    __write_only image2d_t  dst,
	                    __private const int     input_channel)
{
#ifdef AVOID_MEM_BANK_CONFLICT
	__local FLOAT4 src_shared[TILE_DIM16][TILE_DIM16 + 1];
#else
  	__local FLOAT4 src_shared[TILE_DIM16][TILE_DIM16];
#endif
	const int localx         = get_local_id(0), localy = get_local_id(1);
	const int globalx        = get_global_id(0), globaly = get_global_id(1), globalz = get_global_id(2);
	const int grpx           = get_group_id(0), grpy = get_group_id(1);
	const int image_start_x  = mul24(globalz, (int)get_global_size(0));
	const int output_x       = image_start_x + globalx;
	const int output_y       = (globaly << 2);
	const int src_start_y    = image_start_x + (grpx << 4);
	const int filter_start_y = mad24(globalz, (int)get_global_size(1), (grpy << 4));
	const int cur_filter_y   = filter_start_y + localy;
	FLOAT4    acc0           = (FLOAT4)(0.0f);
	FLOAT4    acc1           = (FLOAT4)(0.0f);
	FLOAT4    acc2           = (FLOAT4)(0.0f);
	FLOAT4    acc3           = (FLOAT4)(0.0f);
	
	FLOAT4 fetch_src = RI_F(src,    SAMPLER, (int2)(localx, src_start_y + localy));
	for (int i = TILE_DIM16; i < input_channel; i += TILE_DIM16)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		src_shared[localx][localy]    = fetch_src;
		barrier(CLK_LOCAL_MEM_FENCE);
		fetch_src                     = RI_F(src,    SAMPLER, (int2)(localx + i, src_start_y + localy));
		for (int k = 0; k < TILE_DIM16; k += 4)
		{
			__private int    cur_filter_x = (i - TILE_DIM16) + k;
			__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
			__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
			__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
			__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
			acc0 = mad(filter0_data, src_shared[k][localx].x, acc0);
			acc0 = mad(filter1_data, src_shared[k][localx].y, acc0);
			acc0 = mad(filter2_data, src_shared[k][localx].z, acc0);
			acc0 = mad(filter3_data, src_shared[k][localx].w, acc0);

			acc1 = mad(filter0_data, src_shared[k + 1][localx].x, acc1);
			acc1 = mad(filter1_data, src_shared[k + 1][localx].y, acc1);
			acc1 = mad(filter2_data, src_shared[k + 1][localx].z, acc1);
			acc1 = mad(filter3_data, src_shared[k + 1][localx].w, acc1);

			acc2 = mad(filter0_data, src_shared[k + 2][localx].x, acc2);
			acc2 = mad(filter1_data, src_shared[k + 2][localx].y, acc2);
			acc2 = mad(filter2_data, src_shared[k + 2][localx].z, acc2);
			acc2 = mad(filter3_data, src_shared[k + 2][localx].w, acc2);

			acc3 = mad(filter0_data, src_shared[k + 3][localx].x, acc3);
			acc3 = mad(filter1_data, src_shared[k + 3][localx].y, acc3);
			acc3 = mad(filter2_data, src_shared[k + 3][localx].z, acc3);
			acc3 = mad(filter3_data, src_shared[k + 3][localx].w, acc3);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	src_shared[localx][localy] = fetch_src;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = 0; k < TILE_DIM16; k += 4)
	{
		__private int    cur_filter_x = (input_channel - TILE_DIM16) + k;
		__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
		__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
		__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
		__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
		
		acc0 = mad(filter0_data, src_shared[k][localx].x, acc0);
		acc0 = mad(filter1_data, src_shared[k][localx].y, acc0);
		acc0 = mad(filter2_data, src_shared[k][localx].z, acc0);
		acc0 = mad(filter3_data, src_shared[k][localx].w, acc0);

		acc1 = mad(filter0_data, src_shared[k + 1][localx].x, acc1);
		acc1 = mad(filter1_data, src_shared[k + 1][localx].y, acc1);
		acc1 = mad(filter2_data, src_shared[k + 1][localx].z, acc1);
		acc1 = mad(filter3_data, src_shared[k + 1][localx].w, acc1);

		acc2 = mad(filter0_data, src_shared[k + 2][localx].x, acc2);
		acc2 = mad(filter1_data, src_shared[k + 2][localx].y, acc2);
		acc2 = mad(filter2_data, src_shared[k + 2][localx].z, acc2);
		acc2 = mad(filter3_data, src_shared[k + 2][localx].w, acc2);

		acc3 = mad(filter0_data, src_shared[k + 3][localx].x, acc3);
		acc3 = mad(filter1_data, src_shared[k + 3][localx].y, acc3);
		acc3 = mad(filter2_data, src_shared[k + 3][localx].z, acc3);
		acc3 = mad(filter3_data, src_shared[k + 3][localx].w, acc3);
	}
	
	WI_F(dst, (int2)(output_x, output_y),     acc0);
	WI_F(dst, (int2)(output_x, output_y + 1), acc1);
	WI_F(dst, (int2)(output_x, output_y + 2), acc2);
	WI_F(dst, (int2)(output_x, output_y + 3), acc3);
}

#define TILE_DIM32 32
__kernel 
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(TILE_DIM32, TILE_DIM32, 1)))
#endif
void gemm_local_mem_32x_prefetch(__read_only  image2d_t  src,
	                             __read_only  image2d_t  filter,
	                             __write_only image2d_t  dst,
	                             __private const int     input_channel)
{
#ifdef AVOID_MEM_BANK_CONFLICT
	__local FLOAT4 src_shared[TILE_DIM32][TILE_DIM32 + 1];
#else
  	__local FLOAT4 src_shared[TILE_DIM32][TILE_DIM32];
#endif
	const int localx         = get_local_id(0), localy = get_local_id(1);
	const int globalx        = get_global_id(0), globaly = get_global_id(1), globalz = get_global_id(2);
	const int grpx           = get_group_id(0), grpy = get_group_id(1);
	const int image_start_x  = mul24(globalz, (int)get_global_size(0));
	const int output_x       = image_start_x + globalx;
	const int output_y       = (globaly << 2);
	const int src_start_y    = image_start_x + (grpx << 5);
	const int filter_start_y = mad24(globalz, (int)get_global_size(1), (grpy << 5));
	const int cur_filter_y   = filter_start_y + localy;
	FLOAT4    acc0           = (FLOAT4)(0.0f);
	FLOAT4    acc1           = (FLOAT4)(0.0f);
	FLOAT4    acc2           = (FLOAT4)(0.0f);
	FLOAT4    acc3           = (FLOAT4)(0.0f);
	
	FLOAT4 fetch_src = RI_F(src,    SAMPLER, (int2)(localx, src_start_y + localy));
	for (int i = TILE_DIM32; i < input_channel; i += TILE_DIM32)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		src_shared[localx][localy]    = fetch_src;
		barrier(CLK_LOCAL_MEM_FENCE);
		fetch_src                     = RI_F(src,    SAMPLER, (int2)(localx + i, src_start_y + localy));
		for (int k = 0; k < TILE_DIM32; k += 4)
		{
			__private int    cur_filter_x = (i - TILE_DIM32) + k;
			__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
			__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
			__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
			__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
			acc0 = mad(filter0_data, src_shared[k][localx].x, acc0);
			acc0 = mad(filter1_data, src_shared[k][localx].y, acc0);
			acc0 = mad(filter2_data, src_shared[k][localx].z, acc0);
			acc0 = mad(filter3_data, src_shared[k][localx].w, acc0);

			acc1 = mad(filter0_data, src_shared[k + 1][localx].x, acc1);
			acc1 = mad(filter1_data, src_shared[k + 1][localx].y, acc1);
			acc1 = mad(filter2_data, src_shared[k + 1][localx].z, acc1);
			acc1 = mad(filter3_data, src_shared[k + 1][localx].w, acc1);

			acc2 = mad(filter0_data, src_shared[k + 2][localx].x, acc2);
			acc2 = mad(filter1_data, src_shared[k + 2][localx].y, acc2);
			acc2 = mad(filter2_data, src_shared[k + 2][localx].z, acc2);
			acc2 = mad(filter3_data, src_shared[k + 2][localx].w, acc2);

			acc3 = mad(filter0_data, src_shared[k + 3][localx].x, acc3);
			acc3 = mad(filter1_data, src_shared[k + 3][localx].y, acc3);
			acc3 = mad(filter2_data, src_shared[k + 3][localx].z, acc3);
			acc3 = mad(filter3_data, src_shared[k + 3][localx].w, acc3);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	src_shared[localx][localy] = fetch_src;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = 0; k < TILE_DIM32; k += 4)
	{
		__private int    cur_filter_x = (input_channel - TILE_DIM32) + k;
		__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
		__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
		__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
		__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
		
		acc0 = mad(filter0_data, src_shared[k][localx].x, acc0);
		acc0 = mad(filter1_data, src_shared[k][localx].y, acc0);
		acc0 = mad(filter2_data, src_shared[k][localx].z, acc0);
		acc0 = mad(filter3_data, src_shared[k][localx].w, acc0);

		acc1 = mad(filter0_data, src_shared[k + 1][localx].x, acc1);
		acc1 = mad(filter1_data, src_shared[k + 1][localx].y, acc1);
		acc1 = mad(filter2_data, src_shared[k + 1][localx].z, acc1);
		acc1 = mad(filter3_data, src_shared[k + 1][localx].w, acc1);

		acc2 = mad(filter0_data, src_shared[k + 2][localx].x, acc2);
		acc2 = mad(filter1_data, src_shared[k + 2][localx].y, acc2);
		acc2 = mad(filter2_data, src_shared[k + 2][localx].z, acc2);
		acc2 = mad(filter3_data, src_shared[k + 2][localx].w, acc2);

		acc3 = mad(filter0_data, src_shared[k + 3][localx].x, acc3);
		acc3 = mad(filter1_data, src_shared[k + 3][localx].y, acc3);
		acc3 = mad(filter2_data, src_shared[k + 3][localx].z, acc3);
		acc3 = mad(filter3_data, src_shared[k + 3][localx].w, acc3);
	}
	
	WI_F(dst, (int2)(output_x, output_y),     acc0);
	WI_F(dst, (int2)(output_x, output_y + 1), acc1);
	WI_F(dst, (int2)(output_x, output_y + 2), acc2);
	WI_F(dst, (int2)(output_x, output_y + 3), acc3);
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(4, 16, 1)))
#endif
void gemm_local_mem_opt_4x16(__read_only  image2d_t  src,
	                         __read_only  image2d_t  filter,
	                         __write_only image2d_t  dst,
	                         __private const int     input_channel)
{
#ifdef AVOID_MEM_BANK_CONFLICT
	__local FLOAT4 filter_shared[TILE_DIM16][TILE_DIM16 + 1];
#else
  	__local FLOAT4 filter_shared[TILE_DIM16][TILE_DIM16];
#endif
	const int localx         = get_local_id(0), localy = get_local_id(1);
	const int globalx        = get_global_id(0), globaly = get_global_id(1), globalz = get_global_id(2);
	const int grpx           = get_group_id(0), grpy = get_group_id(1);
	const int image_start_x  = mul24(globalz, (int)get_global_size(0));
	const int output_x       = image_start_x + globalx;
	const int output_y       = (globaly << 2);
	const int src_start_y    = image_start_x + (grpx << 2);
	const int filter_start_y = mad24(globalz, (int)get_global_size(1), (grpy << 4));
	const int cur_filter_y   = filter_start_y + localy;
	FLOAT4    acc0           = (FLOAT4)(0.0f);
	FLOAT4    acc1           = (FLOAT4)(0.0f);
	FLOAT4    acc2           = (FLOAT4)(0.0f);
	FLOAT4    acc3           = (FLOAT4)(0.0f);
	int    filter_pos_x   = ((localy & 3) << 2) + localx;
	int    filter_pos_y   = (localy & 0xFFFFFFFC);

	FLOAT4 fetch_filter_0 = RI_F(filter, SAMPLER, (int2)(filter_pos_x, filter_start_y + filter_pos_y));
	FLOAT4 fetch_filter_1 = RI_F(filter, SAMPLER, (int2)(filter_pos_x, filter_start_y + filter_pos_y + 1));
	FLOAT4 fetch_filter_2 = RI_F(filter, SAMPLER, (int2)(filter_pos_x, filter_start_y + filter_pos_y + 2));
	FLOAT4 fetch_filter_3 = RI_F(filter, SAMPLER, (int2)(filter_pos_x, filter_start_y + filter_pos_y + 3));
	for (int i = TILE_DIM16; i < input_channel; i += TILE_DIM16)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		filter_shared[filter_pos_x][filter_pos_y]     = fetch_filter_0;
		filter_shared[filter_pos_x][filter_pos_y + 1] = fetch_filter_1;
		filter_shared[filter_pos_x][filter_pos_y + 2] = fetch_filter_2;
		filter_shared[filter_pos_x][filter_pos_y + 3] = fetch_filter_3;
		barrier(CLK_LOCAL_MEM_FENCE);
		fetch_filter_0 = RI_F(filter, SAMPLER, (int2)(filter_pos_x + i, filter_start_y + filter_pos_y));
		fetch_filter_1 = RI_F(filter, SAMPLER, (int2)(filter_pos_x + i, filter_start_y + filter_pos_y + 1));
		fetch_filter_2 = RI_F(filter, SAMPLER, (int2)(filter_pos_x + i, filter_start_y + filter_pos_y + 2));
		fetch_filter_3 = RI_F(filter, SAMPLER, (int2)(filter_pos_x + i, filter_start_y + filter_pos_y + 3));
		for (int k = 0; k < TILE_DIM16; k += 4)
		{
			__private int    cur_src_x = (i - TILE_DIM16) + k;
			__private FLOAT4 src0_data = RI_F(src, SAMPLER, (int2)(cur_src_x,     src_start_y + localx));
			__private FLOAT4 src1_data = RI_F(src, SAMPLER, (int2)(cur_src_x + 1, src_start_y + localx));
			__private FLOAT4 src2_data = RI_F(src, SAMPLER, (int2)(cur_src_x + 2, src_start_y + localx));
			__private FLOAT4 src3_data = RI_F(src, SAMPLER, (int2)(cur_src_x + 3, src_start_y + localx));
			acc0 = mad(filter_shared[k][localy],     src0_data.x, acc0);
			acc0 = mad(filter_shared[k + 1][localy], src0_data.y, acc0);
			acc0 = mad(filter_shared[k + 2][localy], src0_data.z, acc0);
			acc0 = mad(filter_shared[k + 3][localy], src0_data.w, acc0);

			acc1 = mad(filter_shared[k][localy],     src1_data.x, acc1);
			acc1 = mad(filter_shared[k + 1][localy], src1_data.y, acc1);
			acc1 = mad(filter_shared[k + 2][localy], src1_data.z, acc1);
			acc1 = mad(filter_shared[k + 3][localy], src1_data.w, acc1);

			acc2 = mad(filter_shared[k][localy],     src2_data.x, acc2);
			acc2 = mad(filter_shared[k + 1][localy], src2_data.y, acc2);
			acc2 = mad(filter_shared[k + 2][localy], src2_data.z, acc2);
			acc2 = mad(filter_shared[k + 3][localy], src2_data.w, acc2);

			acc3 = mad(filter_shared[k][localy],     src3_data.x, acc3);
			acc3 = mad(filter_shared[k + 1][localy], src3_data.y, acc3);
			acc3 = mad(filter_shared[k + 2][localy], src3_data.z, acc3);
			acc3 = mad(filter_shared[k + 3][localy], src3_data.w, acc3);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	filter_shared[filter_pos_x][filter_pos_y]     = fetch_filter_0;
	filter_shared[filter_pos_x][filter_pos_y + 1] = fetch_filter_1;
	filter_shared[filter_pos_x][filter_pos_y + 2] = fetch_filter_2;
	filter_shared[filter_pos_x][filter_pos_y + 3] = fetch_filter_3;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = 0; k < TILE_DIM16; k += 4)
	{
		__private int    cur_src_x = (input_channel - TILE_DIM16) + k;
		__private FLOAT4 src0_data = RI_F(src, SAMPLER, (int2)(cur_src_x,     src_start_y + localx));
		__private FLOAT4 src1_data = RI_F(src, SAMPLER, (int2)(cur_src_x + 1, src_start_y + localx));
		__private FLOAT4 src2_data = RI_F(src, SAMPLER, (int2)(cur_src_x + 2, src_start_y + localx));
		__private FLOAT4 src3_data = RI_F(src, SAMPLER, (int2)(cur_src_x + 3, src_start_y + localx));

		acc0 = mad(filter_shared[k][localy],     src0_data.x, acc0);
		acc0 = mad(filter_shared[k + 1][localy], src0_data.y, acc0);
		acc0 = mad(filter_shared[k + 2][localy], src0_data.z, acc0);
		acc0 = mad(filter_shared[k + 3][localy], src0_data.w, acc0);

		acc1 = mad(filter_shared[k][localy],     src1_data.x, acc1);
		acc1 = mad(filter_shared[k + 1][localy], src1_data.y, acc1);
		acc1 = mad(filter_shared[k + 2][localy], src1_data.z, acc1);
		acc1 = mad(filter_shared[k + 3][localy], src1_data.w, acc1);

		acc2 = mad(filter_shared[k][localy],     src2_data.x, acc2);
		acc2 = mad(filter_shared[k + 1][localy], src2_data.y, acc2);
		acc2 = mad(filter_shared[k + 2][localy], src2_data.z, acc2);
		acc2 = mad(filter_shared[k + 3][localy], src2_data.w, acc2);

		acc3 = mad(filter_shared[k][localy],     src3_data.x, acc3);
		acc3 = mad(filter_shared[k + 1][localy], src3_data.y, acc3);
		acc3 = mad(filter_shared[k + 2][localy], src3_data.z, acc3);
		acc3 = mad(filter_shared[k + 3][localy], src3_data.w, acc3);
	}

	WI_F(dst, (int2)(output_x, output_y),     acc0);
	WI_F(dst, (int2)(output_x, output_y + 1), acc1);
	WI_F(dst, (int2)(output_x, output_y + 2), acc2);
	WI_F(dst, (int2)(output_x, output_y + 3), acc3);
}

#define TILE_DIM8 8
__kernel 
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(8, 8, 1)))
#endif
void gemm_local_mem_8x(__read_only  image2d_t  src,
	                   __read_only  image2d_t  filter,
	                   __write_only image2d_t  dst,
	                   __private const int     input_channel)
{
	__local FLOAT4 src_shared[TILE_DIM8][TILE_DIM8];
	const int localx         = get_local_id(0), localy = get_local_id(1);
	const int globalx        = get_global_id(0), globaly = get_global_id(1), globalz = get_global_id(2);
	const int grpx           = get_group_id(0), grpy = get_group_id(1);
	const int image_start_x  = mul24(globalz, (int)get_global_size(0));
	const int output_x       = image_start_x + globalx;
	const int output_y       = (globaly << 2);
	const int src_start_y    = image_start_x + (grpx << 3);
	const int filter_start_y = mad24(globalz, (int)get_global_size(1), (grpy << 3));
	const int cur_filter_y   = filter_start_y + localy;
	FLOAT4 acc0 = (FLOAT4)(0.0f);
	FLOAT4 acc1 = (FLOAT4)(0.0f);
	FLOAT4 acc2 = (FLOAT4)(0.0f);
	FLOAT4 acc3 = (FLOAT4)(0.0f);
	
	FLOAT4 fetch_src = RI_F(src,    SAMPLER, (int2)(localx, src_start_y + localy));
	for (int i = TILE_DIM8; i < input_channel; i += TILE_DIM8)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		src_shared[localx][localy]    = fetch_src;
		barrier(CLK_LOCAL_MEM_FENCE);
		fetch_src                     = RI_F(src,    SAMPLER, (int2)(localx + i, src_start_y + localy));
		for (int k = 0; k < TILE_DIM8; k += 4)
		{
			__private int    cur_filter_x = (i - TILE_DIM8) + k;
			__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
			__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
			__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
			__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
			acc0 = mad(filter0_data, src_shared[k][localx].x, acc0);
			acc0 = mad(filter1_data, src_shared[k][localx].y, acc0);
			acc0 = mad(filter2_data, src_shared[k][localx].z, acc0);
			acc0 = mad(filter3_data, src_shared[k][localx].w, acc0);

			acc1 = mad(filter0_data, src_shared[k + 1][localx].x, acc1);
			acc1 = mad(filter1_data, src_shared[k + 1][localx].y, acc1);
			acc1 = mad(filter2_data, src_shared[k + 1][localx].z, acc1);
			acc1 = mad(filter3_data, src_shared[k + 1][localx].w, acc1);

			acc2 = mad(filter0_data, src_shared[k + 2][localx].x, acc2);
			acc2 = mad(filter1_data, src_shared[k + 2][localx].y, acc2);
			acc2 = mad(filter2_data, src_shared[k + 2][localx].z, acc2);
			acc2 = mad(filter3_data, src_shared[k + 2][localx].w, acc2);

			acc3 = mad(filter0_data, src_shared[k + 3][localx].x, acc3);
			acc3 = mad(filter1_data, src_shared[k + 3][localx].y, acc3);
			acc3 = mad(filter2_data, src_shared[k + 3][localx].z, acc3);
			acc3 = mad(filter3_data, src_shared[k + 3][localx].w, acc3);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	src_shared[localx][localy] = fetch_src;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = 0; k < TILE_DIM8; k += 4)
	{
		__private int    cur_filter_x = (input_channel - TILE_DIM8) + k;
		__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y));
		__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y));
		__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y));
		__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y));
		
		acc0 = mad(filter0_data, src_shared[k][localx].x, acc0);
		acc0 = mad(filter1_data, src_shared[k][localx].y, acc0);
		acc0 = mad(filter2_data, src_shared[k][localx].z, acc0);
		acc0 = mad(filter3_data, src_shared[k][localx].w, acc0);

		acc1 = mad(filter0_data, src_shared[k + 1][localx].x, acc1);
		acc1 = mad(filter1_data, src_shared[k + 1][localx].y, acc1);
		acc1 = mad(filter2_data, src_shared[k + 1][localx].z, acc1);
		acc1 = mad(filter3_data, src_shared[k + 1][localx].w, acc1);

		acc2 = mad(filter0_data, src_shared[k + 2][localx].x, acc2);
		acc2 = mad(filter1_data, src_shared[k + 2][localx].y, acc2);
		acc2 = mad(filter2_data, src_shared[k + 2][localx].z, acc2);
		acc2 = mad(filter3_data, src_shared[k + 2][localx].w, acc2);

		acc3 = mad(filter0_data, src_shared[k + 3][localx].x, acc3);
		acc3 = mad(filter1_data, src_shared[k + 3][localx].y, acc3);
		acc3 = mad(filter2_data, src_shared[k + 3][localx].z, acc3);
		acc3 = mad(filter3_data, src_shared[k + 3][localx].w, acc3);
	}
	
	WI_F(dst, (int2)(output_x, output_y),     acc0);
	WI_F(dst, (int2)(output_x, output_y + 1), acc1);
	WI_F(dst, (int2)(output_x, output_y + 2), acc2);
	WI_F(dst, (int2)(output_x, output_y + 3), acc3);
}

__kernel 
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(8, 8, 1)))
#endif
void gemm_local_mem_opt_8x_dual_filter(__read_only  image2d_t  src,
	                                   __read_only  image2d_t  filter,
	                                   __write_only image2d_t  dst,
                                       __private const int     src_height_q,
                                       __private const int     input_channel)
{
	__local FLOAT4 src_shared[TILE_DIM8][TILE_DIM8];
	const int localx          = get_local_id(0), localy = get_local_id(1);
	const int globalx         = get_global_id(0), globaly = get_global_id(1), globalz = get_global_id(2);
	const int grpx            = get_group_id(0), grpy = get_group_id(1);
#ifndef CHECK_SRC_BORDER	
	const int  image_start_x   = mul24(globalz, (int)get_global_size(0));
#else
	const int  image_start_x   = mul24(globalz, src_height_q);
#endif
	const int output_x        = image_start_x + globalx;
	const int output_y0       = (((grpy << 4) + localy)     << 2) ;//(globaly << 2);
	const int output_y1       = (((grpy << 4) + localy + 8) << 2) ;//(globaly << 2);
#ifndef CHECK_SRC_BORDER
	const int  src_start_y    = image_start_x + (grpx << 3) + localy;
#else
    int        src_start_y    = image_start_x + (grpx << 3) + localy;
	src_start_y               = select(src_start_y, -1, ((grpx << 3) + localy) >= src_height_q);
#endif
	const int filter_start_y0 = mad24(globalz, ((int)get_global_size(1)) << 1, (grpy << 4));
	const int filter_start_y1 = mad24(globalz, ((int)get_global_size(1)) << 1, (grpy << 4) + 8);
	const int cur_filter_y0   = filter_start_y0 + localy;
	const int cur_filter_y1   = filter_start_y1 + localy;
	FLOAT4 acc0 = (FLOAT4)(0.0f);
	FLOAT4 acc1 = (FLOAT4)(0.0f);
	FLOAT4 acc2 = (FLOAT4)(0.0f);
	FLOAT4 acc3 = (FLOAT4)(0.0f);
	FLOAT4 acc4 = (FLOAT4)(0.0f);
	FLOAT4 acc5 = (FLOAT4)(0.0f);
	FLOAT4 acc6 = (FLOAT4)(0.0f);
	FLOAT4 acc7 = (FLOAT4)(0.0f);
	
	FLOAT4 fetch_src = RI_F(src,    SAMPLER, (int2)(localx, src_start_y));
	for (int i = TILE_DIM8; i < input_channel; i += TILE_DIM8)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		src_shared[localx][localy]    = fetch_src;
		barrier(CLK_LOCAL_MEM_FENCE);
		fetch_src                     = RI_F(src,    SAMPLER, (int2)(localx + i, src_start_y));
		for (int k = 0; k < TILE_DIM8; k += 4)
		{
			__private int    cur_filter_x = (i - TILE_DIM8) + k;
			__private FLOAT4 src0_data    = src_shared[k][localx];
			__private FLOAT4 src1_data    = src_shared[k + 1][localx];
			__private FLOAT4 src2_data    = src_shared[k + 2][localx];
			__private FLOAT4 src3_data    = src_shared[k + 3][localx];
			__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y0));
			__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y0));
			__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y0));
			__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y0));
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
			
			filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y1));
			filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y1));
			filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y1));
			filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y1));
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
	src_shared[localx][localy] = fetch_src;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = 0; k < TILE_DIM8; k += 4)
	{
		__private FLOAT4 src0_data    = src_shared[k][localx];
		__private FLOAT4 src1_data    = src_shared[k + 1][localx];
		__private FLOAT4 src2_data    = src_shared[k + 2][localx];
		__private FLOAT4 src3_data    = src_shared[k + 3][localx];
		
		__private int    cur_filter_x = (input_channel - TILE_DIM8) + k;
		__private FLOAT4 filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y0));
		__private FLOAT4 filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y0));
		__private FLOAT4 filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y0));
		__private FLOAT4 filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y0));
		
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
		
		filter0_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x,     cur_filter_y1));
		filter1_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 1, cur_filter_y1));
		filter2_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 2, cur_filter_y1));
		filter3_data = RI_F(filter, SAMPLER, (int2)(cur_filter_x + 3, cur_filter_y1));
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
	
#ifdef CHECK_SRC_BORDER
	if(globalx < src_height_q)
	{
#endif	
		WI_F(dst, (int2)(output_x, output_y0),      acc0);
		WI_F(dst, (int2)(output_x, output_y0 + 1),  acc1);
		WI_F(dst, (int2)(output_x, output_y0 + 2),  acc2);
		WI_F(dst, (int2)(output_x, output_y0 + 3),  acc3);
		
		WI_F(dst, (int2)(output_x, output_y1),      acc4);
		WI_F(dst, (int2)(output_x, output_y1 + 1),  acc5);
		WI_F(dst, (int2)(output_x, output_y1 + 2),  acc6);
		WI_F(dst, (int2)(output_x, output_y1 + 3),  acc7);
#ifdef CHECK_SRC_BORDER
	}
#endif	
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(8, 8, 1)))
#endif
void gemm_local_mem_opt_8x8_dual_input(__read_only  image2d_t  src,
	                                   __read_only  image2d_t  filter,
	                                   __write_only image2d_t  dst,
									   __private const int     src_height_q,
	                                   __private const int     src_channel)
{
	__local FLOAT4 filter_shared[TILE_DIM8][TILE_DIM8];
	const int  localx          = get_local_id(0), localy = get_local_id(1);
	const int  globalx         = get_global_id(0), globaly = get_global_id(1), globalz = get_global_id(2);
	const int  grpx            = get_group_id(0), grpy = get_group_id(1);
	const int  image_start_x   = mul24(globalz, src_height_q);
	const int  src_pos_x       = (grpx << 4) + localx;
	int        output_x        = image_start_x + src_pos_x;
	const int  output_y        = (globaly << 2);
#ifndef CHECK_SRC_BORDER
	const int  src_start_y0    = output_x;
	const int  src_start_y1    = src_start_y0 + 8;
#else
	const int  src_start_y0    = select(output_x,     -1, src_pos_x       >= src_height_q);
	const int  src_start_y1    = select(output_x + 8, -1, (src_pos_x + 8) >= src_height_q);
#endif
	const int  filter_start_y  = mad24(globalz, (int)get_global_size(1), (grpy << 3));
	const int  cur_filter_y    = filter_start_y + localy;
	
	FLOAT4 acc0 = (FLOAT4)(0.0f);
	FLOAT4 acc1 = (FLOAT4)(0.0f);
	FLOAT4 acc2 = (FLOAT4)(0.0f);
	FLOAT4 acc3 = (FLOAT4)(0.0f);
	FLOAT4 acc4 = (FLOAT4)(0.0f);
	FLOAT4 acc5 = (FLOAT4)(0.0f);
	FLOAT4 acc6 = (FLOAT4)(0.0f);
	FLOAT4 acc7 = (FLOAT4)(0.0f);
	
	FLOAT4 fetch_filter = RI_F(filter,    SAMPLER, (int2)(localx, cur_filter_y));
	for (int i = TILE_DIM8; i < src_channel; i += TILE_DIM8)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		filter_shared[localx][localy]    = fetch_filter;
		barrier(CLK_LOCAL_MEM_FENCE);
		fetch_filter                     = RI_F(filter,    SAMPLER, (int2)(localx + i, cur_filter_y));
		for (int k = 0; k < TILE_DIM8; k += 4)
		{
			int    cur_src_x       = (i - TILE_DIM8) + k;
			int4   cur_src_x4      = (int4)(cur_src_x, cur_src_x + 1, cur_src_x + 2, cur_src_x + 3);
			FLOAT4 filter0_data    = filter_shared[k][localy];
			FLOAT4 filter1_data    = filter_shared[k + 1][localy];
			FLOAT4 filter2_data    = filter_shared[k + 2][localy];
			FLOAT4 filter3_data    = filter_shared[k + 3][localy];
			FLOAT4 src0_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.x, src_start_y0));
			FLOAT4 src1_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.y, src_start_y0));
			FLOAT4 src2_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.z, src_start_y0));
			FLOAT4 src3_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.w, src_start_y0));
			
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
			
			src0_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.x, src_start_y1));
			src1_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.y, src_start_y1));
			src2_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.z, src_start_y1));
			src3_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.w, src_start_y1));
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
	filter_shared[localx][localy] = fetch_filter;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = 0; k < TILE_DIM8; k += 4)
	{
		int    cur_src_x       = (src_channel - TILE_DIM8) + k;
		int4   cur_src_x4      = (int4)(cur_src_x, cur_src_x + 1, cur_src_x + 2, cur_src_x + 3);
		FLOAT4 filter0_data    = filter_shared[k][localy];
		FLOAT4 filter1_data    = filter_shared[k + 1][localy];
		FLOAT4 filter2_data    = filter_shared[k + 2][localy];
		FLOAT4 filter3_data    = filter_shared[k + 3][localy];
		FLOAT4 src0_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.x, src_start_y0));
		FLOAT4 src1_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.y, src_start_y0));
		FLOAT4 src2_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.z, src_start_y0));
		FLOAT4 src3_data       = RI_F(src, SAMPLER, (int2)(cur_src_x4.w, src_start_y0));
		
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
		
		src0_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.x, src_start_y1));
		src1_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.y, src_start_y1));
		src2_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.z, src_start_y1));
		src3_data    = RI_F(src, SAMPLER, (int2)(cur_src_x4.w, src_start_y1));
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

#ifdef CHECK_SRC_BORDER
	if((src_pos_x + 8) < src_height_q)
	{
#endif
		WI_F(dst, (int2)(output_x,     output_y),      acc0);
		WI_F(dst, (int2)(output_x,     output_y + 1),  acc1);
		WI_F(dst, (int2)(output_x,     output_y + 2),  acc2);
		WI_F(dst, (int2)(output_x,     output_y + 3),  acc3);
		output_x += 8;
		WI_F(dst, (int2)(output_x,     output_y),      acc4);
		WI_F(dst, (int2)(output_x,     output_y + 1),  acc5);
		WI_F(dst, (int2)(output_x,     output_y + 2),  acc6);
		WI_F(dst, (int2)(output_x,     output_y + 3),  acc7);
#ifdef CHECK_SRC_BORDER
	}
	else if(src_pos_x < src_height_q)
	{
		WI_F(dst, (int2)(output_x,     output_y),      acc0);
		WI_F(dst, (int2)(output_x,     output_y + 1),  acc1);
		WI_F(dst, (int2)(output_x,     output_y + 2),  acc2);
		WI_F(dst, (int2)(output_x,     output_y + 3),  acc3);
	}
#endif
}