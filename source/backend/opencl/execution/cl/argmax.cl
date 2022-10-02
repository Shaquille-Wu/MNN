#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void argmax_channel(__read_only image2d_t   input, 
                             __write_only image2d_t  output, 
                             __private const int     input_width,
                             __private const int     input_height,      //batch_height
                             __private const int     input_channels) 
{
	int  gx  = get_global_id(0);
	int  gy  = get_global_id(1);
	if((gx >= input_width) || (gy >= input_height))
		return;

	int     input_channel_q    = (input_channels >> 2);
	int     input_channel_tail = (input_channels & 3);
	int4    sel_idx0           = (int4)(0, 1, 2, 3);
	int4    cur_sel            = sel_idx0;
	int     pos_x              = gx;
#ifdef CHECK_IMAGE_BORDER
    int     check_pos          = (pos_x >= input_width);
	int     cur_pos_x          = select(pos_x, -1, check_pos);
#else
	int     cur_pos_x          = pos_x;
#endif

#ifdef CHANNEL_GE_4
	int     i          = 0;
	FLOAT4  in_max0    = RI_F(input, SAMPLER, (int2)(cur_pos_x, gy));
	pos_x             += input_width;
	cur_sel           += (int4)(4);
	for(i = 1 ; i < input_channel_q ; ++i)
	{
#ifdef CHECK_IMAGE_BORDER
		cur_pos_x      = select(pos_x, (int4)(-1), check_pos);
#else
		cur_pos_x      = pos_x;
#endif
		FLOAT4 in0     = RI_F(input, SAMPLER, (int2)(cur_pos_x, gy));
		int4  compare0 = (int4)(in0.x > in_max0.x, 
		                        in0.y > in_max0.y, 
		                        in0.z > in_max0.z, 
		                        in0.w > in_max0.w);
		sel_idx0       = select(sel_idx0, cur_sel, compare0);
		in_max0        = select(in_max0,      in0, compare0);
		pos_x         += (input_width);
		cur_sel       += (int4)(4);
	}
#else
	FLOAT4  in_max0    = (FLOAT4)(-FLT_MAX);
#endif

#ifdef PROC_CHANNEL_TAIL
	{ 
#ifdef CHECK_IMAGE_BORDER
		cur_pos_x         = select(pos_x, -1, check_pos);
#else
		cur_pos_x         = pos_x;
#endif
		FLOAT4 in0        = RI_F(input, SAMPLER, (int2)(cur_pos_x, gy));
#ifdef PROC_CHANNEL_TAIL_3
		int comparex      = (in0.x > in_max0.x);
		int comparey      = (in0.y > in_max0.y);
		int comparez      = (in0.z > in_max0.z);
		sel_idx0.x        = select(sel_idx0.x, cur_sel.x, comparex);
		in_max0.x         = select(in_max0.x,  in0.x,     comparex);
		sel_idx0.y        = select(sel_idx0.y, cur_sel.y, comparey);
		in_max0.y         = select(in_max0.y,  in0.y,     comparey);
		sel_idx0.z        = select(sel_idx0.z, cur_sel.z, comparez);
		in_max0.z         = select(in_max0.z,  in0.z,     comparez);
#else
#ifdef PROC_CHANNEL_TAIL_2
		int comparex      = (in0.x > in_max0.x);
		int comparey      = (in0.y > in_max0.y);
		sel_idx0.x        = select(sel_idx0.x, cur_sel.x, comparex);
		in_max0.x         = select(in_max0.x,  in0.x,     comparex);
		sel_idx0.y        = select(sel_idx0.y, cur_sel.y, comparey);
		in_max0.y         = select(in_max0.y,  in0.y,     comparey);
#else
#ifdef PROC_CHANNEL_TAIL_1
		bool comparex     = (in0.x > in_max0.x);
		sel_idx0.x        = select(sel_idx0.x, cur_sel.x, comparex);
		in_max0.x         = select(in_max0.x,  in0.x,     comparex);
#endif//PROC_CHANNEL_TAIL_1
#endif//PROC_CHANNEL_TAIL_2
#endif//PROC_CHANNEL_TAIL_3
	}
#endif

	FLOAT4  output_res        = (FLOAT4)(sel_idx0.x, 0.0f, 0.0f, 0.0f);
#ifdef PROC_CHANNEL_4
	int     compare_xy        = (in_max0.x < in_max0.y);
	int     result_idx_xy     = select(sel_idx0.x, sel_idx0.y,  compare_xy);
	FLOAT   result_max_xy     = select(in_max0.x,  in_max0.y,   compare_xy);
	int     compare_zw        = (in_max0.z < in_max0.w);
	int     result_idx_zw     = select(sel_idx0.z, sel_idx0.w,  compare_zw);
	FLOAT   result_max_zw     = select(in_max0.z,  in_max0.w,   compare_zw);
	int     compare_xyzw      = (result_max_xy < result_max_zw);
	int     result_idx_xyzw   = select(result_idx_xy,  result_idx_zw,   compare_xyzw);
	output_res.x              = (FLOAT)(result_idx_xyzw);
#else
#ifdef PROC_CHANNEL_3
	int     compare_xy        = (in_max0.x < in_max0.y);
	int     result_idx_xy     = select(sel_idx0.x, sel_idx0.y,  compare_xy);
	FLOAT   result_max_xy     = select(in_max0.x,  in_max0.y,   compare_xy);
	int     compare_xyzw      = (result_max_xy < in_max0.z);
	int     result_idx_xyzw   = select(result_idx_xy,  sel_idx0.z,   compare_xyzw);
	output_res.x              = (FLOAT)(result_idx_xyzw);
#else
#ifdef PROC_CHANNEL_2
	int     compare_xy        = (in_max0.x < in_max0.y);
	int     result_idx_xy     = select(sel_idx0.x, sel_idx0.y,  compare_xy);
	output_res.x              = (FLOAT)(result_idx_xy);
#else  //PROC_CHANNEL_1
	output_res.x              = (FLOAT)(0.0f);
#endif //PROC_CHANNEL_2
#endif //PROC_CHANNEL_3
#endif //PROC_CHANNEL_4

	WI_F(output, (int2)(gx, gy),  output_res);
}