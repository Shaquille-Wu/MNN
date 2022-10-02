#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define  ACTIVE_RELU_4OUT(res0, res1, res2, res3)                              \
    res0 = fmax(res0, (FLOAT4)0);                                              \
    res1 = fmax(res1, (FLOAT4)0);                                              \
    res2 = fmax(res2, (FLOAT4)0);                                              \
    res3 = fmax(res3, (FLOAT4)0);

#define  ACTIVE_RELU6_4OUT(res0, res1, res2, res3)                             \
    res0 = clamp(res0, (FLOAT4)0, (FLOAT4)6);                                  \
    res1 = clamp(res1, (FLOAT4)0, (FLOAT4)6);                                  \
    res2 = clamp(res2, (FLOAT4)0, (FLOAT4)6);                                  \
    res3 = clamp(res3, (FLOAT4)0, (FLOAT4)6);

#define  ACTIVE_LEAKY_RELU_4OUT(res0, res1, res2, res3)                        \
    res0 = select(((FLOAT)(LEAKY_RELU))*res0, res0, res0 >= (FLOAT4)0);        \
    res1 = select(((FLOAT)(LEAKY_RELU))*res1, res1, res1 >= (FLOAT4)0);        \
    res2 = select(((FLOAT)(LEAKY_RELU))*res2, res2, res2 >= (FLOAT4)0);        \
    res3 = select(((FLOAT)(LEAKY_RELU))*res3, res3, res3 >= (FLOAT4)0);

#define  ACTIVE_RELU(res0, res1, res2, res3, res4, res5, res6, res7)           \
    res0 = fmax(res0, (FLOAT4)0);                                              \
    res1 = fmax(res1, (FLOAT4)0);                                              \
    res2 = fmax(res2, (FLOAT4)0);                                              \
    res3 = fmax(res3, (FLOAT4)0);                                              \
    res4 = fmax(res4, (FLOAT4)0);                                              \
    res5 = fmax(res5, (FLOAT4)0);                                              \
    res6 = fmax(res6, (FLOAT4)0);                                              \
    res7 = fmax(res7, (FLOAT4)0);

#define  ACTIVE_RELU6(res0, res1, res2, res3, res4, res5, res6, res7)          \
    res0 = clamp(res0, (FLOAT4)0, (FLOAT4)6);                                  \
    res1 = clamp(res1, (FLOAT4)0, (FLOAT4)6);                                  \
    res2 = clamp(res2, (FLOAT4)0, (FLOAT4)6);                                  \
    res3 = clamp(res3, (FLOAT4)0, (FLOAT4)6);                                  \
    res4 = clamp(res4, (FLOAT4)0, (FLOAT4)6);                                  \
    res5 = clamp(res5, (FLOAT4)0, (FLOAT4)6);                                  \
    res6 = clamp(res6, (FLOAT4)0, (FLOAT4)6);                                  \
    res7 = clamp(res7, (FLOAT4)0, (FLOAT4)6);

#define  ACTIVE_LEAKY_RELU(res0, res1, res2, res3, res4, res5, res6, res7)     \
    res0 = select(((FLOAT)(LEAKY_RELU))*res0, res0, res0 >= (FLOAT4)0);        \
    res1 = select(((FLOAT)(LEAKY_RELU))*res1, res1, res1 >= (FLOAT4)0);        \
    res2 = select(((FLOAT)(LEAKY_RELU))*res2, res2, res2 >= (FLOAT4)0);        \
    res3 = select(((FLOAT)(LEAKY_RELU))*res3, res3, res3 >= (FLOAT4)0);        \
    res4 = select(((FLOAT)(LEAKY_RELU))*res4, res4, res4 >= (FLOAT4)0);        \
    res5 = select(((FLOAT)(LEAKY_RELU))*res5, res5, res5 >= (FLOAT4)0);        \
    res6 = select(((FLOAT)(LEAKY_RELU))*res6, res6, res6 >= (FLOAT4)0);        \
    res7 = select(((FLOAT)(LEAKY_RELU))*res7, res7, res7 >= (FLOAT4)0);

__kernel void conv_3d_s1_out_4channel(__private const int       global_size_dim0,
                                      __private const int       global_size_dim1,
                                      __read_only  image2d_t    input,
                                      __read_only  image2d_t    weights,
#ifdef BIAS
                                      __read_only  image2d_t    bias,
#endif
                                      __write_only image2d_t    output,
                                      __private const int       in_channel_q,
                                      __private const int4      image_shape,
                                      __private const int4      kernel_shape,
                                      __private const int4      padding_shape,
                                      __private const int       out_channel_q)
{
    __private const int gx = get_global_id(0);
    __private const int gy = get_global_id(1);
    if (gx >= global_size_dim0 || gy >= global_size_dim1)
        return;
    __private const int out_channel_q_idx = gx / image_shape.z;
    __private const int pos_x             = gx - mul24(out_channel_q_idx, image_shape.z);
    __private const int h_q               = ((image_shape.y + 3) >> 2);
    __private const int depth_heightq     = mul24(image_shape.x, h_q);
    __private const int batch_idx         = gy / depth_heightq;
    __private const int pos_depth_yq      = gy - mul24(batch_idx, depth_heightq);
    __private const int depth_idx         = pos_depth_yq / h_q;
    __private const int pos_yq_idx        = pos_depth_yq - mul24(depth_idx, h_q);
    __private const int out_y_pos         = (pos_yq_idx << 2);
    __private const int batch_pos_base    = mul24(mul24(batch_idx, image_shape.x), image_shape.y);
    __private const int depth_pos_base    = mad24(depth_idx, image_shape.y, batch_pos_base);
#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_q_idx, 0));
#else
    FLOAT4 out0 = (FLOAT4)0.0f;
#endif
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int in_z_idx        = depth_idx - padding_shape.x;
    const int in_y_idx        = out_y_pos - padding_shape.y;
    const int kernel_size_xyz = mul24(mul24(kernel_shape.x, kernel_shape.y), kernel_shape.z);
    const int kernel_size_yz  = mul24(kernel_shape.y, kernel_shape.z);
    const int weights_z_base  = mul24(out_channel_q_idx, kernel_size_xyz);
    int4      in_y_ofs = (int4)(in_y_idx, in_y_idx + 1, in_y_idx + 2, in_y_idx + 3);
    for (int z = 0; z < kernel_shape.x; z++)
    {
        int        in_z_pos  = in_z_idx + z;
        int4       in_y_pos  = select(in_y_ofs, (int4)(-1), (in_y_ofs < ((int4)(0)) || in_y_ofs >= (int4)(image_shape.y)));
        int4       in_z_flag = (int4)(in_z_pos < 0 || in_z_pos >= image_shape.x);
        const int  in_y_base = select((batch_pos_base + mul24(in_z_pos, image_shape.y)), -1, in_z_flag.x);
        in_y_pos = select(((int4)(batch_pos_base + mul24(in_z_pos, image_shape.y))) + in_y_pos, (int4)(-1), in_z_flag || (in_y_pos < (int4)0));
        const int  weights_y_base = weights_z_base + mul24(z, kernel_size_yz);
        __private FLOAT4 weights0, weights1, weights2, weights3;
        __private FLOAT4 in0, in1, in2, in3;
        int in_channel_q_idx = 0;
#ifdef IN_CHANNEL_LOOP
        for (in_channel_q_idx = 0; in_channel_q_idx < in_channel_q; ++in_channel_q_idx)
        {
#endif
            int        in_x_idx = pos_x - padding_shape.z;
            const int  in_x_base = mul24(in_channel_q_idx, image_shape.z);
            int        weights_x_idx = in_channel_q_idx << 2;
            for (int w = 0; w < kernel_shape.z; w++)
            {
                int cur_x_pos = select(in_x_base + in_x_idx, -1, (in_x_idx < 0 || in_x_idx >= image_shape.z));
                in_x_idx++;

				in1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.x));
				in2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.y));
				in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.z));
				for (int h = 0; h < kernel_shape.y; h++)
				{
                    int    weight_y_idx = weights_y_base + mad24(h, kernel_shape.z, w);
                    in0 = in1;
                    in1 = in2;
                    in2 = in3;
                    int    cur_in_y_pos = in_y_ofs.w + h;
                    cur_in_y_pos = select(in_y_base + cur_in_y_pos, -1, (in_z_flag.x || cur_in_y_pos < 0 || cur_in_y_pos >= image_shape.y));
                    in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos));

                    weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weight_y_idx));
                    weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weight_y_idx));
                    weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weight_y_idx));
                    weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weight_y_idx));

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
                }
            }
#ifdef IN_CHANNEL_LOOP
        }
#endif
    }

#ifdef RELU
    ACTIVE_RELU_4OUT(out0, out1, out2, out3);
#endif

#ifdef RELU6
    ACTIVE_RELU6_4OUT(out0, out1, out2, out3);
#endif

#ifdef LEAKY_RELU
    ACTIVE_LEAKY_RELU_4OUT(out0, out1, out2, out3);
#endif
    const int remain     = image_shape.y - out_y_pos;
    const int output_y   = depth_pos_base + out_y_pos;
    const int out_x_base = mul24(out_channel_q_idx, image_shape.z);
    const int output_idx = out_x_base + pos_x;

    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, output_y),     out0);
        WI_F(output, (int2)(output_idx, output_y + 1), out1);
        WI_F(output, (int2)(output_idx, output_y + 2), out2);
        WI_F(output, (int2)(output_idx, output_y + 3), out3);
    }
    else if (remain == 3) {
        WI_F(output, (int2)(output_idx, output_y),     out0);
        WI_F(output, (int2)(output_idx, output_y + 1), out1);
        WI_F(output, (int2)(output_idx, output_y + 2), out2);
    }
    else if (remain == 2) {
        WI_F(output, (int2)(output_idx, output_y),     out0);
        WI_F(output, (int2)(output_idx, output_y + 1), out1);
    }
    else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_y), out0);
    }
}

__kernel void conv_3d_k3_s1_out_depth_as_channel(__private const int       global_size_dim0,
	                                             __private const int       global_size_dim1,
	                                             __read_only  image2d_t    input,
	                                             __read_only  image2d_t    weights,
#ifdef BIAS
	                                             __read_only  image2d_t    bias,
#endif
	                                             __write_only image2d_t    output,
	                                             __private const int       in_channel_q,
	                                             __private const int4      image_shape,
	                                             __private const int4      padding_shape)
{
	__private const int gx = get_global_id(0);
	__private const int gy = get_global_id(1);
	if (gx >= global_size_dim0 || gy >= global_size_dim1)
		return;

	__private const int  pos_x             = gx;
	__private const int  h_q               = ((image_shape.y + 3) >> 2);
	__private const int  d_q               = ((image_shape.x + 3) >> 2);
	__private const int  depthq_heightq    = mul24(d_q, h_q);
	__private const int  batch_idx         = gy / depthq_heightq;
	__private const int  pos_depthq_yq     = gy - mul24(batch_idx, depthq_heightq);
	__private const int  depth_q_idx       = pos_depthq_yq / h_q;
	__private const int4 depth_idx         = (int4)(depth_q_idx << 2, (depth_q_idx << 2) + 1, (depth_q_idx << 2) + 2, (depth_q_idx << 2) + 3);
	__private const int  pos_yq_idx        = pos_depthq_yq - mul24(depth_q_idx, h_q);
	__private const int  out_y_pos         = (pos_yq_idx << 2);
	__private const int  batch_pos_base    = batch_idx * image_shape.x * image_shape.y;
	__private const int  out_batch_base    = batch_idx * image_shape.y;

#ifdef BIAS
	FLOAT4 out0                           = RI_F(bias, SAMPLER, (int2)(0, 0));
	out0.y                                = out0.x;
	out0.z                                = out0.x;
	out0.w                                = out0.x;
#else
	FLOAT4 out0                           = (FLOAT4)0.0f;
#endif
	FLOAT4 out1                           = out0;
	FLOAT4 out2                           = out0;
	FLOAT4 out3                           = out0;

	const int4  in_z_idx                  = depth_idx - (int4)(padding_shape.x);
	const int   in_y_idx                  = out_y_pos - padding_shape.y;
	int4        in_y_ofs0                 = (int4)(in_y_idx, in_y_idx + 1, in_y_idx + 2, in_y_idx + 3);
	int4        in_y_ofs1                 = in_y_ofs0 + (int4)(4);
	for (int z = 0; z < 3; z++)
	{
		const int4  in_z_pos = in_z_idx + (int4)(z);
		int4  in_y_pos0      = select(in_y_ofs0, (int4)(-1), (in_y_ofs0 < ((int4)(0)) || in_y_ofs0 >= (int4)(image_shape.y)));
		int4  in_y_pos1      = select(in_y_ofs1, (int4)(-1), (in_y_ofs1 < ((int4)(0)) || in_y_ofs1 >= (int4)(image_shape.y)));
		int4  in_z_flag      = (int4)(in_z_pos.x < 0 || in_z_pos.x >= image_shape.x);
		int4  in_y_pos0_0    = select((int4)(batch_pos_base + mul24(in_z_pos.x, image_shape.y)) + in_y_pos0, (int4)(-1), in_z_flag || (in_y_pos0 < (int4)0));
		int4  in_y_pos1_0    = select((int4)(batch_pos_base + mul24(in_z_pos.x, image_shape.y)) + in_y_pos1, (int4)(-1), in_z_flag || (in_y_pos1 < (int4)0));
		
		in_z_flag            = (int4)(in_z_pos.y < 0 || in_z_pos.y >= image_shape.x);
		int4  in_y_pos0_1    = select((int4)(batch_pos_base + mul24(in_z_pos.y, image_shape.y)) + in_y_pos0, (int4)(-1), in_z_flag || (in_y_pos0 < (int4)0));
		int4  in_y_pos1_1    = select((int4)(batch_pos_base + mul24(in_z_pos.y, image_shape.y)) + in_y_pos1, (int4)(-1), in_z_flag || (in_y_pos1 < (int4)0));

		in_z_flag            = (int4)(in_z_pos.z < 0 || in_z_pos.z >= image_shape.x);
		int4  in_y_pos0_2    = select((int4)(batch_pos_base + mul24(in_z_pos.z, image_shape.y)) + in_y_pos0, (int4)(-1), in_z_flag || (in_y_pos0 < (int4)0));
		int4  in_y_pos1_2    = select((int4)(batch_pos_base + mul24(in_z_pos.z, image_shape.y)) + in_y_pos1, (int4)(-1), in_z_flag || (in_y_pos1 < (int4)0));

		in_z_flag            = ((int4)(in_z_pos.w < 0 || in_z_pos.w >= image_shape.x));
		int4  in_y_pos0_3    = select((int4)(batch_pos_base + mul24(in_z_pos.w, image_shape.y)) + in_y_pos0, (int4)(-1), in_z_flag || (in_y_pos0 < (int4)0));
		int4  in_y_pos1_3    = select((int4)(batch_pos_base + mul24(in_z_pos.w, image_shape.y)) + in_y_pos1, (int4)(-1), in_z_flag || (in_y_pos1 < (int4)0));

		FLOAT4 weights0, weights1, weights2;
		for (int in_channel_q_idx = 0; in_channel_q_idx < in_channel_q; ++in_channel_q_idx)
		{
			int        in_x_idx      = pos_x - padding_shape.z;
			const int  in_x_base     = mul24(in_channel_q_idx, image_shape.z);
			int        weights_x_idx = in_channel_q_idx;
			int        weight_y_idx  = mul24(z, 9);
			for (int w = 0; w < 3; w++)
			{
				int cur_x_pos = select(in_x_base + in_x_idx, -1, (in_x_idx < 0 || in_x_idx >= image_shape.z));
				in_x_idx++;

				FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_0.x));
				FLOAT4 in1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_0.y));
				FLOAT4 in2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_0.z));
				FLOAT4 in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_0.w));
				FLOAT4 in4 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_0.x));
				FLOAT4 in5 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_0.y));

				weights0   = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weight_y_idx));
				out0.x    += dot(in0, weights0);
				out1.x    += dot(in1, weights0);
				out2.x    += dot(in2, weights0);
				out3.x    += dot(in3, weights0);

				weights1   = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weight_y_idx + 3));
				out0.x    += dot(in1, weights1);
				out1.x    += dot(in2, weights1);
				out2.x    += dot(in3, weights1);
				out3.x    += dot(in4, weights1);

				weights2  = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weight_y_idx + 6));
				out0.x    += dot(in2, weights2);
				out1.x    += dot(in3, weights2);
				out2.x    += dot(in4, weights2);
				out3.x    += dot(in5, weights2);

				in0       = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_1.x));
				in1       = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_1.y));
				in2       = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_1.z));
				in3       = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_1.w));
				in4       = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_1.x));
				in5       = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_1.y));

				out0.y   += dot(in0, weights0);
				out1.y   += dot(in1, weights0);
				out2.y   += dot(in2, weights0);
				out3.y   += dot(in3, weights0);

				out0.y   += dot(in1, weights1);
				out1.y   += dot(in2, weights1);
				out2.y   += dot(in3, weights1);
				out3.y   += dot(in4, weights1);

				out0.y   += dot(in2, weights2);
				out1.y   += dot(in3, weights2);
				out2.y   += dot(in4, weights2);
				out3.y   += dot(in5, weights2);

				in0 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_2.x));
				in1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_2.y));
				in2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_2.z));
				in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_2.w));
				in4 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_2.x));
				in5 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_2.y));

				out0.z   += dot(in0, weights0);
				out1.z   += dot(in1, weights0);
				out2.z   += dot(in2, weights0);
				out3.z   += dot(in3, weights0);

				out0.z   += dot(in1, weights1);
				out1.z   += dot(in2, weights1);
				out2.z   += dot(in3, weights1);
				out3.z   += dot(in4, weights1);

				out0.z   += dot(in2, weights2);
				out1.z   += dot(in3, weights2);
				out2.z   += dot(in4, weights2);
				out3.z   += dot(in5, weights2);

				in0 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_3.x));
				in1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_3.y));
				in2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_3.z));
				in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0_3.w));
				in4 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_3.x));
				in5 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1_3.y));

				out0.w   += dot(in0, weights0);
				out1.w   += dot(in1, weights0);
				out2.w   += dot(in2, weights0);
				out3.w   += dot(in3, weights0);

				out0.w   += dot(in1, weights1);
				out1.w   += dot(in2, weights1);
				out2.w   += dot(in3, weights1);
				out3.w   += dot(in4, weights1);

				out0.w   += dot(in2, weights2);
				out1.w   += dot(in3, weights2);
				out2.w   += dot(in4, weights2);
				out3.w   += dot(in5, weights2);

				weight_y_idx++;
			}
		}
	}

#ifdef RELU
    ACTIVE_RELU_4OUT(out0, out1, out2, out3);
#endif

#ifdef RELU6
    ACTIVE_RELU6_4OUT(out0, out1, out2, out3);
#endif

#ifdef LEAKY_RELU
    ACTIVE_LEAKY_RELU_4OUT(out0, out1, out2, out3);
#endif

	const int remain    = image_shape.y - out_y_pos;
	const int output_y  = out_batch_base + out_y_pos;
	const int output_x  = mad24(depth_q_idx, image_shape.z, pos_x); 

	if (remain >= 4) {
		WI_F(output, (int2)(output_x, output_y),     out0);
		WI_F(output, (int2)(output_x, output_y + 1), out1);
		WI_F(output, (int2)(output_x, output_y + 2), out2);
		WI_F(output, (int2)(output_x, output_y + 3), out3);
	}
	else if (remain == 3) {
		WI_F(output, (int2)(output_x, output_y),     out0);
		WI_F(output, (int2)(output_x, output_y + 1), out1);
		WI_F(output, (int2)(output_x, output_y + 2), out2);
	}
	else if (remain == 2) {
		WI_F(output, (int2)(output_x, output_y),     out0);
		WI_F(output, (int2)(output_x, output_y + 1), out1);
	}
	else if (remain == 1) {
		WI_F(output, (int2)(output_x, output_y),     out0);
	}
}

__kernel void conv_3d_s1(__private const int       global_size_dim0,
	                     __private const int       global_size_dim1,
	                     __read_only  image2d_t    input,
	                     __read_only  image2d_t    weights,
#ifdef BIAS
	                     __read_only  image2d_t    bias,
#endif
	                     __write_only image2d_t    output,
	                     __private const int       in_channel_q,
	                     __private const int4      image_shape,
	                     __private const int4      kernel_shape,
	                     __private const int4      padding_shape,
	                     __private const int       out_channel_q)
{
	__private const int gx = get_global_id(0);
	__private const int gy = get_global_id(1);
	if (gx >= global_size_dim0 || gy >= global_size_dim1)
		return;
	__private const int out_channel_q_idx  = gx / image_shape.z;
	__private const int out_channel_q_idx0 = out_channel_q_idx << 1;
	__private const int out_channel_q_idx1 = select(out_channel_q_idx0 + 1, -1, (out_channel_q_idx0 + 1) >= out_channel_q);
	__private const int pos_x              = gx - mul24(out_channel_q_idx, image_shape.z);
	__private const int h_q                = ((image_shape.y + 3) >> 2);
	__private const int depth_heightq      = mul24(image_shape.x, h_q);
	__private const int batch_idx          = gy / depth_heightq;
	__private const int pos_depth_yq       = gy - mul24(batch_idx, depth_heightq);
	__private const int depth_idx          = pos_depth_yq / h_q;
	__private const int pos_yq_idx         = pos_depth_yq - mul24(depth_idx, h_q);
	__private const int out_y_pos          = (pos_yq_idx << 2);
	__private const int batch_pos_base     = batch_idx * image_shape.x * image_shape.y;
	__private const int depth_pos_base     = batch_pos_base + depth_idx * image_shape.y;
#ifdef BIAS
	FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_q_idx0, 0));
	FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(out_channel_q_idx1, 0));
#else
	FLOAT4 out0 = (FLOAT4)0.0f;
	FLOAT4 out4 = (FLOAT4)0.0f;
#endif
	FLOAT4 out1 = out0;
	FLOAT4 out2 = out0;
	FLOAT4 out3 = out0;
	FLOAT4 out5 = out4;
	FLOAT4 out6 = out4;
	FLOAT4 out7 = out4;

	const int in_z_idx        = depth_idx - padding_shape.x;
	const int in_y_idx        = out_y_pos - padding_shape.y;
	const int kernel_size_xyz = mul24(mul24(kernel_shape.x, kernel_shape.y), kernel_shape.z);
	const int kernel_size_yz  = mul24(kernel_shape.y, kernel_shape.z);
	const int weights_z_base0 = mul24(out_channel_q_idx0, kernel_size_xyz);
	const int weights_z_base1 = mul24(out_channel_q_idx1, kernel_size_xyz);
	int4      in_y_ofs        = (int4)(in_y_idx, in_y_idx + 1, in_y_idx + 2, in_y_idx + 3);
	for (int z = 0; z < kernel_shape.x; z++)
	{
		int        in_z_pos   = in_z_idx + z;
		int4       in_y_pos   = select(in_y_ofs, (int4)(-1), (in_y_ofs < ((int4)(0)) || in_y_ofs >= (int4)(image_shape.y)));
		int4       in_z_flag  = (int4)(in_z_pos < 0 || in_z_pos >= image_shape.x);
		const int  in_y_base  = select((batch_pos_base + mul24(in_z_pos, image_shape.y)), -1, in_z_flag.x);
		in_y_pos              = select(((int4)(batch_pos_base + mul24(in_z_pos, image_shape.y))) + in_y_pos, (int4)(-1), in_z_flag || (in_y_pos < (int4)0));
		const int  weights_y_base0 = weights_z_base0 + mul24(z, kernel_size_yz);
		const int  weights_y_base1 = weights_z_base1 + mul24(z, kernel_size_yz);
		__private FLOAT4 weights0, weights1, weights2, weights3;
		__private FLOAT4 in0, in1, in2, in3;
		int in_channel_q_idx = 0;
#ifdef IN_CHANNEL_LOOP
		for (in_channel_q_idx = 0; in_channel_q_idx < in_channel_q; ++in_channel_q_idx)
		{
#endif
			int        in_x_idx      = pos_x - padding_shape.z;
			const int  in_x_base     = mul24(in_channel_q_idx, image_shape.z);
			int        weights_x_idx = in_channel_q_idx << 2;
			for (int w = 0; w < kernel_shape.z; w++)
			{
				int cur_x_pos = select(in_x_base + in_x_idx, -1, (in_x_idx < 0 || in_x_idx >= image_shape.z));
				in_x_idx++;
				
				in1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.x));
				in2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.y));
				in3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.z));
				for (int h = 0; h < kernel_shape.y; h++)
				{
					int    weight_y_idx0 = weights_y_base0 + mad24(h, kernel_shape.z, w);
					int    weight_y_idx1 = select(weights_y_base1 + mad24(h, kernel_shape.z, w), -1, out_channel_q_idx1 < 0);
					in0 = in1;
					in1 = in2;
					in2 = in3;
					int    cur_in_y_pos = in_y_ofs.w + h;
					cur_in_y_pos = select(in_y_base + cur_in_y_pos, -1, (in_z_flag.x || cur_in_y_pos < 0 || cur_in_y_pos >= image_shape.y));
					in3          = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos));

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

	const int remain      = image_shape.y - out_y_pos;
	const int output_y    = depth_pos_base + out_y_pos;
	const int out_x_base0 = mul24(out_channel_q_idx0, image_shape.z);
	const int out_x_base1 = select(mul24(out_channel_q_idx1, image_shape.z), -1, out_channel_q_idx1 < 0);
	const int output_idx0 = out_x_base0 + pos_x;
	const int output_idx1 = out_x_base1 + pos_x;

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
			WI_F(output, (int2)(output_idx1, output_y), out4);
	}
}

__kernel void conv_3d_s1_out_depth_as_channel(__private const int       global_size_dim0,
	                                          __private const int       global_size_dim1,
	                                          __read_only  image2d_t    input,
	                                          __read_only  image2d_t    weights,
#ifdef BIAS
	                                          __read_only  image2d_t    bias,
#endif
	                                          __write_only image2d_t    output,
	                                          __private const int       in_channel_q,
	                                          __private const int4      image_shape,
	                                          __private const int4      kernel_shape,
	                                          __private const int4      padding_shape)
{
	__private const int gx = get_global_id(0);
	__private const int gy = get_global_id(1);
	if (gx >= global_size_dim0 || gy >= global_size_dim1)
		return;

	__private const int  pos_x          = gx;
	__private const int  h_q            = ((image_shape.y + 3) >> 2);
	__private const int  d_q            = ((image_shape.x + 3) >> 2);
	__private const int  depthq_heightq = mul24(d_q, h_q);
	__private const int  batch_idx      = gy / depthq_heightq;
	__private const int  pos_depthq_yq  = gy - mul24(batch_idx, depthq_heightq);
	__private const int  depth_q_idx    = pos_depthq_yq / h_q;
	__private const int4 depth_idx      = (int4)(depth_q_idx << 2, (depth_q_idx << 2) + 1, (depth_q_idx << 2) + 2, (depth_q_idx << 2) + 3);
	__private const int  pos_yq_idx     = pos_depthq_yq - mul24(depth_q_idx, h_q);
	__private const int  out_y_pos      = (pos_yq_idx << 2);
	__private const int  batch_pos_base = batch_idx * image_shape.x * image_shape.y;
	__private const int  out_batch_base = batch_idx * image_shape.y;

#ifdef BIAS
	FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(0, 0));
	out0.y      = out0.x;
	out0.z      = out0.x;
	out0.w      = out0.x;
#else
	FLOAT4 out0 = (FLOAT4)0.0f;
#endif
	FLOAT4 out1 = out0;
	FLOAT4 out2 = out0;
	FLOAT4 out3 = out0;

	const int4  in_z_idx        = depth_idx - (int4)(padding_shape.x);
	const int   in_y_idx        = out_y_pos - padding_shape.y;
	const int   kernel_size_yz  = mul24(kernel_shape.y, kernel_shape.z);
	int4        in_y_ofs        = (int4)(in_y_idx, in_y_idx + 1, in_y_idx + 2, in_y_idx + 3);
	for (int z = 0; z < kernel_shape.x; z++)
	{
		const int4  in_z_pos      = in_z_idx + (int4)(z);
		const int4  in_z_flag     = ((in_z_pos < ((int4)0)) | (in_z_pos >= (int4)(image_shape.x)));
		const int4  in_y_flag     = ((in_y_ofs < ((int4)0)) | (in_y_ofs >= (int4)(image_shape.y)));
		const int4  in_y_base     = mad24(in_z_pos, (int4)(image_shape.y), (int4)(batch_pos_base));
		const int4  in_y_pos_0    = select(((int4)(in_y_base.x)) + in_y_ofs, (int4)(-1), ((int4)(in_z_flag.x)) || in_y_flag);
		const int4  in_y_pos_1    = select(((int4)(in_y_base.y)) + in_y_ofs, (int4)(-1), ((int4)(in_z_flag.y)) || in_y_flag);
		const int4  in_y_pos_2    = select(((int4)(in_y_base.z)) + in_y_ofs, (int4)(-1), ((int4)(in_z_flag.z)) || in_y_flag);
		const int4  in_y_pos_3    = select(((int4)(in_y_base.w)) + in_y_ofs, (int4)(-1), ((int4)(in_z_flag.w)) || in_y_flag);
		int         weight_y_base = mul24(z, kernel_size_yz);
		__private FLOAT4 in0, in1, in2, in3, in4, in5, in6, in7;
		__private FLOAT4 in8, in9, in10, in11, in12, in13, in14, in15;
		__private FLOAT4 weights0;
		for (int in_channel_q_idx = 0; in_channel_q_idx < in_channel_q; ++in_channel_q_idx)
		{
			int        in_x_idx      = pos_x - padding_shape.z;
			const int  in_x_base     = mul24(in_channel_q_idx, image_shape.z);
			int        weights_x_idx = in_channel_q_idx;
			for (int w = 0; w < kernel_shape.z; w++)
			{
				int cur_x_pos = select(in_x_base + in_x_idx, -1, (in_x_idx < 0 || in_x_idx >= image_shape.z));
				in_x_idx++;

				in1  = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_0.x));
				in2  = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_0.y));
				in3  = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_0.z));

				in5  = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_1.x));
				in6  = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_1.y));
				in7  = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_1.z));

				in9  = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_2.x));
				in10 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_2.y));
				in11 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_2.z));

				in13 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_3.x));
				in14 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_3.y));
				in15 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos_3.z));
				for (int h = 0; h < kernel_shape.y; h++)
				{
					int    weight_y_idx = weight_y_base + mad24(h, kernel_shape.z, w);
					in0   = in1;
					in1   = in2;
					in2   = in3;
					in4   = in5;
					in5   = in6;
					in6   = in7;
					in8   = in9;
					in9   = in10;
					in10  = in11;
					in12  = in13;
					in13  = in14;
					in14  = in15;

					int    cur_in_y_pos0 = in_y_ofs.w + h;
					int4   cur_in_y_flag = (int4)((cur_in_y_pos0 < 0) | (cur_in_y_pos0 >= image_shape.y));
					int4   cur_in_y_pos1 = select(in_y_base + ((int4)cur_in_y_pos0), (int4)(-1), in_z_flag || cur_in_y_flag);
					in3  = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos1.x));
					in7  = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos1.y));
					in11 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos1.z));
					in15 = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos1.w));

					weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weight_y_idx));
					out0.x += dot(in0,  weights0);
					out1.x += dot(in1,  weights0);
					out2.x += dot(in2,  weights0);
					out3.x += dot(in3,  weights0);
					out0.y += dot(in4,  weights0);
					out1.y += dot(in5,  weights0);
					out2.y += dot(in6,  weights0);
					out3.y += dot(in7,  weights0);
					out0.z += dot(in8,  weights0);
					out1.z += dot(in9,  weights0);
					out2.z += dot(in10, weights0);
					out3.z += dot(in11, weights0);
					out0.w += dot(in12, weights0);
					out1.w += dot(in13, weights0);
					out2.w += dot(in14, weights0);
					out3.w += dot(in15, weights0);
				}
			}
		}
	}

#ifdef RELU
	ACTIVE_RELU_4OUT(out0, out1, out2, out3);
#endif

#ifdef RELU6
	ACTIVE_RELU6_4OUT(out0, out1, out2, out3);
#endif

#ifdef LEAKY_RELU
	ACTIVE_LEAKY_RELU_4OUT(out0, out1, out2, out3);
#endif

	const int remain   = image_shape.y - out_y_pos;
	const int output_y = out_batch_base + out_y_pos;
	const int output_x = mad24(depth_q_idx, image_shape.z, pos_x);

	if (remain >= 4) {
		WI_F(output, (int2)(output_x, output_y), out0);
		WI_F(output, (int2)(output_x, output_y + 1), out1);
		WI_F(output, (int2)(output_x, output_y + 2), out2);
		WI_F(output, (int2)(output_x, output_y + 3), out3);
	}
	else if (remain == 3) {
		WI_F(output, (int2)(output_x, output_y), out0);
		WI_F(output, (int2)(output_x, output_y + 1), out1);
		WI_F(output, (int2)(output_x, output_y + 2), out2);
	}
	else if (remain == 2) {
		WI_F(output, (int2)(output_x, output_y), out0);
		WI_F(output, (int2)(output_x, output_y + 1), out1);
	}
	else if (remain == 1) {
		WI_F(output, (int2)(output_x, output_y), out0);
	}
}