#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void grid_sample_bilinear_padding_zero(__private const int     global_size_dim0,
                                                __private const int     global_size_dim1,
                                                __read_only  image2d_t  input,
												__read_only  image2d_t  grid,
                                                __write_only image2d_t  output,
                                                __private const int4    in_image_shape,
                                                __private const int2    out_image_shape,
                                                __private const float2  inv_factor)
{
    __private const int gx = get_global_id(0);
    __private const int gy = get_global_id(1);
    if (gx >= global_size_dim0 || gy >= global_size_dim1)
        return;
    __private const int out_channel_q_idx = gx / out_image_shape.y;
    __private const int pos_x             = gx - mul24(out_channel_q_idx, out_image_shape.y);
    __private const int batch_idx         = gy / out_image_shape.x;
    __private const int pos_y             = gy - mul24(batch_idx, out_image_shape.x);
    __private const int in_x_base         = mul24(out_channel_q_idx, in_image_shape.z);
    __private const int in_y_base         = mul24(batch_idx,         in_image_shape.y);
	
    __private const FLOAT4 grid_data      = RI_F(grid, SAMPLER, (int2)(pos_x, gy));
    __private FLOAT4 grid_raw_data        = grid_data + ((FLOAT4)(1.0f));
#ifdef ALIGN_CORNERS
    grid_raw_data                         = CONVERT_FLOAT4((float4)(inv_factor.x, inv_factor.y, 0.0, 0.0)) * grid_raw_data;
#else
    grid_raw_data                         = mad(CONVERT_FLOAT4((float4)(inv_factor.x, inv_factor.y, 0.0, 0.0)), grid_raw_data, (FLOAT4)(-0.5));
#endif
    __private int4   grid_raw_data_int        = (int4)(grid_raw_data.x, grid_raw_data.y, grid_raw_data.z, grid_raw_data.w);
    __private FLOAT4 grid_raw_data_int_float  = (FLOAT4)(grid_raw_data_int.x, grid_raw_data_int.y, grid_raw_data_int.z, grid_raw_data_int.w);
    grid_raw_data_int                         = select(grid_raw_data_int, grid_raw_data_int - (int4)1, grid_raw_data_int_float > grid_raw_data);
    grid_raw_data_int_float                   = (FLOAT4)(grid_raw_data_int.x, grid_raw_data_int.y, grid_raw_data_int.z, grid_raw_data_int.w);
    FLOAT            u                        = grid_raw_data.x - grid_raw_data_int_float.x;
    FLOAT            v                        = grid_raw_data.y - grid_raw_data_int_float.y;
    __private int2   input_pos                = (int2)(grid_raw_data_int.x, grid_raw_data_int.y);
    __private int2   input_pos_plus           = input_pos + ((int2)1);
    __private int2   pos_base                 = (int2)(in_x_base, in_y_base);
    input_pos                                 = select(pos_base + input_pos,      (int2)(-1), input_pos      < ((int2)0) || input_pos      >= ((int2)(in_image_shape.z, in_image_shape.y)));
    input_pos_plus                            = select(pos_base + input_pos_plus, (int2)(-1), input_pos_plus < ((int2)0) || input_pos_plus >= ((int2)(in_image_shape.z, in_image_shape.y)));
    __private FLOAT4 in_data00                = RI_F(input, SAMPLER, (int2)(input_pos.x,  input_pos.y));
    __private FLOAT4 in_data01                = RI_F(input, SAMPLER, (int2)(input_pos_plus.x, input_pos.y));
    __private FLOAT4 in_data10                = RI_F(input, SAMPLER, (int2)(input_pos.x, input_pos_plus.y));
    __private FLOAT4 in_data11                = RI_F(input, SAMPLER, (int2)(input_pos_plus.x, input_pos_plus.y));
    __private FLOAT4 res0                     = in_data00 * (((FLOAT)1.0f) - u) * (((FLOAT)1.0f) - v);
    __private FLOAT4 res1                     = in_data01 * u * (((FLOAT)1.0f) - v);
    __private FLOAT4 res2                     = in_data10 * (((FLOAT)1.0f) - u) * v;
    __private FLOAT4 res3                     = in_data11 * u * v;
    __private FLOAT4 res                      = res0 + res1 + res2 + res3;

#ifdef CHANNEL_TAIL_3
    res.w = (FLOAT)0.0f;
#else
#ifdef CHANNEL_TAIL_2
    res.z = (FLOAT)0.0f;
    res.w = (FLOAT)0.0f;
#else
#ifdef CHANNEL_TAIL_1
    res.y = (FLOAT)0.0f;
    res.z = (FLOAT)0.0f;
    res.w = (FLOAT)0.0f;
#endif
#endif
#endif	
    WI_F(output, (int2)(gx, gy), res);
}