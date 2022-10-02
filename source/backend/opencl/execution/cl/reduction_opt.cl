// TODO: use INIT_SCALAR_VALUE, OPERATOR, FINAL_OPERATOR_ON_CHANNEL macro abstract and simplify code
// TODO: support reduce dims include batch
// TODO: support keep_dim=False
// TODO: fix channel reduce result re-pack problem
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void reduct_general_mean_opt(__read_only image2d_t  input,
                                      __write_only image2d_t output,
                                      __private const int    batch,
                                      __private const int    height,
                                      __private const int    width,
                                      __private const int    channel)
{
    const int x_idx     = get_global_id(0);
    const int y_idx     = get_global_id(1);
    const int batch_pos = y_idx / height;
    const int y_pos     = y_idx - mul24(batch_pos, height);
    const int x_pos     = x_idx;
    const int channel_q = channel >> 2;
    int       i = 0;
    if (y_pos >= height || x_pos >= width || batch_pos >= batch)
        return;

    FLOAT4 sum = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (i = 0; i < channel_q; i++)
    {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(mad24(i, width, x_pos), mad24(batch_pos, height, y_pos)));
        sum = sum + in;
    }
    FLOAT res = sum.x + sum.y + sum.z + sum.w;
#ifdef REDUCTION_TAIL
    FLOAT4 in = RI_F(input, SAMPLER, (int2)(mad24(i, width, x_pos), mad24(batch_pos, height, y_pos)));
#ifdef REDUCTION_TAIL_3
    res = res + (in.x + in.y + in.z);
#else
#ifdef REDUCTION_TAIL_2
    res = res + (in.x + in.y);
#else
#ifdef REDUCTION_TAIL_1
    res = res + in.x;
#endif
#endif
#endif
#endif
#ifndef INV_SCALE
#define INV_SCALE    1.0f
#endif
    WI_F(output, (int2)(x_pos, y_idx), (FLOAT4)(((FLOAT)INV_SCALE)*res, 0.0f, 0.0f, 0.0f));
}

__kernel void reduct_general_sum_opt(__read_only image2d_t  input,
                                     __write_only image2d_t output,
	                                 __private const int    batch,
                                     __private const int    height,
                                     __private const int    width,
                                     __private const int    channel) 
{
    const int x_idx     = get_global_id(0);
    const int y_idx     = get_global_id(1);
    const int batch_pos = y_idx / height;
    const int y_pos     = y_idx - mul24(batch_pos, height);
    const int x_pos     = x_idx;
    const int channel_q = channel >> 2;
    int       i         = 0;
    if (y_pos >= height || x_pos >= width || batch_pos >= batch)
        return;

    FLOAT4 sum = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (i = 0; i < channel_q; i++) 
    {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(mad24(i, width, x_pos), mad24(batch_pos, height, y_pos)));
        sum = sum + in;
    }
    FLOAT res = sum.x + sum.y + sum.z + sum.w;
#ifdef REDUCTION_TAIL
    FLOAT4 in = RI_F(input, SAMPLER, (int2)(mad24(i, width, x_pos), mad24(batch_pos, height, y_pos)));
#ifdef REDUCTION_TAIL_3
    res = res + (in.x + in.y + in.z);
#else
#ifdef REDUCTION_TAIL_2
    res = res + (in.x + in.y);
#else
#ifdef REDUCTION_TAIL_1
    res = res + in.x;
#endif
#endif
#endif
#endif

    WI_F(output, (int2)(x_pos, y_idx), (FLOAT4)(res, 0.0f, 0.0f, 0.0f));
}

__kernel void reduct_general_l1_opt(__read_only image2d_t  input,
                                    __write_only image2d_t output,
                                    __private const int    batch,
                                    __private const int    height,
                                    __private const int    width,
                                    __private const int    channel)
{
    const int x_idx     = get_global_id(0);
    const int y_idx     = get_global_id(1);
    const int batch_pos = y_idx / height;
    const int y_pos     = y_idx - mul24(batch_pos, height);
    const int x_pos     = x_idx;
    const int channel_q = channel >> 2;
    int       i = 0;
    if (y_pos >= height || x_pos >= width || batch_pos >= batch)
        return;

    FLOAT4 sum = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (i = 0; i < channel_q; i++)
    {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(mad24(i, width, x_pos), mad24(batch_pos, height, y_pos)));
        sum = sum + fabs(convert_float4(in));
    }
    FLOAT res = sum.x + sum.y + sum.z + sum.w;
#ifdef REDUCTION_TAIL
    FLOAT4 in = RI_F(input, SAMPLER, (int2)(mad24(i, width, x_pos), mad24(batch_pos, height, y_pos)));
    in = fabs(convert_float4(in));
#ifdef REDUCTION_TAIL_3
    res = res + (in.x + in.y + in.z);
#else
#ifdef REDUCTION_TAIL_2
    res = res + (in.x + in.y);
#else
#ifdef REDUCTION_TAIL_1
    res = res + in.x;
#endif
#endif
#endif
#endif

    WI_F(output, (int2)(x_pos, y_idx), (FLOAT4)(res, 0.0f, 0.0f, 0.0f));
}