#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define READ_INPUT_IMAGE(i, base)                                                                         \
    int in_width_value##i = in_width##i + base;                                                           \
    in_width_value##i =                                                                                   \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_shape.y)); \
    in##i = RI_F(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);    

#define CALCULATE_OUTPUT_OPT(i)                  \
    out##i = mad(in_sm##i[local_idx].x, weights0, out##i); \
    out##i = mad(in_sm##i[local_idx].y, weights1, out##i); \
    out##i = mad(in_sm##i[local_idx].z, weights2, out##i); \
    out##i = mad(in_sm##i[local_idx].w, weights3, out##i);   

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define UNIT 4

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
void conv_2d_1x1_mali(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks, __read_only image2d_t input,
                          #ifdef BUFFER_INP_FP32
                          __global const float *kernel_ptr,
                          __global const float *bias_ptr,
                          #else
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          #endif
                          __write_only image2d_t output,
                          __private const int in_c_block, __private const int out_h,
                          __private const int out_w) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;

    const int out_w4_idx = mul24(out_w_idx, 4);

    #ifdef BUFFER_INP_FP32
    FLOAT4 out0 = CONVERT_FLOAT4(vload4(out_c_idx, (__global float *)bias_ptr));
    #else
    FLOAT4 out0 = vload4(out_c_idx, (__global FLOAT *)bias_ptr);
    #endif
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    FLOAT4 weights0;
    FLOAT4 weights1;
    FLOAT4 weights2;
    FLOAT4 weights3;

    FLOAT4 in0; 
    FLOAT4 in1; 
    FLOAT4 in2;
    FLOAT4 in3; 

    FLOAT16 weight16;

    const int intput_width_idx0 = out_w4_idx;
    const int intput_width_idx1 = out_w4_idx + 1;
    const int intput_width_idx2 = out_w4_idx + 2;
    const int intput_width_idx3 = out_w4_idx + 3;

    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        int input_width_base  = mul24(in_channel_block_idx, out_w);

        int offset = mad24(out_c_idx, in_c_block, in_channel_block_idx)*4;
        in0 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx0, out_b_h_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx1, out_b_h_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx2, out_b_h_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx3, out_b_h_idx));

        #ifdef BUFFER_INP_FP32
        weights0 = CONVERT_FLOAT4(vload4(offset, (__global float *)kernel_ptr));
        weights1 = CONVERT_FLOAT4(vload4(offset + 1, (__global float *)kernel_ptr));
        weights2 = CONVERT_FLOAT4(vload4(offset + 2, (__global float *)kernel_ptr));
        weights3 = CONVERT_FLOAT4(vload4(offset + 3, (__global float *)kernel_ptr));
        #else
        weights0 = vload4(offset, (__global FLOAT *)kernel_ptr);
        weights1 = vload4(offset + 1, (__global FLOAT *)kernel_ptr);
        weights2 = vload4(offset + 2, (__global FLOAT *)kernel_ptr);
        weights3 = vload4(offset + 3, (__global FLOAT *)kernel_ptr);
        #endif
        
        out0.x += dot(weights0, in0);
        out0.y += dot(weights1, in0);
        out0.z += dot(weights2, in0);
        out0.w += dot(weights3, in0);

        out1.x += dot(weights0, in1);
        out1.y += dot(weights1, in1);
        out1.z += dot(weights2, in1);
        out1.w += dot(weights3, in1);

        out2.x += dot(weights0, in2);
        out2.y += dot(weights1, in2);
        out2.z += dot(weights2, in2);
        out2.w += dot(weights3, in2);

        out3.x += dot(weights0, in3);
        out3.y += dot(weights1, in3);
        out3.z += dot(weights2, in3);
        out3.w += dot(weights3, in3);

    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

//Shaquille, Added 20210220 Start
#ifdef LEAKY_RELU
	out0 = select(((FLOAT)(LEAKY_RELU))*out0, out0, out0 >= (FLOAT4)((FLOAT)0));
	out1 = select(((FLOAT)(LEAKY_RELU))*out1, out1, out1 >= (FLOAT4)((FLOAT)0));
	out2 = select(((FLOAT)(LEAKY_RELU))*out2, out2, out2 >= (FLOAT4)((FLOAT)0));
	out3 = select(((FLOAT)(LEAKY_RELU))*out3, out3, out3 >= (FLOAT4)((FLOAT)0));
#endif
//Shaquille, Added 20210220 End

    const int out_x_base = out_c_idx*out_w;

    const int remain = out_w - out_w4_idx;
    int output_idx   = out_x_base + out_w4_idx;
    
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
        WI_F(output, (int2)(output_idx + 2, out_b_h_idx), out2);
        WI_F(output, (int2)(output_idx + 3, out_b_h_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
        WI_F(output, (int2)(output_idx + 2, out_b_h_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
    }

}

__kernel void conv_2d_1x1_local(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                          __read_only image2d_t bias,
                          __write_only image2d_t output,
                          __private const int in_c_block, __private const int out_h,
                          __private const int out_w) {

    const int row = get_local_id(0); 
    const int col = get_local_id(1); 

    const int out_c_idx = get_global_id(0); //c/4
    const int out_w_idx = get_global_id(1); //w
    const int out_b_h_idx  = get_global_id(2); //b h

    DEAL_NON_UNIFORM_DIM3(out_c_idx, out_w_idx, out_b_h_idx);

    const int out_w4_idx = mul24(out_w_idx, 4);

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_c_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    FLOAT4 weights0;
    FLOAT4 weights1;
    FLOAT4 weights2;
    FLOAT4 weights3;
    
    __local FLOAT4 in_sm0[UNIT*UNIT];
    __local FLOAT4 in_sm1[UNIT*UNIT];
    __local FLOAT4 in_sm2[UNIT*UNIT];
    __local FLOAT4 in_sm3[UNIT*UNIT];

    int tiles = (in_c_block + UNIT -1)/ UNIT;

    const int col_x_unit = mul24(col, UNIT);
    const int in_index = col_x_unit + row;

    for (int t = 0; t < tiles; ++t) {
        
        int in_c = mad24(t, UNIT, row);
        int in_c_w_idx = mad24(in_c, out_w, out_w4_idx);

        in_sm0[in_index] = RI_F(input, SAMPLER, (int2)(in_c_w_idx, out_b_h_idx));
        in_sm1[in_index] = RI_F(input, SAMPLER, (int2)(in_c_w_idx+1, out_b_h_idx));
        in_sm2[in_index] = RI_F(input, SAMPLER, (int2)(in_c_w_idx+2, out_b_h_idx));
        in_sm3[in_index] = RI_F(input, SAMPLER, (int2)(in_c_w_idx+3, out_b_h_idx));

        barrier(CLK_GLOBAL_MEM_FENCE);

        int kernel_index = mul24(t, UNIT*4);

        for(int k = 0; k < UNIT; k++){

            __private int kernel_cx4 = mad24(k, 4, kernel_index);
            __private int local_idx = col_x_unit + k;

            weights0 = RI_F(weights, SAMPLER, (int2)(kernel_cx4++, out_c_idx));
            weights1 = RI_F(weights, SAMPLER, (int2)(kernel_cx4++, out_c_idx));
            weights2 = RI_F(weights, SAMPLER, (int2)(kernel_cx4++, out_c_idx));
            weights3 = RI_F(weights, SAMPLER, (int2)(kernel_cx4++, out_c_idx));

            CALCULATE_OUTPUT_OPT(0);
            CALCULATE_OUTPUT_OPT(1);
            CALCULATE_OUTPUT_OPT(2);
            CALCULATE_OUTPUT_OPT(3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

//Shaquille, Added 20210220 Start
#ifdef LEAKY_RELU
	out0 = select(((FLOAT)(LEAKY_RELU))*out0, out0, out0 >= (FLOAT4)((FLOAT)0));
	out1 = select(((FLOAT)(LEAKY_RELU))*out1, out1, out1 >= (FLOAT4)((FLOAT)0));
	out2 = select(((FLOAT)(LEAKY_RELU))*out2, out2, out2 >= (FLOAT4)((FLOAT)0));
	out3 = select(((FLOAT)(LEAKY_RELU))*out3, out3, out3 >= (FLOAT4)((FLOAT)0));
#endif
//Shaquille, Added 20210220 End

    const int out_x_base = out_c_idx*out_w;

    const int remain = out_w - out_w4_idx;
    int output_idx   = out_x_base + out_w4_idx;
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
        WI_F(output, (int2)(output_idx + 2, out_b_h_idx), out2);
        WI_F(output, (int2)(output_idx + 3, out_b_h_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
        WI_F(output, (int2)(output_idx + 2, out_b_h_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
    }

}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_1x1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                          __read_only image2d_t bias,
                          __write_only image2d_t output,
                          __private const int2 input_shape,
                          __private const int in_channel_block, __private const int2 output_shape,
                          __private const int2 stride_shape,
                          __private const int output_width_4) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int output_channel_block_idx = output_channel_width_idx / output_width_4;
    const int output_width_block_idx   = output_channel_width_idx % output_width_4;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int intput_width_idx0 = mul24(output_width_block_idx, stride_shape.y*4);
    int intput_width_idx1 = intput_width_idx0 + stride_shape.y;
    int intput_width_idx2 = intput_width_idx1 + stride_shape.y;
    int intput_width_idx3 = intput_width_idx2 + stride_shape.y;

    intput_width_idx0 = select(intput_width_idx0, INT_MIN, intput_width_idx0 >= input_shape.y);
    intput_width_idx1 = select(intput_width_idx1, INT_MIN, intput_width_idx1 >= input_shape.y);
    intput_width_idx2 = select(intput_width_idx2, INT_MIN, intput_width_idx2 >= input_shape.y);
    intput_width_idx3 = select(intput_width_idx3, INT_MIN, intput_width_idx3 >= input_shape.y);

    int batch_index            = output_batch_height_idx / output_shape.x;
    int input_height_block_idx = mul24((output_batch_height_idx % output_shape.x), stride_shape.x) + batch_index * input_shape.x;

    FLOAT4 in0;
    FLOAT4 in1;
    FLOAT4 in2;
    FLOAT4 in3;
    FLOAT4 weights0;
    FLOAT4 weights1;
    FLOAT4 weights2;
    FLOAT4 weights3;

    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_block; ++in_channel_block_idx) {
        int input_width_base  = in_channel_block_idx * input_shape.y;
        int weights_width_base = in_channel_block_idx << 2;
        in0 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx0, input_height_block_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx1, input_height_block_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx2, input_height_block_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx3, input_height_block_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 0, output_channel_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 1, output_channel_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 2, output_channel_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 3, output_channel_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

//Shaquille, Added 20210220 Start
#ifdef LEAKY_RELU
	out0 = select(((FLOAT)(LEAKY_RELU))*out0, out0, out0 >= (FLOAT4)((FLOAT)0));
	out1 = select(((FLOAT)(LEAKY_RELU))*out1, out1, out1 >= (FLOAT4)((FLOAT)0));
	out2 = select(((FLOAT)(LEAKY_RELU))*out2, out2, out2 >= (FLOAT4)((FLOAT)0));
	out3 = select(((FLOAT)(LEAKY_RELU))*out3, out3, out3 >= (FLOAT4)((FLOAT)0));
#endif
//Shaquille, Added 20210220 End

#ifdef BN_AFTER_RELU
	const int out_channel_q = global_size_dim0 / output_width_4;
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, out_channel_q, output_channel_block_idx);
#endif

    const int out_x_base = mul24(output_channel_block_idx, output_shape.y);
    int out_x_idx        = output_width_block_idx << 2;

    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base + out_x_idx;
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t weights,
#ifdef BIAS
                      __read_only image2d_t bias,
#endif
                      __write_only image2d_t output,
                      __private const int2 input_shape,
                      __private const int in_channel_block_length,
                      __private const int2 output_shape,
                      __private const int2 weights_shape,
                      __private const int2 stride_shape,
                      __private const int2 padding_shape,
                      __private const int2 dilation_shape,
                      __private const int out_width_blocks) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int out_channel_block_idx  = output_channel_width_idx / out_width_blocks;
    const int out_height_block_idx   = output_channel_width_idx % out_width_blocks;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
#else
    FLOAT4 out0 = (FLOAT4)0;
#endif
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0          = mad24(out_height_block_idx, stride_shape.y<<2, -padding_shape.y);
    int in_width1          = in_width0 + stride_shape.y;
    int in_width2          = in_width0 + stride_shape.y * 2;
    int in_width3          = in_width0 + stride_shape.y * 3;
    
    const int height_start = mad24((output_batch_height_idx % output_shape.x), stride_shape.x, -padding_shape.x);
    int in_height_start    = mad24(select(0, (-height_start + dilation_shape.x - 1) / dilation_shape.x, height_start < 0), dilation_shape.x, height_start);
    int in_height_end      = min(mad24(weights_shape.x, dilation_shape.x, height_start), input_shape.x);

    const int batch_idx          = mul24((output_batch_height_idx / output_shape.x), input_shape.x);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(weights_shape.y, weights_shape.x)) + mul24(select(0, (-height_start + dilation_shape.x - 1) / dilation_shape.x, height_start < 0), weights_shape.y);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
        const int in_idx = mul24(in_channel_block_idx, input_shape.y);
        int weights_x_idx = in_channel_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_shape.x) {
            int in_hb_value = iy + batch_idx;
            for (int w = 0; w < weights_shape.y; w++) {
                int input_width_base = mul24(w, dilation_shape.y);
                READ_INPUT_IMAGE(0, input_width_base);
                READ_INPUT_IMAGE(1, input_width_base);
                READ_INPUT_IMAGE(2, input_width_base);
                READ_INPUT_IMAGE(3, input_width_base);

                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx)); 
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx)); 
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx)); 
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));

//Shaquille, Modidifed 20201012 Start
#if 0
                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
#else
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
#endif
//Shaquille, Modidifed 20201012 End
            }
        }
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

//Shaquille, Added 20210220 Start
#ifdef LEAKY_RELU
	out0 = select(((FLOAT)(LEAKY_RELU))*out0, out0, out0 >= (FLOAT4)((FLOAT)0));
	out1 = select(((FLOAT)(LEAKY_RELU))*out1, out1, out1 >= (FLOAT4)((FLOAT)0));
	out2 = select(((FLOAT)(LEAKY_RELU))*out2, out2, out2 >= (FLOAT4)((FLOAT)0));
	out3 = select(((FLOAT)(LEAKY_RELU))*out3, out3, out3 >= (FLOAT4)((FLOAT)0));
#endif
//Shaquille, Added 20210220 End

#ifdef BN_AFTER_RELU
	const int out_channel_q = global_size_dim0 / out_width_blocks;
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, out_channel_q, out_channel_block_idx);
#endif
    const int out_x_base = mul24(out_channel_block_idx, output_shape.y);
    int out_x_idx        = out_height_block_idx << 2;

    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base + out_x_idx;
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    }
}


//Shaquille, Added 20201015 Start
__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_stride_1x1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, 
                        __read_only image2d_t weights,
#ifdef BIAS
                        __read_only image2d_t bias,
#endif
                        __write_only image2d_t output,
                        __private const int2 input_shape,
                        __private const int in_channel_block_length,
                        __private const int2 output_shape,
                        __private const int2 weights_shape,
                        __private const int2 padding_shape,
                        __private const int out_width_blocks)
{
	const int output_channel_width_idx = get_global_id(0);
	const int output_batch_height_idx = get_global_id(1);
	DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

#ifndef DUAL_OUTPUT_DATA
	const int  out_channel_block_idx  = output_channel_width_idx / out_width_blocks;
#else
    const int  out_channel_block_idx  = (output_channel_width_idx / out_width_blocks) << 1;
    const int  out_channel_block_idx2 = out_channel_block_idx + 1;
#endif
	const int  out_height_block_idx   = output_channel_width_idx % out_width_blocks;

#ifdef BIAS
	FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx,  0));
#ifdef DUAL_OUTPUT_DATA
	FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx2, 0));
#endif
#else
	FLOAT4 out0 = (FLOAT4)0;
#ifdef DUAL_OUTPUT_DATA
	FLOAT4 out4 = (FLOAT4)0;
#endif
#endif
	FLOAT4 out1 = out0;
	FLOAT4 out2 = out0;
	FLOAT4 out3 = out0;
#ifdef DUAL_OUTPUT_DATA
	FLOAT4 out5 = out4;
	FLOAT4 out6 = out4;
	FLOAT4 out7 = out4;
#endif

	int in_width0 = mad24(out_height_block_idx, 4, -padding_shape.y);
	int in_width1 = in_width0 + 1;
	int in_width2 = in_width0 + 2;
	int in_width3 = in_width0 + 3;

	const int height_start   = mad24((output_batch_height_idx % output_shape.x), 1, -padding_shape.x);
	int in_height_start      = mad24(select(0, -height_start, height_start < 0), 1, height_start);
	int in_height_end        = min(mad24(weights_shape.x, 1, height_start), input_shape.x);

	const int batch_idx      = mul24((output_batch_height_idx / output_shape.x), input_shape.x);
	const int weights_h_idx  = mul24(out_channel_block_idx,  mul24(weights_shape.y, weights_shape.x)) + mul24(select(0, -height_start, height_start < 0), weights_shape.y);
#ifdef DUAL_OUTPUT_DATA
    const int weights_h_idx2 = mul24(out_channel_block_idx2, mul24(weights_shape.y, weights_shape.x)) + mul24(select(0, -height_start, height_start < 0), weights_shape.y);
#endif
	FLOAT4 in0, in1, in2, in3;
	FLOAT4 weights0, weights1, weights2, weights3;
	int in_channel_block_idx = 0;
#ifdef IN_CHANNEL_LOOP
	for (in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
#endif
            const int in_idx = mul24(in_channel_block_idx, input_shape.y);
            int weights_x_idx  = in_channel_block_idx << 2;
            int weights_y_idx  = weights_h_idx;
#ifdef DUAL_OUTPUT_DATA
            int weights_y_idx2 = weights_h_idx2;
#endif
            for (int iy = in_height_start; iy < in_height_end; iy += 1) {
                int in_hb_value = iy + batch_idx;
                int4 in_width_value = (int4)(in_width0, in_width1, in_width2, in_width3);
                int4 cur_src     = select((int4)(in_idx) + in_width_value, (int4)(-1), (in_width_value < (int4)(0)) || (in_width_value >= (int4)(input_shape.y)));
                in1              = RI_F(input, SAMPLER, (int2)(cur_src.x, in_hb_value));
                in2              = RI_F(input, SAMPLER, (int2)(cur_src.y, in_hb_value));
                in3              = RI_F(input, SAMPLER, (int2)(cur_src.z, in_hb_value));
                for (int w = 0; w < weights_shape.y; w++) {
                    int input_width_base = w;
                    in0             = in1;
                    in1             = in2;
                    in2             = in3;
                    int cur_src_x   = in_width_value.w + w;
                    cur_src_x       = select(in_idx + cur_src_x, -1, (cur_src_x < 0) || (cur_src_x >= input_shape.y));
                    in3             = RI_F(input, SAMPLER, (int2)(cur_src_x, in_hb_value));

                    weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx));
                    weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                    weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                    weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));

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

#ifdef DUAL_OUTPUT_DATA
                    weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx2));
                    weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx2));
                    weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx2));
                    weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx2++));

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
#endif
			}
		}
#ifdef IN_CHANNEL_LOOP
	}
#endif

#ifdef RELU
	out0 = fmax(out0, (FLOAT4)0);
	out1 = fmax(out1, (FLOAT4)0);
	out2 = fmax(out2, (FLOAT4)0);
	out3 = fmax(out3, (FLOAT4)0);
#ifdef DUAL_OUTPUT_DATA
	out4 = fmax(out4, (FLOAT4)0);
	out5 = fmax(out5, (FLOAT4)0);
	out6 = fmax(out6, (FLOAT4)0);
	out7 = fmax(out7, (FLOAT4)0);
#endif
#endif

#ifdef RELU6
	out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
	out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
	out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
	out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#ifdef DUAL_OUTPUT_DATA
	out4 = clamp(out4, (FLOAT4)0, (FLOAT4)6);
	out5 = clamp(out5, (FLOAT4)0, (FLOAT4)6);
	out6 = clamp(out6, (FLOAT4)0, (FLOAT4)6);
	out7 = clamp(out7, (FLOAT4)0, (FLOAT4)6);
#endif
#endif

#ifdef LEAKY_RELU
	out0 = select(((FLOAT)(LEAKY_RELU))*out0, out0, out0 >= (FLOAT4)((FLOAT)0));
	out1 = select(((FLOAT)(LEAKY_RELU))*out1, out1, out1 >= (FLOAT4)((FLOAT)0));
	out2 = select(((FLOAT)(LEAKY_RELU))*out2, out2, out2 >= (FLOAT4)((FLOAT)0));
	out3 = select(((FLOAT)(LEAKY_RELU))*out3, out3, out3 >= (FLOAT4)((FLOAT)0));
#ifdef DUAL_OUTPUT_DATA
	out4 = select(((FLOAT)(LEAKY_RELU))*out4, out4, out4 >= (FLOAT4)((FLOAT)0));
	out5 = select(((FLOAT)(LEAKY_RELU))*out5, out5, out5 >= (FLOAT4)((FLOAT)0));
	out6 = select(((FLOAT)(LEAKY_RELU))*out6, out6, out6 >= (FLOAT4)((FLOAT)0));
	out7 = select(((FLOAT)(LEAKY_RELU))*out7, out7, out7 >= (FLOAT4)((FLOAT)0));
#endif
#endif

#ifdef BN_AFTER_RELU
#ifndef DUAL_OUTPUT_DATA
	const int out_channel_q = (global_size_dim0 / out_width_blocks);
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, out_channel_q, out_channel_block_idx);
#else
	const int out_channel_q = 2 * (global_size_dim0 / out_width_blocks);
	CAL_BN_AFTER_RELU(out0, out1, out2, out3, bias, out_channel_q, out_channel_block_idx);
	if (out_channel_block_idx2 >= 0)
	{
		CAL_BN_AFTER_RELU(out4, out5, out6, out7, bias, out_channel_q, out_channel_block_idx2);
	}
#endif
#endif

	const int out_x_base  = mul24(out_channel_block_idx, output_shape.y);
#ifdef DUAL_OUTPUT_DATA
	const int out_x_base2 = mul24(out_channel_block_idx2, output_shape.y);
#endif
	int out_x_idx = out_height_block_idx << 2;
	const int remain = output_shape.y - out_x_idx;
	int output_idx  = out_x_base + out_x_idx;
#ifdef DUAL_OUTPUT_DATA
	int output_idx2 = out_x_base2 + out_x_idx;
#endif
	if (remain >= 4) {
		WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
		WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
		WI_F(output, (int2)(output_idx + 2,  output_batch_height_idx), out2);
		WI_F(output, (int2)(output_idx + 3,  output_batch_height_idx), out3);
#ifdef DUAL_OUTPUT_DATA
		WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
		WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
		WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
		WI_F(output, (int2)(output_idx2 + 3, output_batch_height_idx), out7);
#endif
	}
	else if (remain == 3) {
		WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
		WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
		WI_F(output, (int2)(output_idx + 2,  output_batch_height_idx), out2);
#ifdef DUAL_OUTPUT_DATA
		WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
		WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
		WI_F(output, (int2)(output_idx2 + 2, output_batch_height_idx), out6);
#endif
	}
	else if (remain == 2) {
		WI_F(output, (int2)(output_idx,      output_batch_height_idx), out0);
		WI_F(output, (int2)(output_idx + 1,  output_batch_height_idx), out1);
#ifdef DUAL_OUTPUT_DATA
		WI_F(output, (int2)(output_idx2,     output_batch_height_idx), out4);
		WI_F(output, (int2)(output_idx2 + 1, output_batch_height_idx), out5);
#endif
	}
	else if (remain == 1) {
		WI_F(output, (int2)(output_idx,    output_batch_height_idx), out0);
#ifdef DUAL_OUTPUT_DATA
		WI_F(output, (int2)(output_idx2,   output_batch_height_idx), out4);
#endif
	}
}
//Shaquille, Added 20201015 End