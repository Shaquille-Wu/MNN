#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define READ_INPUT_IMAGE(i, base)                                                                         \
    int inOffset##i = inWidthOffset##i + base;                                                           \
    inOffset##i =                                                                                   \
        select(inCurIdx + inOffset##i, -1, (inOffset##i < 0 || inOffset##i >= inputShape.y)); \
    inValue##i = RI_F(input, SAMPLER, (int2)(inOffset##i, inHeightIdx));

#define CALCULATE_OUTPUT(i)                  \
    outValue##i = mad(inValue##i.x, weights0, outValue##i); \
    outValue##i = mad(inValue##i.y, weights1, outValue##i); \
    outValue##i = mad(inValue##i.z, weights2, outValue##i); \
    outValue##i = mad(inValue##i.w, weights3, outValue##i);

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

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
void depthwise_conv2d_s1_opt(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t filter,
#ifndef NO_BIAS
                             __read_only image2d_t bias,
#endif							 
                             __write_only image2d_t output,
                             __private const int2 inputShape,
                             __private const int inChannelBlocks, 
                             __private const int2 outputShape,
                             __private const int2 filterShape,
                             __private const int2 paddingShape) 
{
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int ow5                      = (outputShape.y + 4) / 5;
    const int outChannelBlockIdx = outChannelWidthIdx / ow5;
    const int outWidthBlockidx   = outChannelWidthIdx % ow5;

    const int inChannelBlockIdx  = outChannelBlockIdx;

#ifndef NO_BIAS
    FLOAT4 outValue0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
#else
    FLOAT4 outValue0 = (FLOAT4)(0.0f);
#endif	
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;
	FLOAT4 outValue4 = outValue0;

    const int outWidthBlockidx5 = mul24(outWidthBlockidx, 5);
    const int inWidthOffset0    = outWidthBlockidx5 - paddingShape.y;
    const int inWidthOffset1    = inWidthOffset0 + 1;
    const int inWidthOffset2    = inWidthOffset0 + 2;
    const int inWidthOffset3    = inWidthOffset0 + 3;
	const int inWidthOffset4    = inWidthOffset0 + 4;

    int heightIdx            = outHeightBlockIdx % outputShape.x - paddingShape.x;
    const int outBatchIdx    = mul24((outHeightBlockIdx / outputShape.x), inputShape.x);
    const int inCurIdx       = mul24(inChannelBlockIdx, inputShape.y);

	const int4 inWidthOffset = (int4)(inWidthOffset0, inWidthOffset1, inWidthOffset2, inWidthOffset3);
	const int4 inWidthIdx    = select((int4)(inCurIdx) + inWidthOffset, (int4)(-1), (inWidthOffset < (int4)(0) || inWidthOffset >= (int4)(inputShape.y)));

    FLOAT4 inValue0, inValue1, inValue2, inValue3, inValue4;
    for (int kh = 0; kh < filterShape.x; kh++) {
        int inHeightIdx = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= inputShape.x));
        heightIdx++;
        inValue1       = RI_F(input, SAMPLER, (int2)(inWidthIdx.x, inHeightIdx));
        inValue2       = RI_F(input, SAMPLER, (int2)(inWidthIdx.y, inHeightIdx));
        inValue3       = RI_F(input, SAMPLER, (int2)(inWidthIdx.z, inHeightIdx));
		inValue4       = RI_F(input, SAMPLER, (int2)(inWidthIdx.w, inHeightIdx));
        for (int kw = 0; kw < filterShape.y; kw++) {

            int filterIdx   = mad24(kh, filterShape.y, kw);
            inValue0 = inValue1;
            inValue1 = inValue2;
            inValue2 = inValue3;
			inValue3 = inValue4;

            int cur_inWidthIdx = inWidthOffset4 + kw;
            cur_inWidthIdx     = select(inCurIdx + cur_inWidthIdx, -1, (cur_inWidthIdx < 0 || cur_inWidthIdx >= inputShape.y));
            inValue4           = RI_F(input, SAMPLER, (int2)(cur_inWidthIdx, inHeightIdx));

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            outValue0 = mad(inValue0, weights, outValue0);
            outValue1 = mad(inValue1, weights, outValue1);
            outValue2 = mad(inValue2, weights, outValue2);
            outValue3 = mad(inValue3, weights, outValue3);
			outValue4 = mad(inValue4, weights, outValue4);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
	outValue4 = fmax(outValue4, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
	outValue4 = clamp(outValue4, (FLOAT4)0, (FLOAT4)6);
#endif

#ifdef LEAKY_RELU
    outValue0 = select(((FLOAT)(LEAKY_RELU))*outValue0, outValue0, outValue0 >= (FLOAT4)0);
    outValue1 = select(((FLOAT)(LEAKY_RELU))*outValue1, outValue1, outValue1 >= (FLOAT4)0);
    outValue2 = select(((FLOAT)(LEAKY_RELU))*outValue2, outValue2, outValue2 >= (FLOAT4)0);
    outValue3 = select(((FLOAT)(LEAKY_RELU))*outValue3, outValue3, outValue3 >= (FLOAT4)0);
    outValue4 = select(((FLOAT)(LEAKY_RELU))*outValue4, outValue4, outValue4 >= (FLOAT4)0);
#endif

    const int remain     = outputShape.y - outWidthBlockidx5;
    int outWidthIdx      = mad24(outChannelBlockIdx, outputShape.y, outWidthBlockidx5);
    if (remain >= 5) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
		WI_F(output, (int2)(outWidthIdx + 4, outHeightBlockIdx), outValue4);
    } else if (remain == 4) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
		WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
		WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
		WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
    }
}

__kernel
#if SET_ATTRIBUTE
//__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d_k3_s1_5in_opt(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t filter,
#ifndef NO_BIAS
                                    __read_only image2d_t bias,
#endif	
                                    __write_only image2d_t output,
                                    __private const int2 inputShape,
                                    __private const int inChannelBlocks, 
                                    __private const int2 outputShape,
                                    __private const int2 filterShape,
                                    __private const int2 paddingShape) 
{
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int ow5                      = (outputShape.y + 4) / 5;
    const int outChannelBlockIdx = outChannelWidthIdx / ow5;
    const int outWidthBlockidx   = outChannelWidthIdx % ow5;

    const int inChannelBlockIdx  = outChannelBlockIdx;

#ifndef NO_BIAS
    FLOAT4 outValue0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
#else
	FLOAT4 outValue0 = (FLOAT4)(0.0f);
#endif	
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;
	FLOAT4 outValue4 = outValue0;

    const int  outWidthBlockidx5 = mul24(outWidthBlockidx, 5);
    const int  inWidthOffset     = outWidthBlockidx5 - 1;
	const int4 inWidthOffset0    = (int4)(inWidthOffset, inWidthOffset + 1, inWidthOffset + 2, inWidthOffset + 3);
	const int4 inWidthOffset1    = inWidthOffset0 + (int4)(4);

    int heightIdx            = outHeightBlockIdx % outputShape.x - 1;
    const int outBatchIdx    = mul24((outHeightBlockIdx / outputShape.x), inputShape.x);
    const int inCurIdx       = mul24(inChannelBlockIdx, inputShape.y);
	
    const int4 inWidthIdx0   = select((int4)(inCurIdx) + inWidthOffset0, (int4)(-1), (((int4)(inWidthOffset0)) < ((int4)(0)) || ((int4)(inWidthOffset0)) >= (int4)(inputShape.y)));
	const int4 inWidthIdx1   = select((int4)(inCurIdx) + inWidthOffset1, (int4)(-1), (((int4)(inWidthOffset1)) < ((int4)(0)) || ((int4)(inWidthOffset1)) >= (int4)(inputShape.y)));
    int    filterIdx = 0;
	for (int kh = 0; kh < filterShape.x; kh++) {
        int inHeightIdx = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= inputShape.x));
        heightIdx++;
		
		const FLOAT4 inValue0       = RI_F(input, SAMPLER, (int2)(inWidthIdx0.x, inHeightIdx));
        const FLOAT4 inValue1       = RI_F(input, SAMPLER, (int2)(inWidthIdx0.y, inHeightIdx));
        const FLOAT4 inValue2       = RI_F(input, SAMPLER, (int2)(inWidthIdx0.z, inHeightIdx));
        const FLOAT4 inValue3       = RI_F(input, SAMPLER, (int2)(inWidthIdx0.w, inHeightIdx));
		const FLOAT4 inValue4       = RI_F(input, SAMPLER, (int2)(inWidthIdx1.x, inHeightIdx));
		const FLOAT4 inValue5       = RI_F(input, SAMPLER, (int2)(inWidthIdx1.y, inHeightIdx));
		const FLOAT4 inValue6       = RI_F(input, SAMPLER, (int2)(inWidthIdx1.z, inHeightIdx));
		
		FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx ++, inChannelBlockIdx));
		outValue0 = mad(inValue0, weights, outValue0);
		outValue1 = mad(inValue1, weights, outValue1);
		outValue2 = mad(inValue2, weights, outValue2);
		outValue3 = mad(inValue3, weights, outValue3);
		outValue4 = mad(inValue4, weights, outValue4);
		
		weights = RI_F(filter, SAMPLER, (int2)(filterIdx ++, inChannelBlockIdx));
		outValue0 = mad(inValue1, weights, outValue0);
		outValue1 = mad(inValue2, weights, outValue1);
		outValue2 = mad(inValue3, weights, outValue2);
		outValue3 = mad(inValue4, weights, outValue3);
		outValue4 = mad(inValue5, weights, outValue4);
		
		weights = RI_F(filter, SAMPLER, (int2)(filterIdx ++, inChannelBlockIdx));
		outValue0 = mad(inValue2, weights, outValue0);
		outValue1 = mad(inValue3, weights, outValue1);
		outValue2 = mad(inValue4, weights, outValue2);
		outValue3 = mad(inValue5, weights, outValue3);
		outValue4 = mad(inValue6, weights, outValue4);
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
	outValue4 = fmax(outValue4, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
	outValue4 = clamp(outValue4, (FLOAT4)0, (FLOAT4)6);
#endif

#ifdef LEAKY_RELU
    outValue0 = select(((FLOAT)(LEAKY_RELU))*outValue0, outValue0, outValue0 >= (FLOAT4)0);
    outValue1 = select(((FLOAT)(LEAKY_RELU))*outValue1, outValue1, outValue1 >= (FLOAT4)0);
    outValue2 = select(((FLOAT)(LEAKY_RELU))*outValue2, outValue2, outValue2 >= (FLOAT4)0);
    outValue3 = select(((FLOAT)(LEAKY_RELU))*outValue3, outValue3, outValue3 >= (FLOAT4)0);
    outValue4 = select(((FLOAT)(LEAKY_RELU))*outValue4, outValue4, outValue4 >= (FLOAT4)0);
#endif

    const int remain     = outputShape.y - outWidthBlockidx5;
    int outWidthIdx      = mad24(outChannelBlockIdx, outputShape.y, outWidthBlockidx5);
    if (remain >= 5) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
		WI_F(output, (int2)(outWidthIdx + 4, outHeightBlockIdx), outValue4);
    } else if (remain == 4) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
		WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
		WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
		WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx,     outHeightBlockIdx), outValue0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d_s1_4row(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t filter,
#ifndef NO_BIAS
	                          __read_only image2d_t bias,
#endif							 
	                          __write_only image2d_t output,
	                          __private const int2 inputShape,
	                          __private const int inChannelBlocks,
	                          __private const int2 outputShape,
	                          __private const int2 filterShape,
	                          __private const int2 paddingShape)
{
    const int outChannelWidthIdx  = get_global_id(0);
    const int outHeightBlockIdx   = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int       out_height_q        = (outputShape.x + 3) >> 2;
    const int out_channel_blk_idx = outChannelWidthIdx / outputShape.y;
    const int out_x_pos           = outChannelWidthIdx - mul24(out_channel_blk_idx, outputShape.y);
    const int out_batch_idx       = outHeightBlockIdx / out_height_q;
    const int out_batch_base      = mul24(out_batch_idx, out_height_q);
    const int out_y_pos           = (outHeightBlockIdx - out_batch_base) << 2;

#ifndef NO_BIAS
    FLOAT4    outValue0     = RI_F(bias, SAMPLER, (int2)(out_channel_blk_idx, 0));
#else
    FLOAT4    outValue0     = (FLOAT4)(0.0f);
#endif	
    FLOAT4    outValue1     = outValue0;
    FLOAT4    outValue2     = outValue0;
    FLOAT4    outValue3     = outValue0;

    int       in_width_idx  = out_x_pos - paddingShape.y;
    const int in_height_idx = out_y_pos - paddingShape.x;

    const int in_x_base     = mul24(out_channel_blk_idx, inputShape.y);

    int4      in_y_ofs      = (int4)(in_height_idx, in_height_idx + 1, in_height_idx + 2, in_height_idx + 3);
    int4      in_y_pos      = select(out_batch_base + in_y_ofs, -1, (in_y_ofs < ((int4)(0)) || in_y_ofs >= (int4)(inputShape.x)));
    FLOAT4 inValue0, inValue1, inValue2, inValue3, inValue4;
    for (int kw = 0; kw < filterShape.y; kw++) {
        int cur_x_pos = select(in_x_base + in_width_idx, -1, (in_width_idx < 0 || in_width_idx >= inputShape.y));
        in_width_idx++;

        inValue1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.x));
        inValue2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.y));
        inValue3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos.z));
        for (int kh = 0; kh < filterShape.x; kh++) {

            int filterIdx = mad24(kh, filterShape.y, kw);
            inValue0 = inValue1;
            inValue1 = inValue2;
            inValue2 = inValue3;

            int cur_in_y_pos = in_y_ofs.w + kh;
            cur_in_y_pos     = select(out_batch_base + cur_in_y_pos, -1, (cur_in_y_pos < 0 || cur_in_y_pos >= inputShape.x));
            inValue3         = RI_F(input, SAMPLER, (int2)(cur_x_pos, cur_in_y_pos));

            FLOAT4 weights   = RI_F(filter, SAMPLER, (int2)(filterIdx, out_channel_blk_idx));

            outValue0 = mad(inValue0, weights, outValue0);
            outValue1 = mad(inValue1, weights, outValue1);
            outValue2 = mad(inValue2, weights, outValue2);
            outValue3 = mad(inValue3, weights, outValue3);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

#ifdef LEAKY_RELU
    outValue0 = select(((FLOAT)(LEAKY_RELU))*outValue0, outValue0, outValue0 >= (FLOAT4)0);
    outValue1 = select(((FLOAT)(LEAKY_RELU))*outValue1, outValue1, outValue1 >= (FLOAT4)0);
    outValue2 = select(((FLOAT)(LEAKY_RELU))*outValue2, outValue2, outValue2 >= (FLOAT4)0);
    outValue3 = select(((FLOAT)(LEAKY_RELU))*outValue3, outValue3, outValue3 >= (FLOAT4)0);
#endif

#ifdef BN_AFTER_RELU
	const int out_channel_q = global_size_dim0 / outputShape.y;
	CAL_BN_AFTER_RELU(outValue0, outValue1, outValue2, outValue3, bias, out_channel_q, out_channel_blk_idx);
#endif

    const int remain   = outputShape.x - out_y_pos;
    const int output_y = out_batch_base + out_y_pos;
    if (remain >= 4) {
        WI_F(output, (int2)(outChannelWidthIdx, output_y),     outValue0);
        WI_F(output, (int2)(outChannelWidthIdx, output_y + 1), outValue1);
        WI_F(output, (int2)(outChannelWidthIdx, output_y + 2), outValue2);
        WI_F(output, (int2)(outChannelWidthIdx, output_y + 3), outValue3);
    }
    else if (remain == 3) {
        WI_F(output, (int2)(outChannelWidthIdx, output_y),     outValue0);
        WI_F(output, (int2)(outChannelWidthIdx, output_y + 1), outValue1);
        WI_F(output, (int2)(outChannelWidthIdx, output_y + 2), outValue2);
    }
    else if (remain == 2) {
        WI_F(output, (int2)(outChannelWidthIdx, output_y),     outValue0);
        WI_F(output, (int2)(outChannelWidthIdx, output_y + 1), outValue1);
    }
    else if (remain == 1) {
        WI_F(output, (int2)(outChannelWidthIdx, output_y),     outValue0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d_k3_s1_4row(GLOBAL_SIZE_2_DIMS
                                 __read_only image2d_t  input,
                                 __read_only image2d_t  filter,
                                 __read_only image2d_t  bias,
                                 __write_only image2d_t output,
                                 __private const int2   inputShape,
                                 __private const int    inChannelBlocks,
                                 __private const int2   outputShape,
                                 __private const int2   filterShape,
                                 __private const int2   paddingShape)
{
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    const int outChannelBlockIdx = outChannelWidthIdx / outputShape.y;
    const int out_x_pos          = (outChannelWidthIdx - mul24(outChannelBlockIdx, outputShape.y));
    const int out_height_q       = (outputShape.x + 3) >> 2;
    const int out_batch_idx      = outHeightBlockIdx / out_height_q;
    const int out_batch_base     = mul24(out_batch_idx, out_height_q);
    const int out_y_pos          = (outHeightBlockIdx - out_batch_base) << 2;

    FLOAT4    outValue0          = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4    outValue1          = outValue0;
    FLOAT4    outValue2          = outValue0;
    FLOAT4    outValue3          = outValue0;

    int       in_width_idx       = out_x_pos - 1;
    const int in_height_idx      = out_y_pos - 1;

    const int in_x_base = mul24(outChannelBlockIdx, inputShape.y);

    int4      in_y_ofs0 = (int4)(in_height_idx, in_height_idx + 1, in_height_idx + 2, in_height_idx + 3);
    int4      in_y_ofs1 = in_y_ofs0 + (int4)(4);
    int4      in_y_pos0 = select(out_batch_base + in_y_ofs0, -1, (in_y_ofs0 < ((int4)(0)) || in_y_ofs0 >= (int4)(inputShape.x)));
    int4      in_y_pos1 = select(out_batch_base + in_y_ofs1, -1, (in_y_ofs1 < ((int4)(0)) || in_y_ofs1 >= (int4)(inputShape.x)));
    int       filterIdx = 0;
    for (int kh = 0; kh < 3; ++kh)
    {
        int cur_x_pos = select(in_x_base + in_width_idx, -1, (in_width_idx < 0 || in_width_idx >= inputShape.y));
        in_width_idx++;

        FLOAT4 inValue0 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0.x));
        FLOAT4 inValue1 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0.y));
        FLOAT4 inValue2 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0.z));
        FLOAT4 inValue3 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos0.w));
        FLOAT4 inValue4 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1.x));
        FLOAT4 inValue5 = RI_F(input, SAMPLER, (int2)(cur_x_pos, in_y_pos1.y));
        FLOAT4 weights  = RI_F(filter, SAMPLER, (int2)(filterIdx, outChannelBlockIdx));
        outValue0 = mad(inValue0, weights, outValue0);
        outValue1 = mad(inValue1, weights, outValue1);
        outValue2 = mad(inValue2, weights, outValue2);
        outValue3 = mad(inValue3, weights, outValue3);

        weights = RI_F(filter, SAMPLER, (int2)(filterIdx + 3, outChannelBlockIdx));
        outValue0 = mad(inValue1, weights, outValue0);
        outValue1 = mad(inValue2, weights, outValue1);
        outValue2 = mad(inValue3, weights, outValue2);
        outValue3 = mad(inValue4, weights, outValue3);

        weights = RI_F(filter, SAMPLER, (int2)(filterIdx + 6, outChannelBlockIdx));
        outValue0 = mad(inValue2, weights, outValue0);
        outValue1 = mad(inValue3, weights, outValue1);
        outValue2 = mad(inValue4, weights, outValue2);
        outValue3 = mad(inValue5, weights, outValue3);

        filterIdx++;
	}

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

#ifdef LEAKY_RELU
    outValue0 = select(((FLOAT)(LEAKY_RELU))*outValue0, outValue0, outValue0 >= (FLOAT4)0);
    outValue1 = select(((FLOAT)(LEAKY_RELU))*outValue1, outValue1, outValue1 >= (FLOAT4)0);
    outValue2 = select(((FLOAT)(LEAKY_RELU))*outValue2, outValue2, outValue2 >= (FLOAT4)0);
    outValue3 = select(((FLOAT)(LEAKY_RELU))*outValue3, outValue3, outValue3 >= (FLOAT4)0);
#endif

#ifdef BN_AFTER_RELU
	const int out_channel_q = global_size_dim0 / outputShape.y;
	CAL_BN_AFTER_RELU(outValue0, outValue1, outValue2, outValue3, bias, out_channel_q, outChannelBlockIdx);
#endif

    const int remain   = outputShape.x - out_y_pos;
    const int output_y = out_batch_base + out_y_pos;
    if (remain >= 4) {
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos),     outValue0);
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos + 1), outValue1);
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos + 2), outValue2);
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos + 3), outValue3);
    }
    else if (remain == 3) {
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos),     outValue0);
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos + 1), outValue1);
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos + 2), outValue2);
    }
    else if (remain == 2) {
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos),     outValue0);
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos + 1), outValue1);
    }
    else if (remain == 1) {
        WI_F(output, (int2)(outChannelWidthIdx, out_y_pos),     outValue0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d_k3_s2_opt(GLOBAL_SIZE_2_DIMS
                                __read_only image2d_t  input, 
								__read_only image2d_t  filter,
                                __read_only image2d_t  bias,
                                __write_only image2d_t output,
                                __private const int2   inputShape,
                                __private const int    inChannelBlocks, 
							    __private const int2   outputShape) 
{
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int ow4                      = (outputShape.y + 3) >> 2;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx  = outChannelBlockIdx;

    FLOAT4 outValue0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;

    const int inWidthOffset0   = (outWidthBlockidx << 3) - 1 ;
    int       heightIdx        = ((outHeightBlockIdx % outputShape.x) << 1) - 1;
	
    const int outBatchIdx      = mul24((outHeightBlockIdx / outputShape.x), inputShape.x);
	
    const int inCurIdx         = mul24(inChannelBlockIdx, inputShape.y);
	
	int4      in_width_offset0 = (int4)(inWidthOffset0, inWidthOffset0 + 1, inWidthOffset0 + 2, inWidthOffset0 + 3);
	int4      in_width_offset1 = in_width_offset0 + (int4)(4);
	int4      in_x_pos_0       = select((int4)(inCurIdx) + in_width_offset0,  -1, (in_width_offset0  < ((int4)(0)) || in_width_offset0  >= (int4)(inputShape.y)));
	int4      in_x_pos_1       = select((int4)(inCurIdx) + in_width_offset1,  -1, (in_width_offset1  < ((int4)(0)) || in_width_offset1  >= (int4)(inputShape.y)));
	int       in_x_pos_2       = select(inCurIdx + in_width_offset1.w + 1,    -1, ((in_width_offset1.w + 1)  < 0)  || ((in_width_offset1.w + 1)  >= inputShape.y));
	int       filterIdx        = 0;
    for (int kh = 0; kh < 3; ++kh) 
	{
        int inHeightIdx = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= inputShape.x));
        heightIdx++;

		FLOAT4 inValue0 = RI_F(input, SAMPLER, (int2)(in_x_pos_0.x, inHeightIdx));
		FLOAT4 inValue1 = RI_F(input, SAMPLER, (int2)(in_x_pos_0.y, inHeightIdx));
		FLOAT4 inValue2 = RI_F(input, SAMPLER, (int2)(in_x_pos_0.z, inHeightIdx));
		FLOAT4 inValue3 = RI_F(input, SAMPLER, (int2)(in_x_pos_0.w, inHeightIdx));
		FLOAT4 inValue4 = RI_F(input, SAMPLER, (int2)(in_x_pos_1.x, inHeightIdx));
		FLOAT4 inValue5 = RI_F(input, SAMPLER, (int2)(in_x_pos_1.y, inHeightIdx));
		FLOAT4 inValue6 = RI_F(input, SAMPLER, (int2)(in_x_pos_1.z, inHeightIdx));
		FLOAT4 inValue7 = RI_F(input, SAMPLER, (int2)(in_x_pos_1.w, inHeightIdx));
		FLOAT4 inValue8 = RI_F(input, SAMPLER, (int2)(in_x_pos_2,   inHeightIdx));
		FLOAT4 weights  = RI_F(filter, SAMPLER, (int2)(filterIdx++, inChannelBlockIdx));
		outValue0 = mad(inValue0, weights, outValue0);
		outValue1 = mad(inValue2, weights, outValue1);
		outValue2 = mad(inValue4, weights, outValue2);
		outValue3 = mad(inValue6, weights, outValue3);
		
		weights  = RI_F(filter, SAMPLER, (int2)(filterIdx++, inChannelBlockIdx));
		outValue0 = mad(inValue1, weights, outValue0);
		outValue1 = mad(inValue3, weights, outValue1);
		outValue2 = mad(inValue5, weights, outValue2);
		outValue3 = mad(inValue7, weights, outValue3);

		weights  = RI_F(filter, SAMPLER, (int2)(filterIdx++, inChannelBlockIdx));
		outValue0 = mad(inValue2, weights, outValue0);
		outValue1 = mad(inValue4, weights, outValue1);
		outValue2 = mad(inValue6, weights, outValue2);
		outValue3 = mad(inValue8, weights, outValue3);
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

#ifdef LEAKY_RELU
    outValue0 = select(((FLOAT)(LEAKY_RELU))*outValue0, outValue0, outValue0 >= (FLOAT4)0);
    outValue1 = select(((FLOAT)(LEAKY_RELU))*outValue1, outValue1, outValue1 >= (FLOAT4)0);
    outValue2 = select(((FLOAT)(LEAKY_RELU))*outValue2, outValue2, outValue2 >= (FLOAT4)0);
    outValue3 = select(((FLOAT)(LEAKY_RELU))*outValue3, outValue3, outValue3 >= (FLOAT4)0);
#endif

#ifdef BN_AFTER_RELU
	const int out_channel_q = global_size_dim0 / ow4;
	CAL_BN_AFTER_RELU(outValue0, outValue1, outValue2, outValue3, bias, out_channel_q, outChannelBlockIdx);
#endif

    const int outWidthBlockidx4        = outWidthBlockidx << 2;
    const int remain = outputShape.y - outWidthBlockidx4;
    int outWidthIdx   = mul24(outChannelBlockIdx, outputShape.y) + outWidthBlockidx4;
    if (remain >= 4) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d_k5_s2_opt(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t filter,
#ifndef NO_BIAS
                               __read_only image2d_t bias,
#endif	
                               __write_only image2d_t output,
                               __private const int2 inputShape,
                               __private const int inChannelBlocks, 
							   __private const int2 outputShape) 
{
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int ow4                      = (outputShape.y + 3) >> 2;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx  = outChannelBlockIdx;
#ifndef NO_BIAS
    FLOAT4 outValue0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
#else
	FLOAT4 outValue0 = (FLOAT4)(0.0f);
#endif	
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;

    const int inWidthOffset0    = (outWidthBlockidx << 3) - 2 ;
    const int inWidthOffset1    = inWidthOffset0 + 2;
    const int inWidthOffset2    = inWidthOffset0 + 4;
    const int inWidthOffset3    = inWidthOffset0 + 6;
    int       heightIdx        = ((outHeightBlockIdx % outputShape.x) << 1) - 2;
	
    const int outBatchIdx      = mul24((outHeightBlockIdx / outputShape.x), inputShape.x);
	
    const int inCurIdx         = mul24(inChannelBlockIdx, inputShape.y);
	
	int4      in_width_offset  = (int4)(inWidthOffset0, inWidthOffset1, inWidthOffset2, inWidthOffset3);
	int4      in_width_offset2 = in_width_offset + (int4)(1);
	int4      in_start_x       = select((int4)(inCurIdx) + in_width_offset,  -1, (in_width_offset  < ((int4)(0)) || in_width_offset  >= (int4)(inputShape.y)));
	int4      in_start_x2      = select((int4)(inCurIdx) + in_width_offset2, -1, (in_width_offset2 < ((int4)(0)) || in_width_offset2 >= (int4)(inputShape.y)));
    FLOAT4 inValue0, inValue1, inValue2, inValue3, inValue4, inValue5, inValue6, inValue7;
	int    filterIdx   = 0;
    for (int kh = 0; kh < 5; ++kh) 
	{
        int inHeightIdx = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= inputShape.x));
        heightIdx++;

		inValue1 = RI_F(input, SAMPLER, (int2)(in_start_x.x,  inHeightIdx));
		inValue5 = RI_F(input, SAMPLER, (int2)(in_start_x2.x, inHeightIdx));
		inValue2 = RI_F(input, SAMPLER, (int2)(in_start_x.y,  inHeightIdx));
		inValue6 = RI_F(input, SAMPLER, (int2)(in_start_x2.y, inHeightIdx));
		inValue3 = RI_F(input, SAMPLER, (int2)(in_start_x.z,  inHeightIdx));
		inValue7 = RI_F(input, SAMPLER, (int2)(in_start_x2.z, inHeightIdx));
        for (int kw = 0; kw < 5; ++kw) 
		{
			int cur_src_x   = in_width_offset.w + kw;
			cur_src_x       = select(inCurIdx + cur_src_x, -1, (cur_src_x < 0) || (cur_src_x >= inputShape.y));
			__private FLOAT4 cur_in0, cur_in1, cur_in2, cur_in3;
			if(0 == (kw&1))
			{
				inValue0     = inValue1;
				inValue1     = inValue2;
				inValue2     = inValue3;
				inValue3     = RI_F(input, SAMPLER, (int2)(cur_src_x, inHeightIdx));
				cur_in0      = inValue0;
				cur_in1      = inValue1;
				cur_in2      = inValue2;
				cur_in3      = inValue3;
			}
			else
			{
				inValue4     = inValue5;
				inValue5     = inValue6;
				inValue6     = inValue7;
				inValue7     = RI_F(input, SAMPLER, (int2)(cur_src_x, inHeightIdx));
				cur_in0      = inValue4;
				cur_in1      = inValue5;
				cur_in2      = inValue6;
				cur_in3      = inValue7;
			}
			
            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx++, inChannelBlockIdx));
			
            outValue0 = mad(cur_in0, weights, outValue0);
            outValue1 = mad(cur_in1, weights, outValue1);
            outValue2 = mad(cur_in2, weights, outValue2);
            outValue3 = mad(cur_in3, weights, outValue3);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

#ifdef LEAKY_RELU
    outValue0 = select(((FLOAT)(LEAKY_RELU))*outValue0, outValue0, outValue0 >= (FLOAT4)0);
    outValue1 = select(((FLOAT)(LEAKY_RELU))*outValue1, outValue1, outValue1 >= (FLOAT4)0);
    outValue2 = select(((FLOAT)(LEAKY_RELU))*outValue2, outValue2, outValue2 >= (FLOAT4)0);
    outValue3 = select(((FLOAT)(LEAKY_RELU))*outValue3, outValue3, outValue3 >= (FLOAT4)0);
#endif

#ifdef BN_AFTER_RELU
	const int out_channel_q = global_size_dim0 / ow4;
	CAL_BN_AFTER_RELU(outValue0, outValue1, outValue2, outValue3, bias, out_channel_q, outChannelBlockIdx);
#endif

    const int outWidthBlockidx4        = outWidthBlockidx << 2;
    const int remain = outputShape.y - outWidthBlockidx4;
    int outWidthIdx   = mul24(outChannelBlockIdx, outputShape.y) + outWidthBlockidx4;
    if (remain >= 4) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
    }
}