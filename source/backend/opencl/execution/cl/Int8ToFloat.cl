#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void int8_to_float(
                        __private const int global_size_dim0, __private const int global_size_dim1,
                        __private const int global_size_dim2,
                        __global char* input_ptr, __write_only image2d_t output,
                        #ifdef BUFFER_INP_FP32
                        __global float* scale_ptr,
                        #else
                        __global FLOAT* scale_ptr,
                        #endif
                        __private const int height,
                        __private const int width) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);
    
    if (channel_block_idx < global_size_dim0 && w < global_size_dim1 && hb < global_size_dim2) {
        
        int index = channel_block_idx*height*width + hb*width + w;
        char4 in = vload4(index, input_ptr);

        #ifdef BUFFER_INP_FP32
        FLOAT4 scale = CONVERT_FLOAT4(vload4(channel_block_idx, (__global float *)scale_ptr));
        #else
        FLOAT4 scale = vload4(channel_block_idx, (__global FLOAT *)scale_ptr);
        #endif
        FLOAT4 result_float = CONVERT_FLOAT4(convert_int4_rte(in)) * scale;

        const int pos  = mad24(channel_block_idx, width, w);
        WI_F(output, (int2)(pos, hb), result_float);
        
    }
    
}
