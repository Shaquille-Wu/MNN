#include "quantize_speedup.hpp"
#include <emmintrin.h>
#include <smmintrin.h>
#include <math.h>

int fake_quantize_data(float* src, int count, float scale, float clamp_value)
{
    float          scale_inv        = 1.0f / scale;
    __m128         plus_value       = _mm_set_ps1(clamp_value);
    __m128         minus_value      = _mm_set_ps1(-clamp_value);
    __m128         plus_half        = _mm_set_ps1(0.5f);
    __m128         minus_half       = _mm_set_ps1(-0.5f);
    __m128         scale_value      = _mm_set_ps1(scale);
    __m128         scale_inv_value  = _mm_set_ps1(scale_inv);
    __m128         zero_data        = _mm_set_ps1(0.0f);
    int            count_aligned_16 = (((count + 15) >> 4) << 4);
    int            count_aligned_4  = (((count + 3) >> 2) << 2);
    int            i                = 0;
    float*         src_ptr          = src;
    for(i = 0 ; i < count_aligned_16 ; i += 16)
    {
        __m128 data0       = _mm_loadu_ps(src_ptr);
        __m128 data1       = _mm_loadu_ps(src_ptr + 4);
        __m128 data2       = _mm_loadu_ps(src_ptr + 8);
        __m128 data3       = _mm_loadu_ps(src_ptr + 12);
        data0              = _mm_mul_ps(data0, scale_inv_value);
        data1              = _mm_mul_ps(data1, scale_inv_value);
        data2              = _mm_mul_ps(data2, scale_inv_value);
        data3              = _mm_mul_ps(data3, scale_inv_value);
        __m128 cur_sign0   = _mm_cmplt_ps(data0, zero_data);
        __m128 cur_sign1   = _mm_cmplt_ps(data1, zero_data);
        __m128 cur_sign2   = _mm_cmplt_ps(data2, zero_data);
        __m128 cur_sign3   = _mm_cmplt_ps(data3, zero_data);
        cur_sign0          = _mm_blendv_ps(plus_half, minus_half, cur_sign0);
        cur_sign1          = _mm_blendv_ps(plus_half, minus_half, cur_sign1);
        cur_sign2          = _mm_blendv_ps(plus_half, minus_half, cur_sign2);
        cur_sign3          = _mm_blendv_ps(plus_half, minus_half, cur_sign3);
        data0              = _mm_add_ps(data0, cur_sign0);
        data1              = _mm_add_ps(data1, cur_sign1);
        data2              = _mm_add_ps(data2, cur_sign2);
        data3              = _mm_add_ps(data3, cur_sign3);
        data0              = _mm_round_ps(data0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        data1              = _mm_round_ps(data1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        data2              = _mm_round_ps(data2, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        data3              = _mm_round_ps(data3, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

        data0              = _mm_min_ps(data0, plus_value);
        data1              = _mm_min_ps(data1, plus_value);
        data2              = _mm_min_ps(data2, plus_value);
        data3              = _mm_min_ps(data3, plus_value);
        data0              = _mm_max_ps(data0, minus_value);
        data1              = _mm_max_ps(data1, minus_value);
        data2              = _mm_max_ps(data2, minus_value);
        data3              = _mm_max_ps(data3, minus_value);

        data0              = _mm_mul_ps(data0, scale_value);
        data1              = _mm_mul_ps(data1, scale_value);
        data2              = _mm_mul_ps(data2, scale_value);
        data3              = _mm_mul_ps(data3, scale_value);

        _mm_storeu_ps(src_ptr,      data0);
        _mm_storeu_ps(src_ptr + 4,  data1);
        _mm_storeu_ps(src_ptr + 8,  data2);
        _mm_storeu_ps(src_ptr + 12, data3);

        src_ptr += 16;
    }

    for(; i < count_aligned_4 ; i += 4)
    {
        __m128 data0       = _mm_loadu_ps(src_ptr);
        data0              = _mm_mul_ps(data0, scale_inv_value);
        __m128 cur_sign0   = _mm_cmplt_ps(data0, zero_data);
        cur_sign0          = _mm_blendv_ps(plus_half, minus_half, cur_sign0);
        data0              = _mm_add_ps(data0, cur_sign0);
        data0              = _mm_round_ps(data0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        data0              = _mm_mul_ps(data0, scale_value);
        _mm_storeu_ps(src_ptr, data0);
        src_ptr += 4;
    }

    for(; i < count ; i ++)
    {
        float data = *src_ptr;
        data       = data * scale_inv;
        data       = roundf(data);
        data       = data * scale;
        *src_ptr   = data;
        src_ptr   += 1;
    }

    return 0;
}

int   select_min_max(float const* src, int count, float* min, float* max)
{
    int            count_aligned_16 = (((count + 15) >> 4) << 4);
    int            count_aligned_4  = (((count + 3) >> 2) << 2);
    int            i                = 0;
    float const*   src_ptr          = src;
    __m128         max_value        = _mm_set_ps1(-100000.0f);
    __m128         min_value        = _mm_set_ps1(100000.0f);

    for(i = 0 ; i < count_aligned_16 ; i += 16)
    {
        __m128 data0       = _mm_loadu_ps(src_ptr);
        __m128 data1       = _mm_loadu_ps(src_ptr + 4);
        __m128 data2       = _mm_loadu_ps(src_ptr + 8);
        __m128 data3       = _mm_loadu_ps(src_ptr + 12);
        max_value          = _mm_max_ps(max_value, data0);
        min_value          = _mm_min_ps(min_value, data0);
        max_value          = _mm_max_ps(max_value, data1);
        min_value          = _mm_min_ps(min_value, data1);
        max_value          = _mm_max_ps(max_value, data2);
        min_value          = _mm_min_ps(min_value, data2);
        max_value          = _mm_max_ps(max_value, data3);
        min_value          = _mm_min_ps(min_value, data3);
        src_ptr += 16;
    }

    for(; i < count_aligned_4 ; i += 4)
    {
        __m128 data0       = _mm_loadu_ps(src_ptr);
        max_value          = _mm_max_ps(max_value, data0);
        min_value          = _mm_min_ps(min_value, data0);
        src_ptr += 4;
    }

    float res0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float res1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    _mm_storeu_ps(res0, max_value);
    _mm_storeu_ps(res1, min_value);

    float res_max = res0[0];
    float res_min = res1[0];
    if(count_aligned_4 > 0)
    {
        res_max       = (res_max < res0[1] ? res0[1] : res_max);
        res_max       = (res_max < res0[2] ? res0[2] : res_max);
        res_max       = (res_max < res0[3] ? res0[3] : res_max);
        res_min       = (res_min > res1[1] ? res1[1] : res_min);
        res_min       = (res_min > res1[2] ? res1[2] : res_min);
        res_min       = (res_min > res1[3] ? res1[3] : res_min);
    }

    for(; i < count ; i ++)
    {
        res_max  = (res_max < (*src_ptr) ? (*src_ptr) : res_max);
        res_min  = (res_min > (*src_ptr) ? (*src_ptr) : res_min);
        src_ptr += 1;
    }

    *min = res_min;
    *max = res_max;
}