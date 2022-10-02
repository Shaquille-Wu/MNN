//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2019/08/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <emmintrin.h>
#include <string.h>
#include <algorithm>
//Shaquille, Added 20210112 Start
#include <math.h>
//Shaquille, Added 20210112 End
#include "core/Macro.h"
#include "FunctionSummary.hpp"
void _SSE_MNNInt8ToInt16(int16_t* dest, const int8_t* source, size_t count) {
    int countC16 = count / 16;
    int countR = count % 16;
    auto zero = _mm_set1_epi8(0);
    for (int i = 0; i < countC16; ++i) {
        auto s = _mm_castps_si128(_mm_loadu_ps((float*)source));
        auto d0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, s), 8);
        auto d1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, s), 8);
        _mm_storeu_ps((float*)dest, _mm_castsi128_ps(d0));
        _mm_storeu_ps((float*)dest + 4, _mm_castsi128_ps(d1));

        dest += 16;
        source += 16;
    }
    for (int i = 0; i < countR; ++i) {
        dest[i] = source[i];
    }
}

void _SSE_MNNAddBias(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNAddBiasRelu(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNAddBiasRelu6(float* dst, const float* bias, size_t planeNumber, size_t biasNumber) {
    auto maxV = _mm_set1_ps(0.0f);
    auto minV = _mm_set1_ps(6.0f);
    for (int z = 0; z < biasNumber; ++z) {
        auto biasV   = _mm_loadu_ps(bias + 4 * z);
        float* dst_z = dst + planeNumber * 4 * z;
        for (int p = 0; p < planeNumber; ++p) {
            auto dstV = _mm_add_ps(_mm_loadu_ps(dst_z + 4 * p), biasV);
            dstV      = _mm_max_ps(dstV, maxV);
            dstV      = _mm_min_ps(dstV, minV);
            _mm_storeu_ps(dst_z + 4 * p, dstV);
        }
    }
}

void _SSE_MNNCopyC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_storeu_ps(d, _mm_loadu_ps(s));
    }
}

void _SSE_MNNAddC4WithStride(const float* source, float* dest, size_t srcStride, size_t dstStride, size_t count) {
    for (int i = 0; i < count; ++i) {
        auto s = source + i * srcStride;
        auto d = dest + i * dstStride;
        _mm_storeu_ps(d, _mm_add_ps(_mm_loadu_ps(s), _mm_loadu_ps(d)));
    }
}

void _SSE_MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    auto zero = _mm_set1_ps(0.0f);
    for (int j = 0; j < depthQuad; j++) {
        auto slopeZ       = _mm_loadu_ps(slope + 4 * j);
        const float* srcZ = src + 4 * j * sizeQuad;
        float* dstZ       = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            auto src   = _mm_loadu_ps(srcZ + 4 * i);
            auto mask0 = _mm_cmplt_ps(src, zero);
            auto mask1 = _mm_cmpge_ps(src, zero);
            auto other = _mm_mul_ps(src, slopeZ);
            _mm_storeu_ps(dstZ + 4 * i, _mm_add_ps(_mm_and_ps(other, mask0), _mm_and_ps(src, mask1)));
        }
    }
}

void _SSE_MNNConvRunForLineDepthwise(float* dst, const float* src, const float* weight, size_t width, size_t src_w_setup,
                                size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, size_t height,
                                     size_t srcHStep, size_t dstHStep) {
    int dx, fx, fy;
    const int unit = 8;
    int widthUnit = width / unit;
    int widthRemain = width - widthUnit * unit;
    const float* weight_z = weight;
    bool need4 = widthRemain >= 4;
    if (need4) {
        widthRemain-=4;
    }
    for (int y = 0; y < height; ++y) {
        auto srcY = src + y * srcHStep;
        auto dstY = dst + y * dstHStep;
        for (dx = 0; dx < widthUnit; ++dx) {
            auto dstValue0 = _mm_set1_ps(0.0f);
            auto dstValue1 = _mm_set1_ps(0.0f);
            auto dstValue2 = _mm_set1_ps(0.0f);
            auto dstValue3 = _mm_set1_ps(0.0f);
            auto dstValue4 = _mm_set1_ps(0.0f);
            auto dstValue5 = _mm_set1_ps(0.0f);
            auto dstValue6 = _mm_set1_ps(0.0f);
            auto dstValue7 = _mm_set1_ps(0.0f);
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + 4 * fx;
                    auto weightValue = _mm_loadu_ps(weight_x);
                    dstValue0 = _mm_add_ps(dstValue0, _mm_mul_ps(_mm_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm_add_ps(dstValue1, _mm_mul_ps(_mm_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm_add_ps(dstValue2, _mm_mul_ps(_mm_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm_add_ps(dstValue3, _mm_mul_ps(_mm_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                    dstValue4 = _mm_add_ps(dstValue4, _mm_mul_ps(_mm_loadu_ps(src_x + 4 * src_w_setup), weightValue));
                    dstValue5 = _mm_add_ps(dstValue5, _mm_mul_ps(_mm_loadu_ps(src_x + 5 * src_w_setup), weightValue));
                    dstValue6 = _mm_add_ps(dstValue6, _mm_mul_ps(_mm_loadu_ps(src_x + 6 * src_w_setup), weightValue));
                    dstValue7 = _mm_add_ps(dstValue7, _mm_mul_ps(_mm_loadu_ps(src_x + 7 * src_w_setup), weightValue));
                }
            }
            _mm_storeu_ps(dstY + 4 * 0, dstValue0);
            _mm_storeu_ps(dstY + 4 * 1, dstValue1);
            _mm_storeu_ps(dstY + 4 * 2, dstValue2);
            _mm_storeu_ps(dstY + 4 * 3, dstValue3);
            _mm_storeu_ps(dstY + 4 * 4, dstValue4);
            _mm_storeu_ps(dstY + 4 * 5, dstValue5);
            _mm_storeu_ps(dstY + 4 * 6, dstValue6);
            _mm_storeu_ps(dstY + 4 * 7, dstValue7);
            dstY += 4 * unit;
            srcY += unit * src_w_setup;
        }
        if (need4) {
            auto dstValue0 = _mm_set1_ps(0.0f);
            auto dstValue1 = _mm_set1_ps(0.0f);
            auto dstValue2 = _mm_set1_ps(0.0f);
            auto dstValue3 = _mm_set1_ps(0.0f);
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = srcY + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* src_x    = src_y + fx * dilateX_step;
                    const float* weight_x = weight_y + 4 * fx;
                    auto weightValue = _mm_loadu_ps(weight_x);
                    dstValue0 = _mm_add_ps(dstValue0, _mm_mul_ps(_mm_loadu_ps(src_x + 0 * src_w_setup), weightValue));
                    dstValue1 = _mm_add_ps(dstValue1, _mm_mul_ps(_mm_loadu_ps(src_x + 1 * src_w_setup), weightValue));
                    dstValue2 = _mm_add_ps(dstValue2, _mm_mul_ps(_mm_loadu_ps(src_x + 2 * src_w_setup), weightValue));
                    dstValue3 = _mm_add_ps(dstValue3, _mm_mul_ps(_mm_loadu_ps(src_x + 3 * src_w_setup), weightValue));
                }
            }
            _mm_storeu_ps(dstY + 4 * 0, dstValue0);
            _mm_storeu_ps(dstY + 4 * 1, dstValue1);
            _mm_storeu_ps(dstY + 4 * 2, dstValue2);
            _mm_storeu_ps(dstY + 4 * 3, dstValue3);
            dstY += 4 * 4;
            srcY += 4 * src_w_setup;
        }
        for (dx = 0; dx < widthRemain; ++dx) {
            float* dst_x          = dstY + dx * 4;
            auto dstValue = _mm_set1_ps(0.0f);
            const float* src_z    = srcY + src_w_setup * dx;
            const float* weight_z = weight;
            for (fy = 0; fy < fh; ++fy) {
                const float* src_y    = src_z + fy * dilateY_step;
                const float* weight_y = weight_z + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const float* weight_x = weight_y + 4 * fx;
                    const float* src_x    = src_y + fx * dilateX_step;
                    dstValue = _mm_add_ps(dstValue, _mm_mul_ps(_mm_loadu_ps(src_x), _mm_loadu_ps(weight_x)));
                }
            }
            _mm_storeu_ps(dst_x, dstValue);
        }
    }
}

void _SSE_MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    auto count = countC8 * 2;
    auto p0    = _mm_set1_ps(parameters[0]);
    auto p1    = _mm_set1_ps(parameters[1]);
    auto p2    = _mm_set1_ps(parameters[2]);
    auto p3    = _mm_set1_ps(parameters[3]);
    auto p4    = _mm_set1_ps(parameters[4]);
    auto p5    = _mm_set1_ps(parameters[5]);
    auto p6    = _mm_set1_ps(parameters[6]);
    auto p7    = _mm_set1_ps(parameters[7]);
    auto xMax  = _mm_set1_ps(87);
    auto xMin  = _mm_set1_ps(-87);
    auto basic = _mm_set1_epi32(1 << 23);
    for (int i = 0; i < count; ++i) {
        auto x            = _mm_xor_ps(_mm_loadu_ps(source + i * 4), _mm_set1_ps(-0.f));
        x                 = _mm_max_ps(x, xMin);
        x                 = _mm_min_ps(x, xMax);
        auto div          = _mm_mul_ps(x, p1);
//Shaquille, Modified 20210112 Start
#if 0
		auto divInt       = _mm_cvtps_epi32(div);
#else
		auto divInt       = _mm_cvtps_epi32(_mm_round_ps(div, 3));
#endif
//Shaquille, Modified 20210112 End
        div               = _mm_cvtepi32_ps(divInt);
        auto div2         = _mm_add_epi32(divInt, _mm_set1_epi32(127));
        div2 = _mm_mullo_epi32(div2, basic);
        auto expBasic  = _mm_castsi128_ps(div2);
        auto xReamin   = _mm_sub_ps(x, _mm_mul_ps(div, p0));
        auto t         = xReamin;
        auto c0        = _mm_mul_ps(p7, t);
        auto c1        = _mm_add_ps(c0, p6);
        auto c2        = _mm_mul_ps(c1, t);
        auto c3        = _mm_add_ps(c2, p5);
        auto c4        = _mm_mul_ps(c3, t);
        auto c5        = _mm_add_ps(c4, p4);
        auto c6        = _mm_mul_ps(c5, t);
        auto c7        = _mm_add_ps(c6, p3);
        auto c8        = _mm_mul_ps(c7, t);
        auto c9        = _mm_add_ps(c8, p2);
        auto expRemain = c9;
        _mm_store_ps(dest + 4 * i, _mm_mul_ps(expBasic, expRemain));
    }
}

//Shaquille, Modified 20210111 Start
#ifndef MNN_ORION_INT8_OPT
void _SSE_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minV, ssize_t maxV, ssize_t zeroPoint) {
    __m128i zero = _mm_set1_epi32(0);
    __m128 minValue = _mm_set1_ps(minV);
    __m128 maxValue = _mm_set1_ps(maxV);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    __m128 scaleValue = _mm_loadu_ps(scalep);
    int32_t temp[4];

    for (int i = 0; i < sizeQuad; ++i) {
        __m128 f0 = _mm_loadu_ps(src + 4 * i);
        f0 = _mm_mul_ps(f0, scaleValue);
        f0 = _mm_min_ps(f0, maxValue);
        f0 = _mm_max_ps(f0, minValue);
        auto m0 = _mm_cmplt_ps(f0, _mm_castsi128_ps(zero));
        m0 = _mm_blendv_ps(plus, minus, m0);
        f0 = _mm_add_ps(f0, m0);
        // 3: _MM_FROUND_TO_ZERO
        auto d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
        *(__m128i*)temp = d0;
        for (int j=0; j<4; ++j) {
            dst[4*i+j] = temp[j];
        }
    }
}

void _SSE_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t sizeQuad, ssize_t zeroPoint) {
    auto sizeC4 = sizeQuad / 4;
    auto sizeRemain = sizeQuad % 4;
    __m128i zero = _mm_set1_epi32(0);
    __m128 scaleValue = _mm_loadu_ps(scale);
    for (int i = 0; i < sizeC4; ++i) {
        auto s = _mm_castps_si128(_mm_loadu_ps((const float*)(src)));
        auto s0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, s), 8);
        auto s1_16 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, s), 8);
        auto s0_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s0_16), 16);
        auto s1_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s0_16), 16);
        auto s2_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s1_16), 16);
        auto s3_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s1_16), 16);
        auto s0_f = _mm_cvtepi32_ps(s0_32);
        auto s1_f = _mm_cvtepi32_ps(s1_32);
        auto s2_f = _mm_cvtepi32_ps(s2_32);
        auto s3_f = _mm_cvtepi32_ps(s3_32);
        _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 2, _mm_mul_ps(s2_f, scaleValue));
        _mm_storeu_ps(dst + 4 * 3, _mm_mul_ps(s3_f, scaleValue));
        src += 16;
        dst += 16;
    }
    if (sizeRemain > 0) {
        int8_t srcTemp[128];
        ::memcpy(srcTemp, src, sizeRemain * 4);
        auto s = *(__m128i*)srcTemp;
        auto s0_16 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, s), 8);
        auto s1_16 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, s), 8);
        auto s0_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s0_16), 16);
        auto s1_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s0_16), 16);
        auto s2_32 = _mm_srai_epi32(_mm_unpacklo_epi16(zero, s1_16), 16);
        auto s3_32 = _mm_srai_epi32(_mm_unpackhi_epi16(zero, s1_16), 16);
        auto s0_f = _mm_cvtepi32_ps(s0_32);
        auto s1_f = _mm_cvtepi32_ps(s1_32);
        auto s2_f = _mm_cvtepi32_ps(s2_32);
        auto s3_f = _mm_cvtepi32_ps(s3_32);
        switch (sizeRemain) {
            case 3:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 2, _mm_mul_ps(s2_f, scaleValue));
                break;
            case 2:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                _mm_storeu_ps(dst + 4 * 1, _mm_mul_ps(s1_f, scaleValue));
                break;
            case 1:
                _mm_storeu_ps(dst + 4 * 0, _mm_mul_ps(s0_f, scaleValue));
                break;
            default:
                break;
        }
    }
}

#else
void _SSE_MNNFloat2Int8(const float* src, int8_t* dst, size_t sizeQuad, const float* scalep, ssize_t minValue, ssize_t maxValue, ssize_t zeroPoint)
{
	size_t   i          = 0;
	__m128   scale_p    = _mm_loadu_ps(scalep);
	__m128i  min_val    = _mm_set_epi32(minValue, minValue, minValue, minValue);
	__m128i  max_val    = _mm_set_epi32(maxValue, maxValue, maxValue, maxValue);
	__m128i  all_one    = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
	__m128   plus_half  = _mm_set1_ps(0.5f);
	__m128   minus_half = _mm_set1_ps(-0.5f);
	__m128   zero_data  = _mm_set1_ps(0.0f);
	for (i = 0; i < sizeQuad; ++i)
	{
		__m128   cur_data   = _mm_loadu_ps(src + 4 * i);
		__m128   cur_res_f  = _mm_mul_ps(cur_data, scale_p);
		__m128   cur_sign   = _mm_cmplt_ps(cur_res_f, zero_data);
		cur_sign            = _mm_blendv_ps(plus_half, minus_half, cur_sign);
		cur_res_f           = _mm_add_ps(cur_res_f, cur_sign);
		__m128i  cur_res    = _mm_cvtps_epi32(_mm_round_ps(cur_res_f, 3));
		cur_res             = _mm_max_epi32(cur_res, min_val);
		cur_res             = _mm_min_epi32(cur_res, max_val);
		__m128i  int16_data = _mm_packs_epi32(cur_res, cur_res);
		__m128i  int8_data  = _mm_packs_epi16(int16_data, int16_data);
		((int*)dst)[i]      = _mm_cvtsi128_si32(int8_data);
	}
}

void  _SSE_MNNInt8ScaleToFloat(float* dst, const int8_t* src, const float* scale, size_t size, ssize_t zeroPoint)
{
    __m128        scale_p   = _mm_loadu_ps(scale);
    size_t        size_q    = (size >> 2);
    size_t        size_tail = size - (size_q << 2);
    size_t        i         = 0;
    size_t        j         = 0;
    __m128i       mask0     = _mm_set1_epi32(0xFF00FF00);
    __m128i       mask1     = _mm_set1_epi32(0xFFFF0000);
    for (i = 0; i < size_q; i++, j += 16)
    {
        __m128i   int8_data_raw   = _mm_loadu_si128((__m128i*)(src + j));
#ifdef _MSC_VER
        __m128i   cur_int8_data0  = _mm_slli_epi16(int8_data_raw, 8);     // 0  2  4  6  8  10  12  14
        __m128i   cur_int8_data1  = _mm_and_si128(int8_data_raw, mask0);  // 1  3  5  7  9  11  13  15
        __m128i   cur_int8_data00 = _mm_slli_epi32(cur_int8_data0, 16);   // 0     4     8      12
        __m128i   cur_int8_data10 = _mm_slli_epi32(cur_int8_data1, 16);   // 1     5     9      13
        __m128i   cur_int8_data01 = _mm_and_si128(cur_int8_data0, mask1); // 2     6    10      14
        __m128i   cur_int8_data11 = _mm_and_si128(cur_int8_data1, mask1); // 3     7    11      15
        cur_int8_data00 = _mm_srai_epi32(cur_int8_data00, 24);
        cur_int8_data10 = _mm_srai_epi32(cur_int8_data10, 24);
        cur_int8_data01 = _mm_srai_epi32(cur_int8_data01, 24);
        cur_int8_data11 = _mm_srai_epi32(cur_int8_data11, 24);
        __m128    data0 = _mm_cvtepi32_ps(cur_int8_data00);
        __m128    data1 = _mm_cvtepi32_ps(cur_int8_data10);
        __m128    data2 = _mm_cvtepi32_ps(cur_int8_data01);
        __m128    data3 = _mm_cvtepi32_ps(cur_int8_data11);
        _MM_TRANSPOSE4_PS(data0, data1, data2, data3);
#else
        __m128i   cur_int8_data00 = _mm_cvtepi8_epi32(int8_data_raw);
        __m128i   cur_int8_data01 = _mm_cvtepi8_epi32(_mm_srli_si128(int8_data_raw, 4));
        __m128i   cur_int8_data10 = _mm_cvtepi8_epi32(_mm_srli_si128(int8_data_raw, 8));
        __m128i   cur_int8_data11 = _mm_cvtepi8_epi32(_mm_srli_si128(int8_data_raw, 12));
        __m128    data0 = _mm_cvtepi32_ps(cur_int8_data00);
        __m128    data1 = _mm_cvtepi32_ps(cur_int8_data01);
        __m128    data2 = _mm_cvtepi32_ps(cur_int8_data10);
        __m128    data3 = _mm_cvtepi32_ps(cur_int8_data11);
#endif

        data0 = _mm_mul_ps(data0, scale_p);
        data1 = _mm_mul_ps(data1, scale_p);
        data2 = _mm_mul_ps(data2, scale_p);
        data3 = _mm_mul_ps(data3, scale_p);
        _mm_storeu_ps(dst, data0);
        _mm_storeu_ps(dst + 4, data1);
        _mm_storeu_ps(dst + 8, data2);
        _mm_storeu_ps(dst + 12, data3);
        dst += 16;
    }

    size_q = (size_q << 2);
    j      = 0;
    for (i = size_q; i < size; i++, j += 4)
    {
        dst[j]     = static_cast<float>(src[4 * i])     * scale[0];
        dst[j + 1] = static_cast<float>(src[4 * i + 1]) * scale[1];
        dst[j + 2] = static_cast<float>(src[4 * i + 2]) * scale[2];
        dst[j + 3] = static_cast<float>(src[4 * i + 3]) * scale[3];
    }
}
#endif
//Shaquille, Modified 20210111 End

//Shaquille, Added 20201208 Start
void _SSE_MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber, size_t biasNumber)
{
	size_t i = 0, j = 0;
	int    plane_number_8 = ((planeNumber >> 3) << 3);
	for (i = 0; i < biasNumber; i++)
	{
		float*              cur_dst   = dst + 4 * planeNumber * i;
		float const*        cur_src   = src + 4 * planeNumber * i;
		unsigned long long  dst_addr  = (unsigned long long)cur_dst;
		unsigned long long  src_addr  = (unsigned long long)cur_src;
		__m128              cur_bias  = _mm_loadu_ps(bias + 4 * i);
		__m128              cur_alpha = _mm_loadu_ps(alpha + 4 * i);
		if (0 == (dst_addr & 0xF) && 0 == (src_addr & 0xF))
		{
			for (j = 0; j < plane_number_8; j += 8)
			{
				__m128   cur_data0 = _mm_load_ps(cur_src);
				__m128   cur_data1 = _mm_load_ps(cur_src + 4);
				__m128   cur_data2 = _mm_load_ps(cur_src + 8);
				__m128   cur_data3 = _mm_load_ps(cur_src + 12);
				__m128   cur_data4 = _mm_load_ps(cur_src + 16);
				__m128   cur_data5 = _mm_load_ps(cur_src + 20);
				__m128   cur_data6 = _mm_load_ps(cur_src + 24);
				__m128   cur_data7 = _mm_load_ps(cur_src + 28);
				__m128   cur_res0  = _mm_mul_ps(cur_data0, cur_alpha);
				__m128   cur_res1  = _mm_mul_ps(cur_data1, cur_alpha);
				__m128   cur_res2  = _mm_mul_ps(cur_data2, cur_alpha);
				__m128   cur_res3  = _mm_mul_ps(cur_data3, cur_alpha);
				__m128   cur_res4  = _mm_mul_ps(cur_data4, cur_alpha);
				__m128   cur_res5  = _mm_mul_ps(cur_data5, cur_alpha);
				__m128   cur_res6  = _mm_mul_ps(cur_data6, cur_alpha);
				__m128   cur_res7  = _mm_mul_ps(cur_data7, cur_alpha);

				cur_res0 = _mm_add_ps(cur_res0, cur_bias);
				cur_res1 = _mm_add_ps(cur_res1, cur_bias);
				cur_res2 = _mm_add_ps(cur_res2, cur_bias);
				cur_res3 = _mm_add_ps(cur_res3, cur_bias);
				cur_res4 = _mm_add_ps(cur_res4, cur_bias);
				cur_res5 = _mm_add_ps(cur_res5, cur_bias);
				cur_res6 = _mm_add_ps(cur_res6, cur_bias);
				cur_res7 = _mm_add_ps(cur_res7, cur_bias);
				_mm_store_ps(cur_dst, cur_res0);
				_mm_store_ps(cur_dst + 4,  cur_res1);
				_mm_store_ps(cur_dst + 8,  cur_res2);
				_mm_store_ps(cur_dst + 12, cur_res3);
				_mm_store_ps(cur_dst + 16, cur_res4);
				_mm_store_ps(cur_dst + 20, cur_res5);
				_mm_store_ps(cur_dst + 24, cur_res6);
				_mm_store_ps(cur_dst + 28, cur_res7);
				cur_dst += 32;
				cur_src += 32;
			}
		}
		else
		{
			for (j = 0; j < plane_number_8; j += 8)
			{
				__m128   cur_data0 = _mm_loadu_ps(cur_src);
				__m128   cur_data1 = _mm_loadu_ps(cur_src + 4);
				__m128   cur_data2 = _mm_loadu_ps(cur_src + 8);
				__m128   cur_data3 = _mm_loadu_ps(cur_src + 12);
				__m128   cur_data4 = _mm_loadu_ps(cur_src + 16);
				__m128   cur_data5 = _mm_loadu_ps(cur_src + 20);
				__m128   cur_data6 = _mm_loadu_ps(cur_src + 24);
				__m128   cur_data7 = _mm_loadu_ps(cur_src + 28);
				__m128   cur_res0  = _mm_mul_ps(cur_data0, cur_alpha);
				__m128   cur_res1  = _mm_mul_ps(cur_data1, cur_alpha);
				__m128   cur_res2  = _mm_mul_ps(cur_data2, cur_alpha);
				__m128   cur_res3  = _mm_mul_ps(cur_data3, cur_alpha);
				__m128   cur_res4  = _mm_mul_ps(cur_data4, cur_alpha);
				__m128   cur_res5  = _mm_mul_ps(cur_data5, cur_alpha);
				__m128   cur_res6  = _mm_mul_ps(cur_data6, cur_alpha);
				__m128   cur_res7  = _mm_mul_ps(cur_data7, cur_alpha);

				cur_res0 = _mm_add_ps(cur_res0, cur_bias);
				cur_res1 = _mm_add_ps(cur_res1, cur_bias);
				cur_res2 = _mm_add_ps(cur_res2, cur_bias);
				cur_res3 = _mm_add_ps(cur_res3, cur_bias);
				cur_res4 = _mm_add_ps(cur_res4, cur_bias);
				cur_res5 = _mm_add_ps(cur_res5, cur_bias);
				cur_res6 = _mm_add_ps(cur_res6, cur_bias);
				cur_res7 = _mm_add_ps(cur_res7, cur_bias);
				_mm_storeu_ps(cur_dst, cur_res0);
				_mm_storeu_ps(cur_dst + 4, cur_res1);
				_mm_storeu_ps(cur_dst + 8, cur_res2);
				_mm_storeu_ps(cur_dst + 12, cur_res3);
				_mm_storeu_ps(cur_dst + 16, cur_res4);
				_mm_storeu_ps(cur_dst + 20, cur_res5);
				_mm_storeu_ps(cur_dst + 24, cur_res6);
				_mm_storeu_ps(cur_dst + 28, cur_res7);
				cur_dst += 32;
				cur_src += 32;
			}
		}
		for (; j < planeNumber; j++)
		{
			__m128   cur_data = _mm_loadu_ps(cur_src);
			__m128   cur_res  = _mm_mul_ps(cur_data, cur_alpha);
			cur_res           = _mm_add_ps(cur_res, cur_bias);
			_mm_storeu_ps(cur_dst, cur_res);
			cur_dst += 4;
			cur_src += 4;
		}
	}
}

void _SSE_MNNRoundFloat(const float* src, float* dst, size_t size)
{
    size_t  i       = 0;
    size_t  size_16 = size - (size&0xF);
    for (i = 0; i < size_16; i += 16)
    {
        __m128  cur0 = _mm_loadu_ps(src + i);
        __m128  cur1 = _mm_loadu_ps(src + i + 4);
        __m128  cur2 = _mm_loadu_ps(src + i + 8);
        __m128  cur3 = _mm_loadu_ps(src + i + 12);
        __m128  res0 = _mm_round_ps(cur0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128  res1 = _mm_round_ps(cur1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128  res2 = _mm_round_ps(cur2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128  res3 = _mm_round_ps(cur3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_ps(dst + i,      res0);
        _mm_storeu_ps(dst + i + 4,  res1);
        _mm_storeu_ps(dst + i + 8,  res2);
        _mm_storeu_ps(dst + i + 12, res3);
    }
    for (; i < size; i++)
        dst[i] = roundf(src[i]);
}

void _SSE_MNNSignFloat(const float* src, float* dst, size_t size)
{
    size_t        i         = 0;
    size_t        size_16   = size - (size & 0xF);
    const __m128  res_plus  = _mm_set1_ps(1.0f);
    const __m128  res_minus = _mm_set1_ps(-1.0f);
    const __m128  zero      = _mm_setzero_ps();
    __m128  plus_mask       = _mm_set1_ps(0.0f);
    __m128  minus_mask      = _mm_set1_ps(0.0f);
    __m128  res             = _mm_set1_ps(0.0f);
	for (i = 0; i < size_16; i += 16)
	{
        __m128 cur0       = _mm_loadu_ps(src + i);
        __m128 cur1       = _mm_loadu_ps(src + i + 4);
        __m128 cur2       = _mm_loadu_ps(src + i + 8);
        __m128 cur3       = _mm_loadu_ps(src + i + 12);

#define COMPUTE_SIGN(cur)                                                      \
        plus_mask        = _mm_cmpgt_ps(cur, zero);                            \
        minus_mask       = _mm_cmplt_ps(cur, zero);                            \
        res              = _mm_and_ps(res_plus, plus_mask);                    \
        cur              = _mm_or_ps(res, _mm_and_ps(res_minus, minus_mask));  \
		
        COMPUTE_SIGN(cur0);
        COMPUTE_SIGN(cur1);
        COMPUTE_SIGN(cur2);
        COMPUTE_SIGN(cur3);

        _mm_storeu_ps(dst + i,      cur0);
        _mm_storeu_ps(dst + i + 4,  cur1);
        _mm_storeu_ps(dst + i + 8,  cur2);
        _mm_storeu_ps(dst + i + 12, cur3);
	}
    for (; i < size; i++)
    {
        if (src[i] > 0.0f)
            dst[i] = 1.0f;
        else if (src[i] < 0.0f)
            dst[i] = -1.0f;
        else
            dst[i] = 0.0f;
    }
}

void _SSE_MNNScaleAddInt8(int8_t*         dst, 
	                      const int8_t*   src0, 
	                      const int8_t*   src1, 
	                      const float*    scale0, 
	                      const float*    scale1, 
	                      const float*    outputScale, 
	                      const size_t    size)
{
    size_t   i          = 0;
    size_t   size_4     = size - (size & 3);
    __m128   scale0_val = _mm_loadu_ps(scale0);
    __m128   scale1_val = _mm_loadu_ps(scale1);
    __m128i  min_val    = _mm_set1_epi8(-127);
	__m128   plus_half  = _mm_set1_ps(0.5f);
	__m128   minus_half = _mm_set1_ps(-0.5f);
	__m128   zero_data  = _mm_set1_ps(0.0f);
    __m128   out_scale  = _mm_loadu_ps(outputScale);
    int8_t const*  src_data0 = src0;
    int8_t const*  src_data1 = src1;
    int8_t*        dst_data  = dst;

    for (i = 0; i < size_4; i += 4)
    {
        __m128i  val0  = _mm_loadu_si128((const __m128i*)src_data0);
        __m128i  val1  = _mm_loadu_si128((const __m128i*)src_data1);

        __m128i  val00  = _mm_cvtepi8_epi32(val0);
        __m128i  val01  = _mm_cvtepi8_epi32(_mm_srli_si128(val0, 4));
        __m128i  val02  = _mm_cvtepi8_epi32(_mm_srli_si128(val0, 8));
        __m128i  val03  = _mm_cvtepi8_epi32(_mm_srli_si128(val0, 12));
        __m128   data00 = _mm_cvtepi32_ps(val00);
        __m128   data01 = _mm_cvtepi32_ps(val01);
        __m128   data02 = _mm_cvtepi32_ps(val02);
        __m128   data03 = _mm_cvtepi32_ps(val03);


        __m128i  val10  = _mm_cvtepi8_epi32(val1);
        __m128i  val11  = _mm_cvtepi8_epi32(_mm_srli_si128(val1, 4));
        __m128i  val12  = _mm_cvtepi8_epi32(_mm_srli_si128(val1, 8));
        __m128i  val13  = _mm_cvtepi8_epi32(_mm_srli_si128(val1, 12));
        __m128   data10 = _mm_cvtepi32_ps(val10);
        __m128   data11 = _mm_cvtepi32_ps(val11);
        __m128   data12 = _mm_cvtepi32_ps(val12);
        __m128   data13 = _mm_cvtepi32_ps(val13);

        data00          = _mm_mul_ps(data00, scale0_val);
        data01          = _mm_mul_ps(data01, scale0_val);
        data02          = _mm_mul_ps(data02, scale0_val);
        data03          = _mm_mul_ps(data03, scale0_val);
        data10          = _mm_mul_ps(data10, scale1_val);
        data11          = _mm_mul_ps(data11, scale1_val);
        data12          = _mm_mul_ps(data12, scale1_val);
        data13          = _mm_mul_ps(data13, scale1_val);
        data00          = _mm_add_ps(data00, data10);
        data01          = _mm_add_ps(data01, data11);
        data02          = _mm_add_ps(data02, data12);
        data03          = _mm_add_ps(data03, data13);

        data00          = _mm_mul_ps(data00, out_scale);
        data01          = _mm_mul_ps(data01, out_scale);
        data02          = _mm_mul_ps(data02, out_scale);
        data03          = _mm_mul_ps(data03, out_scale);

//Shaquille, Modified 20210203 Start
        __m128 cur_sign0   = _mm_cmplt_ps(data00, zero_data);
        __m128 cur_sign1   = _mm_cmplt_ps(data01, zero_data);
        __m128 cur_sign2   = _mm_cmplt_ps(data02, zero_data);
        __m128 cur_sign3   = _mm_cmplt_ps(data03, zero_data);
        cur_sign0          = _mm_blendv_ps(plus_half, minus_half, cur_sign0);
        cur_sign1          = _mm_blendv_ps(plus_half, minus_half, cur_sign1);
        cur_sign2          = _mm_blendv_ps(plus_half, minus_half, cur_sign2);
        cur_sign3          = _mm_blendv_ps(plus_half, minus_half, cur_sign3);

        data00             = _mm_add_ps(data00, cur_sign0);
        data01             = _mm_add_ps(data01, cur_sign1);
        data02             = _mm_add_ps(data02, cur_sign2);
        data03             = _mm_add_ps(data03, cur_sign3);

        data00             = _mm_round_ps(data00, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        data01             = _mm_round_ps(data01, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        data02             = _mm_round_ps(data02, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        data03             = _mm_round_ps(data03, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
//Shaquille, Modified 20210203 End
        val00           = _mm_cvtps_epi32(data00);
        val01           = _mm_cvtps_epi32(data01);
        val02           = _mm_cvtps_epi32(data02);
        val03           = _mm_cvtps_epi32(data03);

        val10           = _mm_packs_epi32(val00, val01);
        val11           = _mm_packs_epi32(val02, val03);
        val12           = _mm_packs_epi16(val10, val11);
        val12           = _mm_max_epi8(val12, min_val);

        _mm_storeu_si128((__m128i*)dst_data, val12);

        src_data0 += 16;
        src_data1 += 16;
        dst_data  += 16;
    }

    for (; i < size; i++)
    {
        __m128i  v0    = _mm_set_epi32(src0[4 * i + 3], src0[4 * i + 2], src0[4 * i + 1], src0[4 * i]);
        __m128i  v1    = _mm_set_epi32(src1[4 * i + 3], src1[4 * i + 2], src1[4 * i + 1], src1[4 * i]);
        __m128   d0    = _mm_cvtepi32_ps(v0);
        __m128   d1    = _mm_cvtepi32_ps(v1);
        d0             = _mm_mul_ps(d0, scale0_val);
        d1             = _mm_mul_ps(d1, scale1_val);
        d0             = _mm_add_ps(d0, d1);
        d0             = _mm_mul_ps(d0, out_scale);
//Shaquille, Modified 20210203 Start
        __m128 cur_sign= _mm_cmplt_ps(d0, zero_data);
        cur_sign       = _mm_blendv_ps(plus_half, minus_half, cur_sign);
        d0             = _mm_add_ps(d0, cur_sign);
        d0             = _mm_round_ps(d0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
//Shaquille, Modified 20210203 End
        v0             = _mm_cvtps_epi32(d0);
        v0             = _mm_packs_epi32(v0, v0);
        v0             = _mm_packs_epi16(v0, v0);
        v0             = _mm_max_epi8(v0, min_val);
        ((int*)dst)[i] = _mm_cvtsi128_si32(v0);
    }
}
//Shaquille, Added 20201208 End

void _SSE_MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dstO, const int8_t* srcO, const int8_t* weightO, const QuanPostTreatParameters* parameters, size_t width, size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step) {
    auto dst = dstO;
    auto src = (const int16_t*)srcO;
    int widthC4 = width / 4;
    int widthRemain = width % 4;
    auto weight = (const int16_t*)weightO;
    auto biasValue = _mm_castps_si128(_mm_loadu_ps((const float*)parameters->bias));
    //auto biasValue = *(__m128i*)parameters->bias;
    auto scaleValue = _mm_loadu_ps((const float*)parameters->scale);
    __m128i d0, d1, d2, d3;
    int dx, fx, fy;
    __m128i srcValue0;
    auto srcTemp0 = (int64_t*)(&srcValue0);
    __m128i srcValue1;
    auto srcTemp1 = (int64_t*)(&srcValue1);
    __m128i weightValue;
    auto weightTemp = (int64_t*)(&weightValue);
    __m128i zero = _mm_xor_si128(srcValue1, srcValue1);
    __m128 zero128 = _mm_set1_ps(0.0f);
    auto minValue = _mm_set1_epi8(parameters->minValue);
    auto maxValue = _mm_set1_epi8(parameters->maxValue);
    __m128 plus = _mm_set1_ps(0.5f);
    __m128 minus = _mm_set1_ps(-0.5f);
    if (4 == src_w_step) {
        // Stride = 1
        for (dx = 0; dx < widthC4; ++dx) {
            d0 = biasValue;
            d1 = biasValue;
            d2 = biasValue;
            d3 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    auto s0_16 = _mm_castps_si128(_mm_loadu_ps((float*)src_x));
                    auto s1_16 = _mm_castps_si128(_mm_loadu_ps((float*)src_x + 4));
                    auto s0_32 = _mm_unpacklo_epi16(s0_16, zero);
                    auto s1_32 = _mm_unpackhi_epi16(s0_16, zero);
                    auto s2_32 = _mm_unpacklo_epi16(s1_16, zero);
                    auto s3_32 = _mm_unpackhi_epi16(s1_16, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                    d2 = _mm_add_epi32(d2, _mm_madd_epi16(weightValue, s2_32));
                    d3 = _mm_add_epi32(d3, _mm_madd_epi16(weightValue, s3_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(d0));
            dst += 16;
            src += src_w_step * 4;
        }
    } else {
        for (dx = 0; dx < widthC4; ++dx) {
            d0 = biasValue;
            d1 = biasValue;
            d2 = biasValue;
            d3 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + src_w_step * 1);
                    srcTemp1[0] = *(int64_t*)(src_x + src_w_step * 2);
                    srcTemp1[1] = *(int64_t*)(src_x + src_w_step * 3);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm_unpackhi_epi16(srcValue0, zero);
                    auto s2_32 = _mm_unpacklo_epi16(srcValue1, zero);
                    auto s3_32 = _mm_unpackhi_epi16(srcValue1, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                    d2 = _mm_add_epi32(d2, _mm_madd_epi16(weightValue, s2_32));
                    d3 = _mm_add_epi32(d3, _mm_madd_epi16(weightValue, s3_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            __m128 f3 = _mm_cvtepi32_ps(d3);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            f3 = _mm_mul_ps(f3, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            auto m3 = _mm_cmplt_ps(f3, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            m3 = _mm_blendv_ps(plus, minus, m3);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            f3 = _mm_add_ps(f3, m3);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            d3 = _mm_cvtps_epi32(_mm_round_ps(f3, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(dst), _mm_castsi128_ps(d0));
            dst += 16;
            src += src_w_step * 4;
        }
    }
    switch (widthRemain) {
        case 3:
        {
            d0 = biasValue;
            d1 = biasValue;
            d2 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + src_w_step * 1);
                    srcTemp1[0] = *(int64_t*)(src_x + src_w_step * 2);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm_unpackhi_epi16(srcValue0, zero);
                    auto s2_32 = _mm_unpacklo_epi16(srcValue1, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                    d2 = _mm_add_epi32(d2, _mm_madd_epi16(weightValue, s2_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            __m128 f2 = _mm_cvtepi32_ps(d2);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            f2 = _mm_mul_ps(f2, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            auto m2 = _mm_cmplt_ps(f2, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            m2 = _mm_blendv_ps(plus, minus, m2);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            f2 = _mm_add_ps(f2, m2);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            d2 = _mm_cvtps_epi32(_mm_round_ps(f2, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int8_t temp[128];
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(d0));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        case 2:
        {
            d0 = biasValue;
            d1 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    srcTemp0[1] = *(int64_t*)(src_x + src_w_step * 1);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    auto s1_32 = _mm_unpackhi_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                    d1 = _mm_add_epi32(d1, _mm_madd_epi16(weightValue, s1_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            __m128 f1 = _mm_cvtepi32_ps(d1);
            f0 = _mm_mul_ps(f0, scaleValue);
            f1 = _mm_mul_ps(f1, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            auto m1 = _mm_cmplt_ps(f1, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            m1 = _mm_blendv_ps(plus, minus, m1);
            f0 = _mm_add_ps(f0, m0);
            f1 = _mm_add_ps(f1, m1);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            d1 = _mm_cvtps_epi32(_mm_round_ps(f1, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d2 = _mm_packs_epi32(d2, d3);
            d0 = _mm_packs_epi16(d0, d2);
            int8_t temp[128];
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(d0));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        case 1:
        {
            d0 = biasValue;

            auto dst_x          = dst;
            const auto src_z    = src;
            for (fy = 0; fy < fh; ++fy) {
                const auto src_y    = src_z + fy * dilateY_step;
                const auto weight_y = weight + fy * fw * 4;
                for (fx = 0; fx < fw; ++fx) {
                    const auto src_x    = src_y + fx * dilateX_step;
                    srcTemp0[0] = *(int64_t*)(src_x);
                    auto s0_32 = _mm_unpacklo_epi16(srcValue0, zero);
                    const auto weight_x = weight_y + 4 * fx;
                    weightTemp[0] = *(int64_t*)weight_x;
                    weightValue = _mm_unpacklo_epi16(weightValue, zero);
                    d0 = _mm_add_epi32(d0, _mm_madd_epi16(weightValue, s0_32));
                }
            }
            __m128 f0 = _mm_cvtepi32_ps(d0);
            f0 = _mm_mul_ps(f0, scaleValue);
            auto m0 = _mm_cmplt_ps(f0, zero128);
            m0 = _mm_blendv_ps(plus, minus, m0);
            f0 = _mm_add_ps(f0, m0);
            // 3: _MM_FROUND_TO_ZERO
            d0 = _mm_cvtps_epi32(_mm_round_ps(f0, 3));
            
            // Int32 -> Int8
            d0 = _mm_packs_epi32(d0, d1);
            d0 = _mm_packs_epi16(d0, d2);
            int8_t temp[128];
            d0 = _mm_min_epi8(d0, maxValue);
            d0 = _mm_max_epi8(d0, minValue);

            _mm_storeu_ps((float*)(temp), _mm_castsi128_ps(d0));
            ::memcpy(dst, temp, widthRemain * 4 * sizeof(int8_t));
            break;
        }
        default:
            break;
    }
}
void _SSE_MNNComputeMatMulForE_1(const float* A, const float* B, float* C, const float* biasPtr, const MatMulParam* param, size_t tId) {
    auto l = param->l;
    auto h = param->h;
    auto numberThread = param->numberThread;
    auto lC4 = l / 4;
    auto lR = lC4 * 4;
    if (param->BTranspose) {
        for (int y=tId; y<h; y+=numberThread) {
            auto sumValue = _mm_set1_ps(0.0f);
            auto by = B + y * l;
            for (int x=0; x<lC4; ++x) {
                sumValue = _mm_add_ps(sumValue, _mm_mul_ps(_mm_loadu_ps(A + x * 4), _mm_loadu_ps(by + x * 4)));
            }
            float sumRemain = 0.0f;
            for (int x=lR; x<l; ++x) {
                sumRemain = sumRemain + A[x] * by[x];
            }
            if (nullptr != biasPtr) {
                sumRemain += biasPtr[y];
            }
            sumValue = _mm_hadd_ps(sumValue, sumValue);
            sumValue = _mm_hadd_ps(sumValue, sumValue);
            auto s = _mm_cvtss_f32(sumValue);
            C[y] = sumRemain + s;
        }
    } else {
        auto hC4 = h / 4;
        auto hR = hC4 * 4;
        for (int y=tId; y<hC4; y+=numberThread) {
            auto bs = B + 4 * y;
            auto sumValue = _mm_set1_ps(0.0f);
            if (biasPtr != nullptr) {
                sumValue = _mm_loadu_ps(biasPtr + 4 * y);
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = _mm_add_ps(sumValue, _mm_mul_ps(_mm_set1_ps(A[x]), _mm_loadu_ps(bs + h * x)));
            }
            _mm_storeu_ps(C + 4 * y, sumValue);
        }
        for (int y=hR + tId; y<h; y+=numberThread) {
            auto bs = B + y;
            float sumValue = 0.0f;
            if (biasPtr != nullptr) {
                sumValue = biasPtr[y];
            }
            auto srcY = A + y * l;
            for (int x=0; x<l; ++x) {
                sumValue = sumValue + A[x] * bs[h * x];
            }
            C[y] = sumValue;
        }
    }
}

extern "C" {
void MNNInt8ToUInt8(void* ptr, int count) {
    auto src = (int8_t*)ptr;
    auto dst = (uint8_t*)ptr;
    int c16 = count / 16;
    count = count % 16;
    auto zero = _mm_set1_epi8(0);
    auto offset = _mm_set1_epi16(128);
    for (int v = 0; v < c16; ++v) {
        auto i8Value = _mm_loadu_si128((__m128i*)(src));
        auto i16Value0 = _mm_srai_epi16(_mm_unpacklo_epi8(zero, i8Value), 8);
        auto i16Value1 = _mm_srai_epi16(_mm_unpackhi_epi8(zero, i8Value), 8);
        i16Value0 = _mm_add_epi16(i16Value0, offset);
        i16Value1 = _mm_add_epi16(i16Value1, offset);
        i8Value = _mm_packus_epi16(i16Value0, i16Value1);
        _mm_storeu_si128((__m128i*)dst, i8Value);
        dst += 16;
        src += 16;
    }
    for (int v = 0; v < count; ++v) {
        dst[v] = (int)src[v] + 128;
    }
}
}

