#include "tensor_convert.h"
#if (__arm__ || __aarch64__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include <arm_neon.h>
#define   orion_loadu_ps    vld1q_f32
#define   orion_storeu_ps   vst1q_f32
#define   orion_float32x4   float32x4_t
#else
#include <emmintrin.h>
#define   orion_loadu_ps    _mm_loadu_ps
#define   orion_storeu_ps   _mm_storeu_ps
#define   orion_float32x4   __m128
#endif

static void nc4hw4_to_nhwc_float_channel4_width4(float const* nc4hw4, int h, int w, int c, float* nhwc)
{
    int  nc4hw4_channel_4  = c >> 2;
    int  dst_pitch         = w * c;
    int  i = 0, j = 0, k = 0;
    for (k = 0; k < nc4hw4_channel_4; k++)
    {
        int src_offset = ((k * w * h) << 2);
        for (i = 0; i < h; i++)
        {
            for (j = 0; j < w; j += 4)
            {
                int              src_pos    = src_offset + ((i * w + j) << 2);
                int              dst_pos    = i * dst_pitch + j * c + (k << 2);
                orion_float32x4  cur_data_0 = orion_loadu_ps(nc4hw4 + src_pos);
                orion_float32x4  cur_data_1 = orion_loadu_ps(nc4hw4 + src_pos + 4);
                orion_float32x4  cur_data_2 = orion_loadu_ps(nc4hw4 + src_pos + 8);
                orion_float32x4  cur_data_3 = orion_loadu_ps(nc4hw4 + src_pos + 12);

                orion_storeu_ps(nhwc + dst_pos,         cur_data_0);
                orion_storeu_ps(nhwc + dst_pos + c,     cur_data_1);
                orion_storeu_ps(nhwc + dst_pos + 2 * c, cur_data_2);
                orion_storeu_ps(nhwc + dst_pos + 3 * c, cur_data_3);
            }
        }
    }
}

static void nc4hw4_to_nhwc_float_width4(float const* nc4hw4, int h, int w, int c, float* nhwc)
{
    int  nc4hw4_channel_4    = c >> 2;
    int  nc4hw4_channel_tail = c - (nc4hw4_channel_4 << 2);
    int  dst_pitch = w * c;
    int  i = 0, j = 0, k = 0;
    for (k = 0; k < nc4hw4_channel_4; k++)
    {
        int src_offset = ((k * w * h) << 2);
        for (i = 0; i < h; i++)
        {
            for (j = 0; j < w; j += 4)
            {
                int              src_pos    = src_offset + ((i * w + j) << 2);
                int              dst_pos    = i * dst_pitch + j * c + (k << 2);
                orion_float32x4  cur_data_0 = orion_loadu_ps(nc4hw4 + src_pos);
                orion_float32x4  cur_data_1 = orion_loadu_ps(nc4hw4 + src_pos + 4);
                orion_float32x4  cur_data_2 = orion_loadu_ps(nc4hw4 + src_pos + 8);
                orion_float32x4  cur_data_3 = orion_loadu_ps(nc4hw4 + src_pos + 12);

                orion_storeu_ps(nhwc + dst_pos,         cur_data_0);
                orion_storeu_ps(nhwc + dst_pos + c,     cur_data_1);
                orion_storeu_ps(nhwc + dst_pos + 2 * c, cur_data_2);
                orion_storeu_ps(nhwc + dst_pos + 3 * c, cur_data_3);
            }
        }
    }

    int src_offset = nc4hw4_channel_4 * 4 * w * h;
    for (i = 0; i < h; i++)
    {
        for (j = 0; j < w; j += 4)
        {
            for (k = 0; k < nc4hw4_channel_tail; k++)
            {
                int  src_pos = src_offset + i * w * 4 + 4 * j + k;
                int  dst_pos = i * dst_pitch + j * c + 4 * nc4hw4_channel_4 + k;
                nhwc[dst_pos]         = nc4hw4[src_pos];
                nhwc[dst_pos + c]     = nc4hw4[src_pos + 4];
                nhwc[dst_pos + 2 * c] = nc4hw4[src_pos + 8];
                nhwc[dst_pos + 3 * c] = nc4hw4[src_pos + 12];
            }
        }
    }
}

static void nc4hw4_to_nhwc_float_unalinged(float const* nc4hw4, int h, int w, int c, float* nhwc)
{
	int  nc4hw4_channel_4    = c >> 2;
	int  nc4hw4_channel_tail = c - (nc4hw4_channel_4 << 2);
	int  dst_pitch           = w * c;
	int  i = 0, j = 0, k = 0;
	for (k = 0; k < nc4hw4_channel_4; k++)
	{
		int src_offset = ((k * w * h) << 2);
		for (i = 0; i < h; i++)
		{
			for (j = 0; j < w; j ++)
			{
				int              src_pos    = src_offset + ((i * w + j) << 2);
				int              dst_pos    = i * dst_pitch + j * c + (k << 2);
				orion_float32x4  cur_data_0 = orion_loadu_ps(nc4hw4 + src_pos);
				orion_storeu_ps(nhwc + dst_pos, cur_data_0);
			}
		}
	}

	int src_offset = nc4hw4_channel_4 * 4 * w * h;
	for (i = 0; i < h; i++)
	{
		for (j = 0; j < w; j ++)
		{
			for (k = 0; k < nc4hw4_channel_tail; k++)
			{
				int  src_pos  = src_offset + i * w * 4 + 4 * j + k;
				int  dst_pos  = i * dst_pitch + j * c + 4 * nc4hw4_channel_4 + k;
				nhwc[dst_pos] = nc4hw4[src_pos];
			}
		}
	}
}

void  orion_nc4hw4_to_nhwc_float(float const* nc4hw4, int h, int w, int c, float* nhwc)
{
    int  is_width_aligned   = (0 == (w & 3));
    int  is_channel_aligned = (0 == (c & 3));
    if ((1 == is_width_aligned) && (1 == is_channel_aligned))
        nc4hw4_to_nhwc_float_channel4_width4(nc4hw4, h, w, c, nhwc);
    else if ((1 == is_width_aligned) && (0 == is_channel_aligned))
        nc4hw4_to_nhwc_float_width4(nc4hw4, h, w, c, nhwc);
    else
        nc4hw4_to_nhwc_float_unalinged(nc4hw4, h, w, c, nhwc);
}