#ifndef __QUANTIZE_SPEEDUP_H__
#define __QUANTIZE_SPEEDUP_H__

int   fake_quantize_data(float* src, int count, float scale, float clamp_value);

int   select_min_max(float const* src, int count, float* min, float* max);

#endif