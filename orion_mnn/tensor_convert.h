#ifndef __TENSOR_CONVERT_H__
#define __TENSOR_CONVERT_H__

/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

 /**
  * @file tensor_convert.h
  * @brief This header file defines NC4HW4 to NHWC
  * @author Wu Xiao(wuxiao@ainirobot.com)
  * @date 2021-01-04
  */

#ifndef __cplusplus
extern "C" {
#endif

void  orion_nc4hw4_to_nhwc_float(float const* src, int h, int w, int c, float* dst);

#ifndef __cplusplus
}
#endif
#endif