//
//  CPUReduction.cpp
//  MNN
//
//  Created by MNN on 2018/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUReduction.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include <cmath>
#include <algorithm>
#include "core/OpCommonUtils.hpp"
#define UNIT 4
#define UNIT_DUP(value) \
    { (value), (value), (value), (value) }

namespace MNN {
// outside, axis, inside

class Reduction : public Execution {
public:
    Reduction(Backend* backend, const Op* op) : Execution(backend) {
        // Do nothing
        mAxis = op->main_as_ReductionParam()->dim()->data()[0];
    }
    virtual ~Reduction() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input  = inputs[0];
        auto output = outputs[0];
        auto typeCode = input->getType().code;
        auto src = inputs[0];
        int outside = 1;
        for(int i=0; i<mAxis; ++i) {
            outside *= input->length(i);
        }
        int inside = 1;
        for(int i=mAxis+1; i<input->dimensions(); ++i) {
            inside *= input->length(i);
        }
        auto axis = input->length(mAxis);
        auto dst = output;
        //MNN_ASSERT(output->elementSize() == inside * outside);
        auto src_fmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto dst_fmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
        if (src_fmt == MNN_DATA_FORMAT_NC4HW4 && dst_fmt == MNN_DATA_FORMAT_NC4HW4)
        {
            if (halide_type_float == typeCode) {
                this->onnc4hw4Reduce(src->host<float>(), dst->host<float>(), inputs[0]->batch(), inputs[0]->channel(), inputs[0]->height(), inputs[0]->width());
            }
            else if (halide_type_int == typeCode) {
                this->onnc4hw4Reduce(src->host<int32_t>(), dst->host<int32_t>(), inputs[0]->batch(), inputs[0]->channel(), inputs[0]->height(), inputs[0]->width());
            }
        }
        else
        {
            if (halide_type_float == typeCode) {
                this->onReduce(src->host<float>(), dst->host<float>(), inside, outside, axis);
            }
            else if (halide_type_int == typeCode) {
                this->onReduce(src->host<int32_t>(), dst->host<int32_t>(), inside, outside, axis);
            }
        }

        return NO_ERROR;
    }
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axis) const     = 0;
    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outsize, int axis) const = 0;

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const      = 0;
    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const  = 0;
private:
    int mAxis = -1;
};

class MeanReduce : public Reduction {
public:
    MeanReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MeanReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi+=numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                if (inside % 4 == 0) {
                    ::memcpy(dstOutSide, srcOutSide, inside * sizeof(float));
                    for (int a = 1; a < axisSize; ++a) {
                        auto srcAxis = srcOutSide + a * inside;
                        MNNMatrixAddCommon(dstOutSide, dstOutSide, srcAxis, inside, 0, 0, 0, 1);
                    }
                    float divide = 1.0f / (float)axisSize;
                    for (int i=0; i<inside; ++i) {
                        dstOutSide[i] = dstOutSide[i] * divide;
                    }
                } else {
                    for (int ii = 0; ii < inside; ++ii) {
                        auto srcInside = srcOutSide + ii;
                        auto dstInside = dstOutSide + ii;
                        float summer   = 0.0f;
                        for (int a = 0; a < axisSize; ++a) {
                            summer += srcInside[a * inside];
                        }
                        *dstInside = summer / (float)axisSize;
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t summer = 0;
                for (int a = 0; a < axisSize; ++a) {
                    summer += srcInside[a * inside];
                }
                *dstInside = summer / axisSize;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override 
    { 
        int          cq      = c >> 2;
        int          c_tail  = c - (cq << 2);
        float const* src_ptr = src;
        float*       dst_ptr = dst;
        float        scale   = 1.0f / (float)(c);
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int   dst_pos = i * w * 4 + 4 * j;
                float res = 0.0f;
                int   k = 0;
                int   cus_pos = 0;
                for (k = 0; k < cq; k++)
                {
                    cus_pos = k * w * h * 4 + i * 4 * w + 4 * j;
                    float a = src_ptr[cus_pos];
                    float b = src_ptr[cus_pos + 1];
                    float c = src_ptr[cus_pos + 2];
                    float d = src_ptr[cus_pos + 3];
                    float s = a + b + c + d;
                    res = res + s;
                }
                cus_pos = k * w * h * 4 + i * 4 * w + 4 * j;
                for (int m = 0; m < c_tail; m++)
                    res = res + src_ptr[cus_pos + m];

                dst_ptr[dst_pos]     = res * scale;
                dst_ptr[dst_pos + 1] = 0.0f;
                dst_ptr[dst_pos + 2] = 0.0f;
                dst_ptr[dst_pos + 3] = 0.0f;
            }
        }
    };

    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override 
    { 
        int            cq      = c >> 2;
        int            c_tail  = c - (cq << 2);
        int32_t const* src_ptr = src;
        int32_t*       dst_ptr = dst;
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int     dst_pos = i * w * 4 + 4 * j;
                int32_t res     = 0;
                int     k       = 0;
                int     cus_pos = 0;
                for (k = 0; k < cq; k++)
                {
                    cus_pos   = k * w * h * 4 + i * 4 * w + 4 * j;
                    int32_t a = src_ptr[cus_pos];
                    int32_t b = src_ptr[cus_pos + 1];
                    int32_t c = src_ptr[cus_pos + 2];
                    int32_t d = src_ptr[cus_pos + 3];
                    int32_t s = a + b + c + d;
                    res = res + s;
                }
                cus_pos = k * w * h * 4 + i * 4 * w + 4 * j;
                for (int m = 0; m < c_tail; m++)
                    res = res + src_ptr[cus_pos + m];
                dst_ptr[dst_pos]     = res / c;
                dst_ptr[dst_pos + 1] = 0;
                dst_ptr[dst_pos + 2] = 0;
                dst_ptr[dst_pos + 3] = 0;
            }
        }
    };
};

class SumReduce : public Reduction {
public:
    SumReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~SumReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi+=numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                if (inside % 4 == 0) {
                    ::memcpy(dstOutSide, srcOutSide, inside * sizeof(float));
                    for (int a = 1; a < axisSize; ++a) {
                        auto srcAxis = srcOutSide + a * inside;
                        MNNMatrixAddCommon(dstOutSide, dstOutSide, srcAxis, inside, 0, 0, 0, 1);
                    }
                } else {
                    for (int ii = 0; ii < inside; ++ii) {
                        auto srcInside = srcOutSide + ii;
                        auto dstInside = dstOutSide + ii;
                        float summer   = 0.0f;
                        for (int a = 0; a < axisSize; ++a) {
                            summer += srcInside[a * inside];
                        }
                        *dstInside = summer;
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t summer = 0;
                for (int a = 0; a < axisSize; ++a) {
                    summer += srcInside[a * inside];
                }
                *dstInside = summer;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override 
    { 
        int          cq      = c >> 2;
        int          c_tail  = c - (cq << 2);
        float const* src_ptr = src;
        float*       dst_ptr = dst;
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int   dst_pos = i * w * 4 + 4 * j;
                float res     = 0.0f;
                int   k       = 0;
                int   cus_pos = 0;
                for (k = 0; k < cq; k++)
                {
                    cus_pos = k * w * h * 4 + i * 4 * w + 4 * j;
                    float a = src_ptr[cus_pos];
                    float b = src_ptr[cus_pos + 1];
                    float c = src_ptr[cus_pos + 2];
                    float d = src_ptr[cus_pos + 3];
                    float s = a + b + c + d;
                    res = res + s;
                }
                cus_pos = k * w * h * 4 + i * 4 * w + 4 * j;
                for (int m = 0; m < c_tail; m++)
                    res = res + src_ptr[cus_pos + m];
                dst_ptr[dst_pos]     = res;
                dst_ptr[dst_pos + 1] = 0.0f;
                dst_ptr[dst_pos + 2] = 0.0f;
                dst_ptr[dst_pos + 3] = 0.0f;
            }
        }
    };

    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override 
    { 
        int            cq      = c >> 2;
        int            c_tail  = c - (cq << 2);
        int32_t const* src_ptr = src;
        int32_t*       dst_ptr = dst;
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int     dst_pos = i * w * 4 + 4 * j;
                int32_t res     = 0;
                int     k       = 0;
                int     cus_pos = 0;
                for (k = 0; k < cq; k++)
                {
                    cus_pos = k * w * h * 4 + i * 4 * w + 4 * j;
                    int32_t a = src_ptr[cus_pos];
                    int32_t b = src_ptr[cus_pos + 1];
                    int32_t c = src_ptr[cus_pos + 2];
                    int32_t d = src_ptr[cus_pos + 3];
                    int32_t s = a + b + c + d;
                    res       = res + s;
                }
                cus_pos = k * w * h * 4 + i * 4 * w + 4 * j;
                for (int m = 0; m < c_tail; m++)
                    res = res + src_ptr[cus_pos + m];
                dst_ptr[dst_pos]     = res;
                dst_ptr[dst_pos + 1] = 0;
                dst_ptr[dst_pos + 2] = 0;
                dst_ptr[dst_pos + 3] = 0;
            }
        }
    };
};

class MinReduce : public Reduction {
public:
    MinReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MinReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float Min      = srcInside[0];
                if (1 == inside) {
                    int32_t inputCountUnit = axisSize / (UNIT * 2);
                    int32_t remain         = axisSize - (inputCountUnit * UNIT * 2);
                    float minArray[UNIT]   = UNIT_DUP(Min);
                    MNNMinFloat((float*)srcInside, minArray, inputCountUnit);

                    for (int i = 0; i < UNIT; i++) {
                        Min = std::min(Min, minArray[i]);
                    }
                    if (remain > 0) {
                        int currentIndex = inputCountUnit * UNIT * 2;
                        for (int i = 0; i < remain; i++) {
                            float currentInputData = srcInside[currentIndex + i];
                            Min                    = std::min(Min, currentInputData);
                        }
                    }
                } else {
                    for (int a = 0; a < axisSize; ++a) {
                        Min = std::min(Min, srcInside[a * inside]);
                    }
                }
                *dstInside = Min;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t Min    = srcInside[0];
                for (int a = 0; a < axisSize; ++a) {
                    Min = std::min(Min, srcInside[a * inside]);
                }
                *dstInside = Min;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override { ; };
    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override { ; };
};

class MaxReduce : public Reduction {
public:
    MaxReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MaxReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float Max      = srcInside[0];
                if (1 == inside) {
                    int32_t inputCountUnit = axisSize / (UNIT * 2);
                    int32_t remain         = axisSize - (inputCountUnit * UNIT * 2);
                    float maxArray[UNIT]   = UNIT_DUP(Max);

                    MNNMaxFloat((float*)srcInside, maxArray, inputCountUnit);

                    for (int i = 0; i < UNIT; i++) {
                        Max = std::max(Max, maxArray[i]);
                    }
                    if (remain > 0) {
                        int currentIndex = inputCountUnit * UNIT * 2;
                        for (int i = 0; i < remain; i++) {
                            float currentInputData = srcInside[currentIndex + i];
                            Max                    = std::max(Max, currentInputData);
                        }
                    }
                } else {
                    for (int a = 0; a < axisSize; ++a) {
                        Max = std::max(Max, srcInside[a * inside]);
                    }
                }
                *dstInside = Max;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t Max    = srcInside[0];
                for (int a = 0; a < axisSize; ++a) {
                    Max = std::max(Max, srcInside[a * inside]);
                }
                *dstInside = Max;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override { ; };
    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override { ; };
};

class ProdReduce : public Reduction {
public:
    ProdReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ProdReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float product  = 1.0f;
                for (int a = 0; a < axisSize; ++a) {
                    product *= srcInside[a * inside];
                }
                *dstInside = product;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t product = 1;
                for (int a = 0; a < axisSize; ++a) {
                    product *= srcInside[a * inside];
                }
                *dstInside = product;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override { ; };
    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override { ; };
};

class AnyReduce : public Reduction {
public:
    AnyReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ AnyReduce() = default;
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        MNN_ASSERT(false);
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t result = 0;
                for (int a = 0; a < axisSize; ++a) {
                    if (srcInside[a * inside] > 0) {
                        result = 1;
                        break;
                    }
                }
                *dstInside = result;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override { ; };
    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override { ; };
};

class AllReduce : public Reduction {
public:
    AllReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ AllReduce() = default;
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        MNN_ASSERT(false);
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t result = 1;
                for (int a = 0; a < axisSize; ++a) {
                    if (srcInside[a * inside] == 0) {
                        result = 0;
                        break;
                    }
                }
                *dstInside = result;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override { ; };
    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override { ; };
};

class L1Reduce : public Reduction {
public:
    L1Reduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~L1Reduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi += numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                for (int ii = 0; ii < inside; ++ii) {
                    auto srcInside = srcOutSide + ii;
                    auto dstInside = dstOutSide + ii;
                    float summer = 0.0f;
                    for (int a = 0; a < axisSize; ++a) {
                        summer += fabs(srcInside[a * inside]);
                    }
                    *dstInside = summer;
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t summer = 0;
                for (int a = 0; a < axisSize; ++a) {
                    summer += abs(srcInside[a * inside]);
                }
                *dstInside = summer;
            }
        }
    }

    virtual void onnc4hw4Reduce(const float* src, float* dst, int n, int c, int h, int w) const override
    {
        int          cq            = c >> 2;
        int          c_tail        = c - (cq << 2);
        float const* src_ptr       = src;
        float*       dst_ptr       = dst;
        int          src_line_size = 4 * cq * w;
        int          dst_line_size = 4 * w;
        for (int m = 0; m < n; m++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    int   dst_pos = m * h * dst_line_size + i * w * 4 + 4 * j;
                    float res     = 0.0f;
                    int   k       = 0;
                    int   cus_pos = 0;
                    for (k = 0; k < cq; k++)
                    {
                        cus_pos = m * cq * h * src_line_size + k * w * h * 4 + i * 4 * w + 4 * j;
                        float a = fabs(src_ptr[cus_pos]);
                        float b = fabs(src_ptr[cus_pos + 1]);
                        float c = fabs(src_ptr[cus_pos + 2]);
                        float d = fabs(src_ptr[cus_pos + 3]);
                        float s = a + b + c + d;
                        res = res + s;
                    }
                    cus_pos = m * cq * h * src_line_size + k * w * h * 4 + i * 4 * w + 4 * j;
                    for (int m = 0; m < c_tail; m++)
                        res = res + fabs(src_ptr[cus_pos + m]);
                    dst_ptr[dst_pos]     = res;
                    dst_ptr[dst_pos + 1] = 0.0f;
                    dst_ptr[dst_pos + 2] = 0.0f;
                    dst_ptr[dst_pos + 3] = 0.0f;
                }
            }
        }
    };

    virtual void onnc4hw4Reduce(const int32_t* src, int32_t* dst, int n, int c, int h, int w) const override
    {
        int            cq            = c >> 2;
        int            c_tail        = c - (cq << 2);
        int32_t const* src_ptr       = src;
        int32_t*       dst_ptr       = dst;
        int            src_line_size = 4 * cq * w;
        int            dst_line_size = 4 * w;
        for (int m = 0; m < n; m++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    int     dst_pos = m * h * dst_line_size + i * w * 4 + 4 * j;
                    int32_t res     = 0;
                    int     k       = 0;
                    int     cus_pos = 0;
                    for (k = 0; k < cq; k++)
                    {
                        cus_pos = m * cq * h * src_line_size + k * w * h * 4 + i * 4 * w + 4 * j;
                        int32_t a = abs(src_ptr[cus_pos]);
                        int32_t b = abs(src_ptr[cus_pos + 1]);
                        int32_t c = abs(src_ptr[cus_pos + 2]);
                        int32_t d = abs(src_ptr[cus_pos + 3]);
                        int32_t s = a + b + c + d;
                        res = res + s;
                    }
                    cus_pos = m * cq * h * src_line_size + k * w * h * 4 + i * 4 * w + 4 * j;
                    for (int m = 0; m < c_tail; m++)
                        res = res + abs(src_ptr[cus_pos + m]);
                    dst_ptr[dst_pos]     = res;
                    dst_ptr[dst_pos + 1] = 0;
                    dst_ptr[dst_pos + 2] = 0;
                    dst_ptr[dst_pos + 3] = 0;
                }
            }
        }
    };
};

Execution* CPUReductionCreator::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const MNN::Op* op, Backend* backend) const {
    auto type = inputs[0]->getType();
    if (type.bits != 32) {
        return nullptr;
    }
    if (type.code != halide_type_float && type.code != halide_type_int) {
        return nullptr;
    }
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            return new MeanReduce(backend, op);
        case ReductionType_SUM:
            return new SumReduce(backend, op);
        case ReductionType_MINIMUM:
            return new MinReduce(backend, op);
        case ReductionType_MAXIMUM:
            return new MaxReduce(backend, op);
        case ReductionType_PROD:
            return new ProdReduce(backend, op);
        case ReductionType_ANY:
            return new AnyReduce(backend, op);
        case ReductionType_ALL:
            return new AllReduce(backend, op);
        case ReductionType_L1:
            return new L1Reduce(backend, op);
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

REGISTER_CPU_OP_CREATOR(CPUReductionCreator, OpType_Reduction);

} // namespace MNN
