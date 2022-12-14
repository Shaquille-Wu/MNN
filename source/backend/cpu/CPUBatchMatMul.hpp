//
//  CPUBatchMatMul.hpp
//  MNN
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBatchMatMul_hpp
#define CPUBatchMatMul_hpp

#include "backend/cpu/CPUMatMul.hpp"

namespace MNN {

class CPUBatchMatMul : public Execution {
public:
    CPUBatchMatMul(Backend *backend, bool adjX, bool adjY);
    virtual ~CPUBatchMatMul() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mBatch;
    struct Unit {
        std::shared_ptr<Execution> mMatMul;
        std::vector<Tensor*> mTempInputs;
        std::vector<Tensor*> mTempOutputs;
        std::unique_ptr<Tensor> mMatrixA;
        std::unique_ptr<Tensor> mMatrixB;
        std::unique_ptr<Tensor> mMatrixC;
    };
    std::vector<Unit> mUnits;
};

} // namespace MNN

#endif /* CPUBatchMatMul_hpp */
