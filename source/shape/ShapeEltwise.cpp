//
//  ShapeEltwise.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
// Size Computer
class EltWiseComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        TensorUtils::copyShape(inputs[0], outputs[0], true);
        outputs[0]->buffer().type = inputs[0]->getType();
//Shaquille, Modified 20210226 Start
        if (inputs.size() > 1)
        {
            auto in_tensor_fmt0 = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            auto in_tensor_fmt1 = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
            auto out_tensor_fmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat; 
            if (in_tensor_fmt0 != in_tensor_fmt1)
            {
                if (MNN_DATA_FORMAT_NC4HW4 == in_tensor_fmt0 || MNN_DATA_FORMAT_NC4HW4 == in_tensor_fmt1)
                    TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            }
        }
//Shaquille, Modified 20210226 End
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto size = (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
        return size * (inputs.size() - 1);
    }
};

REGISTER_SHAPE(EltWiseComputer, OpType_Eltwise);
REGISTER_SHAPE(EltWiseComputer, OpType_SpatialProduct);
} // namespace MNN
