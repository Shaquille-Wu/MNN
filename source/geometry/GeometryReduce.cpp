//
//  GeometryReduce.cpp
//  MNN
//
//  Created by MNN on 2020/06/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryReduce : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(inputs.size() >= 1);
        auto reduceDims      = OpCommonUtils::computeReduceDims(inputs, op);
        auto reduct          = op->main_as_ReductionParam();
//Shaquille, Added 20210212 Start
        int   max_dim        = op->main_as_ReductionParam()->dim()->data()[0];
        bool  keep_dims      = op->main_as_ReductionParam()->keepDims();
        if (true == keep_dims && 1 == (int)(inputs.size()) && 4 == inputs[0]->dimensions())
        {
            auto tensor_fmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            if (MNN_DATA_FORMAT_NC4HW4 == tensor_fmt ||
                MNN_DATA_FORMAT_NCHW == tensor_fmt)
            {
                if(1 == max_dim)
                {
                    Command cmd;
                    cmd.op      = op;
                    cmd.inputs  = std::move(inputs);
                    cmd.outputs = std::move(outputs);
                    res.command.emplace_back(std::move(cmd));
                    return true;
                }
			}
            else
            {
                if(3 == max_dim)
                {
                    Command cmd;
                    cmd.op      = op;
                    cmd.inputs  = std::move(inputs);
                    cmd.outputs = std::move(outputs);
                    res.command.emplace_back(std::move(cmd));
                    return true;
                }
            }
        }
//Shaquille, Added 20210212 End
        auto reductOp        = reduct->operation();
        Tensor* currentInput = inputs[0];
        MNN_ASSERT(reduceDims.size() > 0);
        for (int i = 0; i < reduceDims.size(); ++i) {
            auto& iter   = reduceDims[i];
            auto inside  = std::get<2>(iter);
            auto outside = std::get<0>(iter);
            auto axis    = std::get<1>(iter);
            
            std::shared_ptr<Tensor> inputTensor(
                Tensor::createDevice({outside, axis, inside}, inputs[0]->getType()));
            auto des        = TensorUtils::getDescribe(inputTensor.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions    = {TensorUtils::makeFullSlice(currentInput)};
            res.extras.emplace_back(inputTensor);
            std::shared_ptr<Tensor> outputTensor(
                Tensor::createDevice({outside, 1, inside}, inputs[0]->getType()));
            res.extras.emplace_back(outputTensor);

            // Create Command
            {
                auto cmd = GeometryComputerUtils::makeReduce(reductOp, inputTensor.get(), outputTensor.get());
                res.command.emplace_back(std::move(cmd));
            }
            currentInput = outputTensor.get();
            // Ref output
            if (i == reduceDims.size() - 1) {
                auto outputDes        = TensorUtils::getDescribe(outputs[0]);
                outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                outputDes->regions    = {TensorUtils::makeFullSlice(outputTensor.get())};
            }
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryReduce);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Reduction});
}

REGISTER_GEOMETRY(GeometryReduce, _create);

} // namespace MNN
