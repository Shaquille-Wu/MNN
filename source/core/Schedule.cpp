//
//  Schedule.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Schedule.hpp"
#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include "core/DirectedAcyclicGraph.hpp"
#include "core/Macro.h"
#include "core/RuntimeFactory.hpp"
#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
#include "utils/InitNet.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
//#define MNN_AUTO_CHECK_COST
namespace MNN {

class OpNodeDef : public NodeDef<Op*> {
public:
    OpNodeDef(Op* op) {
        this->op = op;
    }

public:
    virtual shared_ptr<Node<Op*>> makeNode() override {
        shared_ptr<Node<Op*>> ptr = make_shared<Node<Op*>>();
        ptr->setData(this->op);
        return ptr;
    }

private:
    Op* op;
};

MNNForwardType Schedule::getApprociateType(const ScheduleConfig& config) {
    MNNForwardType type = config.type;
    // FIXME: Support Auto determine
    if (MNN_FORWARD_AUTO == config.type) {
        // Search Backend Exclude MNN_FORWARD_CPU
        for (int i = 1; i < MNN_FORWARD_ALL; ++i) {
            if (MNNGetExtraRuntimeCreator((MNNForwardType)i) != nullptr) {
                type = (MNNForwardType)i;
                break;
            }
        }
    }
    auto creator = MNNGetExtraRuntimeCreator(type);
    if (nullptr == creator) {
        MNN_PRINT("Can't Find type=%d backend, use %d instead\n", type, config.backupType);
        type = config.backupType;
    }
    return type;
}

static bool _setUpTensorInfo(std::vector<std::shared_ptr<Tensor>>& allTensors, const Net* net) {
    bool valid    = true;
    auto& tensors = allTensors;
    tensors.resize(net->tensorName()->size());
    if (net->usage() == Usage_INFERENCE_STATIC) {
        // static model will set all tensors' shape
        auto describes = net->extraTensorDescribe();
        std::vector<const TensorDescribe*> des(tensors.size());
        for (int i = 0; i < describes->size(); i++) {
            int index  = describes->GetAs<TensorDescribe>(i)->index();
            des[index] = describes->GetAs<TensorDescribe>(i);
        }
        for (int i = 0; i < tensors.size(); ++i) {
            auto blob = des[i]->blob();
            if (auto idims = blob->dims()) {
                tensors[i].reset(new Tensor(idims->size()));
                auto& tb = tensors[i]->buffer();
                for (int d = 0; d < idims->size(); d++) {
                    tb.dim[d].extent = idims->Get(d);
                }
            } else {
                tensors[i].reset(new Tensor(1));
            }
            tensors[i]->setType(blob->dataType());
        }
        for (int i = 0; i < tensors.size(); ++i) {
            auto blob                                                   = des[i]->blob();
            TensorUtils::getDescribe(tensors[i].get())->dimensionFormat = blob->dataFormat();
            if (auto regions = des[i]->regions()) {
                auto& regs = TensorUtils::getDescribe(tensors[i].get())->regions;
                TensorUtils::getDescribe(tensors[i].get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                regs.reserve(regions->size());
                for (int r = 0; r < regions->size(); r++) {
                    auto region = regions->GetAs<Region>(r);
                    Tensor::InsideDescribe::Region reg;
                    reg.origin     = tensors[region->origin()].get();
                    reg.src.offset = region->src()->offset();
                    reg.dst.offset = region->dst()->offset();
                    for (int d = 0; d < 3; d++) {
                        reg.size[d]       = region->size()->data()[d];
                        reg.src.stride[d] = region->src()->stride()->data()[d];
                        reg.dst.stride[d] = region->dst()->stride()->data()[d];
                    }
                    regs.emplace_back(std::move(reg));
                }
            }
        }
        for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
            auto op = net->oplists()->GetAs<Op>(opIndex);
            if (OpType_Const == op->type()) {
                MNN_ASSERT(nullptr != op->outputIndexes());
                auto index                                            = op->outputIndexes()->data()[0];
                TensorUtils::getDescribe(tensors[index].get())->usage = Tensor::InsideDescribe::CONSTANT;
            }
        }
    } else {
        // Dynamic Model just set input tensor's shape
        valid = initTensors(tensors, net);
    }
    return valid;
}


//Shaquille, Added 20201102 Start
#if 1
static bool _setUpOCLFilteredTensorInfo(std::vector<std::shared_ptr<Tensor>>&  allTensors, 
	                                    const Net*                             net, 
	                                    std::vector<int>&                      filter2raw_tensor_idx,
	                                    std::vector<int>&                      raw2filter_tensor_idx,
									    std::map<int, int>&                    raw_cvt_output_2_input_idx) 
{
	bool valid    = true;
	auto& tensors = allTensors;
	filter2raw_tensor_idx.resize(0);
	raw2filter_tensor_idx.resize(0);
	raw_cvt_output_2_input_idx.clear();
	bool normal_proc = true;

	int                    valid_tensor_cnt = 0;
	int                    all_tensor_cnt   = (int)(net->tensorName()->size());
	std::map<int, int>     convert_op_2_input_tensor_idx;
	std::map<int, int>     convert_op_2_output_tensor_idx;
	int                    opIndex          = 0;
	for (opIndex = 0; opIndex < net->oplists()->size(); ++opIndex)
	{
		auto op = net->oplists()->GetAs<Op>(opIndex);
		if (OpType_ConvertTensor == op->type())
		{
			MNN_CHECK((nullptr != op->inputIndexes()), "cannot process zero input for convert_tensor_op.");
			MNN_CHECK((1 == op->inputIndexes()->size()), "convert_tensor_op's input should be 1.");
			if (nullptr == op->inputIndexes())
				break;
			if (1 != op->inputIndexes()->size())
				break;
			if (1 != op->outputIndexes()->size())
				break;

			auto data_i = op->inputIndexes()->data();
			convert_op_2_input_tensor_idx[opIndex] = data_i[0];

			std::vector<int>  output_idx;
			auto data_o = op->outputIndexes()->data();
			convert_op_2_output_tensor_idx[opIndex] = data_o[0];

			raw_cvt_output_2_input_idx[data_o[0]] = data_i[0];
		}
	}
	if (opIndex >= net->oplists()->size())
		normal_proc = false;

	if (false == normal_proc)
	{
		for (int i = 0; i < all_tensor_cnt; i++)
		{
			if (convert_op_2_output_tensor_idx.end() == convert_op_2_output_tensor_idx.find(i))
				valid_tensor_cnt++;
		}
		tensors.resize(valid_tensor_cnt);
		filter2raw_tensor_idx.resize(valid_tensor_cnt);
		raw2filter_tensor_idx.resize(all_tensor_cnt, -1);
		int j = 0;
		for (int i = 0; i < all_tensor_cnt; i++)
		{
			if (raw_cvt_output_2_input_idx.end() == raw_cvt_output_2_input_idx.find(i))
			{
				filter2raw_tensor_idx[j] = i;
				raw2filter_tensor_idx[i] = j;
				j++;
			}
			else
				raw2filter_tensor_idx[i] = -1;
		}

		if (net->usage() == Usage_INFERENCE_STATIC) 
		{
			// static model will set all tensors' shape
			auto describes = net->extraTensorDescribe();
			std::vector<const TensorDescribe*> des(tensors.size());
			for (int m = 0; m < describes->size(); m++) 
			{
				int index = describes->GetAs<TensorDescribe>(m)->index();
				des[index] = describes->GetAs<TensorDescribe>(m);
			}
			for (int m = 0; m < tensors.size(); ++m) 
			{
				auto blob = des[m]->blob();
				if (auto idims = blob->dims()) 
				{
					tensors[m].reset(new Tensor(idims->size(), Tensor::CAFFE_C4));
					auto& tb = tensors[m]->buffer();
					for (int d = 0; d < idims->size(); d++) 
					{
						tb.dim[d].extent = idims->Get(d);
					}
				}
				else 
				{
					tensors[m].reset(new Tensor(1, Tensor::CAFFE_C4));
				}
				tensors[m]->setType(DataType_DT_FLOAT);
			}
			for (int m = 0; m < tensors.size(); ++m) 
			{
				auto blob = des[m]->blob();
				TensorUtils::getDescribe(tensors[m].get())->dimensionFormat = blob->dataFormat();
				if (auto regions = des[m]->regions()) 
				{
					auto& regs = TensorUtils::getDescribe(tensors[m].get())->regions;
					TensorUtils::getDescribe(tensors[m].get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
					regs.reserve(regions->size());
					for (int r = 0; r < regions->size(); r++) 
					{
						auto region = regions->GetAs<Region>(r);
						Tensor::InsideDescribe::Region reg;
						reg.origin = tensors[region->origin()].get();
						reg.src.offset = region->src()->offset();
						reg.dst.offset = region->dst()->offset();
						for (int d = 0; d < 3; d++) 
						{
							reg.size[d] = region->size()->data()[d];
							reg.src.stride[d] = region->src()->stride()->data()[d];
							reg.dst.stride[d] = region->dst()->stride()->data()[d];
						}
						regs.emplace_back(std::move(reg));
					}
				}
			}

			opIndex = 0;
			for (opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) 
			{
				auto op = net->oplists()->GetAs<Op>(opIndex);
				if (OpType_Const == op->type()) 
				{
					MNN_ASSERT(nullptr != op->outputIndexes());
					auto index = op->outputIndexes()->data()[0];
					index      = raw2filter_tensor_idx[index];
					if(index >= 0)
						TensorUtils::getDescribe(tensors[index].get())->usage = Tensor::InsideDescribe::CONSTANT;
				}
			}
		}
		else 
		{
			// Dynamic Model just set input tensor's shape
			for (int m = 0; m < (int)(tensors.size()); m++)
			{
				tensors[m].reset(new Tensor(4, Tensor::CAFFE_C4)); // NCHW, TODO
				tensors[m]->setType(DataType_DT_FLOAT);
			}
			valid = initTensors_filter_cvt(tensors, net, false, raw2filter_tensor_idx);
		}
	}
	else
	{
		filter2raw_tensor_idx.resize(0);
		raw2filter_tensor_idx.resize(0);

		tensors.resize(net->tensorName()->size());
		if (net->usage() == Usage_INFERENCE_STATIC) {
			// static model will set all tensors' shape
			auto describes = net->extraTensorDescribe();
			std::vector<const TensorDescribe*> des(tensors.size());
			for (int i = 0; i < describes->size(); i++) {
				int index = describes->GetAs<TensorDescribe>(i)->index();
				des[index] = describes->GetAs<TensorDescribe>(i);
			}
			for (int i = 0; i < tensors.size(); ++i) {
				auto blob = des[i]->blob();
				if (auto idims = blob->dims()) {
					tensors[i].reset(new Tensor(idims->size()));
					auto& tb = tensors[i]->buffer();
					for (int d = 0; d < idims->size(); d++) {
						tb.dim[d].extent = idims->Get(d);
					}
				}
				else {
					tensors[i].reset(new Tensor(1));
				}
				tensors[i]->setType(blob->dataType());
			}
			for (int i = 0; i < tensors.size(); ++i) {
				auto blob = des[i]->blob();
				TensorUtils::getDescribe(tensors[i].get())->dimensionFormat = blob->dataFormat();
				if (auto regions = des[i]->regions()) {
					auto& regs = TensorUtils::getDescribe(tensors[i].get())->regions;
					TensorUtils::getDescribe(tensors[i].get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
					regs.reserve(regions->size());
					for (int r = 0; r < regions->size(); r++) {
						auto region = regions->GetAs<Region>(r);
						Tensor::InsideDescribe::Region reg;
						reg.origin = tensors[region->origin()].get();
						reg.src.offset = region->src()->offset();
						reg.dst.offset = region->dst()->offset();
						for (int d = 0; d < 3; d++) {
							reg.size[d] = region->size()->data()[d];
							reg.src.stride[d] = region->src()->stride()->data()[d];
							reg.dst.stride[d] = region->dst()->stride()->data()[d];
						}
						regs.emplace_back(std::move(reg));
					}
				}
			}
			for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
				auto op = net->oplists()->GetAs<Op>(opIndex);
				if (OpType_Const == op->type()) {
					MNN_ASSERT(nullptr != op->outputIndexes());
					auto index = op->outputIndexes()->data()[0];
					TensorUtils::getDescribe(tensors[index].get())->usage = Tensor::InsideDescribe::CONSTANT;
				}
			}
		}
		else {
			// Dynamic Model just set input tensor's shape
			valid = initTensors(tensors, net);
		}
	}
	return valid;
}
#endif

//Shaquille, Added 20201102 End
static int _findOpPosition(const std::string& opName, const Net* net) {
    for (int i = 0; i < net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (opName == op->name()->str()) {
            return i;
        }
    }
    return -1;
}

static bool _validateOp(const Op* op) {
    if (nullptr == op->inputIndexes() && nullptr == op->outputIndexes()) {
        return false;
    }
    if (nullptr == op->name()) {
        return false;
    }
    return true;
}

static vector<Op*> generateOneSchedulePath(const Net* net, const int begin, const int end,
                                           const vector<shared_ptr<Tensor>>& allTensors) {
    vector<Op*> oplists;
    for (int i = begin; i < end; ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->type() == OpType_Input || !_validateOp(op)) {
            continue;
        }
        oplists.emplace_back(const_cast<Op*>(op));
    }
    return oplists;
}

static vector<vector<Op*>> generateSchedulePath(const Net* net, const ScheduleConfig& configs,
                                                const vector<shared_ptr<Tensor>>& allTensors) {
    vector<vector<Op*>> oplists;
    vector<string> inputs(configs.path.inputs);
    vector<string> outputs(configs.path.outputs);
    auto maxSize = std::max(inputs.size(), outputs.size());
    inputs.resize(maxSize);
    outputs.resize(maxSize);

    for (int i = 0; i < inputs.size(); i++) {
        string in  = inputs[i];
        string out = outputs[i];
        int start  = 0;
        int end    = net->oplists()->size();
        if (in.length() > 0) {
            auto pos = _findOpPosition(in, net);
            if (-1 == pos) {
                MNN_PRINT("Can't find %s op as start op\n", in.c_str());
            } else {
                start = pos;
            }
        }
        if (out.length() > 0) {
            auto pos = _findOpPosition(out, net);
            if (-1 == pos) {
                MNN_PRINT("Can't find %s op as end op\n", out.c_str());
            } else {
                end = pos + 1;
            }
        }
        if (start > end) {
            MNN_PRINT("op order incorrect end op '%s' before begin op '%s',please check!\n", out.c_str(), in.c_str());
        } else {
            vector<Op*> path = generateOneSchedulePath(net, start, end, allTensors);
            oplists.emplace_back(path);
        }
    }

    return oplists;
}

static void generateScheduleGraph(vector<const Op*>& ops, const Net* net, const ScheduleConfig& configs,
                                  const vector<shared_ptr<Tensor>>& allTensors) {
    if (configs.path.inputs.empty() && configs.path.outputs.empty()) {
        // Use Default Linear schedule
        ops.clear();
        ops.reserve(net->oplists()->size());
        for (int i = 0; i < net->oplists()->size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (op->type() != OpType_Input) {
                ops.emplace_back(op);
            }
        }
        return;
    }
    vector<vector<Op*>> paths = generateSchedulePath(net, configs, allTensors);

    unique_ptr<DirectedAcyclicGraph<Op*>> graph(new DirectedAcyclicGraph<Op*>());

    // add Node
    unordered_map<Op*, shared_ptr<Node<Op*>>> opMaps;
    for (vector<Op*> path : paths) {
        for (Op* op : path) {
            if (opMaps.find(op) == opMaps.end()) {
                OpNodeDef def(op);
                shared_ptr<Node<Op*>> n = graph->AddNode(def);
                opMaps.insert(make_pair(op, n));
            }
        }
    }

    // add edges
    for (vector<Op*> path : paths) {
        shared_ptr<Node<Op*>> pre = nullptr;
        for (Op* op : path) {
            shared_ptr<Node<Op*>> n = opMaps[op];
            if (nullptr == pre) {
                pre = n;
            } else {
                graph->AddEdge(pre, n);
                pre = n;
            }
        }
    }
    ops.clear();
    vector<shared_ptr<Node<Op*>>> order;
    if (graph->GetPostOrder(order)) {
        for (shared_ptr<Node<Op*>> n : order) {
            ops.emplace_back(n->getData());
        }
    } else {
        MNN_PRINT("op graph have cycle,schedule failed\n");
    }
}

static vector<Schedule::PipelineInfo> _scheduleUnit(const Net* net, const ScheduleConfig& configs,
                                                    const vector<shared_ptr<Tensor>>& allTensors) {
    vector<Schedule::PipelineInfo> oplists;
    vector<const Op*> ops;
    generateScheduleGraph(ops, net, configs, allTensors);
    initPipelineInfosFromOps(oplists, ops, allTensors);
    return oplists;
}


//Shaquille, Added 20201102 Start
static vector<Schedule::PipelineInfo> _scheduleFilteredUnit(const Net*                         net, 
	                                                        const ScheduleConfig&              configs,
	                                                        const vector<shared_ptr<Tensor>>&  allTensors,
	                                                        const std::vector<int>&            raw2filtered_tensor_idx,
	                                                        const std::vector<int>&            filter2raw_tensor_idx,
	                                                        const std::map<int, int>&          raw_cvt_output_2_input_idx)
{
	vector<Schedule::PipelineInfo> oplists;
	vector<const Op*> ops;
	generateScheduleGraph(ops, net, configs, allTensors);
	initPipelineInfosFromOps_filter_cvt(oplists, 
		                                ops, 
		                                allTensors, 
		                                raw2filtered_tensor_idx, 
		                                filter2raw_tensor_idx, 
		                                raw_cvt_output_2_input_idx);
	return oplists;
}
//Shaquille, Added 20201102 End

//Shaquille, Modified 20201102 Start
#if 0
Schedule::ScheduleInfo Schedule::schedule(const Net* net, const std::vector<ScheduleConfig>& configs) {
    std::vector<std::shared_ptr<Tensor>> allTensors;

    ScheduleInfo schedule;
    if (nullptr == net->oplists()) {
        MNN_PRINT("Error net for schedule\n");
        return schedule;
    }
    bool valid              = _setUpTensorInfo(allTensors, net);
    schedule.validForResize = valid;

    std::vector<std::pair<Backend::Info, std::vector<Schedule::PipelineInfo>>> result;

    for (auto& config : configs) {
        Backend::Info compute;
        compute.type      = getApprociateType(config);
        compute.numThread = config.numThread;
        compute.user      = config.backendConfig;
        auto oplists      = _scheduleUnit(net, config, allTensors);
        result.emplace_back(std::make_pair(compute, std::move(oplists)));
    }

    schedule.pipelineInfo = std::move(result);

    // get all used op's output, drop unused op, won't change op order. always insert all Input Ops
    std::vector<const Op*> oplists;
    {
        for (std::pair<Backend::Info, vector<Schedule::PipelineInfo>>& pipeline : schedule.pipelineInfo) {
            for (auto& info : pipeline.second) {
                oplists.push_back(info.op);
            }
        }
    }
    // set tensors' input/output usage by oplists info
    setInputOutputForOps(allTensors, oplists, net->usage() == Usage_INFERENCE_STATIC);

    // add output index by config info and outputName
    std::unordered_map<std::string, int> tensorNameIndexMap;
    for (int i = 0; i < net->tensorName()->size(); ++i) {
        tensorNameIndexMap[net->tensorName()->Get(i)->str()] = i;
    }
    for (auto& config : configs) {
        for (const auto& name : config.saveTensors) {
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                } else {
                    schedule.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
                }
            } else {
                MNN_PRINT("Bad outputname: %s\n", name.c_str());
            }
        }
    }
    if (net->outputName()) {
        for (int i = 0; i < net->outputName()->size(); ++i) {
            std::string name = net->outputName()->Get(i)->str();
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                } else {
                    schedule.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
                }
            }
        }
    }
    // add input/output tensor to schedule's input/output
    for (int index = 0; index < allTensors.size(); index++) {
        auto t = allTensors[index].get();
        auto usage = TensorUtils::getDescribe(t)->usage;
        if (usage == Tensor::InsideDescribe::INPUT) {
            schedule.inputTensors.insert(std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
        if (usage == Tensor::InsideDescribe::OUTPUT) {
            schedule.outputTensor.insert(
                       std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
    }
    // move tensors to schedule
    for (auto& t : allTensors) {
        schedule.allTensors.emplace_back(std::make_pair(0, std::move(t)));
    }
    return schedule;
}
#else
Schedule::ScheduleInfo Schedule::schedule(const Net* net, const std::vector<ScheduleConfig>& configs) {
	std::vector<std::shared_ptr<Tensor>> allTensors;
	ScheduleInfo schedule;
	if (nullptr == net->oplists()) {
		MNN_PRINT("Error net for schedule\n");
		return schedule;
	}
	bool valid = false;
	std::vector<int>    filter2raw_tensor_idx;
	std::vector<int>    raw2filter_tensor_idx;
	std::map<int, int>  raw_cvt_output_2_input_idx;
	if (1 == configs.size() && MNN_FORWARD_OPENCL == configs[0].type)
	{
		valid = _setUpOCLFilteredTensorInfo(allTensors,
			                                net,
			                                filter2raw_tensor_idx,
			                                raw2filter_tensor_idx,
			                                raw_cvt_output_2_input_idx);
	}
	else
		valid = _setUpTensorInfo(allTensors, net);
	schedule.validForResize = valid;

	std::vector<std::pair<Backend::Info, std::vector<Schedule::PipelineInfo>>> result;

	for (auto& config : configs) {
		Backend::Info compute;
		compute.type      = getApprociateType(config);
		compute.numThread = config.numThread;
		compute.user      = config.backendConfig;
		vector<Schedule::PipelineInfo> oplists(0);
		if (0 == filter2raw_tensor_idx.size())
			oplists = _scheduleUnit(net, config, allTensors);
		else
			oplists = _scheduleFilteredUnit(net, config, allTensors, raw2filter_tensor_idx, filter2raw_tensor_idx, raw_cvt_output_2_input_idx);
		result.emplace_back(std::make_pair(compute, std::move(oplists)));
	}

	schedule.pipelineInfo = std::move(result);

	// get all used op's output, drop unused op, won't change op order. always insert all Input Ops
	std::vector<const Op*> oplists;
	{
		for (std::pair<Backend::Info, vector<Schedule::PipelineInfo>>& pipeline : schedule.pipelineInfo) {
			for (auto& info : pipeline.second) {
				oplists.push_back(info.op);
			}
		}
	}
	// set tensors' input/output usage by oplists info
	setInputOutputForOps_filter_cvt(allTensors, 
		                            oplists, net->usage() == Usage_INFERENCE_STATIC, 
		                            raw2filter_tensor_idx, 
		                            filter2raw_tensor_idx, 
		                            raw_cvt_output_2_input_idx);

	// add output index by config info and outputName
	std::unordered_map<std::string, int> tensorNameIndexMap;
	for (int i = 0; i < net->tensorName()->size(); ++i) 
	{
		if (raw2filter_tensor_idx.size() > 0)
		{
			if (-1 != raw2filter_tensor_idx[i])
				tensorNameIndexMap[net->tensorName()->Get(i)->str()] = raw2filter_tensor_idx[i];
		}
		else
			tensorNameIndexMap[net->tensorName()->Get(i)->str()] = i;
	}
	for (auto& config : configs) 
	{
		for (const auto& name : config.saveTensors) 
		{
			auto iter = tensorNameIndexMap.find(name);
			if (iter != tensorNameIndexMap.end()) 
			{
				auto t = allTensors[iter->second].get();
				if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) 
					TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
				else 
				{
					int  in_tensor_idx     = iter->second;
					int  raw_in_tensor_idx = iter->second;
					if (raw2filter_tensor_idx.size() > 0)
						raw_in_tensor_idx = filter2raw_tensor_idx[in_tensor_idx];
					schedule.outputTensor.insert(
						std::make_pair(net->tensorName()->GetAsString(raw_in_tensor_idx)->c_str(), t));
				}
			}
			else 
			{
				MNN_PRINT("Bad outputname: %s\n", name.c_str());
			}
		}
	}
	if (net->outputName()) {
		for (int i = 0; i < net->outputName()->size(); ++i) {
			std::string name = net->outputName()->Get(i)->str();
			auto iter = tensorNameIndexMap.find(name);
			if (iter != tensorNameIndexMap.end()) {
				auto t = allTensors[iter->second].get();
				if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
					TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
				}
				else {
					int  out_tensor_idx     = iter->second;
					int  raw_out_tensor_idx = iter->second;
					if (raw2filter_tensor_idx.size() > 0)
						raw_out_tensor_idx = filter2raw_tensor_idx[out_tensor_idx];
					schedule.outputTensor.insert(
						std::make_pair(net->tensorName()->GetAsString(raw_out_tensor_idx)->c_str(), t));
				}
			}
		}
	}

	std::vector<int>  tensor_ref_cnt(allTensors.size(), 0);
	for (int i = 0; i < net->oplists()->size(); ++i)
	{
		auto op = net->oplists()->GetAs<Op>(i);
		if (raw2filter_tensor_idx.size() > 0)
		{
			if (OpType_ConvertTensor == op->type())
				continue;
		}
		if (nullptr != op->inputIndexes())
		{
			auto data = op->inputIndexes()->data();
			for (int j = 0; j < op->inputIndexes()->size(); ++j)
			{
				auto index = data[j];
				if (raw2filter_tensor_idx.size() > 0)
				{
					if (-1 != raw2filter_tensor_idx[index])
					{
						index                  = raw2filter_tensor_idx[index];
						tensor_ref_cnt[index] += 1;
					}
					else
					{
						std::map<int, int>::const_iterator  iter = raw_cvt_output_2_input_idx.find(index);
						if (iter != raw_cvt_output_2_input_idx.end())
						{
							index                  = iter->second;
							index                  = raw2filter_tensor_idx[index];
							tensor_ref_cnt[index] += 1;
						}
						else
						{
							MNN_PRINT("error, cannot find the input of TensorConvertor: %s\n", op->name()->c_str());
						}
					}
				}
				else
					tensor_ref_cnt[index] += 1;
			}
		}
	}

	// add input/output tensor to schedule's input/output
	for (int index = 0; index < allTensors.size(); index++) {
		auto t = allTensors[index].get();
		auto usage = TensorUtils::getDescribe(t)->usage;
		if (usage == Tensor::InsideDescribe::INPUT) {
			int raw_idx = index;
			if (raw2filter_tensor_idx.size() > 0)
				raw_idx = filter2raw_tensor_idx[index];
			schedule.inputTensors.insert(std::make_pair(net->tensorName()->GetAsString(raw_idx)->c_str(), t));
		}
		if (usage == Tensor::InsideDescribe::OUTPUT) {
			if (0 == tensor_ref_cnt[index])
				TensorUtils::getDescribe(t)->fake_used = true;
			int raw_idx = index;
			if (raw2filter_tensor_idx.size() > 0)
				raw_idx = filter2raw_tensor_idx[index];
			schedule.outputTensor.insert(
				std::make_pair(net->tensorName()->GetAsString(raw_idx)->c_str(), t));
		}
	}

	// move tensors to schedule
	for (auto& t : allTensors) {
		schedule.allTensors.emplace_back(std::make_pair(0, std::move(t)));
	}

	return schedule;
}
#endif
//Shaquille, Modified 20201102 End

} // namespace MNN
