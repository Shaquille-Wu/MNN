//
//  InitNet.hpp
//  MNN
//
//  Created by MNN on 2018/09/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/Schedule.hpp"

namespace MNN {

// init Tensors by net
bool initTensors(std::vector<std::shared_ptr<Tensor>>& allTensors, const Net* net);
// init Pipeline Infos by oplist and tensors
void initPipelineInfosFromOps(std::vector<Schedule::PipelineInfo>& infos, std::vector<const Op*>& ops, const std::vector<std::shared_ptr<Tensor>>& allTensors);
// set input and output for allTensors by ops info
void setInputOutputForOps(std::vector<std::shared_ptr<Tensor>>& allTensors, const std::vector<const Op*>& ops, bool isStatic = false);

//Shaquille, Added 20201117 Start
// init Tensors by net
bool initTensors_filter_cvt(std::vector<std::shared_ptr<Tensor>>& allTensors, const Net* net, bool is_normal_proc, std::vector<int> const& raw2filter_tensor_idx);
// init Pipeline Infos by oplist and tensors
void initPipelineInfosFromOps_filter_cvt(std::vector<Schedule::PipelineInfo>&           infos,
	                                     std::vector<const Op*>&                        ops, 
	                                     const std::vector<std::shared_ptr<Tensor>>&    allTensors,
	                                     const std::vector<int>&                        raw2filtered_tensor_idx,
	                                     const std::vector<int>&                        filter2raw_tensor_idx,
	                                     const std::map<int, int>&                      raw_cvt_output_2_input_idx);

void setInputOutputForOps_filter_cvt(std::vector<std::shared_ptr<Tensor>>&    allTensors, 
	                                 const std::vector<const Op*>&            ops, 
	                                 bool                                     isStatic, 
	                                 const std::vector<int>&                  raw2filtered_tensor_idx,
	                                 const std::vector<int>&                  filter2raw_tensor_idx,
	                                 const std::map<int, int>&                raw_cvt_output_2_input_idx);
//Shaquille, Added 20201117 End

// init Pipeline Infos by net and tensors, set input and output info
void initPipelineInfosFromNet(std::vector<Schedule::PipelineInfo>& infos, const Net* net, std::vector<std::shared_ptr<Tensor>>& allTensors);
} // namespace MNN
