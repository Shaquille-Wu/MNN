//
//  CPUGridSample.hpp
//  OrionStar
//
//  Created by Shaquille.Wu on 2021/02/27.
//  Copyright © 2021, OrionStar
//

#ifndef CPUGridSample_hpp
#define CPUGridSample_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {

class CPUGridSample : public Execution {
public:
    CPUGridSample(Backend*  backend, 
                  int       mode          = 0,
                  int       padding_mode  = 0, 
                  bool      align_corners = false);
    virtual ~CPUGridSample();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int    mode_            = 0; // 0:bilinear 1: nereast
    int    padding_mode_    = 0;
    bool   align_corners_   = false;
};

} // namespace MNN

#endif /* CPUInterp_hpp */
