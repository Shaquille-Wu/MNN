#ifndef PRINT_TENSOR_H_
#define PRINT_TENSOR_H_

#include "../source/core/TensorUtils.hpp"

inline void PrintTensor(MNN::Tensor const* tensor, const char* tensor_name, const char* ident)
{
    static const std::string kTensorDimensionType[] = {
        "TENSORFLOW",
        "CAFFE",
        "CAFFE_C4",
    };

    static const std::string kTensorFormatType[] = {
        "NCHW",
        "NHWC",
        "NC4HW4",
        "NHWC4",
        "UNKNOWN",
    };

    printf("%sTensorName:    %s\r\n", ident, tensor_name);
    if(nullptr == tensor)
    {
        printf("%scannot found tensor\r\n", ident);
        return ;
    }
    int  i       = 0;
    MNN::Tensor::InsideDescribe* desc = MNN::TensorUtils::getDescribe(tensor);
    printf("%sDimensionType:  %s\r\n", ident, kTensorDimensionType[tensor->getDimensionType()].c_str());
    printf("%sFormatType:     %s\r\n", ident, kTensorFormatType[desc->dimensionFormat].c_str());
    printf("%sElementSize:    %d\r\n", ident, tensor->getType().bits);
    printf("%sElementCount:   %d\r\n", ident, tensor->elementSize());
    printf("%sBufferSize:     %d\r\n", ident, tensor->size());
    int dims = tensor->dimensions();
    printf("%stensor shape:   ", ident);
    if (dims == 0)
        printf("\t*Scalar*");
    for (i = 0; i < dims; ++i)
        printf("%d, ", tensor->length(i));
    printf("\n");  
    printf("%sstride:         ", ident);
    for(i = 0 ; i < dims ; i ++)
        printf("%d, ", tensor->stride(i));
    printf("\r\n");
    std::vector<int>  cur_shape(4, 1);
    dims = (dims > 4 ? 4 : dims);
    for(i = 0 ; i < dims ; i ++)
        cur_shape[4 - dims + i] = tensor->length(i);

    if(MNN::MNN_DATA_FORMAT_NCHW == desc->dimensionFormat)
    {
        printf("%snorminal shape: %d, %d, %d, %d\n", ident, cur_shape[0], cur_shape[1], cur_shape[2], cur_shape[3]);
        printf("%sdlcv shape:     %d, %d, %d, %d\n", ident, cur_shape[0], cur_shape[1], cur_shape[2], cur_shape[3]);
    }
    else if(MNN::MNN_DATA_FORMAT_NHWC == desc->dimensionFormat)
    {
        printf("%snorminal shape: %d, %d, %d, %d\n", ident, cur_shape[0], cur_shape[2], cur_shape[3], cur_shape[1]);
        printf("%sdlcv shape:     %d, %d, %d, %d\n", ident, cur_shape[0], cur_shape[2], cur_shape[3], cur_shape[1]);
    }
    else if(MNN::MNN_DATA_FORMAT_NC4HW4 == desc->dimensionFormat)
    {
        printf("%snorminal shape: %d, %d, %d, %d\n", ident, cur_shape[0], cur_shape[1], cur_shape[2], cur_shape[3]);
        printf("%sdlcv shape:     %d, %d, %d, %d\n", ident, cur_shape[0], cur_shape[2], cur_shape[3], cur_shape[1]);
    }
    else
        printf("unknown shape\n");
}

#endif