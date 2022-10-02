//
//  Tensor.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <complex.h>
#include <string.h>
#include <MNN/Tensor.hpp>
#include "MNN_generated.h"
#include "core/Backend.hpp"
#include "core/MNNMemoryUtils.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

using namespace std;

namespace MNN {
Tensor::Tensor(int dimSize, DimensionType type) {
    MNN_ASSERT(dimSize <= MNN_MAX_TENSOR_DIM);
    mDescribe          = new InsideDescribe;
    mBuffer.dimensions = dimSize;
    mBuffer.type       = halide_type_of<float>();
    mBuffer.device     = 0;
    mBuffer.host       = nullptr;
    mBuffer.dim        = &mDescribe->dims[0];

    switch (type) {
        case CAFFE:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            break;
        case TENSORFLOW:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            break;
        case CAFFE_C4:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            break;
        default:
            break;
    }
}

Tensor::Tensor(const Tensor* tensor, DimensionType type, bool allocMemory) {
    MNN_ASSERT(tensor != nullptr);

    auto buffer        = tensor->buffer();
    mDescribe          = new InsideDescribe;
    mBuffer.dimensions = buffer.dimensions;
    mBuffer.type       = buffer.type;
    mBuffer.device     = 0;
    mBuffer.host       = nullptr;
    mBuffer.dim        = &mDescribe->dims[0];

    for (int i = 0; i < buffer.dimensions; ++i) {
        mBuffer.dim[i].extent = buffer.dim[i].extent;
    }
    switch (type) {
        case CAFFE:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            break;
        case TENSORFLOW:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            break;
        case CAFFE_C4:
            mDescribe->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
            type                       = CAFFE;
            break;
        default:
            break;
    }

    // format mapping
    auto originType = tensor->getDimensionType();
    if (originType != type && buffer.dimensions >= 4) {
        std::vector<int> axisMap;
        // NCHW -> NHWC
        if (originType == CAFFE) {
            axisMap.push_back(0);
            for (int i = 2; i < buffer.dimensions; ++i) {
                axisMap.push_back(i);
            }
            axisMap.push_back(1);
        }
        // NHWC -> NCHW
        else {
            axisMap.push_back(0);
            axisMap.push_back(buffer.dimensions - 1);
            for (int i = 1; i < buffer.dimensions - 1; ++i) {
                axisMap.push_back(i);
            }
        }
        for (int i = 0; i < buffer.dimensions; ++i) {
            mBuffer.dim[i].extent = buffer.dim[axisMap[i]].extent;
        }
    }
    TensorUtils::setLinearLayout(this);

    if (allocMemory) {
        auto memorySize = size();
        if (memorySize > 0) {
            mDescribe->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
            mBuffer.host          = (uint8_t*)MNNMemoryAllocAlign(size(), MNN_MEMORY_ALIGN_DEFAULT);
            MNN_ASSERT(mBuffer.host != nullptr);
        }
    }
}

//Shaquille, Added 20201107 Start
Tensor::Tensor(struct InsideDescribe*   ion_device_desc,
	           const  Tensor*           tensor,
	           DimensionType            type)
{
	MNN_ASSERT(tensor != nullptr);

	auto buffer        = tensor->buffer();
	mDescribe          = new InsideDescribe;
	mBuffer.dimensions = buffer.dimensions;
	mBuffer.type       = buffer.type;
	mBuffer.device     = 0;
	mBuffer.host       = nullptr;
	mBuffer.dim        = &mDescribe->dims[0];

	for (int i = 0; i < buffer.dimensions; ++i) {
		mBuffer.dim[i].extent = buffer.dim[i].extent;
	}
	switch (type) {
	case CAFFE:
		mDescribe->dimensionFormat = MNN_DATA_FORMAT_NCHW;
		break;
	case TENSORFLOW:
		mDescribe->dimensionFormat = MNN_DATA_FORMAT_NHWC;
		break;
	case CAFFE_C4:
		mDescribe->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
		type = CAFFE;
		break;
	default:
		break;
	}

	// format mapping
	auto originType = tensor->getDimensionType();
	if (originType != type && buffer.dimensions >= 4) {
		std::vector<int> axisMap;
		// NCHW -> NHWC
		if (originType == CAFFE) {
			axisMap.push_back(0);
			for (int i = 2; i < buffer.dimensions; ++i) {
				axisMap.push_back(i);
			}
			axisMap.push_back(1);
		}
		// NHWC -> NCHW
		else {
			axisMap.push_back(0);
			axisMap.push_back(buffer.dimensions - 1);
			for (int i = 1; i < buffer.dimensions - 1; ++i) {
				axisMap.push_back(i);
			}
		}
		for (int i = 0; i < buffer.dimensions; ++i) {
			mBuffer.dim[i].extent = buffer.dim[axisMap[i]].extent;
		}
	}
	TensorUtils::setLinearLayout(this);

	auto memorySize = size();
	if (memorySize > 0) 
	{
		std::pair<uint8_t*, uint64_t>  ion_buf_desc = ion_device_desc->backend->CreateIONBuf(memorySize);
		if (nullptr == ion_buf_desc.first || 0 == ion_buf_desc.second)
		{
			MNN_PRINT("create ion buf failed\n");
			exit(-1);
		}
		//Shaquille, TODO
		mDescribe->memoryType = Tensor::InsideDescribe::MEMORY_HOST_BACKEND_SHARED;
		mBuffer.host          = ion_buf_desc.first;
		mBuffer.device        = ion_buf_desc.second;
		mDescribe->backend    = ion_device_desc->backend;
		is_ion_buf_           = true;
	}
}
//Shaquille, Added 20201107 End

Tensor::~Tensor() {
    if (mBuffer.type.code == halide_type_handle) {
        auto handles = (void**)mBuffer.host;
        for (int i = 0; i < elementSize(); ++i) {
            if (nullptr != handles[i]) {
                mDescribe->extra.handleFreeFunction(handles[i]);
            }
        }
    }
//Shaquille, Modified 20201107 Start
#if 0
    if (mDescribe->memoryType == InsideDescribe::MEMORY_HOST) {
        if (nullptr != mBuffer.host) {
            MNNMemoryFreeAlign(mBuffer.host);
        }
    }
#else
	if (false == is_ion_buf_)
	{
	    if (mDescribe->memoryType == InsideDescribe::MEMORY_HOST) {
	        if (nullptr != mBuffer.host) {
	            MNNMemoryFreeAlign(mBuffer.host);
	        }
	    }
	}
	else
	{
		uint8_t*  host_ptr = (uint8_t*)mBuffer.host;
		if(nullptr != host_ptr)
			mDescribe->backend->ReleaseIONBuf(host_ptr);
	}
#endif
//Shaquille, Modified 20201107 End
    delete mDescribe;
}

Tensor* Tensor::createDevice(const std::vector<int>& dims, halide_type_t type, DimensionType dimType) {
    auto shapeTensor = new Tensor((int)dims.size(), dimType);
    for (int i = 0; i < dims.size(); ++i) {
        shapeTensor->setLength(i, dims[i]);
    }
    shapeTensor->buffer().type = type;
    TensorUtils::setLinearLayout(shapeTensor);
    return shapeTensor;
}

Tensor* Tensor::create(const std::vector<int>& dims, halide_type_t type, void* userData, DimensionType dimType) {
    Tensor shapeTensor((int)dims.size(), dimType);
    for (int i = 0; i < dims.size(); ++i) {
        shapeTensor.setLength(i, dims[i]);
    }
    shapeTensor.buffer().type = type;

    bool ownData = userData == nullptr;
    auto result  = new Tensor(&shapeTensor, dimType, ownData);
    if (nullptr != userData) {
        result->buffer().host = (uint8_t*)userData;
    }
    return result;
}

//Shaquille, Added 20201107 Start
Tensor* Tensor::create_by_backend(struct InsideDescribe*     device_desc,
	                              const std::vector<int>&    shape,
	                              halide_type_t              type,
	                              DimensionType              dimType)
{
	Tensor shapeTensor((int)shape.size(), dimType);
	for (int i = 0; i < shape.size(); ++i) 
		shapeTensor.setLength(i, shape[i]);
	shapeTensor.buffer().type = type;

	Tensor* result = nullptr;
	if (false == device_desc->backend->IsSupportION())
		result = new Tensor(&shapeTensor, dimType, true);
	else
		result = new Tensor(device_desc, &shapeTensor, dimType);

	return result;
}

void Tensor::SyncIONStart(bool finish_cmd_que)
{
	if(nullptr != mDescribe->backend && true == is_ion_buf_)
		mDescribe->backend->SyncIONBufStart(buffer().host, finish_cmd_que);
}

void Tensor::SyncIONEnd(bool finish_cmd_que)
{
	if (nullptr != mDescribe->backend && true == is_ion_buf_)
		mDescribe->backend->SyncIONBufEnd(buffer().host, finish_cmd_que);
}

//Shaquille, Added 20201107

bool Tensor::copyFromHostTensor(const Tensor* hostTensor) {
    auto bn = mDescribe->backend;
    if (nullptr == bn) {
        return false;
    }
    bn->onCopyBuffer(hostTensor, this);
    return true;
}

bool Tensor::copyToHostTensor(Tensor* hostTensor) const {
    auto bn = mDescribe->backend;
    if (nullptr == bn) {
        return false;
    }
    bn->onCopyBuffer(this, hostTensor);
    return true;
}

bool Tensor::copy_to_host_batch(std::vector<Tensor*> const& dev_tensor, std::vector<Tensor*>& host_tensor)
{
	bool      batch_copy  = true;
	Backend*  first_bn    = nullptr;
	int       tensor_cnt  = (int)(host_tensor.size());
	int       i           = 0;
	for (i = 0; i < tensor_cnt; i++)
	{
		if (0 == i)
			first_bn = TensorUtils::getDescribe(host_tensor[i])->backend;
		Backend*  cur_bn = TensorUtils::getDescribe(host_tensor[i])->backend;
		if (cur_bn != first_bn || 
			false == host_tensor[i]->is_ion_buf_ || 
			cur_bn->type() != MNN_FORWARD_OPENCL)
		{
			batch_copy = false;
			break;
		}
	}
	if(nullptr == first_bn)
		batch_copy = false;

	if (false == batch_copy)
	{
		for (i = 0; i < tensor_cnt; i++)
			dev_tensor[i]->copyToHostTensor(host_tensor[i]);
	}
	else
	{
		for (i = 0; i < tensor_cnt; i++)
		{
			bool block_read = (i == (tensor_cnt - 1));
			first_bn->onCopyIONBufferFromDev(dev_tensor[i], host_tensor[i], block_read);
		}
	}

	return true;
}

static Tensor::DimensionType getDimType(const Tensor* origin) {
    auto dimformat = TensorUtils::getDescribe(origin)->dimensionFormat;
    switch (dimformat) {
        case MNN_DATA_FORMAT_NHWC:
            return Tensor::TENSORFLOW;
        case MNN_DATA_FORMAT_NCHW:
            return Tensor::CAFFE;
        case MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        default:
            break;
    }
    return Tensor::CAFFE;
}

//Shaquille, Modified 20201107 Start
#if 0
Tensor* Tensor::createHostTensorFromDevice(const Tensor* device, bool copyContent) {
    auto tensor = Tensor::create(device->shape(), device->getType(), nullptr, TensorUtils::getDimType(device));
    if (copyContent) {
        device->copyToHostTensor(tensor);
    }
    return tensor;
}
#else
Tensor* Tensor::createHostTensorFromDevice(const Tensor*  device, 
	                                       bool           copyContent,
	                                       int            tensor_io_type,
                                           int            tensor_dim_type) 
{
    bool                    create_ion   = false;
    Tensor::InsideDescribe* desc         = TensorUtils::getDescribe(device);
    MNNForwardType          forward_type = desc->backend->type();

    if (device->getType().code == halide_type_float)
    {
        if (Tensor::TENSOR_NORMAL != tensor_io_type && forward_type == MNNForwardType::MNN_FORWARD_OPENCL)
        {
            if (true == desc->backend->IsSupportION())
                create_ion = true;
        }
    }
	Tensor*                 tensor          = nullptr;
    Tensor::DimensionType   expect_dim_type = TensorUtils::getDimType(device);
    if (-1 != tensor_dim_type)
        expect_dim_type = (Tensor::DimensionType)tensor_dim_type;

    if (false == create_ion)
        tensor = Tensor::create(device->shape(), device->getType(), nullptr, expect_dim_type);
    else
        tensor = Tensor::create_by_backend(desc, device->shape(), device->getType(), expect_dim_type);
    if (copyContent) {
        device->copyToHostTensor(tensor);
    }

    return tensor;
}
#endif
//Shaquille, Modified 20201107 End

Tensor::DimensionType Tensor::getDimensionType() const {
    if (mDescribe->dimensionFormat == MNN_DATA_FORMAT_NHWC) {
        return Tensor::TENSORFLOW;
    }
    return Tensor::CAFFE;
}

Tensor::HandleDataType Tensor::getHandleDataType() const {
    if (halide_type_handle != mBuffer.type.code) {
        return HANDLE_NONE;
    }
    return HANDLE_STRING;
}
void Tensor::setType(int type) {
    switch (type) {
        case DataType_DT_DOUBLE:
        case DataType_DT_FLOAT:
            mBuffer.type = halide_type_of<float>();
            break;
        case DataType_DT_BFLOAT16:
            mBuffer.type = halide_type_t(halide_type_float, 16);
            break;
        case DataType_DT_QINT32:
        case DataType_DT_INT32:
        case DataType_DT_BOOL:
        case DataType_DT_INT64:
            mBuffer.type = halide_type_of<int32_t>();
            break;
        case DataType_DT_QINT8:
        case DataType_DT_INT8:
            mBuffer.type = halide_type_of<int8_t>();
            break;
        case DataType_DT_QUINT8:
        case DataType_DT_UINT8:
            mBuffer.type = halide_type_of<uint8_t>();
            break;
        case DataType_DT_QUINT16:
        case DataType_DT_UINT16:
            mBuffer.type = halide_type_of<uint16_t>();
            break;
        case DataType_DT_QINT16:
        case DataType_DT_INT16:
            mBuffer.type = halide_type_of<int16_t>();
            break;
        case DataType_DT_STRING:
            mBuffer.type                  = halide_type_t(halide_type_handle, sizeof(void*) * 8);
            mDescribe->extra.handleFreeFunction = (void (*)(void*))::free;
            break;

        default:
            MNN_PRINT("Unsupported data type!");
            MNN_ASSERT(false);
            break;
    }
}

std::vector<int> Tensor::shape() const {
    std::vector<int> result;
    for (int i = 0; i < mBuffer.dimensions; ++i) {
        result.push_back(mBuffer.dim[i].extent);
    }
    return result;
}
template <typename T>
void printData(const Tensor* tensor, const void* data, const char* fmt) {
    const T* buffer = (const T*)data;
    if (tensor->dimensions() != 4) {
        auto size = tensor->elementSize();
        for (int i = 0; i < size; i++) {
            MNN_PRINT(fmt, buffer[i]);
        }
        MNN_PRINT("\n");
        return;
    }

    auto tf      = tensor->getDimensionType() == Tensor::TENSORFLOW;
    auto batch   = tensor->batch();
    auto channel = tensor->channel();
    auto height  = tensor->height();
    auto width   = tensor->width();

    auto unit = sizeof(T);
    if (tf) {
        auto bytesPerRow   = channel * unit;
        auto bytesPerImage = width * bytesPerRow;
        auto bytesPerBatch = height * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = buffer + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int c = 0; c < channel; c++) {
                        MNN_PRINT(fmt, bytes[h * width * channel + w * channel + c]);
                    }
                    MNN_PRINT("\n");
                }
                MNN_PRINT("--------------\n");
            }
        }
    } else if (TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) { // NC/4HW4
        auto components    = 4;
        auto bytesPerRow   = width * components * unit;
        auto bytesPerImage = height * bytesPerRow;
        auto bytesPerBatch = UP_DIV(channel, 4) * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = buffer + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        auto n = c / components, r = c % components;
                        MNN_PRINT(fmt, bytes[(n * width * height + h * width + w) * components + r]);
                    }
                    MNN_PRINT("\n");
                }
                MNN_PRINT("--------------\n");
            }
        }
    } else { // NCHW
        auto bytesPerRow   = width * unit;
        auto bytesPerImage = height * bytesPerRow;
        auto bytesPerBatch = channel * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = buffer + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        MNN_PRINT(fmt, bytes[c * width * height + h * width + w]);
                    }
                    MNN_PRINT("\n");
                }
                MNN_PRINT("--------------\n");
            }
        }
    }
}
void Tensor::print() const {
    // print dimensions
    MNN_PRINT("====== Tensor %p ======", this);
    MNN_PRINT("\nDimension: ");
    for (int i = 0; i < mBuffer.dimensions; i++) {
        MNN_PRINT("%d, ", mBuffer.dim[i].extent);
    }

    // convert to host if needed
    auto printee = this;
    bool device  = this->buffer().host == NULL && this->buffer().device != 0;
    if (device) {
        printee = this->createHostTensorFromDevice(this, true);
    }
    auto buffer = printee->buffer().host;

    MNN_PRINT("\nData: ");
    if (printee->getType().code == halide_type_int) {
        if (printee->getType().bits == 8) { // int8
            printData<int8_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 16) { // int16
            printData<int16_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 32) { // int32
            printData<int32_t>(printee, buffer, "%d, ");
        } else {
            MNN_PRINT("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_uint) {
        if (printee->getType().bits == 8) { // uint8
            printData<uint8_t>(printee, buffer, "%d, ");
        } else {
            MNN_PRINT("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_float) {
        if (printee->getType().bits == 32) { // float32
            printData<float>(printee, buffer, "%f, ");
        } else {
            MNN_PRINT("\nunsupported data type\n");
        }
    } else {
        MNN_PRINT("\nunsupported data type");
    }

    // clean up
    if (printee != this) {
        delete printee;
    }
}

void Tensor::printShape() const {
    const int dims = this->dimensions();
    MNN_PRINT("\t**Tensor shape**: ");
    if (dims == 0) {
        MNN_PRINT("\t*Scalar*");
    }
    for (int i = 0; i < dims; ++i) {
        MNN_PRINT("%d, ", this->length(i));
    }
    MNN_PRINT("\n");
}

int Tensor::size() const {
    auto dataSize = mBuffer.type.bytes();
    MNN_ASSERT(dataSize >= 1);
    for (int i = 0; i < this->buffer().dimensions; i++) {
        int currentDimSize = mBuffer.dim[i].extent;
        if (mDescribe->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = ALIGN_UP4(currentDimSize);
        }
        dataSize *= currentDimSize;
    }
    return dataSize;
}

} // namespace MNN
