# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class GpuPipeline(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGpuPipeline(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GpuPipeline()
        x.Init(buf, n + offset)
        return x

    # GpuPipeline
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GpuPipeline
    def LocalSize(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # GpuPipeline
    def LocalSizeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # GpuPipeline
    def LocalSizeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuPipeline
    def Key(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # GpuPipeline
    def Metal(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # GpuPipeline
    def MetalAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int8Flags, o)
        return 0

    # GpuPipeline
    def MetalLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuPipeline
    def Vulkan(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # GpuPipeline
    def VulkanAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int8Flags, o)
        return 0

    # GpuPipeline
    def VulkanLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuPipeline
    def OpenglComputeShader(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # GpuPipeline
    def OpenclKernel(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def GpuPipelineStart(builder): builder.StartObject(6)
def GpuPipelineAddLocalSize(builder, localSize): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(localSize), 0)
def GpuPipelineStartLocalSizeVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuPipelineAddKey(builder, key): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(key), 0)
def GpuPipelineAddMetal(builder, metal): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(metal), 0)
def GpuPipelineStartMetalVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def GpuPipelineAddVulkan(builder, vulkan): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(vulkan), 0)
def GpuPipelineStartVulkanVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def GpuPipelineAddOpenglComputeShader(builder, openglComputeShader): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(openglComputeShader), 0)
def GpuPipelineAddOpenclKernel(builder, openclKernel): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(openclKernel), 0)
def GpuPipelineEnd(builder): return builder.EndObject()
