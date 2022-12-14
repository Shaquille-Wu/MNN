# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class GpuFunction(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGpuFunction(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GpuFunction()
        x.Init(buf, n + offset)
        return x

    # GpuFunction
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GpuFunction
    def Stags(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .GpuStage import GpuStage
            obj = GpuStage()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # GpuFunction
    def StagsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GpuFunction
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def GpuFunctionStart(builder): builder.StartObject(2)
def GpuFunctionAddStags(builder, stags): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(stags), 0)
def GpuFunctionStartStagsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GpuFunctionAddName(builder, name): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def GpuFunctionEnd(builder): return builder.EndObject()
