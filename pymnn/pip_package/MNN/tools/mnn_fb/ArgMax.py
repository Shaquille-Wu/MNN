# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class ArgMax(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsArgMax(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ArgMax()
        x.Init(buf, n + offset)
        return x

    # ArgMax
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ArgMax
    def OutMaxVal(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ArgMax
    def TopK(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ArgMax
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ArgMax
    def SoftmaxThreshold(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def ArgMaxStart(builder): builder.StartObject(4)
def ArgMaxAddOutMaxVal(builder, outMaxVal): builder.PrependInt32Slot(0, outMaxVal, 0)
def ArgMaxAddTopK(builder, topK): builder.PrependInt32Slot(1, topK, 0)
def ArgMaxAddAxis(builder, axis): builder.PrependInt32Slot(2, axis, 0)
def ArgMaxAddSoftmaxThreshold(builder, softmaxThreshold): builder.PrependInt32Slot(3, softmaxThreshold, 0)
def ArgMaxEnd(builder): return builder.EndObject()
