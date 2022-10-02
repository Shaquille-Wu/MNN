# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class RNNParam(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRNNParam(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RNNParam()
        x.Init(buf, n + offset)
        return x

    # RNNParam
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RNNParam
    def NumUnits(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # RNNParam
    def IsBidirectionalRNN(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # RNNParam
    def KeepAllOutputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # RNNParam
    def FwGateWeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RNNParam
    def FwGateBias(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RNNParam
    def FwCandidateWeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RNNParam
    def FwCandidateBias(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RNNParam
    def BwGateWeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RNNParam
    def BwGateBias(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RNNParam
    def BwCandidateWeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RNNParam
    def BwCandidateBias(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .Blob import Blob
            obj = Blob()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def RNNParamStart(builder): builder.StartObject(11)
def RNNParamAddNumUnits(builder, numUnits): builder.PrependInt32Slot(0, numUnits, 0)
def RNNParamAddIsBidirectionalRNN(builder, isBidirectionalRNN): builder.PrependBoolSlot(1, isBidirectionalRNN, 0)
def RNNParamAddKeepAllOutputs(builder, keepAllOutputs): builder.PrependBoolSlot(2, keepAllOutputs, 0)
def RNNParamAddFwGateWeight(builder, fwGateWeight): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(fwGateWeight), 0)
def RNNParamAddFwGateBias(builder, fwGateBias): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(fwGateBias), 0)
def RNNParamAddFwCandidateWeight(builder, fwCandidateWeight): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(fwCandidateWeight), 0)
def RNNParamAddFwCandidateBias(builder, fwCandidateBias): builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(fwCandidateBias), 0)
def RNNParamAddBwGateWeight(builder, bwGateWeight): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(bwGateWeight), 0)
def RNNParamAddBwGateBias(builder, bwGateBias): builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(bwGateBias), 0)
def RNNParamAddBwCandidateWeight(builder, bwCandidateWeight): builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(bwCandidateWeight), 0)
def RNNParamAddBwCandidateBias(builder, bwCandidateBias): builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(bwCandidateBias), 0)
def RNNParamEnd(builder): return builder.EndObject()
