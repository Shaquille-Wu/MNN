# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class InnerProduct(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsInnerProduct(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = InnerProduct()
        x.Init(buf, n + offset)
        return x

    # InnerProduct
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # InnerProduct
    def OutputCount(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # InnerProduct
    def BiasTerm(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # InnerProduct
    def WeightSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # InnerProduct
    def Weight(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # InnerProduct
    def WeightAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # InnerProduct
    def WeightLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # InnerProduct
    def Bias(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # InnerProduct
    def BiasAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # InnerProduct
    def BiasLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # InnerProduct
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # InnerProduct
    def Transpose(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # InnerProduct
    def QuanParameter(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .IDSTQuan import IDSTQuan
            obj = IDSTQuan()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def InnerProductStart(builder): builder.StartObject(8)
def InnerProductAddOutputCount(builder, outputCount): builder.PrependInt32Slot(0, outputCount, 0)
def InnerProductAddBiasTerm(builder, biasTerm): builder.PrependInt32Slot(1, biasTerm, 0)
def InnerProductAddWeightSize(builder, weightSize): builder.PrependInt32Slot(2, weightSize, 0)
def InnerProductAddWeight(builder, weight): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(weight), 0)
def InnerProductStartWeightVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def InnerProductAddBias(builder, bias): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(bias), 0)
def InnerProductStartBiasVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def InnerProductAddAxis(builder, axis): builder.PrependInt32Slot(5, axis, 0)
def InnerProductAddTranspose(builder, transpose): builder.PrependBoolSlot(6, transpose, 0)
def InnerProductAddQuanParameter(builder, quanParameter): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(quanParameter), 0)
def InnerProductEnd(builder): return builder.EndObject()
