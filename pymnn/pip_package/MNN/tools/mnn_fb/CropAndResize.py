# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class CropAndResize(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsCropAndResize(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CropAndResize()
        x.Init(buf, n + offset)
        return x

    # CropAndResize
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CropAndResize
    def ExtrapolationValue(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # CropAndResize
    def Method(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def CropAndResizeStart(builder): builder.StartObject(2)
def CropAndResizeAddExtrapolationValue(builder, extrapolationValue): builder.PrependFloat32Slot(0, extrapolationValue, 0.0)
def CropAndResizeAddMethod(builder, method): builder.PrependInt8Slot(1, method, 0)
def CropAndResizeEnd(builder): return builder.EndObject()
