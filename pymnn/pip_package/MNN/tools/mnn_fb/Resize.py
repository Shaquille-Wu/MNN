# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class Resize(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsResize(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Resize()
        x.Init(buf, n + offset)
        return x

    # Resize
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Resize
    def XScale(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # Resize
    def YScale(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def ResizeStart(builder): builder.StartObject(2)
def ResizeAddXScale(builder, xScale): builder.PrependFloat32Slot(0, xScale, 0.0)
def ResizeAddYScale(builder, yScale): builder.PrependFloat32Slot(1, yScale, 0.0)
def ResizeEnd(builder): return builder.EndObject()
