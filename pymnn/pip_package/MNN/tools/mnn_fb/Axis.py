# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class Axis(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsAxis(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Axis()
        x.Init(buf, n + offset)
        return x

    # Axis
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Axis
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def AxisStart(builder): builder.StartObject(1)
def AxisAddAxis(builder, axis): builder.PrependInt32Slot(0, axis, 0)
def AxisEnd(builder): return builder.EndObject()
