# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class RoiPooling(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRoiPooling(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RoiPooling()
        x.Init(buf, n + offset)
        return x

    # RoiPooling
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RoiPooling
    def PooledWidth(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # RoiPooling
    def PooledHeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # RoiPooling
    def SpatialScale(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def RoiPoolingStart(builder): builder.StartObject(3)
def RoiPoolingAddPooledWidth(builder, pooledWidth): builder.PrependInt32Slot(0, pooledWidth, 0)
def RoiPoolingAddPooledHeight(builder, pooledHeight): builder.PrependInt32Slot(1, pooledHeight, 0)
def RoiPoolingAddSpatialScale(builder, spatialScale): builder.PrependFloat32Slot(2, spatialScale, 0.0)
def RoiPoolingEnd(builder): return builder.EndObject()
