# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class Rank(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRank(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Rank()
        x.Init(buf, n + offset)
        return x

    # Rank
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def RankStart(builder): builder.StartObject(0)
def RankEnd(builder): return builder.EndObject()
