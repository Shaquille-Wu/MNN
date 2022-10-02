import os
import sys
from os import listdir
from os.path import isfile, join
shaderPath=sys.argv[1]
cppPath=sys.argv[2]
def main():
    shaders=[]
    for root, dirs, files in os.walk(shaderPath):
        for file in files:
            if file.endswith('.mm'):
                shaders.append(os.path.join(root,file))
    with open(cppPath,"w") as f:
        f.write("// This file is generated by Shell for ops register\n")
        f.write("#import \"backend/metal/MetalDefine.h\"\n")
        f.write("   namespace MNN {\n")
        f.write("#if MNN_METAL_ENABLED\n")
        funcs=[]
        for shapath in shaders:
            with open(shapath,"r") as sha:
                lines=sha.readlines()
                for l in lines:
                    if l.startswith("REGISTER_METAL_OP_CREATOR("):
                        x=l.replace("REGISTER_METAL_OP_CREATOR(","").replace(")","").replace(" ","").replace(";","").replace("\n","").split(",")
                        funcname="___"+x[0]+"__"+x[1]+"__();"
                        funcs.append(funcname)
                        f.write("  extern void "+funcname+"\n")
            pass
        f.write("void registerMetalOps() {\n")
        for func in funcs:
            f.write("   "+func+"\n")
        f.write("}\n#endif\n}")

if __name__ == '__main__':
    main()
