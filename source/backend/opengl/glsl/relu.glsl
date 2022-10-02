
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=2) uniform ivec4 imgSize;
layout(location=3) uniform float slope;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 imgSize = imgSize.xyz;
    if(pos.x < imgSize.x && pos.y < imgSize.y)
    {
        vec4 dataIn =  texelFetch(uInput, pos, 0);
        bvec4 lessZero = bvec4(lessThan(dataIn, vec4(0.0)));
        vec4 dataTemp = dataIn * vec4(slope);
        imageStore(uOutput, pos, mix(dataIn, dataTemp, lessZero));
    }
}
