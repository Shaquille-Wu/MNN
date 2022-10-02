#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#if 0
__kernel void winogradTransformSource(__read_only image2d_t uInput, // 0
                                      __write_only image2d_t uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int offsetX, // 9
                                      __private const int offsetY, __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    if (pos.x < unitWidth*unitHeight && pos.y < srcChannelC4) {
        int unitWidth_idx = pos.x % unitWidth;
        int unitHeight_idx = pos.x / unitWidth;
        int2 realPos   = (int2)(unitWidth_idx + offsetX, unitHeight_idx + offsetY);
        int dstXOrigin = pos.y;
        int batchIndex = pos.y / srcChannelC4;
        int srcZ       = pos.y % srcChannelC4;
        int dstYOrigin = unitWidth * unitHeight_idx + unitWidth_idx;
        int dstHeight  = (unitWidth * unitHeight + 3) / 4;
        int dstY       = dstYOrigin / 4;
        int dstX       = dstYOrigin % 4 + 4 * dstXOrigin;

        batchIndex = batchOffset;
        {
            int sxStart = (realPos.x) * 2 - padX;
            int syStart = (realPos.y) * 2 - padY;
            FLOAT4 S00;
            FLOAT4 S10;
            FLOAT4 S20;
            FLOAT4 S30;
            FLOAT4 S01;
            FLOAT4 S11;
            FLOAT4 S21;
            FLOAT4 S31;
            FLOAT4 S02;
            FLOAT4 S12;
            FLOAT4 S22;
            FLOAT4 S32;
            FLOAT4 S03;
            FLOAT4 S13;
            FLOAT4 S23;
            FLOAT4 S33;
            {
                int sx      = 0 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S00         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S10         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S20         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S30         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S01         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S11         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S21         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S31         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S02         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S12         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S22         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S32         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S03         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S13         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S23         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S33         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            FLOAT4 m00 = +S00 - S02;
            FLOAT4 m10 = +S10 - S12;
            FLOAT4 m20 = +S20 - S22;
            FLOAT4 m30 = +S30 - S32;
            FLOAT4 m01 = +(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
            FLOAT4 m11 = +(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
            FLOAT4 m21 = +(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
            FLOAT4 m31 = +(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
            FLOAT4 m02 = -(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
            FLOAT4 m12 = -(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
            FLOAT4 m22 = -(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
            FLOAT4 m32 = -(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
            FLOAT4 m03 = -S01 + S03;
            FLOAT4 m13 = -S11 + S13;
            FLOAT4 m23 = -S21 + S23;
            FLOAT4 m33 = -S31 + S33;
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 0), +m00 - m20);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 1), +(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 2), -(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 3), -m10 + m30);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 4), +m01 - m21);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 5), +(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 6), -(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 7), -m11 + m31);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 8), +m02 - m22);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 9), +(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 10), -(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 11), -m12 + m32);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 12), +m03 - m23);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 13), +(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 14), -(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 15), -m13 + m33);
        }
    }
}
#else
__kernel void winogradTransformSource(__read_only image2d_t uInput, // 0
                                      __write_only image2d_t uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int offsetX, // 9
                                      __private const int offsetY, __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    if (pos.x < unitWidth*unitHeight && pos.y < srcChannelC4) {
        int unitWidth_idx = pos.x % unitWidth;
        int unitHeight_idx = pos.x / unitWidth;
        int2 realPos   = (int2)(unitWidth_idx + offsetX, unitHeight_idx + offsetY);
        int dstXOrigin = pos.y;
        int batchIndex = pos.y / srcChannelC4;
        int srcZ       = pos.y % srcChannelC4;
        int dstYOrigin = unitWidth * unitHeight_idx + unitWidth_idx;
        int dstHeight  = (unitWidth * unitHeight + 3) >> 2;
        int dstY       = dstYOrigin >> 2;
        int dstX       = (dstYOrigin & 3) + (dstXOrigin << 2);

        batchIndex = batchOffset;
        {
            int sxStart = (realPos.x << 1) - padX;
            int syStart = (realPos.y << 1) - padY;
            FLOAT4 S00;
            FLOAT4 S10;
            FLOAT4 S20;
            FLOAT4 S30;
            FLOAT4 S01;
            FLOAT4 S11;
            FLOAT4 S21;
            FLOAT4 S31;
            FLOAT4 S02;
            FLOAT4 S12;
            FLOAT4 S22;
            FLOAT4 S32;
            FLOAT4 S03;
            FLOAT4 S13;
            FLOAT4 S23;
            FLOAT4 S33;
            {
                int sx      = 0 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S00         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S10         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S20         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S30         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S01         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S11         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S21         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S31         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S02         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S12         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S22         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S32         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S03         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S13         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S23         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(mad24(srcZ, srcWidth, sx), -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(mad24(batchIndex, srcHeight, sy), -1, sy < 0 || sy >= srcHeight);
                S33         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            FLOAT4 m00 = +S00 - S02;
            FLOAT4 m10 = +S10 - S12;
            FLOAT4 m20 = +S20 - S22;
            FLOAT4 m30 = +S30 - S32;
            FLOAT4 m01 = (FLOAT)(0.5f) * (S01 + S02);//+(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
            FLOAT4 m11 = (FLOAT)(0.5f) * (S11 + S12);//+(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
            FLOAT4 m21 = (FLOAT)(0.5f) * (S21 + S22);//+(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
            FLOAT4 m31 = (FLOAT)(0.5f) * (S31 + S32);//+(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
            FLOAT4 m02 = (FLOAT)(0.5f) * (S02 - S01);//-(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
            FLOAT4 m12 = (FLOAT)(0.5f) * (S12 - S11);//-(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
            FLOAT4 m22 = (FLOAT)(0.5f) * (S22 - S21);//-(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
            FLOAT4 m32 = (FLOAT)(0.5f) * (S32 - S31);//-(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
            FLOAT4 m03 = -S01 + S03;
            FLOAT4 m13 = -S11 + S13;
            FLOAT4 m23 = -S21 + S23;
            FLOAT4 m33 = -S31 + S33;
            /*
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 0), +m00 - m20);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 1), (FLOAT)(0.5f)*(m20 + m10));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 2), (FLOAT)(0.5f)*(m20 - m10));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 3), -m10 + m30);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 4), +m01 - m21);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 5), (FLOAT)(0.5f)*(m21 + m11));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 6), (FLOAT)(0.5f)*(m21 - m11));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 7), -m11 + m31);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 8), +m02 - m22);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 9),  (FLOAT)(0.5f)*(m22 + m12));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 10), (FLOAT)(0.5f)*(m22 - m12));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 11), -m12 + m32);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 12), +m03 - m23);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 13), (FLOAT)(0.5f)*(m23 + m13));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 14), (FLOAT)(0.5f)*(m23 - m13));
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 15), -m13 + m33);
            */
            int dst_pos_y = dstY;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), +m00 - m20);
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m20 + m10));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m20 - m10));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), -m10 + m30);
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), +m01 - m21);
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m21 + m11));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m21 - m11));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), -m11 + m31);
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), +m02 - m22);
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m22 + m12));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m22 - m12));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), -m12 + m32);
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), +m03 - m23);
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m23 + m13));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), (FLOAT)(0.5f)*(m23 - m13));
            dst_pos_y    += dstHeight;
            WI_F(uOutput, (int2)(dstX, dst_pos_y), -m13 + m33);
        }
    }
}
#endif
