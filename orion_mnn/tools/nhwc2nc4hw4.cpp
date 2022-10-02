
#include <string>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "string_func.h"

int main(int argc, char** argv)
{
    int            i               = 0 ;
    if(argc < 4)
    {
        printf("usage: ./nhwc2nc4hw4 n:h:w:c src_file dst_file");
        return 0;
    }

    std::string   data_shape_str = argv[1];
    std::string   src_file_name  = argv[2];
    std::string   dst_file_name  = argv[3];

    std::vector<std::string>   shape_str_list = split_string(data_shape_str, ':');
    std::vector<int>           src_data_shape({ 1, 1, 1, 1 });
    std::vector<int>           dst_data_shape({ 1, 1, 1, 1 });
    if(shape_str_list.size() > 0)
        src_data_shape[0] = atoi(shape_str_list[0].c_str());
    if(shape_str_list.size() > 1)
        src_data_shape[1] = atoi(shape_str_list[1].c_str());
    if(shape_str_list.size() > 2)
        src_data_shape[2] = atoi(shape_str_list[2].c_str());
    if(shape_str_list.size() > 3)
        src_data_shape[3] = atoi(shape_str_list[3].c_str());

    dst_data_shape    = src_data_shape;
    dst_data_shape[3] = (((src_data_shape[3] + 3) >> 2) << 2);
    
    FILE*  input_file = fopen(src_file_name.c_str(), "rb");
    if(NULL == input_file)
    {
        printf("cannot open input_file %s\r\n", src_file_name.c_str());
        return 0;
    }

    fseek(input_file, 0, SEEK_END);
    int tensor_in_file_size     = (int)(ftell(input_file));
    fseek(input_file, 0, SEEK_SET);

    int raw_expect_size = src_data_shape[0] * src_data_shape[1] * src_data_shape[2] * src_data_shape[3] * sizeof(float);
    if(tensor_in_file_size < raw_expect_size)
    {
        printf("input_file's size is %d, its less than %d\r\n", tensor_in_file_size, raw_expect_size);
        fflush(input_file);
        fclose(input_file);
        return 0;
    }

    unsigned char*  raw_data = new unsigned char[tensor_in_file_size];
    fread(raw_data, raw_expect_size, 1, input_file);
    fflush(input_file);
    fclose(input_file);

    float*   src_data = (float*)(raw_data);
    float*   dst_data = new float[dst_data_shape[0] * dst_data_shape[1] * dst_data_shape[2] * dst_data_shape[3]];

    int      h            = dst_data_shape[1];
    int      w            = dst_data_shape[2];
    int      dst_c        = dst_data_shape[3];
    int      src_c        = src_data_shape[3];
    int      dst_line_cnt = w * dst_c;
    int      src_line_cnt = w * src_c;
    int  dst_buf_size = dst_data_shape[0] * dst_data_shape[1] * dst_data_shape[2] * dst_data_shape[3] * sizeof(float);
    memset(dst_data, 0, dst_buf_size);
    for(int i = 0 ; i < h ; i ++)
    {
        for(int j = 0 ; j < w ; j ++)
        {
            dst_data[i * dst_line_cnt + dst_c * j]     = src_data[i * src_line_cnt + src_c * j];
            dst_data[i * dst_line_cnt + dst_c * j + 1] = src_data[i * src_line_cnt + src_c * j + 1];
            dst_data[i * dst_line_cnt + dst_c * j + 2] = src_data[i * src_line_cnt + src_c * j + 2];
        }
    }

    FILE*  output_file = fopen(dst_file_name.c_str(), "wb");
    if(NULL != output_file)
    {
        fwrite(dst_data, dst_buf_size, 1, output_file);
        printf("NC4HW4 file saved, shape: [ %d, %d, %d, %d ]\r\n", 
               dst_data_shape[0], dst_data_shape[1], dst_data_shape[2], dst_data_shape[3]);
        fflush(output_file);
        fclose(output_file);
    }


    delete[] src_data;
    delete[] dst_data;

    printf("end\r\n");

    return 0;
}