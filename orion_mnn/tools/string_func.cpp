/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file string_func.cpp
 * @brief the implementation for some basic and common operation of string
 * @author wuxiao@ainirobot.com
 * @date 2020-03-12
 */

#include "string_func.h"
#if !defined(_MSC_VER)
#include <dirent.h>
#endif
#include <sys/stat.h>
#include <string.h>
#include <algorithm>
#include <chrono>

std::string trim_string(const std::string& src)
{
    int    str_len     = src.length();
    int    start_pos   = 0;
    int    end_pos     = 0;
    int    i           = 0;
    char*  trim_str    = new char[str_len + 1];
    memset(trim_str, 0, str_len + 1);
    for(i = 0 ; i < str_len ; i ++)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i])
        {
            start_pos = i;
            break;
        }
    }
    for(i = str_len - 1 ; i >= 0 ; i --)
    {
        if(' ' != (src.c_str())[i] && '\r' != (src.c_str())[i] && '\n' != (src.c_str())[i])
        {
            end_pos = i;
            break;
        }
    }

    int trim_len = end_pos - start_pos + 1;
    for(i = 0 ; i < trim_len ; i ++)
        trim_str[i] = (src.c_str())[i + start_pos];

    std::string  result = std::string(trim_str);
    delete[] trim_str;

    return result;
}

std::string               get_file_ext(const std::string& src)
{
    int    len        = src.length();
    char*  ext        = new char[len + 1];
    int    start_pos  = 0;
    int    i          = 0;

    for(i = len - 1 ; i >= 0 ; i --)
    {
        if('.' == src.c_str()[i])
        {
            start_pos = i;
            break;
        }
    }

    memset(ext, 0, len + 1);
    for(i = start_pos ; i < len ; i ++)
    {
        ext[i - start_pos] = src.c_str()[i];
    }

    std::string   result = ext;

    delete[] ext;
    return result;
}

#define MAX_PATH_LEN 512
std::vector<std::string>  travel_image_dir(const std::string&                src_image_dir, 
                                           const std::vector<std::string>&   ext_name_list,
                                           const std::string&                sub_dir)
{
    std::vector<std::string>     image_file_list(0);
#if !defined(_MSC_VER)
    DIR*                         d                  = NULL;
    struct dirent*               dp                 = NULL;
    struct stat                  st;    
    char                         p[MAX_PATH_LEN]    = {0};
    int                          i                  = 0;
    int                          ext_name_list_cnt  = (int)(ext_name_list.size());
    std::string                  cur_sub_dir        = "";

    if(stat(src_image_dir.c_str(), &st) < 0 || !S_ISDIR(st.st_mode)) {
        printf("invalid path: %s\n", src_image_dir.c_str());
        return image_file_list;
    }

    if(!(d = opendir(src_image_dir.c_str()))) {
        printf("opendir[%s] error: %m\n", src_image_dir.c_str());
        return image_file_list;
    }

    while((dp = readdir(d)) != NULL) 
    {
        if((!strncmp(dp->d_name, ".", 1)) || (!strncmp(dp->d_name, "..", 2)))
            continue;

        snprintf(p, sizeof(p) - 1, "%s/%s", src_image_dir.c_str(), dp->d_name);
        stat(p, &st);
        if(!S_ISDIR(st.st_mode)) 
        {
            std::string    cur_file      = dp->d_name;
            int            str_len       = cur_file.length();
            bool           found         = false;
            std::string    cur_file_ext  = get_file_ext(cur_file);

            for(i = 0 ; i < ext_name_list_cnt ; i ++)
            {
                const std::string&  cur_ext_name = ext_name_list.at(i);
                if(0 == strcasecmp(cur_ext_name.c_str(), cur_file_ext.c_str()))
                {
                    found = true;
                    break;
                }
            }
            if(true == found)
            {
                if("" != sub_dir)
                    cur_file = sub_dir + std::string("/") + cur_file;
                image_file_list.push_back(cur_file);
            }
        } 
        else
        {
            if("" == sub_dir)
                cur_sub_dir = dp->d_name;
            else
                cur_sub_dir = sub_dir + "/" + dp->d_name;
            std::vector<std::string> sub_image_file_list = travel_image_dir(std::string(p), ext_name_list, cur_sub_dir);
            image_file_list.insert(image_file_list.end(),sub_image_file_list.begin(),sub_image_file_list.end());
        }
    }
    closedir(d);

    std::sort(image_file_list.begin(), image_file_list.end());
#endif    

	return image_file_list;
}

std::vector<std::string> split_string(const std::string& src, char sperator)
{
    std::vector<std::string>   result;
    std::string                trim_str = trim_string(src);
    int                        i        = 0;
    int                        len      = trim_str.length();
    int                        cur_pos  = 0;
    char*                      cur_str  = 0;
    if("" == trim_str || len <= 0)
        return result ;

    cur_str = new char[len + 1];
    memset(cur_str, 0, len + 1);

    for(i = 0 ; i < len ; i ++)
    {
        if(sperator != trim_str.c_str()[i])
        {
            cur_str[cur_pos] = trim_str.c_str()[i];
            cur_pos ++;
        }
        else
        {
            std::string   cur_trim_str = trim_string(std::string(cur_str));
            if("" != cur_trim_str)
                result.push_back(cur_trim_str);
            memset(cur_str, 0, len + 1);
            cur_pos = 0;
        }
    }

    if(cur_pos > 0)
    {
        std::string   cur_trim_str = trim_string(std::string(cur_str));
        result.push_back(cur_trim_str);
    }

    delete[] cur_str;

    return result;
}

std::string               get_system_time()
{
	auto tt = std::chrono::system_clock::to_time_t
	(std::chrono::system_clock::now());
	struct tm* ptm = localtime(&tt);
	char date[60] = { 0 };
	sprintf(date, "%04d%02d%02d%02d%02d%02d",
		(int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
		(int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
	return std::string(date);
}