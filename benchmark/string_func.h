/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file string_func.h
 * @brief This header file defines some basic and common operation for string
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-05-28
 */

#ifndef  __DLCV_TOOLS_STRING_FUNC_H__
#define  __DLCV_TOOLS_STRING_FUNC_H__

#include <string>
#include <vector>

/*
* @brief remove the spaces for string's head and tail
*
* @param src source string
*
* @return the trimed string
*
*/
std::string               trim_string(const std::string& src);

/*
* @brief get file's ext name
*
* @param src source file name
*
* @return the ext name(with '.')
*
*/
std::string               get_file_ext(const std::string& src);

/*
* @brief get all file name from specific directory which match the ext-name
*
* @param src_image_dir source directory
*
* @param ext_name_list a list witch specified the ext-name
*
* @param sub_dir specified the sub_dir in the 'src_image_dir'
*
* @return the ext name(with '.')
*
*/
std::vector<std::string>  travel_image_dir(const std::string&                src_image_dir, 
                                           const std::vector<std::string>&   ext_name_list,
                                           const std::string&                sub_dir);


std::vector<std::string>  split_string(std::string const& src_string, char sperator);

#endif