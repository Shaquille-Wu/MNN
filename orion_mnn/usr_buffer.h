/*
 * Copyright (C) OrionStar Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file usr_buffer.h
 * @brief This header file defines UsrBuffer struct
 * @author Wu Xiao(wuxiao@ainirobot.com)
 * @date 2020-08-24
 */

#ifndef __ORION_MNN_USR_BUF_H__
#define __ORION_MNN_USR_BUF_H__

namespace vision{

/**
 * @brief UsrBuffer, exchange buffer between application and OrionSnpeImpl
 * 
 */
class UsrBuffer
{
public:
    /**
     * @brief default constructor for OrionSnpeImpl
     */
    UsrBuffer()
    {
        buf_       = nullptr;
        buf_size_  = 0;
    };

    /**
     * @brief destructor
     */   
    virtual ~UsrBuffer()
    {
        buf_       = nullptr;
        buf_size_  = 0;
    }

    /**
     * @brief Construct a new UsrBuffer with buf and buf_size, just shadow copy
     *
     * @param buf pointer of buffer
     * @param buf_size size of buffer
     */
    UsrBuffer(unsigned char* buf, unsigned int buf_size)
    {
        buf_       = buf;
        buf_size_  = buf_size;
    };

    /**
     * @brief copy constructor, just shadow copy
     *
     * @param other other buffer
     */
    UsrBuffer(const UsrBuffer& other)
    {
        buf_       = other.buf_;
        buf_size_  = other.buf_size_;
    } ;

    /**
     * @brief assignment operator, just shadow copy
     *
     * @param other other UsrBuffer
     */
    virtual UsrBuffer& operator = (const UsrBuffer& other)
    {
        if (this == &other)	return *this ;

        buf_       = other.buf_;
        buf_size_  = other.buf_size_;            

        return *this ;
    } ;       

public:
    unsigned char*         buf_;
    unsigned int           buf_size_;
}; // UsrBuffer

} //namespace vision

#endif