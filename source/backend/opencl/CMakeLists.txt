add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_codegen.py"
    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/"
    "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc"
    COMMENT "OpenCL Code Generation"
)
add_custom_target (MNN_CLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
file(GLOB_RECURSE MNN_OpenCL_SRC ${CMAKE_CURRENT_LIST_DIR}/*)

option(MNN_OPENCL_LWS_TUNE "Enable MNN OpenCL Lws Tuning" ON)
option(MNN_OPENCL_PROFILE "Enable MNN OpenCL Kernel Profile" OFF)

IF (MNN_OPENCL_LWS_TUNE)
    add_definitions(-DMNN_OPENCL_LWS_TUNE)
ENDIF()

IF (MNN_OPENCL_PROFILE)
    add_definitions(-DENABLE_OPENCL_TIME_PROFILER)
ENDIF()

if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    add_definitions(-DMNN_USE_LIB_WRAPPER)
    add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=110)
else()
    if (MNN_USE_SYSTEM_LIB)
        find_package(OpenCL REQUIRED)
    else()
        add_definitions(-DMNN_USE_LIB_WRAPPER)
        add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=110)
    endif()
endif()
IF(MNN_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    endif()
    add_library(MNN_CL SHARED ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
    target_link_libraries(MNN_CL MNN ${OpenCL_LIBRARIES})
ELSE()
    add_library(MNN_CL OBJECT ${MNN_OpenCL_SRC} "${CMAKE_CURRENT_LIST_DIR}/execution/cl/opencl_program.cc")
    set(MNN_OCL_LIBS ${OpenCL_LIBRARIES} PARENT_SCOPE)
ENDIF()

#Shaquille, Added 20201108 Start
if (${CMAKE_SYSTEM_NAME} MATCHES "Android") 
    set(RPC_MEM_LIB "${CMAKE_SOURCE_DIR}/3rd_party/rpcmem/${ANDROID_ABI}/rpcmem.a")
    target_link_libraries(MNN_CL ${RPC_MEM_LIB})
endif()
#Shaquille, Added 20201108 End

target_include_directories(MNN_CL PRIVATE
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_LIST_DIR}/schema/current
    ${CMAKE_SOURCE_DIR}/3rd_party/half
)

install (TARGETS MNN_CL DESTINATION lib)
