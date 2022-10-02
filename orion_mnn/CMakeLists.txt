cmake_minimum_required (VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE  ON)

set (ORION_MNN_SRC
	orion_mnn_impl.cpp
	tensor_convert.cpp
)

if(NOT (MSVC OR WIN32))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -frtti -fexceptions")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
endif()

add_library(orion_mnn SHARED ${ORION_MNN_SRC})
target_compile_options(orion_mnn PRIVATE -Wall)
if(NOT (MSVC OR WIN32))
	target_compile_options(orion_mnn PRIVATE -Wno-unused-variable)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x86)|(x64)|(X64)|(amd64)|(AMD64)|(i686)|(i386)")
    if(MSVC)
        target_compile_options(orion_mnn PRIVATE /arch:AVX)
    else()
		target_compile_options(orion_mnn PRIVATE -msse4.1)
        target_compile_options(orion_mnn PRIVATE -mavx2 -mfma)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
	target_compile_options(orion_mnn PRIVATE -mfloat-abi=softfp -mfpu=neon)
endif()
target_link_libraries(orion_mnn ${MNN_DEPS})
if(TARGET_OS STREQUAL "android")
     target_link_libraries(orion_mnn c++_shared)
endif()

add_subdirectory(tools)

install (TARGETS orion_mnn DESTINATION lib)
