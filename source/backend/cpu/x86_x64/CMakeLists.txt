if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)")
    if (APPLE)
        add_definitions(-fno-stack-check) # Workaround a Xcode 11.X bug
    endif()
    message(STATUS "${CMAKE_SYSTEM_PROCESSOR}: Open SSE")
    add_definitions(-DMNN_USE_SSE)
    FILE(GLOB MNN_X8664_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
    if (MSVC)
        FILE(GLOB MNN_AVX_SRC ${CMAKE_CURRENT_LIST_DIR}/avx/*.cpp)
    else()
        FILE(GLOB MNN_AVX_SRC ${CMAKE_CURRENT_LIST_DIR}/avx/*)
        if (MNN_AVX512)
            FILE(GLOB MNN_AVX512_SRC ${CMAKE_CURRENT_LIST_DIR}/avx512/*)
            add_library(MNNAVX512 OBJECT ${MNN_AVX512_SRC})
            target_compile_options(MNNAVX512 PRIVATE -m64 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -DMNN_X86_USE_ASM -mavx512vnni)
        endif()
    endif()
    FILE(GLOB MNN_SSE_SRC ${CMAKE_CURRENT_LIST_DIR}/sse/*)
    add_library(MNNX8664 OBJECT ${MNN_X8664_SRC})
    add_library(MNNAVX OBJECT ${MNN_AVX_SRC})
    add_library(MNNSSE OBJECT ${MNN_SSE_SRC})
    if(MSVC)
        target_compile_options(MNNAVX PRIVATE /arch:AVX)
    else()
        target_compile_options(MNNSSE PRIVATE -msse4.1)
        target_compile_options(MNNAVX PRIVATE -mavx2 -mfma -DMNN_X86_USE_ASM)
    endif()
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNX8664> $<TARGET_OBJECTS:MNNAVX> $<TARGET_OBJECTS:MNNSSE>)
    if (MNN_AVX512)
        target_compile_options(MNNX8664 PRIVATE -DMNN_AVX512)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNAVX512>)
    endif()
endif()
