PROJECT_NAME=orion_mnn
BUILD_TYPE=Debug
BUILD_PLATFORM=x86_64-linux
BUILD_CMD=build
#TARGET_OS support linux and android
TARGET_OS=linux
#TARGET_ARCH support x86_64 and armv8
TARGET_ARCH=x86_64
#ANDROID_NDK_DIR=/home/shaquille/android-ndk-r16b
ANDROID_NDK_DIR=/home/shaquille/android-ndk-r17c

while getopts ":c:t:a:o:q" opt
do
    case $opt in
        c)
        BUILD_CMD=$OPTARG
        ;;    
        t)
        BUILD_TYPE=$OPTARG
        ;;
        a)
        TARGET_ARCH=$OPTARG
        ;;
        o)
        TARGET_OS=$OPTARG
        ;;        
        ?)
        echo "unknow parameter: $opt"
        exit 1;;
    esac
done

CORE_COUNT=$(cat /proc/cpuinfo | grep processor | wc -l)
echo "CORE_COUNT ${CORE_COUNT}"

CUR_DIR_PATH=${PWD}
BUILD_PLATFORM=${TARGET_ARCH}-${TARGET_OS}

mkdir -p ./build/${BUILD_TYPE}/${BUILD_PLATFORM}/${PROJECT_NAME}
cd ./build/${BUILD_TYPE}/${BUILD_PLATFORM}/${PROJECT_NAME}
echo entring ${PWD} for ${BUILD_CMD} ${PROJECT_NAME} start

BUILD_CMD_LINE="-DCMAKE_INSTALL_PREFIX=${PWD}/install"
BUILD_CMD_LINE=" ${BUILD_CMD_LINE} -DTARGET_ARCH=${TARGET_ARCH}"
BUILD_CMD_LINE=" ${BUILD_CMD_LINE} -DTARGET_OS=${TARGET_OS}"
BUILD_CMD_LINE=" ${BUILD_CMD_LINE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
BUILD_CMD_LINE=" ${BUILD_CMD_LINE} -DMNN_BUILD_BENCHMARK=ON"
BUILD_CMD_LINE=" ${BUILD_CMD_LINE} -DMNN_USE_SYSTEM_LIB=OFF"
BUILD_CMD_LINE=" ${BUILD_CMD_LINE} -DMNN_BUILD_DEMO=ON"

if [ "$TARGET_OS" == "android" ] ; then
    if [ "$NDK_ROOT" == "" ] ; then
        export NDK_ROOT=${ANDROID_NDK_DIR}
    else
        ANDROID_NDK_DIR=${NDK_ROOT}
    fi
    if [ "$TARGET_ARCH" == "armv8" ] ; then
        ANDROID_ABI_FORMAT="arm64-v8a"
        ANDROID_API_VERSION=27
    else
        ANDROID_ABI_FORMAT="armeabi-v7a"
        ANDROID_API_VERSION=22
    fi
    EXTRA_BUILD_CMD_LINE=" -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_DIR}/build/cmake/android.toolchain.cmake"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DANDROID_ABI=${ANDROID_ABI_FORMAT}"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DANDROID_NATIVE_API_LEVEL=android-${ANDROID_API_VERSION}"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DANDROID_TOOLCHAIN=clang"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DANDROID_STL=c++_shared"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DMNN_USE_LOGCAT=false"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DMNN_BUILD_FOR_ANDROID_COMMAND=true"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DMNN_OPENCL=ON"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DMNN_OPENCL_LWS_TUNE=ON"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DMNN_VULKAN=ON"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DNATIVE_LIBRARY_OUTPUT=."
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DNATIVE_INCLUDE_OUTPUT=."
    BUILD_CMD_LINE="${BUILD_CMD_LINE} ${EXTRA_BUILD_CMD_LINE}"
elif [ "$TARGET_ARCH" == "armv7" ] || [ "$TARGET_ARCH" == "armv8" ] ; then
    echo "armv7-armv8 platform"
elif [ "$TARGET_ARCH" == "x86_64" ] && [ "$TARGET_OS" == "linux" ] ; then
    echo "x86_64-linux platform"
    EXTRA_BUILD_CMD_LINE=" -DMNN_BUILD_QUANTOOLS=ON"
    EXTRA_BUILD_CMD_LINE=" -DMNN_BUILD_TRAIN=ON ${EXTRA_BUILD_CMD_LINE}"
    EXTRA_BUILD_CMD_LINE=" -DMNN_BUILD_CONVERTER=ON ${EXTRA_BUILD_CMD_LINE}"
    EXTRA_BUILD_CMD_LINE=" -DMNN_EXPR_ENABLE_PROFILER=ON ${EXTRA_BUILD_CMD_LINE}"
    EXTRA_BUILD_CMD_LINE=" -DMNN_TRAIN_DEBUG=ON ${EXTRA_BUILD_CMD_LINE}"
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DNATIVE_LIBRARY_OUTPUT=."
    EXTRA_BUILD_CMD_LINE="${EXTRA_BUILD_CMD_LINE} -DNATIVE_INCLUDE_OUTPUT=."
    BUILD_CMD_LINE="${BUILD_CMD_LINE} ${EXTRA_BUILD_CMD_LINE}"
    echo "BUILD_CMD_LINE: ${BUILD_CMD_LINE}"
else
    echo "unknown platform"
    return
fi

BUILD_CMD_LINE=" ${BUILD_CMD_LINE} ../../../../MNN"

if [ "$BUILD_CMD" == "build" ]; then
    cmake ${BUILD_CMD_LINE}
    rm -rf ../../../../MNN/source/backend/opencl/execution/cl/opencl_program.cc
    make -j${CORE_COUNT}
    make install
elif 
    [ "$BUILD_CMD" == "clean" ]; then
    make clean
    rm -rf ./orion_mnn/tools/CMakeFiles
    rm -rf ./orion_mnn/tools/cmake_install.cmake
    rm -rf ./orion_mnn/tools/Makefile
    rm -rf ./orion_mnn/CMakeFiles
    rm -rf ./orion_mnn/cmake_install.cmake
    rm -rf ./orion_mnn/Makefile
    rm -rf ./tests/CMakeFiles
    rm -rf ./tests/cmake_install.cmake
    rm -rf ./tests/Makefile
    rm -rf ./tools/converter/*
    rm -rf ./tools/CMakeFiles
    rm -rf ./tools/cmake_install.cmake
    rm -rf ./tools/Makefile
    rm -rf ./examples/CMakeFiles
    rm -rf ./examples/cmake_install.cmake
    rm -rf ./examples/Makefile
    rm -rf ./demos/CMakeFiles
    rm -rf ./demos/cmake_install.cmake
    rm -rf ./demos/Makefile
    rm -rf ./source/*
    rm -rf ./express/CMakeFiles
    rm -rf ./express/cmake_install.cmake
    rm -rf ./express/Makefile
    rm -rf ./CMakeFiles
    rm -rf ./CMakeCache.txt
    rm -rf ./cmake_install.cmake
    rm -rf ./CTestTestfile.cmake
    rm -rf ./tests/CTestTestfile.cmake
    rm -rf ./install_manifest.txt
    rm -rf ./install/*
    rm -rf ./Makefile
    rm -rf ../../../../MNN/source/backend/opencl/execution/cl/opencl_program.cc
elif 
    [ "$BUILD_CMD" == "test" ]; then
    make test
else
    echo "unknown cmd"
fi

cd ${CUR_DIR_PATH}
echo leaving ${PWD} for ${BUILD_CMD} ${PROJECT_NAME} end