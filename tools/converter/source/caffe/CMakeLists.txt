protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS
    ${CMAKE_CURRENT_LIST_DIR}/caffe.proto
)
file(GLOB CAFFE_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(MNNConverterCaffe OBJECT ${CAFFE_SRC} ${CAFFE_PROTO_SRCS})
target_include_directories(MNNConverterCaffe PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(MNNConverterCaffe MNNCompress)
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterCaffe>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterCaffe)
