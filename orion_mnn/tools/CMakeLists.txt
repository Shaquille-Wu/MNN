cmake_minimum_required (VERSION 2.8)
set(CMAKE_VERBOSE_MAKEFILE  ON)

add_executable(run_net run_net.cpp string_func.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/cpp/revertMNNModel.cpp)
target_include_directories(run_net PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../tools ${CMAKE_CURRENT_SOURCE_DIR}/../json/include)
target_link_libraries(run_net ${MNN_DEPS})

if(NOT (MSVC OR WIN32))
	add_executable(orion_mnn_run orion_mnn_run.cpp string_func.cpp)
	target_link_libraries(orion_mnn_run PRIVATE orion_mnn)
else()
	add_executable(orion_mnn_run orion_mnn_run.cpp string_func.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../orion_mnn_impl.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../tensor_convert.cpp)
	target_compile_options(orion_mnn_run PRIVATE /arch:AVX)
	target_link_libraries(orion_mnn_run PRIVATE ${MNN_DEPS})
endif()
target_include_directories(orion_mnn_run PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../json/include)

add_executable(display_tensor_shape display_tensor_shape.cpp string_func.cpp)
target_link_libraries(display_tensor_shape PRIVATE ${MNN_DEPS})

add_executable(nhwc2nc4hw4 nhwc2nc4hw4.cpp string_func.cpp)

add_executable(rewrite_dlcv_json rewrite_dlcv_json.cpp string_func.cpp)
target_include_directories(rewrite_dlcv_json PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../tools ${CMAKE_CURRENT_SOURCE_DIR}/../json/include)
target_link_libraries(rewrite_dlcv_json PRIVATE ${MNN_DEPS})

install (TARGETS run_net DESTINATION tools)
install (TARGETS orion_mnn_run DESTINATION tools)
install (TARGETS display_tensor_shape DESTINATION tools)
install (TARGETS nhwc2nc4hw4 DESTINATION tools)
install (TARGETS rewrite_dlcv_json DESTINATION tools)
