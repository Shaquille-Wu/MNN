## MNN的编译

在MNN根目录下配置了一个自动编译的脚本:build_mnn.sh，调用它即可完成MNN的编译工作  
1.进入MNN根目录: cd ./MNN  
2.build_mnn.sh，假如要编译基于x86_64处理器，linux操作系统的，并且是Debug版的liborion_mnn.so,，可以这样：  
```
./build_mnn.sh -c build -a x86_64 -o linux -t Debug       //编译
./build_mnn.sh -c test -a x86_64 -o linux -t Debug        //执行test
./build_mnn.sh -c clean -a x86_64 -o linux -t Debug       //clean掉生成的各种目标文件
```
执行完成后可以在这个目录找到编译生成的文件:./build/COMPILE_TYPE/PLATFORM/vision_graph  
假如要编译基于armv8处理器，android操作系统的，并且是Release版的liborion_mnn.so,，可以这样：  
```
./build_mnn.sh -c build -a armv8 -o android -t Release       //编译
./build_mnn.sh -c clean -a armv8 -o android -t Release       //clean掉生成的各种目标文件
```
3.build_graph.sh的格式: ./build_mnn.sh -c CMD -a ARCH -o OS -t COMPILE_TYPE  
CMD: build, test, clean  
ARCH: x86_64, armv8
OS:   linux, android
COMPILE_TYPE:Debug, Release  
以上要注意区分大小写  
其中目前测试有效的有-a armv8 -o android和-a x86_64 -o linux 

4.Android版本编译前需要配置cmake的toolchain，须在脚本中配置NDK的路径，比如
```
ANDROID_NDK_DIR=/home/shaquille/Android/Sdk/ndk-bundle
```
需要根据实际情况配置这个变量  
如果host机已经设定了环境变量"NDK_ROOT"，则无需填写ANDROID_NDK_DIR，脚本将以host的NDK_ROOT为准进行编译  

## 模型转换
在非移动端的编译选项下（比如x86_64-linux），编译完成后将生成转换工具“MNNConvert”，移动端的编译选项（比如armv8-android）下不会生成转换工具  
转换caffe的命令格式：
```
MNNConvert -f CAFFE --modelFile ${SRC_MODEL_FILE} --prototxt ${CAFFE_PROTO_TXT} --MNNModel ${MNNMODEL_NAME}.mnn --bizCode orion
```
假如需要gpu16，则命令为：
```
MNNConvert -f CAFFE --modelFile ${SRC_MODEL_FILE} --prototxt ${CAFFE_PROTO_TXT} --MNNModel ${MNNMODEL_NAME}.mnn --fp16 --bizCode orion
```

转换onnx的命令格式：
```
MNNConvert -f ONNX --modelFile ${SRC_MODEL_FILE} --MNNModel ${MNNMODEL_NAME}.mnn --bizCode orion
```
假如需要gpu16，则命令为：
```
MNNConvert -f ONNX --modelFile ${SRC_MODEL_FILE} --MNNModel ${MNNMODEL_NAME}.mnn --fp16 --bizCode orion
```
具体格式可以参考MNN的官方文档  
一站式转换工具，可以参考orion_mnn/tools/mnn_model_generator.py，及其相关的README.md  

## 模型转换工具的环境依赖
如果是按照上述编译步骤进行编译的，那么MNNConvert不需要任何环境依赖，其依赖的.so都在编译生成的结果中，具体依赖：
```
linux-vdso.so.1 =>  (0x00007ffda08e1000)
libMNNConvertDeps.so => /home/shaquille/WorkSpace/orion_workspace/mnn_workspace/build/Release/x86_64-linux/orion_mnn/tools/converter/libMNNConvertDeps.so (0x00007f7144dc5000)
libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f7144a43000)
libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f714482d000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f7144463000)
libMNN_Express.so => /home/shaquille/WorkSpace/orion_workspace/mnn_workspace/build/Release/x86_64-linux/orion_mnn/express/libMNN_Express.so (0x00007f71441f5000)
libprotobuf.so.22 => /home/shaquille/WorkSpace/orion_workspace/mnn_workspace/MNN/3rd_party/protobuf/x86_64-linux/lib/libprotobuf.so.22 (0x00007f7143d4d000)
libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f7143b30000)
libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f7143827000)
/lib64/ld-linux-x86-64.so.2 (0x00007f714536c000)
libMNN.so => /home/shaquille/WorkSpace/orion_workspace/mnn_workspace/build/Release/x86_64-linux/orion_mnn/libMNN.so (0x00007f71434cf000)
libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f71432b5000)
```
1)以上，除libMNNConvertDeps.so，libMNN_Express.so，libprotobuf.so.22和libMNN.so，其他都是系统依赖，每个linux都有  
2)libMNNConvertDeps.so，libMNN_Express.so，libprotobuf.so.22和libMNN.so又都在MNN的编译结果中，因此MNN的转换工具相对轻量级，暂时不需要docker环境进行转换  

## 模型量化
# MNN标准量化流程  
MNN支持fp32到in8的量化，量化工具(源码在tools/quantization目录下)，编译生成的量化工具叫quantized.out  
工具命令如下：
```
quantized.out src_model.mnn dst_int8_model.mnn quantized_cfg.json
```
src_model.mnn，即原始的fp32模型文件的路径  
dst_int8_model.mnn，即生成的int8模型文件的路径  
quantized_cfg.json，量化的配置文件，配置文件具体如下：
```
{
    "format":"RGB",
    "mean":[
        122.7717,
        115.9465,
        102.9801
    ],
    "normal":[
        1.0,
        1.0,
        1.0
    ],
    "width":320,
    "height":320,
    "path":"preprocess_image",        //用于校准的图像路径
    "used_image_num":1024,            //1000张左右的图像比较适宜
    "input_raw": true,                //若该值为true，程序将不做预处理，忽略"mean"和"normal"，直接从.raw文件中读取图像数据
    "min_quantize_threshold": 768,    //0-2047，最小不宜小于512，最大不宜超过1024
    "quantize_strategy": "Normal",    //"Normal"，"Layer_By_Layer"，"Refine"，默认"Normal"
    "feature_quantize_method":"KL",
    "weight_quantize_method":"ADMM",
    "debug": true
}
```
正常量化时，需要正确填写以上数值  
# ORION量化流程
1).图像预处理，需要根据orion的dlcv中的设定，将校准图像集中的图像转换成.raw文件，具体引用了tools/quantization/preprocess_image_batch.py文件  
   考虑到图像预处理要跟dlcv中的处理一致，所以preprocess_image_batch.py调用了dlcv的preprocess_image，这点和snpe的调用是一致的；  
2).根据实际情况填写quantized_cfg.json，需要根据实际情况填写"min_quantize_threshold"，这对量化的结果有重要影响，默认是768；  
3).调用quantized.out，生成量化模型；  
4).适当调整"min_quantize_threshold"和"quantize_strategy"，选取一个效果好的输出  
tools/quantization目录下有一个quantize_run.sh文件，用于模型的生成

## MNN 文档链接
MNN's docs are in placed in [Yuque docs here](https://www.yuque.com/mnn/en). 