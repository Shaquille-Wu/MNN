## display_tensor_shape

根据指定的模型文件和core_type，输出其输入输出的Nname/shape/format等信息  
-m  model file name, ".mnn" file.  
-c  core type. 0 for cpu, 3 for gpu(opencl)  
MNN中的"NC4HW"，这种格式名义上shape的排序为NCHW，实际内存的存储方式为NHWC，其中C需要4对齐  
在windows平台上，执行命令时省去-m和-c提示符，以空格为分隔符，取第一个参数为“model file name”，第二个参数为“core type”  

## nhwc2nc4hw4
MNN为了提高其运行效率，定义了一种数据格式:NC4HW4，具体格式定义请参考MNN的官方文档  
在测试过程中，需要将一般定义的数据格式转换成NC4HW4格式，让MNN进行处理，因此提供了这个工具，将nhwc格式的数据格式转换成nc4hw4   
命令格式：
```
nhwc2nc4hw4 n:h:w:c src_file dst_file
```

## run_net
用于模型的快速评估，生成相应的profile信息，具体命令信息：  
-h  help  
-m  model file  
-l  loop count  
-w  warm-up count  
-f  forward_type, 0 for cpu, 3 for opencl, other is invalid  
-t  thread num, it useful for cpu.  
-p  precision mode, 0 for normal(fp16), 1 for high(fp32), 2 for low(fp16), default is 2  
-i  input tensor name, sperator is ":"  
-o  output tensor name, sperator is ":"  
-j  config file(dlcv style), "m:f:t:p:i:o" will be ignored, if this option is selected  
其中，需要指定input tensor name和output tensor name，在不知道模型信息的情况下，可通过display_tensor_shape获取  
-j选项被填写的时候，程序将从json文件（与dlcv定义一致）中读取各项参数，"t:p:i:o"这些选项将被忽略  
在windows平台上，执行命令时省去所有-x提示符，以空格为分隔符，并且只支持三个参数:  
第1个参数为-j的内容，指代“config file(dlcv style)”  
第2个参数为-w的内容，指代“warm-up count”  
第3个参数为-l的内容，指代“loop count”  

## orion_mnn_run
根据指定的json文件，运行相应的模型： 
-j  json file path name, .json file.  
-i  input data file. 如果不输入，则生成相应的随机数   
-w  warm-up count before model execution  
-n  execute loop count of forward  

json文件的示例：  
```
{
    "preprocess":{
        "debug": false,
        "ops":[]
    },
    "inference": {
        "debug": false,
        "engine_param":{
            "core_type":  1,          #0 is cpu, 1 is gpu
            "thread_count": 0,        #0 is auto, we should select suitable value for this field, it is very critical for cpu mode
            "precision":  2,          #0 is normal, 1 is fp32, 2 is fp16, we just select 1 and 2
            "power_mode": 1,          #0 is normal, 1 is high, 2 is low, we just select 1 and 2
            "print_tensor_shape": 0
        },
        "engine": "liborion_snpe.so",
        "model":  "face_ssdlite1_qf_0.35_r2.0_nbn.mnn",
        "inputs":  [{"name":"data",          "shape":[1, 320, 320, 3]}],
        "outputs": [
            {"name":"mbox_conf_flatten", "shape":[1, 1, 1, 3512]},
            {"name":"mbox_loc",          "shape":[1, 1, 1, 7024]}
        ]
    },
    "postprocess":{
        "debug": false,
        "ops":[]
    }
}

```
json文件只要包含dlcv定义的“inference”即可，其他都会被忽略  
在windows平台上，执行命令时省去所有-x提示符，以空格为分隔符，并且只支持4个参数:  
第1个参数为-j的内容，指代“json file path name”   
第2个参数为-i的内容，指代“input data file”  
第3个参数为-w的内容，指代“warm-up count before model execution”  
第4个参数为-n的内容，指代“execute loop count of forward”  

## rewrite_dlcv_json
根据指定的模型文件(.mnn)和json文件，生成与dlcv匹配json文件：  
-h  help.  
-m  model file.  
-s  src json file.  
-d  dst json file.   
-c  core type. cpu or gpu  
-p  precision mode. 0 for normal, 1 for high, 2 for low  

-s选项没有被填写时，程序仅生成dlcv必须的字段，包括preprocess.ops.totensor，inference和postprocess，其中postprocess仅包含一个空的框架  
-s选项被填写时，将生成与用户提供的json文件相匹配的完整的json文件  
在windows平台上，执行命令时省去所有-x提示符，以空格为分隔符，并且只支持4个参数:  
第1个参数为-m的内容，指代“model file”   
第2个参数为-s的内容，指代“src json file”  
第3个参数为-d的内容，指代“dst json file”  
第4个参数为-c的内容，指代“core type”  
第5个参数为-p的内容，指代“precision mode”  

## mnn_model_generator
这是一个py文件，根据指定的原始模型文件，生成相应的MNN模型文件和dlcv json文件：  
-f CAFFE or ONNX", required=True)  
-m model file, it is .caffemodel for CAFFE, and it is .onnx for ONNX"  
-p caffe's .prototxt file   
-o output mnn model's file name   
-s dlcv's json file, it can be empty   
-c mnn convert program file name, it can be empty   
-r rewrite json program file name, it can be empty   
该程序调用MNN的MNNConverter和上面提到的rewrite_dlcv_json，因此需要通过-c和-r选项提供这两个程序的地址，如果没有填写程序在当前工作目录查找  
需要结合MNNConverter和rewrite_dlcv_json的功能介绍，理解本程序的用法  
应用举例，假设如下：  
1).假设有caffe模型(face_det.caffemodel和face_det.prototxt)，在当前目录下生成名为face_det.mnn的模型  
2).同时提供了名为face_det.json的json文件  
3).同时假设MNNConverter和rewrite_dlcv_json都在当前工作目录下  
则命令为：
```
python mnn_model_generator.py -f CAFFE -m face_det.caffemodel -p face_det.prototxt -o face_det.mnn -s face_det.json
```
程序运行成功的话，会在-o选项指定的文件的路径下生成如下文件：  
1)face_det.mnn，32位精度的浮点模型  
2)face_det_fp16.mnn，16位精度的浮点模型  
3)face_det_cpu_fp32.mnn.json，跟32位浮点模型匹配的运行于cpu之上的json文件    
4)face_det_cpu_fp16.mnn.json，跟16位浮点模型匹配的运行于cpu之上的json文件  
5)face_det_gpu_fp32.mnn.json，跟32位浮点模型匹配的运行于gpu之上的json文件    
6)face_det_gpu_fp16.mnn.json，跟16位浮点模型匹配的运行于gpu之上的json文件   
### 注意
mnn生成的tensor的顺序与caffe和onnx都有一些差异，因此在生成的dlv.json文件中，程序自动生成的tensor的顺序未必是后处理程序所期望的顺序  
因此在执行完mnn_model_generator之后需要根据实际的后处理的规则，对output_tensor的顺序进行调整  

## mnn_walkthrough
这是一个py文件，根据指定模型信息，对模型进行一站到底的验证：  
1)各种模式下（core+precision的组合）模型在验证集上测试  
2)根据指定的图片，按照dlcv的要求生成unittest样板数据  
3)对模型在各种模式下进行profiel，生成模型的耗时统计  
用户需要指定相应的模型，json，验证集以及安卓远端的工作目录，如下所示：
```
{
    'test_exe_name'             :'detect_batch',
    'ut_result_generator_exe'   :'test_detector',
    'mnn_profile_exe'           :'mnn_lib/run_net',
    'model_folder_name'         :'face_ssdlite1_qf_0.35_r2.0',
    'model_name'                :'face_ssdlite1_qf_0.35_r2.0_nbn',
    'remote_dlcv_root'          :'/data/local/tmp/shaquille/test_dlcv',
    'remote_model_dir'          :'/data/local/tmp/shaquille/test_dlcv/model_zoo/detection',
    'local_model_dir'           :'D:/WorkSpace/OrionWorkSpace/dlcv_test/model_zoo/detection',
    'remote_test_img_set_dir'   :'/data/local/tmp/shaquille/test_dlcv/test_images/face2_val',
    'local_img_set_dir'         :'D:/face_detect/data_set/face2_val/face2_val/images',
    'ut_debug_img'              :'ssd_fp_debug.png',
    'push_test_img'             : True,
}
```
