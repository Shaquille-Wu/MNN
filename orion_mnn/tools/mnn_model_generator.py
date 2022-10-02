import os
import argparse
import copy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnn model generator')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frame_work',        help="CAFFE or ONNX", required=True)
    parser.add_argument('-m', '--model_file',        help="model file, it is .caffemodel for CAFFE, and it is .onnx for ONNX", required=True)
    parser.add_argument('-p', '--prototxt',          help="caffe's .prototxt file", required=False)
    parser.add_argument('-o', '--output_model_name', help="mnn model's file name", required=True)
    parser.add_argument('-s', '--src_json',          help="dlcv's json file, it can be empty", required=False)
    parser.add_argument('-c', '--mnn_cvt',           help="mnn convert program file name", required=False)
    parser.add_argument('-r', '--rewrite_prog',      help="rewrite json program file name", required=False)

    args = parser.parse_args()

    frame_work    = args.frame_work
    model_file    = args.model_file
    prototxt_file = ''
    if args.prototxt:
        prototxt_file  = args.prototxt
    output_model_name  = args.output_model_name
    src_json           = ''
    if args.src_json:
        src_json       = args.src_json
    mnn_cvt            = ''
    if args.mnn_cvt:
        mnn_cvt        = args.mnn_cvt
    rewrite_json       = ''
    if args.rewrite_prog:
        rewrite_json   = args.rewrite_prog
    
    frame_work         = frame_work.strip()
    model_file         = model_file.strip()
    prototxt_file      = prototxt_file.strip()
    output_model_name  = output_model_name.strip()
    src_json           = src_json.strip()
    mnn_cvt            = mnn_cvt.strip()
    rewrite_json       = rewrite_json.strip()

    supported_frame    = [ 'CAFFE', 'ONNX' ]

    if False == (frame_work in supported_frame):
        raise RuntimeError('frame_work is invalid: ' + frame_work)
    model_file_full_name = os.path.abspath(model_file)
    if not os.path.isfile(model_file_full_name):
        raise RuntimeError('missing src model file: ' + model_file_full_name)
    prototxt_file_full_name = os.path.abspath(prototxt_file)
    if 'CAFFE' == frame_work:
        if '' == prototxt_file:
            raise RuntimeError('prototxt file is empty')
        if not os.path.isfile(prototxt_file_full_name):
            raise RuntimeError('missing prototxt file: ' + prototxt_file_full_name)
    if '' != mnn_cvt:
        mnn_cvt_full_name = os.path.abspath(mnn_cvt)
        mnn_cvt           = mnn_cvt_full_name
        if not os.path.isfile(mnn_cvt_full_name):
            raise RuntimeError('missing mnn_cvt file: ' + mnn_cvt_full_name)
    else:
        mnn_cvt = 'MNNConvter'
    
    if '' != rewrite_json:
        rewrite_json_full_name = os.path.abspath(rewrite_json)
        rewrite_json           = rewrite_json
        if not os.path.isfile(rewrite_json_full_name):
            raise RuntimeError('missing mnn_cvt file: ' + rewrite_json_full_name)
    else:
        rewrite_json = 'rewrite_dlcv_json'

    src_json_full_name = os.path.abspath(src_json)
    if '' != src_json:
        if not os.path.isfile(src_json_full_name):
            raise RuntimeError('missing src_json file: ' + src_json_full_name)

    model_cvt_cmd_normal = ''
    model_cvt_cmd_normal = model_cvt_cmd_normal + '{} -f {} --modelFile {} '.format(mnn_cvt, frame_work, model_file_full_name)
    if 'CAFFE' == frame_work:
        model_cvt_cmd_normal = model_cvt_cmd_normal + '--prototxt {} '.format(prototxt_file_full_name)
    model_cvt_cmd        = model_cvt_cmd_normal + '--MNNModel {} --bizCode orion '.format(output_model_name)
    model_name_pos       = output_model_name.rfind('/')
    mnn_model_file_name  = output_model_name
    model_file_dir       = ''
    if model_name_pos >= 0:
        mnn_model_file_name = output_model_name[model_name_pos + 1:]
        if 0 == model_name_pos:
            model_file_dir = '/'
        else:
            model_file_dir = output_model_name[0:model_name_pos] + '/'
    
    mnn_model_base_name = mnn_model_file_name
    model_name_dot_pos  = mnn_model_file_name.rfind('.mnn')
    if model_name_dot_pos > 0:
        mnn_model_base_name = mnn_model_file_name[0:model_name_dot_pos]
    output_fp16_model_name = '{}{}_fp16.mnn'.format(model_file_dir, mnn_model_base_name)
    model_cvt_cmd_fp16     = model_cvt_cmd_normal + '--fp16 --MNNModel {} --bizCode orion'.format(output_fp16_model_name)

    print(model_cvt_cmd)
    print(model_cvt_cmd_fp16)

    os.system(model_cvt_cmd)
    os.system(model_cvt_cmd_fp16)

    core_list      = [ 'cpu', 'gpu' ]
    precision_list = [    1,     2  ]
    for cur_core in core_list:
        for cur_precision in precision_list:
            dst_json_file = '{}{}'.format(model_file_dir, mnn_model_base_name)
            cur_model     = output_model_name
            dst_json_file = dst_json_file + '_{}'.format(cur_core)
            if 2 == cur_precision:
                dst_json_file = dst_json_file + '_fp16'
                cur_model     = output_fp16_model_name
            else:
                dst_json_file = dst_json_file + '_fp32'
            dst_json_file = dst_json_file + '.mnn.json'
            rewrite_cmd   = '{} -m {} -d {} -c {} -p {} '.format(rewrite_json, cur_model, dst_json_file, cur_core, cur_precision)
            if '' != src_json:
                rewrite_cmd = rewrite_cmd + '-s {}'.format(src_json_full_name)
            os.system(rewrite_cmd)