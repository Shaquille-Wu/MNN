import os


def generate_walkthrough_script(local_model_dir,
                                model_folder_name,
                                remote_model_dir,
                                model_name,
                                remote_image_set_dir,
                                core_type_list,
                                precision_type_list,
                                test_exe_name,
                                batch_script_file_name,
                                ut_result_generate_exe,
                                ut_result_generate_script_file_name,
                                ut_debug_image_name,
                                profile_exe,
                                profile_scripte_file_name):
    fw_batch = open(batch_script_file_name, 'wb')
    fw_batch.write('#!system/bin/sh\n\n')
    fw_batch.write('cur_work_space=$1\n')
    fw_batch.write('if [[ \"\" != $cur_work_space ]]; then\n')
    fw_batch.write('    cd $cur_work_space\n')
    fw_batch.write('fi\n')
    fw_batch.write('TEST_EXE={}\n'.format(test_exe_name))
    fw_batch.write('IMG_DIR={}\n'.format(remote_image_set_dir))
    fw_batch.write('MODEL_FOLDER_NAME={}\n'.format(model_folder_name))
    fw_batch.write('\n')
    
    fw_utrg = open(ut_result_generate_script_file_name, 'wb')
    fw_utrg.write('#!system/bin/sh\n\n')
    fw_utrg.write('cur_work_space=$1\n')
    fw_utrg.write('if [[ \"\" != $cur_work_space ]]; then\n')
    fw_utrg.write('    cd $cur_work_space\n')
    fw_utrg.write('fi\n')
    fw_utrg.write('TEST_EXE={}\n'.format(ut_result_generate_exe))
    fw_utrg.write('IMG_FILE={}/{}\n'.format(remote_model_dir, ut_debug_image_name))
    fw_utrg.write('MODEL_FOLDER_NAME={}\n'.format(model_folder_name))
    fw_utrg.write('\n')

    fw_mnn_profile = open(profile_scripte_file_name, 'wb')
    fw_mnn_profile.write('#!system/bin/sh\n\n')
    fw_mnn_profile.write('cur_work_space=$1\n')
    fw_mnn_profile.write('if [[ \"\" != $cur_work_space ]]; then\n')
    fw_mnn_profile.write('    cd $cur_work_space\n')
    fw_mnn_profile.write('fi\n')
    fw_mnn_profile.write('TEST_EXE={}\n'.format(profile_exe))
    fw_mnn_profile.write('MODEL_FOLDER_NAME={}\n'.format(model_folder_name))
    fw_mnn_profile.write('\n')

    local_model_folder           = '{}/mnn'.format(local_model_dir)
    local_model_folder_full_path = os.path.abspath(local_model_folder)
    if not os.path.isdir(local_model_folder_full_path):
        print('cannot find mnn in model directory: {}'.format(local_model_dir))
        return

    fw_batch.write('MODEL_NAME={}\n'.format(model_name))
    fw_utrg.write('MODEL_NAME={}\n'.format(model_name))
    fw_mnn_profile.write('MODEL_NAME={}\n'.format(model_name))
    fw_batch.write('BATCH_RESULT_DIR=batch_result/${MODEL_NAME}\n')
    fw_batch.write('mkdir -p ${BATCH_RESULT_DIR}\n')
    export_android_so_string  = 'export LD_LIBRARY_PATH=${PWD}/mnn_lib\n'
    fw_batch.write(export_android_so_string)
    fw_utrg.write(export_android_so_string)
    fw_mnn_profile.write(export_android_so_string)

    fw_mnn_profile.write('rm -rf profile_result.txt\n')
    fw_mnn_profile.write('rm -rf profile_result.tmp.txt\n')
    for i in range(len(core_type_list)):
        for j in range(len(precision_type_list)):
            json_file_line = 'JSON_FILE={}/mnn/{}_{}_{}.mnn.json\n'.format(remote_model_dir, model_name, core_type_list[i], precision_type_list[j])
            
            fw_mnn_profile.write('echo \"generate profile for {}_{} for ${{MODEL_NAME}}\"\n'.format(core_type_list[j], precision_type_list[j]))
            fw_mnn_profile.write(json_file_line)
            fw_mnn_profile.write('./${TEST_EXE} -j ${JSON_FILE} -w 10 -l 100 -n 10\n')
            fw_mnn_profile.write('cat profile_result.txt >> profile_result.tmp.txt\n')
            fw_mnn_profile.write('\n')
            
            fw_batch.write('echo \"start test in {}_{} for ${{MODEL_NAME}}\"\n'.format(core_type_list[i], precision_type_list[j]))
            fw_batch.write(json_file_line)
            fw_batch.write('./${TEST_EXE} -j ${JSON_FILE} -i ${IMG_DIR}\n')
            fw_batch.write('cp ./batch_result.txt ./${{BATCH_RESULT_DIR}}/batch_result_{}_{}.txt\n'.format(core_type_list[i], precision_type_list[j]))
            fw_batch.write('\n')
            
            fw_utrg.write('echo \"generate result in {}_{} for ${{MODEL_NAME}}\"\n'.format(core_type_list[j], precision_type_list[j]))
            fw_utrg.write(json_file_line)
            fw_utrg.write('./${TEST_EXE} ${JSON_FILE} ${IMG_FILE}\n')
            fw_utrg.write('cp ./unit_test_result.json {}/mnn/unit_test_result_{}_{}.json\n'.format(remote_model_dir, core_type_list[i], precision_type_list[j]))
            fw_utrg.write('\n')
        fw_batch.write('\n')
        fw_utrg.write('\n')
        fw_mnn_profile.write('\n')
    fw_mnn_profile.write('mv profile_result.tmp.txt {}/mnn/profile_result.txt\n'.format(remote_model_dir))
    fw_batch.close()
    fw_utrg.close()
    fw_mnn_profile.close()

def main():
    #current batch test:
    #    detect_batch
    #    multiclass_batch
    #    reidfeature_batch
    
    mnn_walkthrough_param   = [
#        {
#            'test_exe_name'             :'detect_batch',
#            'ut_result_generator_exe'   :'test_detector',
#            'mnn_profile_exe'           :'mnn_lib/run_net',
#            'model_folder_name'         :'face_ssdlite1_qf_0.35_r2.0',
#            'model_name'                :'face_ssdlite1_qf_0.35_r2.0_nbn',
#            'remote_dlcv_root'          :'/data/local/tmp/shaquille/test_dlcv',
#            'remote_model_dir'          :'/data/local/tmp/shaquille/test_dlcv/model_zoo/detection',
#            'local_model_dir'           :'D:/WorkSpace/OrionWorkSpace/dlcv_test/model_zoo/detection',
#            'remote_test_img_set_dir'   :'/data/local/tmp/shaquille/test_dlcv/test_images/face2_val',
#            'local_img_set_dir'         :'D:/face_detect/data_set/face2_val/face2_val/images',
#            'ut_debug_img'              :'ssd_fp_debug.png',
#            'push_test_img'             : False,
#        },

        {
            'test_exe_name'             :'multiclass_batch',
            'ut_result_generator_exe'   :'',
            'mnn_profile_exe'           :'mnn_lib/run_net',
            'model_folder_name'         :'face_age_gender_1115_mobile_student_lr1_96',
            'model_name'                :'face_age_gender_1115_mobile_student_lr1_96',
            'remote_dlcv_root'          :'/data/local/tmp/shaquille/test_dlcv',
            'remote_model_dir'          :'/data/local/tmp/shaquille/test_dlcv/model_zoo/multiclass_attribute',
            'local_model_dir'           :'D:/WorkSpace/OrionWorkSpace/dlcv_test/model_zoo/multiclass_attribute',
            'remote_test_img_set_dir'   :'/data/local/tmp/shaquille/test_dlcv/test_images/face_age_gender_1115_mobile_student_lr1_96',
            'local_img_set_dir'         :'D:/WorkSpace/data_set/age_gender/face_age_gender_1115_mobile_student_lr1_96/dataset_168_data/images',
            'ut_debug_img'              :'',
            'push_test_img'             : True,
        }
    ]    

    model_count                   = len(mnn_walkthrough_param)
    for m in range(0, model_count):
        mnn_walkthrough_param[m]['remote_model_dir'] = mnn_walkthrough_param[m]['remote_model_dir'] + '/' + mnn_walkthrough_param[m]['model_folder_name']
        mnn_walkthrough_param[m]['local_model_dir']  = mnn_walkthrough_param[m]['local_model_dir']  + '/' + mnn_walkthrough_param[m]['model_folder_name']

    #based on above setting, we should figure out something:
    #in this case, the local image dir name as "images",
    #we should notice, this folder will be xfer into remote, and this "images" will be in "remote_test_img_set_dir"
    #so, in remote, the test image set will be located at /data/local/tmp/shaquille/test_dlcv/test_images/mono_bp_val_4.0/images

    core_type_list      = [ 'cpu',  'gpu' ]
    precision_type_list = [ 'fp32', 'fp16' ]
    for m in range(0, model_count):
        cur_model               = mnn_walkthrough_param[m]
        mkdir_model_folder_cmd  = 'adb shell mkdir -p {}'.format(cur_model['remote_model_dir'])
        push_model_cmd          = 'adb push {}/mnn {}/.'.format(cur_model['local_model_dir'], cur_model['remote_model_dir'])
        push_ut_image_cmd       = ''
        if '' != cur_model['ut_debug_img']:
            push_ut_image_cmd   = 'adb push {}/{} {}/.'.format(cur_model['local_model_dir'], cur_model['ut_debug_img'], cur_model['remote_model_dir'])
        push_image_cmd          = 'adb push {} {}/.'.format(cur_model['local_img_set_dir'], cur_model['remote_test_img_set_dir'])

        #we can skip these cmds for the walkthrought_script_file's generation, if we have a test script
        walkthrough_script_file         = 'test_{}.sh'.format(cur_model['model_folder_name'])
        ut_result_generate_script_file  = 'ut_result_generate_{}.sh'.format(cur_model['model_folder_name'])
        profile_generate_script_file    = 'profile_generate_{}.sh'.format(cur_model['model_folder_name'])
        generate_walkthrough_script(cur_model['local_model_dir'], 
                                    cur_model['model_folder_name'], 
                                    cur_model['remote_model_dir'], 
                                    cur_model['model_name'],                               
                                    cur_model['remote_test_img_set_dir'] + '/images',
                                    core_type_list, 
                                    precision_type_list,
                                    cur_model['test_exe_name'], 
                                    walkthrough_script_file,
                                    cur_model['ut_result_generator_exe'],
                                    ut_result_generate_script_file,
                                    cur_model['ut_debug_img'],
                                    cur_model['mnn_profile_exe'],
                                    profile_generate_script_file)
        pull_batch_result_cmd         = 'adb pull {}/batch_result/{} batch_result/.'.format(cur_model['remote_dlcv_root'], cur_model['model_name'])
        os.system('adb wait-for-device')
        os.system('adb root')
        os.system('adb wait-for-device')
        os.system('adb remount')
        os.system('adb wait-for-device')
        os.system(mkdir_model_folder_cmd)
        os.system('adb wait-for-device')
        os.system(push_model_cmd)
        if '' != push_ut_image_cmd:
            os.system(push_ut_image_cmd)
        os.system('adb wait-for-device shell mkdir -p {}'.format(cur_model['remote_test_img_set_dir']))
        if True == cur_model['push_test_img']:
            os.system(push_image_cmd)
        os.system('adb wait-for-device')

        if '' != cur_model['mnn_profile_exe']:
            print(profile_generate_script_file)
            push_profile_script_cmd = 'adb push {} {}/.'.format(profile_generate_script_file, cur_model['remote_dlcv_root'])
            profile_execute_cmd     = 'adb wait-for-device shell {}/{} {}'.format(cur_model['remote_dlcv_root'], profile_generate_script_file, cur_model['remote_dlcv_root'])
            os.system('adb wait-for-device')
            os.system(push_profile_script_cmd)
            os.system('adb wait-for-device')
            os.system('adb wait-for-device shell chmod +x {}/{}'.format(cur_model['remote_dlcv_root'], profile_generate_script_file))
            os.system(profile_execute_cmd)
            os.system('adb wait-for-device')
            
            profile_file_remote           = cur_model['remote_model_dir'] + '/mnn/profile_result.txt'
            profile_result_file_local     = cur_model['local_model_dir']  + '/mnn/profile_result.txt'
            tuning_file_remote_fp32       = cur_model['remote_model_dir'] + '/mnn/' + cur_model['model_name'] + '.mnn_cache'
            tuning_result_file_local_fp32 = cur_model['local_model_dir']  + '/mnn/' + cur_model['model_name'] + '.mnn_cache'
            tuning_file_remote_fp16       = cur_model['remote_model_dir'] + '/mnn/' + cur_model['model_name'] + '_fp16.mnn_cache'
            tuning_result_file_local_fp16 = cur_model['local_model_dir']  + '/mnn/' + cur_model['model_name'] + '_fp16.mnn_cache'
            pull_profile_reuslt_cmd       = 'adb pull {} {}'.format(profile_file_remote, profile_result_file_local)
            pull_tuning_fp32_reuslt_cmd   = 'adb pull {} {}'.format(tuning_file_remote_fp32, tuning_result_file_local_fp32)
            pull_tuning_fp16_reuslt_cmd   = 'adb pull {} {}'.format(tuning_file_remote_fp16, tuning_result_file_local_fp16)
            os.system(pull_profile_reuslt_cmd)
            os.system(pull_tuning_fp32_reuslt_cmd)
            os.system(pull_tuning_fp16_reuslt_cmd)
            os.system('adb wait-for-device')
            if not os.path.isfile(profile_result_file_local):
                print('warning, cannot find ' + profile_result_file_local)
            if not os.path.isfile(profile_result_file_local):
                print('warning, cannot find ' + tuning_result_file_local_fp32)
            if not os.path.isfile(tuning_result_file_local_fp16):
                print('warning, cannot find ' + tuning_result_file_local_fp16)

        push_walkthrought_script_cmd = 'adb push {} {}/.'.format(walkthrough_script_file, cur_model['remote_dlcv_root'])
        os.system(push_walkthrought_script_cmd)
        os.system('adb wait-for-device')
        os.system('adb wait-for-device shell chmod +x {}/{}'.format(cur_model['remote_dlcv_root'], walkthrough_script_file))
        script_execute_cmd = 'adb wait-for-device shell {}/{} {}'.format(cur_model['remote_dlcv_root'], walkthrough_script_file, cur_model['remote_dlcv_root'])

        os.system(script_execute_cmd)
        os.system('adb wait-for-device')
        os.system('mkdir -p ./batch_result')   
        os.system(pull_batch_result_cmd)
        os.system('adb wait-for-device')

        for i in range(len(core_type_list)):
            for j in range(len(precision_type_list)):
                cur_batch_result_file = './batch_result/{}/batch_result_{}_{}.txt'.format(cur_model['model_name'], core_type_list[i], precision_type_list[j])
                if not os.path.isfile(cur_batch_result_file):
                    print('warning, cannot find ' + cur_batch_result_file)
        
        if '' != cur_model['ut_result_generator_exe'] and '' != cur_model['ut_debug_img']:
            print(ut_result_generate_script_file)
            push_ut_image_cmd    = 'adb push {}/{} {}/.'.format(cur_model['local_model_dir'], cur_model['ut_debug_img'], cur_model['remote_model_dir'])
            push_utrg_script_cmd = 'adb push {} {}/.'.format(ut_result_generate_script_file, cur_model['remote_dlcv_root'])
            utrg_execute_cmd     = 'adb wait-for-device shell {}/{} {}'.format(cur_model['remote_dlcv_root'], ut_result_generate_script_file, cur_model['remote_dlcv_root'])
            os.system('adb wait-for-device')
            os.system(push_ut_image_cmd)
            os.system(push_utrg_script_cmd)
            os.system('adb wait-for-device')
            os.system('adb wait-for-device shell chmod +x {}/{}'.format(cur_model['remote_dlcv_root'], ut_result_generate_script_file))
            os.system(utrg_execute_cmd)
            os.system('adb wait-for-device')
            for i in range(len(core_type_list)):
                for j in range(len(precision_type_list)):
                    cur_ut_result_file_remote = cur_model['remote_model_dir'] + '/mnn/unit_test_result_{}_{}.json'.format(core_type_list[i], precision_type_list[j])
                    cur_ut_result_file_local  = cur_model['local_model_dir']  + '/mnn/unit_test_result_{}_{}.json'.format(core_type_list[i], precision_type_list[j])
                    pull_unit_test_reuslt_cmd = 'adb pull {} {}'.format(cur_ut_result_file_remote, cur_ut_result_file_local)
                    os.system(pull_unit_test_reuslt_cmd)
                    if not os.path.isfile(cur_ut_result_file_local):
                        print('warning, cannot find ' + cur_ut_result_file_local)
       
if __name__ == '__main__':
    main()