//
//  MobilenetV2Utils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MobilenetV2Utils.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include <MNN/expr/NN.hpp>
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "ImageDataset.hpp"
#include "module/PipelineModule.hpp"
#include "MergeQuantRedundantOp.hpp"
#include <fstream>
#include <sstream>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void MobilenetV2Utils::train(std::shared_ptr<Module> model, const int numClasses, const int addToLabel,
                                std::string trainImagesFolder, std::string trainImagesTxt,
                                std::string testImagesFolder, std::string testImagesTxt,
                                const int trainQuantDelayEpoch, const int quantBits) {
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 2);
    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setMomentum(0.9f);
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(0.00004f);

    auto converImagesToFormat  = CV::RGB;
    int resizeHeight           = 224;
    int resizeWidth            = 224;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
    std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
    bool centerOrRandomCrop = false; // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means,cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto trainDataset = ImageDataset::create(trainImagesFolder, trainImagesTxt, datasetConfig.get(), readAllImagesToMemory);
    auto testDataset = ImageDataset::create(testImagesFolder, testImagesTxt, datasetConfig.get(), readAllImagesToMemory);

    const int trainBatchSize = 32;
    const int trainNumWorkers = 4;
    const int testBatchSize = 10;
    const int testNumWorkers = 0;

    auto trainDataLoader = trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);
    auto testDataLoader = testDataset.createLoader(testBatchSize, true, false, testNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();
    const int testIterations = testDataLoader->iterNumber();

    // const int usedSize = 1000;
    // const int testIterations = usedSize / testBatchSize;

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            trainDataLoader->reset();
            model->setIsTraining(true);
            // turn float model to quantize-aware-training model after a delay
            if (epoch == trainQuantDelayEpoch) {
                // turn model to train quant model
                std::static_pointer_cast<PipelineModule>(model)->toTrainQuant(quantBits);
            }
            for (int i = 0; i < trainIterations; i++) {
                AUTOTIME;
                auto trainData  = trainDataLoader->next();
                auto example    = trainData[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                                  _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(_Convert(example.first[0], NC4HW4));
                auto loss    = _CrossEntropy(predict, newTarget);
                // float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                float rate = 1e-5;
                solver->setLearningRate(rate);
                if (solver->currentStep() % 10 == 0) {
                    std::cout << "train iteration: " << solver->currentStep();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate << std::endl;
                }
                solver->step(loss);
            }
        }

        int correct = 0;
        int sampleCount = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        exe->gc(Executor::PART);

        AUTOTIME;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            auto predict    = model->forward(_Convert(example.first[0], NC4HW4));
            predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
            auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
            sampleCount += label->getInfo()->size;
            auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));
            correct += accu->readMap<int32_t>()[0];

            if ((i + 1) % 10 == 0) {
                std::cout << "test iteration: " << (i + 1) << " ";
                std::cout << "acc: " << correct << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%";
                std::cout << std::endl;
            }
        }
        auto accu = (float)correct / testDataLoader->size();
        // auto accu = (float)correct / usedSize;
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;

        {
            auto forwardInput = _Input({1, 3, resizeHeight, resizeWidth}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            Transformer::turnModelToInfer()->onExecute({predict});
            predict->setName("prob");
            Variable::save({predict}, "temp.mobilenetv2.mnn");
        }

        exe->dumpProfile();
    }
}

//Shaquille, Added 20201129 Start
static VARP CreateOrionOneHot(Express::VARP& annotate)
{
    const Variable::Info* annotate_info  = annotate->getInfo();
    INTS                  annotate_dim   = annotate_info->dim;
    const int*            annotate_data  = annotate->readMap<int>();

    Variable::Info        one_hot_info;
    int                   batch_cnt          = annotate_dim[0];
    int                   src_channel_cnt    = 5;
    one_hot_info.order   = annotate_info->order;
    one_hot_info.dim     = annotate_dim ;
    one_hot_info.dim[1]  = src_channel_cnt;
    one_hot_info.type    = halide_type_t(halide_type_float, 32, 1);
    one_hot_info.size    = annotate_dim[0] * src_channel_cnt;

    float*  one_hot_data = new float[one_hot_info.size];
    memset(one_hot_data, 0, one_hot_info.size * sizeof(float));
    for(int i = 0 ; i < annotate_dim[0] ; i ++)
    {
        int  age_idx    = annotate_data[2 * i];
        int  gender_idx = annotate_data[2 * i + 1];
        one_hot_data[i * src_channel_cnt + age_idx]  = 1.0f;
        one_hot_data[i * src_channel_cnt + 4]        = (0 == gender_idx ? 0.0f : 1.0f);
    }

    EXPRP   one_hot_expr  = Expr::create(std::move(one_hot_info), one_hot_data, VARP::INPUT, Expr::COPY);
    VARP    one_hot_var   = Variable::create(one_hot_expr, 0);
    delete[] one_hot_data;

    return one_hot_var;
}

static VARP CreateOrionPredict(Express::VARP& model_predict, bool soft_max)
{
    const Variable::Info* model_predict_info = model_predict->getInfo();
    INTS                  model_predict_dim  = model_predict_info->dim;
    const float*          model_predict_res  = model_predict->readMap<float>();

    Variable::Info        new_predict_info;
    int                   batch_cnt          = model_predict_dim[0];
    int                   src_channel_cnt    = model_predict_dim[1];
    new_predict_info.order   = model_predict_info->order;
    new_predict_info.dim     = model_predict_dim ;
    new_predict_info.dim[1]  = 4;
    new_predict_info.type    = model_predict_info->type;
    new_predict_info.size    = model_predict_dim[0] * 4;
    float*  new_predict_data = new float[new_predict_info.size];
    memset(new_predict_data, 0, new_predict_info.size * sizeof(float));
    int   src_channel_4 = (((src_channel_cnt + 3) >> 2) << 2);
    for(int i = 0 ; i < model_predict_dim[0] ; i ++)
    {
        if(true == soft_max)
        {
            int    j       = 0;
            float  cur_max = model_predict_res[i * src_channel_4];
            for(j = 1 ; j < 4 ; j ++)
            {
                if(cur_max < model_predict_res[i * src_channel_4 + j])
                    cur_max = model_predict_res[i * src_channel_4 + j];
            }

            float  sum     = 0.0f;
            float  sum_inv = 0.0f;
            for(j = 0 ; j < 4 ; j ++)
            {
                float cur_delta = model_predict_res[i * src_channel_4 + j] - cur_max;
                float cur_res   = expf(cur_delta);
                new_predict_data[i * 4 + j]  = cur_res;
                sum                         += cur_res;
            }
            sum_inv = 1.0f/sum;
            for(j = 0 ; j < 4 ; j ++)
            {
                new_predict_data[i * 4 + j]  = new_predict_data[i * 4 + j] * sum_inv;
            }
        }
        else
        {
            for(int j = 0 ; j < 4 ; j ++)
                new_predict_data[i * 4 + j]  = model_predict_res[i * src_channel_4 + j];
        }
    }
    EXPRP new_predict_expr  = Expr::create(std::move(new_predict_info), new_predict_data, VARP::CONSTANT, Expr::COPY);
    VARP  new_predict       = Variable::create(new_predict_expr, 0);
    delete[] new_predict_data;
    return new_predict;
}

static VARP OrionArgMax(Express::VARP& model_predict)
{
    const Variable::Info* model_predict_info = model_predict->getInfo();
    INTS                  model_predict_dim  = model_predict_info->dim;
    const float*          model_predict_res  = model_predict->readMap<float>();

    Variable::Info        new_predict_info;
    int                   batch_cnt          = model_predict_dim[0];
    int                   src_channel_cnt    = model_predict_dim[1];
    new_predict_info.order   = model_predict_info->order;
    new_predict_info.dim     = { model_predict_dim[0] } ;
    new_predict_info.type    = halide_type_t(halide_type_int, 32);
    new_predict_info.size    = model_predict_dim[0];
    int*  new_predict_data = new int[new_predict_info.size];
    memset(new_predict_data, 0, new_predict_info.size * sizeof(int));
    for(int i = 0 ; i < model_predict_dim[0] ; i ++)
    {
        int    j       = 0;
        float  cur_max = model_predict_res[i * src_channel_cnt];
        int    max_idx = 0;
        for(j = 1 ; j < 4 ; j ++)
        {
            if(cur_max < model_predict_res[i * src_channel_cnt + j])
            {
                max_idx = j;
                cur_max = model_predict_res[i * src_channel_cnt + j];
            } 
            new_predict_data[i] = max_idx;
        }
    }
    EXPRP new_predict_expr  = Expr::create(std::move(new_predict_info), new_predict_data, VARP::CONSTANT, Expr::COPY);
    VARP  new_predict       = Variable::create(new_predict_expr, 0);
    delete[] new_predict_data;
    return new_predict;
}

static void statistic_orion_acc(Express::VARP const& model_predict, Express::VARP const& label, int& sampleCount, int& age_correct, int& gender_correct, int& correct)
{
    const Variable::Info* predict_info    = model_predict->getInfo();
    INTS                  predict_dim     = predict_info->dim;
    const float*          predict_res     = model_predict->readMap<float>();
    int                   batch_cnt       = predict_dim[0];

    int const*            label_data          = label->readMap<int>();

    int                   age_ok                    = 0;
    int                   gender_ok                 = 0;
    int                   total_ok                  = 0;
    int                   predict_channel_4         = (((predict_dim[1] + 3) >> 2) << 2);

    sampleCount += batch_cnt;
    for(int i = 0 ; i < batch_cnt ; i ++)
    {
        int     age_idx     = label_data[2 * i];
        int     gender_idx  = label_data[2 * i + 1];

        float   age_max     = predict_res[i * predict_channel_4];
        int     age_max_idx = 0;
        for(int j = 1; j < 4 ; j ++)
        {
            if(age_max < predict_res[i * predict_channel_4 + j])
            {
                age_max     = predict_res[i * predict_channel_4 + j];
                age_max_idx = j;
            }
        }

        int     gender_sel = (predict_res[i * predict_channel_4 + 4] > 0.5f ? 1 : 0) ;
        if(age_max_idx == age_idx && gender_idx == gender_sel)
        {
            total_ok  += 1;
            age_ok    += 1;
            gender_ok += 1;
        }
        else if(age_max_idx == age_idx)
        {
            age_ok    += 1;
        }
        else if(gender_idx == gender_sel)
        {
            gender_ok += 1;
        }
    }
    
    age_correct    += age_ok;
    gender_correct += gender_ok;
    correct        += total_ok;
}

void MobilenetV2Utils::orion_train(std::shared_ptr<Module> model, const int numClasses, const int addToLabel,
                                   std::string trainImagesFolder, std::string trainImagesTxt,
                                   std::string testImagesFolder, std::string testImagesTxt,
                                   const int trainQuantDelayEpoch, const int quantBits) {
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 4);
    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setMomentum(0.9f);
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(0.00004f);

    auto converImagesToFormat  = CV::RGB;
    int resizeHeight           = 224;
    int resizeWidth            = 224;
    //std::vector<float> means = {127.5, 127.5, 127.5};
    //std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
    //std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
    std::vector<float> means        = { 103.53f, 116.28f, 123.675f } ;
    std::vector<float> scales       = { 1.0f/57.375f, 1.0f/57.12f, 1.0f/58.395f};
    std::vector<float> cropFraction = { 1.0f, 1.0f };
    bool centerOrRandomCrop = false; // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means,cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto trainDataset = ImageDataset::create(trainImagesFolder, trainImagesTxt, datasetConfig.get(), readAllImagesToMemory);
    auto testDataset = ImageDataset::create(testImagesFolder, testImagesTxt, datasetConfig.get(), readAllImagesToMemory);

    const int trainBatchSize  = 40;
    const int trainNumWorkers = 4;
    const int testBatchSize   = 10;
    const int testNumWorkers  = 0;

    auto trainDataLoader      = trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);
    auto testDataLoader       = testDataset.createLoader(testBatchSize, true, false, testNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();
    const int testIterations  = testDataLoader->iterNumber();

    // const int usedSize = 1000;
    // const int testIterations = usedSize / testBatchSize;

    for (int epoch = 0; epoch < 2; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
#if 1
        {
            AUTOTIME;
            trainDataLoader->reset();
            model->setIsTraining(true);
            // turn float model to quantize-aware-training model after a delay
            if (epoch == trainQuantDelayEpoch) {
                // turn model to train quant model
                std::static_pointer_cast<PipelineModule>(model)->toTrainQuant(quantBits);
            }
            for (int i = 0; i < trainIterations; i++) {
                AUTOTIME;
                auto trainData  = trainDataLoader->next();
                auto example    = trainData[0];
                // Compute One-Hot
/*
                auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                                  _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));
*/
                auto newTarget   = CreateOrionOneHot(example.second[0]);
                auto predict     = model->forward(_Convert(example.first[0], NC4HW4));
/*
                {
                    const Variable::Info* newTarget_info   = newTarget->getInfo();
                    INTS                  newTarget_dim    = newTarget_info->dim;
                    const float*          newTarget_res    = newTarget->readMap<float>();
                    const Variable::Info* new_predict_info = predict->getInfo();
                    INTS                  new_predict_dim  = new_predict_info->dim;
                    const float*          new_predict_res  = predict->readMap<float>();
                    float loss_sum = 0.0f;
                    for(int i = 0 ; i < newTarget_dim[0] ; i ++)
                    {
                        for(int j = 0 ; j < 5 ; j ++)
                        {
                            float cur_prob   = new_predict_res[8 * i + j];
                            float cur_flag   = newTarget_res[5 * i + j];
                            loss_sum      += (cur_prob - cur_flag) * (cur_prob - cur_flag);
                        }
                    }
                    printf("cur_loss_sum %.6f, %.6f\n", loss_sum, loss_sum / (float)(newTarget_dim[0])) ;
                }
*/
                auto loss    = _BCE_CrossEntropy(predict, newTarget);
                //auto loss         = _MSE(predict, newTarget);
                //float rate   = LrScheduler::inv(1e-6, solver->currentStep(), 1e-4, 0.75);
                float rate = 1e-6;
                solver->setLearningRate(rate);
                //if (solver->currentStep() % 10 == 0) {
                    std::cout << "train iteration: " << solver->currentStep();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate << std::endl;
                //}
                bool res = solver->step(loss);
                //printf("step.res %d\n", res);
            }
        }
#endif
        int age_correct = 0;
        int gender_correct = 0;
        int correct = 0;
        int sampleCount = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        exe->gc(Executor::PART);

        printf("start test\n");
        AUTOTIME;
        for (int i = 0; i < testIterations; i++) {
            auto data        = testDataLoader->next();
            auto example     = data[0];
            auto predict     = model->forward(_Convert(example.first[0], NC4HW4));

            statistic_orion_acc(predict, example.second[0], sampleCount, age_correct, gender_correct, correct);
            /*
            auto new_predict = CreateOrionPredict(predict, false);
            new_predict      = OrionArgMax(new_predict); // (N, numClasses) --> (N)
            const Variable::Info* new_predict_info1 = new_predict->getInfo();
            auto label       = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
            sampleCount     += label->getInfo()->size;
            auto accu        = _Equal(new_predict, label).sum({});
            const Variable::Info* accu_info = accu->getInfo();
            correct         += accu->readMap<int32_t>()[0];
            */
            if ((i + 1) % 10 == 0) {
                std::cout << "test iteration: " << (i + 1) << std::endl;
                std::cout << "age_acc:    " << age_correct    << "/" << sampleCount << " = " << float(age_correct) / sampleCount * 100 << "%" << std::endl;
                std::cout << "gender_acc: " << gender_correct << "/" << sampleCount << " = " << float(gender_correct) / sampleCount * 100 << "%" << std::endl;
                std::cout << "acc:        " << correct        << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%" << std::endl;
            }
        }
        auto accu = (float)correct / testDataLoader->size();
        // auto accu = (float)correct / usedSize;
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;

        {
            auto forwardInput = _Input({1, 3, resizeHeight, resizeWidth}, NC4HW4);
            forwardInput->setName("data");
            auto predict     = model->forward(forwardInput);
            Transformer::turnModelToInfer()->onExecute({predict});
            predict->setName("prob");

            std::string output_quantized_mnn = "temp.mobilenetv2.mnn";
            Variable::save({predict}, output_quantized_mnn.c_str());

            {
                std::unique_ptr<MNN::NetT> netT;
                {
                    std::ifstream input(output_quantized_mnn);
                    std::ostringstream outputOs;
                    outputOs << input.rdbuf();
                    netT = MNN::UnPackNet(outputOs.str().c_str());
                }

                // temp build net for inference
                flatbuffers::FlatBufferBuilder builder(1024);
                auto offset = MNN::Net::Pack(builder, netT.get());
                builder.Finish(offset);
                int size      = builder.GetSize();
                auto ocontent = builder.GetBufferPointer();

                // model buffer for creating mnn Interpreter
                std::unique_ptr<uint8_t> modelForInference(new uint8_t[size]);
                memcpy(modelForInference.get(), ocontent, size);

                std::unique_ptr<uint8_t> modelOriginal(new uint8_t[size]);
                memcpy(modelOriginal.get(), ocontent, size);

                netT.reset();
                netT = MNN::UnPackNet(modelOriginal.get());

                MergeQuantRedundantOp::Merge(netT.get());

                flatbuffers::FlatBufferBuilder builderOutput(1024);
                builderOutput.ForceDefaults(true);
                auto len = MNN::Net::Pack(builderOutput, netT.get());
                builderOutput.Finish(len);
                {
                    std::ofstream output(output_quantized_mnn);
                    output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
                }
            }
        }

        exe->dumpProfile();
    }
}
//Shaquille, Added 20201129 End