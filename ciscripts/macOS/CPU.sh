./schema/generate.sh
mkdir macosbuild
cd macosbuild
cmake ../ -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_SUPPORT_TFLITE_QUAN=ON -DMNN_BUILD_TEST=ON -DMNN_BUILD_BENCHMARK=ON
make -j8
