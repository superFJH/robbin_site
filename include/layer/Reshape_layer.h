//
// Created by fjh on 17-11-3. Copyright (c) 2017 ThinkForce, Inc. All Rights Reserved
//

#ifndef MOBILE_DEEP_LEARNING_RESHAPE_LAYER_H
#define MOBILE_DEEP_LEARNING_RESHAPE_LAYER_H
#include "commons/commons.h"
#include "base/layer.h"
namespace mdl{
    class Reshape_layer :public Layer{
    public:
        Reshape_layer(const Json& config);
        ~Reshape_layer();
        std::vector<int> shape;
        void forward(int thread_num);
    };
}
#endif //MOBILE_DEEP_LEARNING_RESHAPE_LAYER_H
