//
// Created by fjh on 17-11-3. Copyright (c) 2017 ThinkForce, Inc. All Rights Reserved
//

#ifndef MOBILE_DEEP_LEARNING_NORMALIZE_H
#define MOBILE_DEEP_LEARNING_NORMALIZE_H
#include "commons/commons.h"
#include "base/layer.h"

namespace mdl{
    class Normalize_layer : public Layer {
    public:
        int value;
        bool across_spatial;
        bool channel_shared;
        Normalize_layer(const Json& config);
        ~Normalize_layer();
        void forward(int threadint);
    };
}
#endif //MOBILE_DEEP_LEARNING_NORMALIZE_H
