//
// Created by fjh on 17-11-3. Copyright (c) 2017 ThinkForce, Inc. All Rights Reserved
//
#include "layer/Reshape_layer.h"

namespace mdl{
    Reshape_layer::Reshape_layer(const Json &config) : Layer(config){
    }
    Reshape_layer::~Reshape_layer() {}
    void Reshape_layer::forward(int thread_num) {
        _output[0]->set_data(_input[0]->get_data());
    }
}
