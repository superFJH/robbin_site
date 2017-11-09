/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "layer/permute.h"
namespace mdl {
    PermuteLayer::PermuteLayer(const Json &config):Layer(config) {
        for(int i=0;i<config["param"]["order"].array_items().size();i++) {
            order.push_back(config["param"]["order"].array_items()[i].int_value());
        }
    }

    PermuteLayer::~PermuteLayer() {}
    void Permute(const int count, Mtype* bottom_data, const bool forward,
                 const int* permute_order, const int* old_steps, const int* new_steps,
                 const int num_axes, Mtype* top_data) {
        for (int i = 0; i < count; ++i) {
            int old_idx = 0;
            int idx = i;
            for (int j = 0; j < num_axes; ++j) {
                int order = permute_order[j];
                old_idx += (idx / new_steps[j]) * old_steps[order];
                idx %= new_steps[j];
            }
            if (forward) {
                top_data[i] = bottom_data[old_idx];
            } else {
                bottom_data[old_idx] = top_data[i];
            }
        }
    }
    void PermuteLayer::forward(int thread_num) {
        std::vector<int> new_step,old_step;
        for(int i=0;i<_output[0]->get_dimensions().size();i++){
            if(i==_output[0]->get_dimensions().size()-1){
                new_step.push_back(1);
            }
            else{
                new_step.push_back(_output[0]->count(i+1));
            }
        }
        for(int i=0;i<_input[0]->get_dimensions().size();i++){
            if(i==_input[0]->get_dimensions().size()-1){
                old_step.push_back(1);
            }
            else{
                old_step.push_back(_input[0]->count(i+1));
            }
        }
       Permute(_output[0]->count(0),_input[0]->get_data(),true,order.data(),old_step.data(),new_step.data(),4,_output[0]->get_data());
    }
}