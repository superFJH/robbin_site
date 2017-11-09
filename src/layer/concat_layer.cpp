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
#include "layer/concat_layer.h"

namespace mdl {
    ConcatLayer::ConcatLayer(const Json &config): Layer(config) {
        assure_memory();
        _layer_type = LayerType::CONCAT;
        axis=(config["param"]["axis"].is_null())?-1:config["param"]["axis"].int_value();
    }

    void ConcatLayer::forward(int thread_num) {
        if (_input.size() == 1) {
            return;
        }
        if (axis == -1) {
            int offset = 0;
            int output_offset = _output[0]->dimension(1);
            Mtype *output_data = _output[0]->get_data();
            int count = _input[0]->count(0, 1);
            int input_size = _input[0]->count(2);
            for (auto &src: _input) {
                int input_offset = src->dimension(1);
                Mtype *input_data = src->get_data();
                for (int i = 0; i < count; i++) {
                    Mtype *input_data_start = input_data + i * input_offset * input_size;
                    std::copy(input_data_start, input_data_start + input_offset * input_size,
                              output_data + (i * output_offset + offset) * input_size);
                }
                offset += input_offset;
            }
            descript();
        }
        else{
            int offset = 0;
            int output_offset = _output[0]->dimension(axis);
            Mtype *output_data = _output[0]->get_data();
            int all_count=_output[0]->count(0,axis-1);

            for (auto &src: _input) {
                int count = src->count(0, axis-1);
                int input_size = 0;
                int input_offset = 0;
                if(all_count!=count){std::cout<<"error you need make all of the input be same size of 0~axis-1";exit(0);}
                if(axis==src->get_dimensions().size()-1){
                    input_size = 1;
                    input_offset = src->dimension(axis);
                }else {
                    input_size = src->count(axis);
                    input_offset = src->dimension(axis);
                }
                Mtype *input_data = src->get_data();
                for (int i = 0; i < count; i++) {
                    Mtype *input_data_start = input_data + i * input_offset * input_size;
                    std::copy(input_data_start, input_data_start + input_offset * input_size,
                              output_data + (i * output_offset + offset) * input_size);
                }
                offset += input_offset;
            }
            descript();
        }
    }
};
