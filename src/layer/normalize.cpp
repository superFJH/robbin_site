//
// Created by fjh on 17-11-3. Copyright (c) 2017 ThinkForce, Inc. All Rights Reserved
//

#include <layer/normalize.h>
namespace mdl{
    Normalize_layer::Normalize_layer(const Json &config) : Layer(config) {
        auto &param=config["param"];
        if(param["scale_filler"]["type"]=="constant"){
            value=param["scale_filler"]["value"].number_value();
        }
        channel_shared=param["channel_shared"].bool_value();
        across_spatial=param["across_spatial"].bool_value();
       // _weight.resize(_input.size());
    }
    Normalize_layer::~Normalize_layer() {

    }
    void Normalize_layer::forward(int threadint) {//现在只能实现单个系数的scale，2.0版本会加上对fill_scale 中多个系数加入的改进
        Mtype *data=_input[0]->get_data();
        Mtype *dataout=new Mtype[_input[0]->count(1)];
        for(int i=0;i<_input[0]->count(0);i++){
            *(dataout+i)=pow(*(data+i),2);
        }
        if (across_spatial) {
            std::vector<Mtype>scale(_input[0]->dimension(1));
            std::fill(scale.begin(),scale.end(),0);
            for(int i=0;i<scale.size();i++){
                for(int j=0;j<_input[0]->count(2);j++){
                    scale[i]+=*(dataout+j+i*(_input[0]->count(2)));
                }
                scale[i]=sqrt(scale[i]);
            }
            for(int i=0;i<scale.size();i++){
                for(int j=0;j<_input[0]->count(2);j++){
                    *(dataout+j+i*(_input[0]->count(2)))=*(dataout+j+i*(_input[0]->count(2)))*value/scale[i];
                }
            }
            _output[0]->set_data(dataout);
        } else {
            std::vector<Mtype>scale(_input[0]->count(2));
            std::fill(scale.begin(),scale.end(),0);
            for(int i=0;i<_input[0]->count(2);i++){
                for(int j=0;j<_input[0]->dimension(1);j++){
                    scale[i]+=*(dataout+i+j*(_input[0]->count(2)));
                }
                scale[i]=sqrt(scale[i]);
            }
            for(int i=0;i<_input[0]->dimension(1);i++){
                for(int j=0;j<_input[0]->count(2);j++){
                    *(dataout+j+i*(_input[0]->count(2)))=*(data+j+i*(_input[0]->count(2)))*value/scale[j];
                }
            }
            _output[0]->set_data(dataout);
        }
        // scale the output

    }
};
