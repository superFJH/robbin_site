//
// Created by fjh on 17-11-3.
//

#ifndef MOBILE_DEEP_LEARNING_PRIORBOX_LAYER_H
#define MOBILE_DEEP_LEARNING_PRIORBOX_LAYER_H
#include "commons/commons.h"
#include "base/layer.h"
namespace mdl{
    class PriorBox_layer :public Layer{
    public:
        PriorBox_layer(const Json& config);
        ~PriorBox_layer();
        bool flip_;
        int num_priors_;
        bool clip_;
        vector<float> variance_;
        vector<float> min_sizes_;
        vector<float> max_sizes_;
        vector<float> aspect_ratios_;
        int img_w_;
        int img_h_;
        float step_w_;
        float step_h_;

        float offset_;
        void forward(int numthread);
    };
}
#endif //MOBILE_DEEP_LEARNING_PRIORBOX_LAYER_H
