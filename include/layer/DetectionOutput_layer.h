//
// Created by fjh on 17-11-3. Copyright (c) 2017 ThinkForce, Inc. All Rights Reserved
//

#ifndef MOBILE_DEEP_LEARNING_DETECTIONOUTPUT_LAYER_H
#define MOBILE_DEEP_LEARNING_DETECTIONOUTPUT_LAYER_H
#include "commons/commons.h"
#include "base/layer.h"
#include "math/bbox_util.hpp"
namespace mdl{
    class DetectionOutput_layer :public Layer{
    public:
        DetectionOutput_layer(const Json& config);
        ~DetectionOutput_layer();
        int num_classes_;
        bool share_location_;
        int num_loc_classes_;
        int background_label_id_;
        CodeType code_type_;
        bool variance_encoded_in_target_;
        int keep_top_k_;
        float confidence_threshold_;

        int num_;
        int num_priors_;

        float nms_threshold_;
        int top_k_;
        float eta_;

        bool need_save_;
        string output_directory_;
        string output_name_prefix_;
        string output_format_;
        map<int, string> label_to_name_;
        map<int, string> label_to_display_name_;
        std::vector<string> names_;
        std::vector<std::pair<int, int> > sizes_;
        int num_test_image_;
        int name_count_;
        bool has_resize_;


        bool visualize_;
        float visualize_threshold_;
        void forward(int numthread);
    };
}
#endif //MOBILE_DEEP_LEARNING_DETECTIONOUTPUT_LAYER_H
