//
// Created by fjh on 17-11-3. Copyright (c) 2017 ThinkForce, Inc. All Rights Reserved
//
#include <layer/PriorBox_layer.h>
namespace mdl{
    PriorBox_layer::PriorBox_layer(const Json &config) :Layer(config) {
        auto & prior_box_param =config["param"];
        if(prior_box_param["min_size"].is_array()) {
            for (int i = 0; i < prior_box_param["min_size"].array_items().size(); ++i) {
                min_sizes_.push_back(prior_box_param["min_size"].array_items()[i].int_value());
            }
        }
        else{
            min_sizes_.push_back(prior_box_param["min_size"].int_value());
        }
        aspect_ratios_.clear();
        aspect_ratios_.push_back(1.);
      //  flip_ = prior_box_param["clip"].string_value()=="True"?true:false;
        /* 还不能支持 多个比率的回归
        for (int i = 0; i < prior_box_param.aspect_ratio_size(); ++i) {
            float ar = prior_box_param.aspect_ratio(i);
            bool already_exist = false;
            for (int j = 0; j < aspect_ratios_.size(); ++j) {
                if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
                    already_exist = true;
                    break;
                }
            }
            if (!already_exist) {
                aspect_ratios_.push_back(ar);
                if (flip_) {
                    aspect_ratios_.push_back(1./ar);
                }
            }
        }
        */
        num_priors_ = aspect_ratios_.size() * min_sizes_.size();
        if (prior_box_param["max_size"].is_array()) {

            CHECK_EQ(prior_box_param["min_size"].array_items().size(), prior_box_param["max_size"].array_items().size());
            for (int i = 0; i < prior_box_param["max_size"].array_items().size(); ++i) {
                max_sizes_.push_back(prior_box_param["max_size"].array_items()[i].int_value());
                num_priors_ += 1;
            }
        }
        else{
            max_sizes_.push_back(prior_box_param["max_size"].int_value());
            num_priors_+=1;
        }
        clip_ = prior_box_param["clip"].bool_value();
        if (prior_box_param["variance"].array_items().size() > 1) {
            // Must and only provide 4 variance.
            CHECK_EQ(prior_box_param["variance"].array_items().size() , 4);
            for (int i = 0; i < prior_box_param["variance"].array_items().size(); ++i) {
                CHECK_GT(prior_box_param["variance"].array_items()[i].number_value() , 0);
                variance_.push_back(prior_box_param["variance"].array_items()[i].number_value());
            }
        } else if (prior_box_param["variance"].array_items().size()  == 1) {
            CHECK_GT(prior_box_param["variance"].array_items()[0].number_value(), 0);
            variance_.push_back(prior_box_param["variance"].array_items()[0].number_value());
        } else {
            // Set default to 0.1.
            variance_.push_back(0.1);
        }
/*
        if (prior_box_param["img_h"].is_null() || prior_box_param.has_img_w()) {
            img_h_ = prior_box_param.img_h();
            img_w_ = prior_box_param.img_w();
        } else if (prior_box_param.has_img_size()) {
            const int img_size = prior_box_param.img_size();
            img_h_ = img_size;
            img_w_ = img_size;
        } else {
            img_h_ = 0;
            img_w_ = 0;
        }
*/
        if (!prior_box_param["step_h"].is_null() || !prior_box_param["step_w"].is_null()) {
            step_h_ = prior_box_param["step_h"].number_value();
            step_w_ = prior_box_param["step_w"].number_value();
        } else if (!prior_box_param["step"].is_null()) {
            const float step = prior_box_param["step"].number_value();
            step_h_ = step;
            step_w_ = step;
        } else {
            step_h_ = 0;
            step_w_ = 0;
        }

        offset_ = prior_box_param["offset"].number_value();
    }
    PriorBox_layer::~PriorBox_layer() {}
    void PriorBox_layer::forward(int numthread) {
        const int layer_width = _input[0]->dimension(2);
        const int layer_height =_input[0]->dimension(3);
        int img_width, img_height;
        if (img_h_ == 0 || img_w_ == 0) {
            img_width = _input[1]->dimension(2);
            img_height = _input[1]->dimension(3);
        } else {
            img_width = img_w_;
            img_height = img_h_;
        }
        float step_w, step_h;
        if (step_w_ == 0 || step_h_ == 0) {
            step_w = static_cast<float>(img_width) / layer_width;
            step_h = static_cast<float>(img_height) / layer_height;
        } else {
            step_w = step_w_;
            step_h = step_h_;
        }
        Mtype* out_data = _output[0]->get_data();
        int dim = layer_height * layer_width * num_priors_ * 4;
        int idx = 0;
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                float center_x = (w + offset_) * step_w;
                float center_y = (h + offset_) * step_h;
                float box_width, box_height;
                for (int s = 0; s < min_sizes_.size(); ++s) {
                    int min_size_ = min_sizes_[s];
                    // first prior: aspect_ratio = 1, size = min_size
                    box_width = box_height = min_size_;
                    // xmin
                    out_data[idx++] = (center_x - box_width / 2.) / img_width;
                    // ymin
                    out_data[idx++] = (center_y - box_height / 2.) / img_height;
                    // xmax
                    out_data[idx++] = (center_x + box_width / 2.) / img_width;
                    // ymax
                    out_data[idx++] = (center_y + box_height / 2.) / img_height;

                    if (max_sizes_.size() > 0) {
                        CHECK_EQ(min_sizes_.size(), max_sizes_.size());
                        int max_size_ = max_sizes_[s];
                        // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        box_width = box_height = sqrt(min_size_ * max_size_);
                        // xmin
                        out_data[idx++] = (center_x - box_width / 2.) / img_width;
                        // ymin
                        out_data[idx++] = (center_y - box_height / 2.) / img_height;
                        // xmax
                        out_data[idx++] = (center_x + box_width / 2.) / img_width;
                        // ymax
                        out_data[idx++] = (center_y + box_height / 2.) / img_height;
                    }

                    // rest of priors
                    for (int r = 0; r < aspect_ratios_.size(); ++r) {
                        float ar = aspect_ratios_[r];
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }
                        box_width = min_size_ * sqrt(ar);
                        box_height = min_size_ / sqrt(ar);
                        // xmin
                        out_data[idx++] = (center_x - box_width / 2.) / img_width;
                        // ymin
                        out_data[idx++] = (center_y - box_height / 2.) / img_height;
                        // xmax
                        out_data[idx++] = (center_x + box_width / 2.) / img_width;
                        // ymax
                        out_data[idx++] = (center_y + box_height / 2.) / img_height;
                    }
                }
            }
        }
        // clip the prior's coordidate such that it is within [0, 1]
        if (clip_) {
            for (int d = 0; d < _output[0]->count(0); ++d) {
                out_data[d] = std::min<Mtype>(std::max<Mtype>(out_data[d], 0.), 1.);
            }
        }
        // set the variance.
        Mtype *top_data=_output[0]->get_data();
        out_data += _output[0]->count(2);
        if (variance_.size() == 1) {
            for (int d = 0; d < _output[0]->count(0); ++d) {
                out_data[d] = Mtype(variance_[0]);
            }
        } else {
            int count = 0;
            for (int h = 0; h < layer_height; ++h) {
                for (int w = 0; w < layer_width; ++w) {
                    for (int i = 0; i < num_priors_; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            top_data[count] = variance_[j];
                            ++count;
                        }
                    }
                }
            }
        }
    }
}
