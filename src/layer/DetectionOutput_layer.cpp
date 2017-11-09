//
// Created by fjh on 17-11-3. Copyright (c) 2017 ThinkForce, Inc. All Rights Reserved
//

#include "layer/DetectionOutput_layer.h"
#include <boost/shared_ptr.hpp>
namespace mdl{
    DetectionOutput_layer::DetectionOutput_layer(const Json& config):Layer(config){
        auto &param=config["param"];
        num_classes_ = param["num_classes"].number_value();
        share_location_ = true;
        num_loc_classes_ = share_location_ ? 1 : num_classes_;
        background_label_id_ = param["background_label_id"].int_value();
        code_type_ = CodeType::PriorBoxParameter_CodeType_CENTER_SIZE;
        variance_encoded_in_target_ =
                param["variance_encoded_in_target"].bool_value();
        keep_top_k_ = param["keep_top_k"].int_value();
        confidence_threshold_ = param["confidence_threshold"].is_null() ?
                                -FLT_MAX:param["confidence_threshold"].number_value() ;
        // Parameters used in nms.
        nms_threshold_ = 0.45;//param["nms_threshold"].number_value();
        CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
        eta_ = 1.0;
        //CHECK_GT(eta_, 0.);
        //CHECK_LE(eta_, 1.);
        top_k_ = -1;
        if (!param["top_k"].is_null()) {
            top_k_ = 400;//param["top_k"].number_value();
        }
        top_k_ = 400;
       
    }
    template <typename T>
    bool SortScorePairDescend(const std::pair<float, T>& pair1,
                              const std::pair<float, T>& pair2) {
        return pair1.first > pair2.first;
    }

// Explicit initialization.
    template bool SortScorePairDescend(const std::pair<float, int>& pair1,
                                       const std::pair<float, int>& pair2);
    template bool SortScorePairDescend(const std::pair<float, std::pair<int, int> >& pair1,
                                       const std::pair<float, std::pair<int, int> >& pair2);
    DetectionOutput_layer::~DetectionOutput_layer() {}
    void DetectionOutput_layer::forward(int numthread) {
        const Mtype* loc_data = _input[0]->get_data();
        const Mtype* conf_data = _input[1]->get_data();
        const Mtype* prior_data = _input[2]->get_data();
        const int num = _output[0]->dimension(0);
        num_priors_ = (_input[2]->dimension(2)) / 4;
        // Retrieve all location predictions.
        vector<LabelBBox> all_loc_preds;
        GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                          share_location_, &all_loc_preds);

        // Retrieve all confidences.
        std::vector<std::map<int, vector<float> > > all_conf_scores;
        GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                            &all_conf_scores);

        // Retrieve all prior bboxes. It is same within a batch since we assume all
        // images in a batch are of same dimension.
        vector<NormalizedBBox> prior_bboxes;
        vector<vector<float> > prior_variances;
        GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

        // Decode all loc predictions to bboxes.
        vector<LabelBBox> all_decode_bboxes;
        const bool clip_bbox = false;
        DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                        share_location_, num_loc_classes_, background_label_id_,
                        code_type_, variance_encoded_in_target_, clip_bbox,
                        &all_decode_bboxes);

        int num_kept = 0;
        vector<map<int, vector<int> > > all_indices;
        for (int i = 0; i < num; ++i) {
            const LabelBBox& decode_bboxes = all_decode_bboxes[i];
            const map<int, vector<float> >& conf_scores = all_conf_scores[i];
            map<int, vector<int> > indices;
            int num_det = 0;
            for (int c = 0; c < num_classes_; ++c) {
                if (c == background_label_id_) {
                    // Ignore background class.
                    continue;
                }
                if (conf_scores.find(c) == conf_scores.end()) {
                    // Something bad happened if there are no predictions for current label.
                    LOG(FATAL) << "Could not find confidence predictions for label " << c;
                }
                const vector<float>& scores = conf_scores.find(c)->second;
                int label = share_location_ ? -1 : c;
                if (decode_bboxes.find(label) == decode_bboxes.end()) {
                    // Something bad happened if there are no predictions for current label.
                    LOG(FATAL) << "Could not find location predictions for label " << label;
                    continue;
                }
                const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
                ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
                             top_k_, &(indices[c]));
                num_det += indices[c].size();
            }
            if (keep_top_k_ > -1 && num_det > keep_top_k_) {
                std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
                for (map<int, vector<int> >::iterator it = indices.begin();
                     it != indices.end(); ++it) {
                    int label = it->first;
                    const vector<int>& label_indices = it->second;
                    if (conf_scores.find(label) == conf_scores.end()) {
                        // Something bad happened for current label.
                        LOG(FATAL) << "Could not find location predictions for " << label;
                        continue;
                    }
                    const vector<float>& scores = conf_scores.find(label)->second;
                    for (int j = 0; j < label_indices.size(); ++j) {
                        int idx = label_indices[j];
                        CHECK_LT(idx, scores.size());
                        score_index_pairs.push_back(std::make_pair(
                                scores[idx], std::make_pair(label, idx)));
                    }
                }
                // Keep top k results per image.
                std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                          SortScorePairDescend<std::pair<int, int> >);
                score_index_pairs.resize(keep_top_k_);
                // Store the new indices.
                map<int, vector<int> > new_indices;
                for (int j = 0; j < score_index_pairs.size(); ++j) {
                    int label = score_index_pairs[j].second.first;
                    int idx = score_index_pairs[j].second.second;
                    new_indices[label].push_back(idx);
                }
                all_indices.push_back(new_indices);
                num_kept += keep_top_k_;
            } else {
                all_indices.push_back(indices);
                num_kept += num_det;
            }
        }

        vector<int> top_shape(2, 1);
        top_shape.push_back(num_kept);
        top_shape.push_back(7);
        Mtype* top_data;
        if (num_kept == 0) {
            LOG(INFO) << "Couldn't find any detections";
            top_shape[2] = num;
            _output[0]->resize(top_shape);
            top_data = _output[0]->get_data();
            for(int i=0;i<_output[0]->count(0);i++){
                top_data[i]=-1;
            }
            // Generate fake results per image.
            for (int i = 0; i < num; ++i) {
                top_data[0] = i;
                top_data += 7;
            }
        } else {
            _output[0]->resize(top_shape);
            top_data = _output[0]->get_data();
        }

        int count = 0;
        for (int i = 0; i < num; ++i) {
            const map<int, vector<float> >& conf_scores = all_conf_scores[i];
            const LabelBBox& decode_bboxes = all_decode_bboxes[i];
            for (map<int, vector<int> >::iterator it = all_indices[i].begin();
                 it != all_indices[i].end(); ++it) {
                int label = it->first;
                if (conf_scores.find(label) == conf_scores.end()) {
                    // Something bad happened if there are no predictions for current label.
                    LOG(FATAL) << "Could not find confidence predictions for " << label;
                    continue;
                }
                const vector<float>& scores = conf_scores.find(label)->second;
                int loc_label = share_location_ ? -1 : label;
                if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                    // Something bad happened if there are no predictions for current label.
                    LOG(FATAL) << "Could not find location predictions for " << loc_label;
                    continue;
                }
                const vector<NormalizedBBox>& bboxes =
                        decode_bboxes.find(loc_label)->second;
                vector<int>& indices = it->second;
                for (int j = 0; j < indices.size(); ++j) {
                    int idx = indices[j];
                    top_data[count * 7] = i;
                    top_data[count * 7 + 1] = label;
                    top_data[count * 7 + 2] = scores[idx];
                    const NormalizedBBox& bbox = bboxes[idx];
                    top_data[count * 7 + 3] = bbox.xmin();
                    top_data[count * 7 + 4] = bbox.ymin();
                    top_data[count * 7 + 5] = bbox.xmax();
                    top_data[count * 7 + 6] = bbox.ymax();

                    ++count;
                }
            }

        }

    }
}