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
#include <iostream>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "net.h"
#include "base/matrix.h"
#include "loader/loader.h"
#include "math/gemm.h"
#include <glog/logging.h>
using namespace std;
#define num_channels_ 3
#define input_geometry_ cv::Size(640,360)
int run();

int main(int argc,char**argv) {
    int run_count = 1;
    google::InitGoogleLogging(argv[0]);
    for (int i = 0; i < run_count; i++) {
        cout << "start running cycle : " << i << endl;
//        EXCEPTION_HEADER
        run();
//        EXCEPTION_FOOTER
        cout << "end running cycle : " << i << endl;
    }
}

bool is_equal(float a, float b) {
    // epsilon is too strict about correctness
    // return abs(a - b) <= std::numeric_limits<float>::epsilon();
    return abs(a - b) <= 0.001;
}

bool is_correct_result(vector<float> &result) {
    // // the correct result without quantification is : 87.5398 103.573 209.723 196.812
    // vector<float> correct_result{87.5398, 103.573, 209.723, 196.812};
    // the correct result with quantification is : 87.4985 103.567 209.752 196.71
    vector<float> correct_result{64.777, 101.88, 210.735, 199.144};
    if (result.size() != 4) {
        return false;
    }
    for (int i = 0; i < 4; i++) {
        if (!is_equal(result[i], correct_result[i])) {
            return false;
        }
    }
    return true;
}

int find_max(vector<float> data) {
    float max = -1000000;
    int index = 0;
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] > max) {
            max = data[i];
            index = i + 1;
        }
    }
    return index;
}
void WrapInputLayer(std::vector<cv::Mat>* input_channels,Mtype*in_data) {

    int width = 640;
    int height = 360;
    float* input_data = in_data;
    for (int i = 0; i < 3; ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}
void Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels,Mtype*in_data) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

   // cv::Mat sample_normalized;
  //  cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == in_data)
    << "Input channels are not wrapping the input layer of the network.";
}

int run() {
    // thread num should set 1 while using mobilenet & resnet

        cv::Mat img=cv::imread("/home/fjh/123.jpg");
        cv::Mat img2;
        cv::resize(img,img2,cv::Size(360,640));
    Mtype *img_data=new Mtype[640*360*3];
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels,img_data);
    Preprocess(img2,&input_channels,img_data);

        int thread_num = 1;
    if (mdl::Gemmer::gemmers.size() == 0) {
        for (int i = 0; i < max(thread_num, 3); i++) {
            mdl::Gemmer::gemmers.push_back(new mdl::Gemmer());
        }
    }
    mdl::Loader *loader = mdl::Loader::shared_instance();
    std::string prefix("/home/fjh/CLionProjects/mobile-deep-learning/cmake-build-debug/tools/build/");
    auto t1 = mdl::time();
    bool load_success = loader->load(prefix + "model.min.json", prefix + "data.min.bin");
    auto t2 = mdl::time();
    cout << "load time : " << mdl::time_diff(t1, t2) << "ms" << endl;
    if (!load_success) {
        cout << "load failure" << endl;
        loader->clear();
        return -1;
    }
    if (!loader->get_loaded()) {
        LOG(FATAL)<<"loader is not loaded yet";
    }
    mdl::Net *net = new mdl::Net(loader->_model);
    net->set_thread_num(thread_num);
    int count = 1;
    double total = 0;
    vector<Mtype > result;
    for (int i = 0; i < count; i++) {
        Time t1 = mdl::time();
        result = net->predict(img_data);
        Time t2 = mdl::time();
        double diff = mdl::time_diff(t1, t2);
        total += diff;
    }
    cout << "total cost: " << total / count << "ms." << endl;
    for (int num: result) {
        cout << num << " ";
    }
    cout <<endl;
    // uncomment while testing clacissification models
//    cout << "the max prob index = "<<find_max(result)<<endl;
    cout << "Done!" << endl;
//    cout << "it " << (is_correct_result(result) ? "is" : "isn't") << " a correct result." << endl;
    loader->clear();
    delete net;
    return 0;
}
