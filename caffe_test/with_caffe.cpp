//
// Created by fjh on 17-10-31.
//

#include <caffe/caffe.hpp>
#include <net.h>
#include <boost/shared_ptr.hpp>
#include <loader/loader.h>
#include <commons/commons.h>
#include <math/gemm.h>
#include <opencv2/core.hpp>
#include <opencv2/shape.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#define num_channels_ 3
#define input_geometry_ cv::Size(640,360)
mdl::Loader* load_net;
cv::Mat mean_;
void init(){
    mdl::Gemmer::gemmers.push_back(new mdl::Gemmer);
    load_net=mdl::Loader::shared_instance();
    load_net->load("tool/model.min.json","tool/data.min.bin");
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

     cv::Mat sample_normalized;
     cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == in_data)
    << "Input channels are not wrapping the input layer of the network.";
}
void SetMean(const string& mean_value) {
    cv::Scalar channel_mean;

    if (!mean_value.empty()) {
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
                                                                    "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
    else{
        LOG(INFO)<<"there is no mean_value been set!";

    }
}
void caffe_WrapInputLayer(std::vector<cv::Mat>* input_channels,caffe::Blob<float>*input_layer) {

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void caffe_Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels,caffe::Blob<float>*input_layer) {
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

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == input_layer->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
int main(){



    boost::shared_ptr<caffe::Net<float>> net_;
    net_.reset(new caffe::Net<float>("/home/fjh/CLionProjects/mobile-deep-learning/cmake-build-debug/tools/build/E23.prototxt",caffe::TEST));
    net_->CopyTrainedLayersFrom("/home/fjh/CLionProjects/mobile-deep-learning/cmake-build-debug/tools/build/E23.caffemodel");
    cv::Mat img=cv::imread("/home/fjh/123.jpg");
    cv::Mat img2;
    SetMean("104,117,123");
    cv::resize(img,img2,cv::Size(640,360));
    float *img_data=new float[640*360*3];
    std::vector<cv::Mat> input_channels;
    caffe_WrapInputLayer(&input_channels,net_->input_blobs()[0]);
    caffe_Preprocess(img2,&input_channels,net_->input_blobs()[0]);
    net_->Forward();
    Mtype *img_data2=new Mtype[640*360*3];
    std::vector<cv::Mat> input_channels2;
    WrapInputLayer(&input_channels2,img_data2);
    Preprocess(img2,&input_channels2,img_data2);

    int thread_num = 1;
    if (mdl::Gemmer::gemmers.size() == 0) {

        mdl::Gemmer::gemmers.push_back(new mdl::Gemmer());

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
/*
        for(int j=0;j<net->_layers.size()-1;j++) {
            if(net_->has_blob(net->_layers[j]->name())) {
               // std::cout<<net->_layers[j]->name()<<std::endl;
                for (int k = 0; k < net->_layers[j]->input().size(); k++) {
                    Mtype *input_data=new Mtype[net->_layers[j]->input()[k]->count(0)];

                    if(net_->has_blob(net->_layers[j]->input()[k]->get_name())) {
                        std::cout<<net->_layers[j]->input()[k]->get_name()<<"  ";
                        for (int l = 0; l < net->_layers[j]->input()[k]->count(0); l++) {
                            input_data[l] = (Mtype) *(
                                    net_->blob_by_name(net->_layers[j]->input()[k]->get_name())->cpu_data() + l);
                        }
                        net->_layers[j]->input()[k]->set_data(input_data);
                    }
                  //  std::cout<<std::endl;
                }
                net->_layers[j]->forward();
                for (int k = 0; k < net->_layers[j]->output().size(); k++) {
                    int error_perlayer=0;
                    for(int l=0;l<net->_layers[j]->output()[k]->count(0);l++){
                        if(abs(abs(*(net->_layers[j]->output()[k]->get_data()+l)-*(net_->blob_by_name(net->_layers[j]->name())->cpu_data()+l))/(*(net->_layers[j]->output()[k]->get_data()+l)))>0.01){
                            // std::cout<<*(net->_layers[j]->output()[k]->get_data()+l)<<"    "<<*(net_->blob_by_name(net->_layers[j]->name())->cpu_data()+l)<<std::endl;
                            //std::cout<<abs(*(net->_layers[j]->output()[k]->get_data()+l)-*(net_->blob_by_name(net->_layers[j]->name())->cpu_data()+l))/(*(net->_layers[j]->output()[k]->get_data()+l))<<std::endl;
                            error_perlayer++;
                        }
                        //std::cout<<*(net->_layers[j]->output()[k]->get_data()+l)<<"    "<<*(net_->blob_by_name(net->_layers[j]->name())->cpu_data()+l)<<std::endl;
                    }
                   // std::cout<<net->_layers[j]->output()[0]->descript_dimention()<<std::endl;
                    //std::cout<<net_->blob_by_name(net->_layers[j]->name())->shape_string()<<std::endl;
                    std::cout<<net->_layers[j]->name()<<"::error_ratio:    "<<(error_perlayer*1.0)/net->_layers[j]->output()[k]->count(0)<<std::endl;
                    //net->_layers[j]->input()[k]->set_data(input_data);
                }
            }
            for(int k=0;k<net->_layers[j]->input()[0]->count();k++){

            }
        }
        net->_layers[net->_layers.size()-1]->forward();
        */
/*
        Mtype *input_data=new Mtype[loader->_matrices["mbox_loc"]->count(0)];
        std::cout<<net_->has_blob("mbox_loc")<<std::endl;

        for (int k = 0; k < loader->_matrices["mbox_loc"]->count(0); k++) {
            input_data[k] = net_->bottom_vecs()[net_->layers().size()-1][0]->cpu_data()[k];
            //std::cout<<input_data[k]<<std::endl;
        }
        Mtype *input_data2=new Mtype[loader->_matrices["mbox_conf_flatten"]->count(0)];
        for (int k = 0; k < loader->_matrices["mbox_conf_flatten"]->count(0); k++) {
            input_data2[k] = net_->bottom_vecs()[net_->layers().size()-1][1]->cpu_data()[k];
        }
        Mtype *input_data3=new Mtype[loader->_matrices["mbox_priorbox"]->count(0)];
        for (int k = 0; k < loader->_matrices["mbox_priorbox"]->count(0); k++) {
            input_data3[k] = net_->bottom_vecs()[net_->layers().size()-1][2]->cpu_data()[k];
        }
        loader->_matrices["mbox_loc"]->set_data(input_data);
        loader->_matrices["mbox_conf_flatten"]->set_data(input_data2);
        loader->_matrices["mbox_priorbox"]->set_data(input_data3);
        net->_layers[net->_layers.size()-1]->forward();
        */
        net->predict(img_data2);
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
    int num_weight=0;

    for(int i=0;i<net_->blob_names().size();i++){
        mdl::Matrix *layer_data=loader->_matrices[net_->blob_names()[i]];
            if (layer_data!= nullptr) {
                int error_perlayer=0;

                for(int j=0;j<layer_data->count(0);j++){
                    if(abs(abs(*(layer_data->get_data()+j)-*(net_->blob_by_name(net_->blob_names()[i])->cpu_data()+j))/(*(layer_data->get_data()+j)))>0.01){
                       // std::cout<<*(layer_data->get_data()+j)<<"    "<<*(net_->blob_by_name(net_->blob_names()[i])->cpu_data()+j)<<std::endl;
                        error_perlayer++;
                    }
                    //std::cout<<*(layer_data->get_data()+j)<<"    "<<*(net_->blob_by_name(net_->blob_names()[i])->cpu_data()+j)<<std::endl;
                }
                std::cout<<layer_data->get_name()<<"::error_ratio:    "<<(error_perlayer*1.0)/layer_data->count(0)<<std::endl;
              //  std::cout<<layer_data->descript_dimention()<<"     "<<net_->blob_by_name(net_->blob_names()[i])->shape_string()<<std::endl;
            }

    }


    Mtype* get_data=net->_layers[net->_layers.size()-1]->output()[0]->get_data();
    for(int i=0;i<net->_layers[net->_layers.size()-1]->output()[0]->dimension(2);i++) {
        float score = get_data[2];
        if (score >= 0.7) {
            int xmin = static_cast<int>(get_data[3] * img.cols);
            int ymin = static_cast<int>(get_data[4] * img.rows);
            int xmax = static_cast<int>(get_data[5] * img.cols);
            int ymax = static_cast<int>(get_data[6] * img.rows);
            cv::rectangle(img, cv::Rect(xmin, ymin, (xmax - xmin), (ymax - ymin)), cv::Scalar(255, 0, 0), 2);
        }
        get_data+=7;
    }
    cv::imshow("a",img);
    cv::waitKey(0);
    /*
    for(int i=0;i<net->_layers.size();i++){
        if(net->_layers[i]->layer_type()==mdl::LayerType::CONVOLUTION){
            int error_perlayer=0;
            for(int j=0;j<net->_layers[i]->_weight[0]->count(0);j++){
                if(abs(abs(*(net_->params()[num_weight].get()->cpu_data()+j)-*(net->_layers[i]->_weight[0]->get_data()+j))/(*(net_->params()[num_weight].get()->cpu_data()+j)))>0.01){
                    error_perlayer++;
                }
              //  std::cout<<*(net_->params()[num_weight].get()->cpu_data()+j)<<"    "<<*(net->_layers[i]->_weight[0]->get_data()+j)<<std::endl;
            }
            std::cout<<net->_layers[i]->name()<<"::error_ratio:    "<<(error_perlayer*1.0)/net->_layers[i]->_weight[0]->count(0)<<std::endl;
            num_weight+=2;
            if(num_weight==20)num_weight+=1;
        }
    }
*/

    loader->clear();
    delete net;



    return 1;
}