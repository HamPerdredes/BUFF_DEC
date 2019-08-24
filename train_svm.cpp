#include <iostream>
#include <cstdio>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace cv::ml;
using namespace std;

int main()
{
    Mat train_img;
    Mat train_data;
    Mat classes;
    vector<int> labels;
    string dir_path="./data/";
    string pos_data=dir_path+"pos/*.jpg";
    string neg_data=dir_path+"neg/*.jpg";
    vector< String > pos_files,neg_files;
    glob(pos_data,pos_files);
    glob(neg_data,neg_files);
    for(int i =0;i<pos_files.size();i++)
    {
        //是否仍需要二值化？
        Mat src=imread(pos_files[i],0);
        src=src.reshape(0,1);
        train_img.push_back(src);
        labels.push_back(1);
    }
    for(int i =0;i<neg_files.size();i++)
    {
        Mat src=imread(neg_files[i],0);
        src=src.reshape(0,1);
        train_img.push_back(src);
        labels.push_back(0);
    }
    Mat(train_img).copyTo(train_data);
    train_data.convertTo(train_data,CV_32FC1);
    Mat(labels).copyTo(classes);

    Ptr<cv::ml::SVM> svm=cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::Types::C_SVC);
	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
    Ptr<TrainData> tData = TrainData::create(train_data,ROW_SAMPLE,classes);
    cout<<"start training"<<endl;
    svm->train(tData);
    cout<<"end training"<<endl;
    //test result
    vector<String> val_files;
    string val_data=dir_path+"*.jpg";
    glob(val_data,val_files);
    float correct=0;
    for(int i=0;i<val_files.size();i++)
    {
        Mat src=imread(val_files[i],0);
        src=src.reshape(0,1);
        src.convertTo(src,CV_32FC1);
        float result=svm->predict(src);
        if(result==0)
            correct++;
    }
    cout<<"on validation set,the correct rate is "<<correct/val_files.size()<<endl;
    svm->save("svm_trained.xml");
    return 0;
}