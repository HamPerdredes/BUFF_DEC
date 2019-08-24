#ifndef _BUFF_HPP_
#define _BUFF_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void fillHole(Mat srcBw, Mat &dstBw);

class BUFF{
    public:
        BUFF(string video_path,string chosen_color);
        void process();

    private:
        void get_center_R();
        void init_img();
        void find_armor();
        void check_armor();
        int center_confidence;
        int gray_thresh,color_thresh;
        double total_cost;
        Ptr<ml::SVM> svm;
        string color_type;
        VideoCapture cap;
        Point2f center_R;
        bool is_center_ok,is_save_data,is_armor_detect;
        int center_dec_count,KF_confidence;
        float h_w_max,h_w_min,min_area,max_area,min_center,max_center;
        Mat cur_img,origin_img,result_img,YUV_img,gray_img,filled_img;
        Mat YUV_bin,gray_bin,B_R_img;
        Mat measurement;
        Mat kernel,erode_kernel;
        vector<Mat> YUV_img_channel,BGR_channel;
        vector<RotatedRect> possible_armor;
        Point2f warp_src_points[3];
        Point2f warp_dst_points[3];
        int frame_count;
        KalmanFilter armor_KF;
        Point armor_center;
};

#endif