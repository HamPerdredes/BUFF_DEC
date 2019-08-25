#include "buff.hpp"

#include <iostream>
#include "math.h"
#include <opencv2/opencv.hpp>
#include "sys/time.h"
using namespace cv;
using namespace std;

//23号：考虑引入卡尔曼滤波
BUFF::BUFF(string video_path,string chosen_color):
armor_KF(4,2,0,CV_32F),center_R(Point2f(1,1)),measurement(Mat::zeros(2,1,CV_32F))
{
    is_armor_detect=false;
    KF_confidence=0;
    total_cost=0;
    frame_count=0;
    center_confidence=0;
    is_save_data=false;
    is_center_ok=false;
    svm=ml::SVM::load("svm_trained.xml");
    h_w_max=3.0;
    h_w_min=1.5;
    min_area=30.0;//30       //主要根据实际图像调整 min/max area of armor and center,以及调整kernel和thresh的值
    max_area=800.0;//900  
    min_center=50.0;//50.0
    max_center=300.0;//300.0
    warp_dst_points[0]=Point2f(134,0);
    warp_dst_points[1]=Point2f(134,34);
    warp_dst_points[2]=Point2f(0,34);
    color_type=chosen_color;
    center_dec_count=0;
    gray_thresh=120;//120           这两个阈值和kernel的size联调
    color_thresh=40;//40
    kernel=getStructuringElement(0,Size(6,6));//6 越大，对装甲提取效果更好
    erode_kernel=getStructuringElement(0,Size(3,3));//3 对中心的提取效果有影响
    cap.open(video_path);
    if(!cap.isOpened()){
        cout<<"unable to open the video"<<endl;
        return  ;
    }
    armor_KF.transitionMatrix=(Mat_<float>(4, 4) << 1, 0, 1, 0,0, 1, 0, 1,0, 0, 1, 0,0, 0, 0, 1);//A 状态转移矩阵
    setIdentity(armor_KF.measurementMatrix);//H=[1,0,0,0;0,1,0,0] 测量矩阵
    setIdentity(armor_KF.processNoiseCov, Scalar::all(1e-5));//Q高斯白噪声，单位阵
    setIdentity(armor_KF.measurementNoiseCov, Scalar::all(1e-1));//R高斯白噪声，单位阵
    setIdentity(armor_KF.errorCovPost, Scalar::all(1));//P后验误差估计协方差矩阵，初始化为单位阵
    randn(armor_KF.statePost, Scalar::all(0), Scalar::all(0.1));//初始化状态为随机值
}

void BUFF::process()
{
    if(color_type!="red" && color_type!="blue"){
        cout<<"incorrect type of color"<<endl;
        return  ;
    }
    while(1)//优化整体架构
    {
        struct timeval st,end;
        gettimeofday(&st,nullptr);
        cap>>origin_img;//tested on 1080*1920
        cout<<origin_img.size<<endl;
        if(origin_img.empty())
            break;
        init_img();
        find_armor();
        get_center_R();
        if(is_center_ok ){
            check_armor();
            circle(origin_img,center_R,2,Scalar(255),5);
            is_center_ok=false;
        }
        if(KF_confidence>0 && !is_armor_detect){//缺少center，本质是没有检测到装甲板
        Mat prediction=armor_KF.predict();
        circle(origin_img,Point((int)prediction.at<float>(0), (int)prediction.at<float>(1)),5,Scalar(0,255,0),5);
        measurement.at<float>(0)=(int)prediction.at<float>(0);
        measurement.at<float>(1)= (int)prediction.at<float>(1);//KF跟的效果不行
        armor_KF.correct(measurement);
    }
        is_armor_detect=false;
        imshow("original",origin_img);
        //imshow("detected_armor",result_img);
        gettimeofday(&end,nullptr);
        double result=(end.tv_sec-st.tv_sec)*1000.+(end.tv_usec-st.tv_usec)/1000;
        total_cost=total_cost+result/1000;
        frame_count++;
        double ave=total_cost/frame_count;
        cout<<"average cost "<<ave<<" s"<<endl;
        cout<<"total_frame:"<<frame_count<<"  total time:"<<total_cost<<endl;
        waitKey(1); 
    }
}
void BUFF::init_img()//根据能量机关颜色对图像进行分割，得到二值化图像
{
    resize(origin_img,origin_img,Size(0.5*origin_img.cols,0.5*origin_img.rows));//0.5 0.5
    cvtColor(origin_img,YUV_img,CV_BGR2YUV);
    split(YUV_img,YUV_img_channel);
    gray_img=YUV_img_channel[0];
    split(origin_img,BGR_channel);
    if(color_type=="red"){
        cur_img=YUV_img_channel[2]-YUV_img_channel[1];
        B_R_img=BGR_channel[2]-BGR_channel[0];}//blue  20red 02blue
    else{    //blue
        cur_img=YUV_img_channel[1]-YUV_img_channel[2];
        B_R_img=BGR_channel[0]-BGR_channel[2];}
    //imshow("BR_IMG",B_R_img);
    Mat YUV_temp,gray_temp;
    threshold(cur_img,YUV_bin,color_thresh,255,0);
    threshold(gray_img,gray_bin,gray_thresh,255,0);
    threshold(B_R_img,B_R_img,gray_thresh,255,0);
    dilate(YUV_bin,YUV_temp,kernel); 
    dilate(gray_bin,gray_temp,kernel);
    cur_img=YUV_temp & gray_temp;
    //imshow("BR_IMG",B_R_img);
    //imshow("YUV_after_dilate",YUV_temp);
    //imshow("gray_after_dilate",gray_temp);
    fillHole(cur_img,filled_img);
}

void BUFF::find_armor()
{
    possible_armor.clear();//clean the vector
    cur_img=filled_img-cur_img;
    imshow("img for armor_find",cur_img);
    vector<vector<Point> > contours;
    findContours(cur_img,contours,RETR_LIST, CHAIN_APPROX_NONE);
    result_img=cur_img.clone();
    result_img.setTo(0);
    for(int i=0;i<contours.size();i++)
    {
        double area=fabs(contourArea(contours[i]));//countNonZero
        if(area<min_area || area>max_area)
            continue;
        RotatedRect rect=minAreaRect(contours[i]);
        float h_w=0;
        if(rect.size.width>rect.size.height)
            h_w=rect.size.width/rect.size.height;
        else
            h_w=rect.size.height/rect.size.width;
        if(h_w >h_w_max || h_w< h_w_min)
            continue;
        Point2f* vertices=new Point2f[4];
        rect.points(vertices);
        for(int i=0;i<4;i++)//画出疑似装甲板
            line(result_img,vertices[i],vertices[(i+1)%4],Scalar(255),2);   //再找圆心，考虑到相对静止，只找有限次数,再用SVM和KF
        possible_armor.push_back(rect);
    }
}

void BUFF::get_center_R()//直接划分四个象限，计算0到2pi的旋转角，看圆心直径到装甲板夹角是否为90
{
    vector< vector<Point> > contours;
    //Mat temp;
    erode(filled_img,filled_img,erode_kernel);//看具体图像比例，或者设置一个erode的size 
    //imshow("img for center detect",filled_img);//filled_img
    findContours(filled_img,contours,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    int center_max_size=0;
    for(int i=0;i<contours.size();i++)
    {
        double area=fabs(contourArea(contours[i]));
        if(area > max_center || area <min_center)
            continue;
        RotatedRect rect=minAreaRect(contours[i]);       
        float h_w;
        if(rect.size.height > rect.size.width)
            h_w=rect.size.height/rect.size.width;
        else
            h_w=rect.size.width/rect.size.height;
        if(h_w<1.2 && fabs(45-fabs(rect.angle))>35)//1.2
        {
            Point2f* vertices=new Point2f[4];
            rect.points(vertices);
            for(int i=0;i<4;i++)//画出疑似圆心
                line(origin_img,vertices[i],vertices[(i+1)%4],Scalar(200,0,200),2);          
            for(int j=0;j<possible_armor.size();j++)//圆心检测的太晚了,什么原因？？角度似乎有问题
            {
                double theta=atan2(rect.center.y-possible_armor[j].center.y,rect.center.x-possible_armor[j].center.x)/3.1415*180;
                if(theta< -90)
                    theta+=180;
                if(theta>90)
                    theta-=180;
                cout<<"theta="<<theta<<endl;
                double armor_ang;
                if(possible_armor[j].size.width>possible_armor[j].size.height)
                    armor_ang=possible_armor[j].angle+90;
                else
                    armor_ang=possible_armor[j].angle;
                double temp=fabs(armor_ang-theta);
                cout<<"armor_ang="<<armor_ang<<endl;
                cout<<temp<<endl;
                if(temp<10 || (armor_ang==-90 && fabs(temp-180)<10))
                    if(rect.size.area()>center_max_size){
                        is_center_ok=true;
                        center_max_size=rect.size.area();
                        center_R=rect.center;
                    }
            }
        }
    }
}

void BUFF::check_armor()//SVM检验装甲叶片，看是否要打击
{
    for(int i=0;i<possible_armor.size();i++)
    {
        Point2f* vertices=new Point2f[4];
        possible_armor[i].points(vertices);
        vector<Point2f> armor_swing;
        if(possible_armor[i].size.height>possible_armor[i].size.width){//line01 or 23
            float r_01=pow((center_R.x-vertices[0].x),2)+pow((center_R.y-vertices[0].y),2);
            float r_23=pow((center_R.x-vertices[3].x),2)+pow((center_R.y-vertices[3].y),2);
            if(r_01>r_23){
                float delta_x=(vertices[2].x-vertices[3].x)/2;
                float delta_y=(vertices[2].y-vertices[3].y)/2;
                armor_swing.push_back(vertices[2]);
                armor_swing.push_back(vertices[3]);
                armor_swing.push_back(Point2f(center_R.x-delta_x,center_R.y-delta_y));
                armor_swing.push_back(Point2f(center_R.x+delta_x,center_R.y+delta_y));
            }
            else
            {
                float delta_x=(vertices[1].x-vertices[0].x)/2;
                float delta_y=(vertices[1].y-vertices[0].y)/2;
                armor_swing.push_back(vertices[1]);
                armor_swing.push_back(vertices[0]);
                armor_swing.push_back(Point2f(center_R.x-delta_x,center_R.y-delta_y));
                armor_swing.push_back(Point2f(center_R.x+delta_x,center_R.y+delta_y));
            }
        }
        else//line12 or 30
        {
            float r_12=pow((center_R.x-vertices[1].x),2)+pow((center_R.y-vertices[1].y),2);
            float r_30=pow((center_R.x-vertices[0].x),2)+pow((center_R.y-vertices[0].y),2);
            if(r_12>r_30){
                float delta_x=(vertices[3].x-vertices[0].x)/2;
                float delta_y=(vertices[3].y-vertices[0].y)/2;
                armor_swing.push_back(vertices[3]);
                armor_swing.push_back(vertices[0]);
                armor_swing.push_back(Point2f(center_R.x-delta_x,center_R.y-delta_y));
                armor_swing.push_back(Point2f(center_R.x+delta_x,center_R.y+delta_y));
            }
            else{
                float delta_x=(vertices[2].x-vertices[1].x)/2;
                float delta_y=(vertices[2].y-vertices[1].y)/2;
                armor_swing.push_back(vertices[2]);
                armor_swing.push_back(vertices[1]);
                armor_swing.push_back(Point2f(center_R.x-delta_x,center_R.y-delta_y));
                armor_swing.push_back(Point2f(center_R.x+delta_x,center_R.y+delta_y));
            }
        }
        Mat possible_swing(35,135,CV_32FC1);
        for(int i=0;i<4;i++)//画出疑似装甲板叶片
            line(origin_img,armor_swing[i],armor_swing[(i+1)%4],Scalar(0,255,0),2); 
        for(int i=0;i<3;i++)//调整抠出的装甲叶片尺寸
            warp_src_points[i]=armor_swing[i];
        Mat warp_mat=getAffineTransform(warp_src_points,warp_dst_points);
        warpAffine(B_R_img,possible_swing,warp_mat,possible_swing.size());//test
        //warpAffine(gray_bin,possible_swing,warp_mat,possible_swing.size());
        //imshow("possilbe_swing",possible_swing);
        if(is_save_data){
            string file_route="./data/"+to_string(frame_count)+".jpg";
            //frame_count++;
            imwrite(file_route,possible_swing);
        }
        possible_swing=possible_swing.reshape(0,1);
        possible_swing.convertTo(possible_swing,CV_32FC1);
        float result=svm->predict(possible_swing);
        if(result==1){//show result
            is_armor_detect=true;
            armor_center.x=(armor_swing[0].x+armor_swing[1].x)/2;
            armor_center.y=(armor_swing[0].y+armor_swing[1].y)/2;
            circle(origin_img,armor_center,5,Scalar(0,0,255),5);//这里稍微调整一下
            Mat prediction=armor_KF.predict();
            Point KF_predict = Point((int)prediction.at<float>(0), (int)prediction.at<float>(1));
            if( (pow((KF_predict.x-armor_center.x),2)+pow((KF_predict.y-armor_center.y),2))  < 100.0)//50
                KF_confidence++;
            else
                KF_confidence=0;
            measurement.at<float>(0)=armor_center.x;
            measurement.at<float>(1)=armor_center.y;//KF跟的效果不行
            armor_KF.correct(measurement);
        }
    } 
}

void fillHole(Mat srcBw, Mat &dstBw)//孔洞填补
{
    Size m_Size = srcBw.size();
    Mat Temp=Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
    cv::floodFill(Temp, Point(0, 0), Scalar(255));
    Mat cutImg;
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
    dstBw = srcBw | (~cutImg);
}


