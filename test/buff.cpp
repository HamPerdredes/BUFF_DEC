#include "buff.hpp"

#include <iostream>
#include "math.h"
#include <opencv2/opencv.hpp>
#include "thread"
#include "omp.h"
#include "time.h"
using namespace cv;
using namespace std;
//考虑结合卡尔曼滤波和SVM，在装甲板分割丢失的情况下，利用卡尔曼滤波,提高稳定性
BUFF::BUFF(string video_path,string chosen_color)
{
    total_cost=0;
    frame_count=0;
    center_confidence=0;
    is_save_data=false;
    svm=ml::SVM::load("svm_trained.xml");
    center_R=Point2f(1,1);
    h_w_max=3.0;
    h_w_min=1.5;
    min_area=200.0;
    max_area=900.0;
    warp_dst_points[0]=Point2f(134,0);
    warp_dst_points[1]=Point2f(134,34);
    warp_dst_points[2]=Point2f(0,34);
    color_type=chosen_color;
    center_dec_count=0;
    gray_thresh=160;//150 对颜色划分还不够完美
    color_thresh=40;//50
    kernel=getStructuringElement(0,Size(8,8));//red 10   blue 8  实质上还是颜色阈值的设置，可以稍微调高一点
    cap.open(video_path);
    if(!cap.isOpened())
    {
        cout<<"unable to open the video"<<endl;
        return  ;
    }
}

void BUFF::process()//优化整体方案，提速,编写技术文档
{
    if(color_type!="red" && color_type!="blue")
    {
        cout<<"incorrect type of color"<<endl;
        return  ;
    }
    while(1)//输出时间，调整整体架构，优化速度等
    {
        clock_t st=clock();
        cap>>origin_img;//1080*1920
        cout<<origin_img.size<<endl;
        if(origin_img.empty())
            break;
        init_img();
        clock_t st_0=clock();
        find_armor();
        clock_t st_1=clock();
        get_center_R();
        if(is_center_ok)
            check_armor();
        circle(origin_img,center_R,2,Scalar(255),5);
        imshow("original",origin_img);
        imshow("result",result_img);
        clock_t end=clock();
        total_cost+=(double)(end-st)/CLOCKS_PER_SEC;
        frame_count++;
        double ave=total_cost/frame_count;
        cout<<"init image cost "<<(double)(st_0-st)/CLOCKS_PER_SEC<<" s"<<endl;
        cout<<"find possible armor cost "<<(double)(st_1-st_0)/CLOCKS_PER_SEC<<" s"<<endl;
        cout<<"find center and check armor cost "<<(double)(end-st_1)/CLOCKS_PER_SEC<<" s"<<endl;
        cout<<"average cost "<<ave<<" s"<<endl;
        waitKey(1); 
    }
}

void BUFF::init_img()//根据能量机关颜色对图像进行分割，得到二值化图像
{
    resize(origin_img,origin_img,Size(0.5*origin_img.cols,0.5*origin_img.rows));//0.5 0.5
    cvtColor(origin_img,YUV_img,CV_BGR2YUV);
    split(YUV_img,YUV_img_channel);
    gray_img=YUV_img_channel[0];
    if(color_type=="red")
        cur_img=YUV_img_channel[2]-YUV_img_channel[1];
    else    //blue
        cur_img=YUV_img_channel[1]-YUV_img_channel[2];
    int thresh_0=255,type_0=0;
    int thresh_1=255,type_1=0;
    std::thread t_0(cv::threshold,cur_img,YUV_bin,color_thresh,255,0);
    std::thread t_1(cv::threshold,gray_img,gray_bin,gray_thresh,255,0);
    //std::thread t_0(cv::threshold,std::ref(cur_img),std::ref(YUV_bin),std::ref(color_thresh),255,0);
    //std::thread t_1(cv::threshold,std::ref(gray_img),std::ref(gray_bin),std::ref(gray_thresh),255,0);
    t_0.join();
    t_1.join();
    
    //threshold(cur_img,YUV_bin,color_thresh,255,0);
    //threshold(gray_img,gray_bin,gray_thresh,255,0);//需要对图像填补完全
    Mat YUV_temp,gray_temp;
    dilate(YUV_bin,YUV_temp,kernel); //暗光环境下稳定性不行,考虑如何提高
    dilate(gray_bin,gray_temp,kernel);
    cur_img=YUV_temp & gray_temp;
    //cur_img=YUV_bin & gray_bin;
    fillHole(cur_img,filled_img);
}

void BUFF::find_armor()
{
    possible_armor.clear();//clean the vector
    cur_img=filled_img-cur_img;
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
        for(int i=0;i<4;i++)
            line(result_img,vertices[i],vertices[(i+1)%4],Scalar(255),2);   //再找圆心，考虑到相对静止，只找有限次数,再用SVM和KF
        possible_armor.push_back(rect);
    }
}

void BUFF::get_center_R()//直接划分四个象限，计算0到2pi的旋转角，看圆心直径到装甲板夹角是否为90
{
    vector< vector<Point> > contours;
    findContours(filled_img,contours,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    int center_max_size=0;
    for(int i=0;i<contours.size();i++)
    {
        double area=fabs(contourArea(contours[i]));
        if(area > 510 || area <100)
            continue;
        RotatedRect rect=minAreaRect(contours[i]);
        float h_w;
        if(rect.size.height > rect.size.width)
            h_w=rect.size.height/rect.size.width;
        else
            h_w=rect.size.width/rect.size.height;
        if(h_w<1.2)
        {
            Point2f* vertices=new Point2f[4];
            rect.points(vertices);
            for(int i=0;i<4;i++)
                line(origin_img,vertices[i],vertices[(i+1)%4],Scalar(255),2);                  
            for(int j=0;j<possible_armor.size();j++)//圆心检测的太晚了,什么原因？？角度似乎有问题
            {
                double theta=atan2(rect.center.y-possible_armor[j].center.y,rect.center.x-possible_armor[j].center.x)/3.1415*180;
                if(theta< -90)
                    theta+=180;
                if(theta>90)
                    theta-=180;
                float armor_ang;
                if(possible_armor[j].size.width>possible_armor[j].size.height)
                    armor_ang=possible_armor[j].angle+90;
                else
                    armor_ang=possible_armor[j].angle;
                //cout<<fabs(armor_ang-theta)<<endl;
                if(fabs(armor_ang-theta)>10)
                    continue;
                if(rect.size.area()>center_max_size)
                {
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
        if(possible_armor[i].size.height>possible_armor[i].size.width)//line01 or 23
        {
            float r_01=pow((center_R.x-vertices[0].x),2)+pow((center_R.y-vertices[0].y),2);
            float r_23=pow((center_R.x-vertices[3].x),2)+pow((center_R.y-vertices[3].y),2);
            if(r_01>r_23)
            {
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
            if(r_12>r_30)
            {
                float delta_x=(vertices[3].x-vertices[0].x)/2;
                float delta_y=(vertices[3].y-vertices[0].y)/2;
                armor_swing.push_back(vertices[3]);
                armor_swing.push_back(vertices[0]);
                armor_swing.push_back(Point2f(center_R.x-delta_x,center_R.y-delta_y));
                armor_swing.push_back(Point2f(center_R.x+delta_x,center_R.y+delta_y));
            }
            else
            {
                float delta_x=(vertices[2].x-vertices[1].x)/2;
                float delta_y=(vertices[2].y-vertices[1].y)/2;
                armor_swing.push_back(vertices[2]);
                armor_swing.push_back(vertices[1]);
                armor_swing.push_back(Point2f(center_R.x-delta_x,center_R.y-delta_y));
                armor_swing.push_back(Point2f(center_R.x+delta_x,center_R.y+delta_y));
            }
        }
        Mat possible_swing(35,135,CV_32FC1);
        for(int i=0;i<4;i++)//show result
                line(origin_img,armor_swing[i],armor_swing[(i+1)%4],Scalar(255),2); 
        for(int i=0;i<3;i++)
            warp_src_points[i]=armor_swing[i];
        Mat warp_mat=getAffineTransform(warp_src_points,warp_dst_points);
        warpAffine(gray_bin,possible_swing,warp_mat,possible_swing.size());
        //test
        possible_swing=possible_swing.reshape(0,1);
        possible_swing.convertTo(possible_swing,CV_32FC1);
        float result=svm->predict(possible_swing);
        if(result==1)
            circle(origin_img,Point((armor_swing[0].x+armor_swing[1].x)/2,(armor_swing[0].y+armor_swing[1].y)/2),5,Scalar(0,255,0));
        if(is_save_data)
        {
            string file_route="./data/"+to_string(frame_count)+".jpg";
            frame_count++;
            imwrite(file_route,possible_swing);
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


