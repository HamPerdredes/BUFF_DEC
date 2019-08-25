#ifndef BIGBUFF_ARMORDETECTOR_H
#define BIGBUFF_ARMORDETECTOR_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#define PI 3.1415
using namespace cv;
using namespace std;

typedef struct
{
    cv::RotatedRect armor;
    vector<Point2f> points;
    Point2f center;
    double angle;
    double area;
    double angle_360;
}ARMOR;

typedef struct
{
    double angle_360;
    Point2f center;
}Aim;

class buff{
public:
    buff();
    buff(string ws);
    ~buff();

    void setSrcImg(cv::Mat src);
    void process(int buffDirection);
    Point2f getCircleCenter();
    Point2f getAim();
    double getAimAngle();
    double getTargetAngle();
    float getDelta_yaw();
    Point2f getTarget();
    bool circleCenterOK();
	
    Mat intriMatrix;
    Point2f center;
    void SetType(int type);

private:
    void initFrame();
    bool findCircleCenter();
    void getPossibleArmor();
    void selectAim();
    void shootTrategy();
    void fillHole(Mat srcBw, Mat &dstBw);
    void adjustRect(cv:: RotatedRect &rect);
    int color = 1;

    Aim aim, target;

    cv::Mat m_frame, m_element_dilate, m_element_erode;
    cv::Mat m_mat_yuv, m_mat_blue, m_mat_red, m_mat_color, m_mat_gray, m_color_area, m_mat_bin, m_init_area;
    std::vector<cv::Mat> m_mat_yuv_channels;
    vector<ARMOR> armors;
    int numLightArmor = 0;
    int color_thresh, gray_thresh, s_AreaThresh, b_AreaThresh, s_dis, b_dis;
    int s_CenterThresh, b_CenterThresh;
    float CenterMaxSize = 0.0;
    double radial;
    int direction = 0;

    Mat fullArea_armor, dilate_img, res, fullArea_arrow, arrow, empty;
    Point2f circleCenter;

    bool isLocked = false;
    vector<Point2f> arrowPoint;

};
#endif //BIGBUFF_ARMORDETECTOR_H
buff::buff() {
	color_thresh=57;
	gray_thresh=53;
	s_AreaThresh=38;
	b_AreaThresh=150;
	s_CenterThresh=10;
	b_CenterThresh=40;
	s_dis=30;
	b_dis=80;
}

buff::~buff() {

}

buff::buff(string ws)
{
    FileStorage fs(ws, FileStorage::READ);
    fs["camera_matrix"] >> intriMatrix;
    string parameter_path;
    fs["buff_parameter_name"] >> parameter_path;
    FileStorage fs_buff(parameter_path, FileStorage::READ);
    fs_buff["color_threshold"] >> color_thresh;
    fs_buff["gray_threshold"] >> gray_thresh;
    fs_buff["s_AreaThresh"] >> s_AreaThresh;
    fs_buff["b_AreaThresh"] >> b_AreaThresh;
    fs_buff["s_CenterThresh"] >> s_CenterThresh;
    fs_buff["b_CenterThresh"] >> b_CenterThresh;
    fs_buff["s_dis"] >> s_dis;
    fs_buff["b_dis"] >> b_dis;
	
	center = Point2f(intriMatrix.ptr<double>(0)[2]/2,intriMatrix.ptr<double>(1)[2]/2);
}

void buff::SetType(int type){
    color = type;         // 0-blue  1-red;
}

void buff::setSrcImg(cv::Mat src) {
    m_frame = src;
    pyrDown(m_frame, m_frame);
    cout << "m_frame.cols: " << m_frame.cols << ", " << "m_frame.rows: " << m_frame.rows << endl;
    aim.center = Point2f(0, 0);
}

void buff::process(int buffDirection) {
    direction = buffDirection;
    std::cout << "direction_1 is: " << direction << std::endl;
    initFrame();
    getPossibleArmor();
    if(armors.size() > 0)
    {
        if(findCircleCenter())
        {
            selectAim();
	circle(m_frame, target.center, 10, Scalar(255, 0, 0), 10);
    	cv::imshow("result", m_frame);
    	waitKey(0);
        }        
    
    
    }
}

void buff::initFrame() {
    cv::cvtColor(m_frame, m_mat_yuv, COLOR_BGR2YUV);
    cv::split(m_mat_yuv, m_mat_yuv_channels);
    m_mat_blue = m_mat_yuv_channels[1] - m_mat_yuv_channels[2];
    m_mat_red = m_mat_yuv_channels[2] - m_mat_yuv_channels[1];
    m_mat_gray = m_mat_yuv_channels[0];
    if(color == 0)    m_mat_color = m_mat_blue;
    else  m_mat_color = m_mat_red;

    cv::threshold(m_mat_color, m_color_area, color_thresh, 255, 0);
    cv::threshold(m_mat_gray, m_mat_bin, gray_thresh, 255, 0);
    m_init_area = m_mat_bin & m_color_area;
}

bool buff::findCircleCenter() {
    circleCenter = Point2f(0.0,0.0);
    vector<vector<Point>> contoursTemp;
    vector<float> error;
    float totalError = 30000;
    findContours(m_init_area, contoursTemp, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<pair<Point2f, double>> circleCenter_temp;
    for(int i=0; i < contoursTemp.size(); i++)
    {
        //cout << "center size: " << contoursTemp.size() << endl;
        error.push_back(0);
        vector<Point> *contourTemp = &contoursTemp[i];
        RotatedRect Rect_temp = cv::minAreaRect(*contourTemp);
        cv::Point2f vertics_temp[4];
        if(Rect_temp.size.height < Rect_temp.size.width)
            swap(Rect_temp.size.height, Rect_temp.size.width);
        double ratio_hw = Rect_temp.size.height/Rect_temp.size.width;
        if(ratio_hw < 1.5 && Rect_temp.size.area() < b_CenterThresh && Rect_temp.size.area() > s_CenterThresh)
        {
            //cout << armors.size() << endl;
            for(int j = 0; j < armors.size(); j++)
            {
                double theta = atan2(Rect_temp.center.y - armors[j].center.y, Rect_temp.center.x - armors[j].center.x)/3.1415*180;
                double dist = sqrt((Rect_temp.center.y - armors[j].center.y) * (Rect_temp.center.y - armors[j].center.y) + (Rect_temp.center.x - armors[j].center.x) * (Rect_temp.center.x - armors[j].center.x));
                if(theta < -90)
                    theta = 180 + theta;
                if(theta > 90)
                    theta = theta - 180;
                float dif_theta = theta - armors[j].angle;
                if(dif_theta < -90)
			dif_theta += 180;
                else if(dif_theta > 90)
                        dif_theta -= 180;
                error[i] += abs(dif_theta);
                //cout << "dif_theta is: " << dif_theta << endl;
                //cout << "dist is: " << dist << endl;
                if(abs(dif_theta) > 10 || dist < s_dis || dist > b_dis)
                {
                    error[i] += 30000;
                    break;
                }
                
            }
            if(error[i] < totalError)
            {
                totalError = error[i];
                circleCenter = Rect_temp.center;;
            }
        }
    }
    circle(m_frame, circleCenter, 2, Scalar(0, 255, 0), 2);
    if(circleCenter == Point2f(0, 0))
    {
        //cout << "find no circleCenter!\n";
        return false;
    }
    else
    {
        cout << "find a center!\n";
        return true;
    }
}

void buff::getPossibleArmor() {
    armors.clear();
    fillHole(m_init_area, fullArea_armor);
    res = fullArea_armor - m_init_area;

    vector<vector<Point>> contoursTemp;
    findContours(res, contoursTemp, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contoursTemp.size(); i++) {
        ARMOR armor_temp;
        vector<Point> *contourTemp = &contoursTemp[i];
        RotatedRect Rect_temp = cv::minAreaRect(*contourTemp);
        cv::Point2f vertices_temp[4];
        Rect_temp.points(vertices_temp);
        adjustRect(Rect_temp);
        double ratio_hw = Rect_temp.size.height / Rect_temp.size.width;
        if (ratio_hw > 1.2 && ratio_hw < 3.5 && Rect_temp.size.area() > s_AreaThresh &&
            Rect_temp.size.area() < b_AreaThresh) {
            for (int j = 0; j < 4; j++)
                armor_temp.points.push_back(vertices_temp[j]);
            armor_temp.armor = Rect_temp;
            armor_temp.angle = Rect_temp.angle;
            armor_temp.area = Rect_temp.size.area();
            armor_temp.center = Rect_temp.center;
            armors.push_back(armor_temp);
        }
    }
}


void buff::selectAim()
{
    float MinRatio = 1.0;
    cv::Point2f vertices_temp[4];
    for(int i = 0; i < armors.size(); i++)
    {
        armors[i].points.push_back(circleCenter);
        RotatedRect Rect_temp = cv::minAreaRect(armors[i].points);
        Rect_temp.points(vertices_temp);
        Mat rot_mat = getRotationMatrix2D(Rect_temp.center, Rect_temp.angle, 1);
        Mat rotate;
        Size dst_sz(res.size());
        warpAffine(res, rotate, rot_mat, dst_sz);
        Mat ROI = rotate(Rect(Rect_temp.center.x - (Rect_temp.size.width/2), Rect_temp.center.y - (Rect_temp.size.height/2), Rect_temp.size.width, Rect_temp.size.height));
        int nonzero_num = countNonZero(ROI);
        float  ratio = nonzero_num/(float)(ROI.cols * ROI.rows);
        if (ratio < MinRatio && ratio < 0.3)
        {
            MinRatio = ratio;
            aim.center = armors[i].center;
            aim.angle_360 =  atan2(circleCenter.y - armors[i].center.y, circleCenter.x - armors[i].center.x)/3.1415*180;
            aim.angle_360 = aim.angle_360 + 270;
            if(aim.angle_360 >= 360)  aim.angle_360 -= 360;
        }
    }
    cout <<"------------------------------\n\n";
    radial = sqrt((aim.center.x - circleCenter.x)*(aim.center.x - circleCenter.x) + 
                      (aim.center.y - circleCenter.y)*(aim.center.y - circleCenter.y));
    //cout << "radial is: " << radial << std::endl;
    //std::cout << "direction_2 is: " << direction << std::endl;
    if(direction == 1)
        target.angle_360 = (int(aim.angle_360 + 10)/60 + 1) * 60;  
    else 
        target.angle_360 = (int(aim.angle_360 - 10)/60 + 0) * 60; 
    if(target.angle_360 >= 360)  target.angle_360 -= 360;         
    //cout << "target angle is: " << target.angle_360 << endl;
    target.center.x = circleCenter.x + radial * sin(target.angle_360/180*M_PI);
    target.center.y = circleCenter.y - radial * cos(target.angle_360/180*M_PI);
    //circle(m_frame, target.center , 2, Scalar(0, 255, 0), 2);
    //circle(m_frame, circleCenter , 2, Scalar(255, 0, 0), 2);
    //imshow("m_frame", m_frame);
    //waitKey(1);

}

Point2f buff::getCircleCenter() {
    return buff::circleCenter;
}

double buff::getAimAngle()
{
    return aim.angle_360;
}

Point2f buff::getAim() {
    return aim.center;
}

double buff::getTargetAngle()
{
    return target.angle_360;
}

float buff::getDelta_yaw()
{
	float delta_x = target.center.x - center.x;
	//float delta_yaw = atan2(delta_x, intriMatrix.ptr<double>(0)[0]/2);
        float delta_yaw = atan2(delta_x, 512);
        //cout << "delta_yaw is: " << delta_yaw * 180 / M_PI << endl;
	return delta_yaw;
}

Point2f buff::getTarget()
{
    return target.center;
}

void buff::fillHole(Mat srcBw, Mat &dstBw)
{
    Size m_Size = srcBw.size();
    Mat Temp=Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255));

    Mat cutImg;
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

bool buff::circleCenterOK()
{
    if(circleCenter == Point2f(0.0, 0.0))
         return false;
    else
         return true;
}
void buff::adjustRect(cv:: RotatedRect &rect)
{
    if(rect.size.width > rect.size.height)
    {
        auto temp = rect.size.height;
        rect.size.height = rect.size.width;
        rect.size.width = temp;
        rect.angle += 90;
        if(rect.angle > 180)
            rect.angle -= 180;
    }

    if(rect.angle > 90)
        rect.angle -= 90;
    else if(rect.angle < -90)
        rect.angle += 90;   // 左灯条角度为负, 右灯条角度为正
}
int main()
{
buff BUFF;
Mat temp;
VideoCapture capture("./blue_ss.mov");//"/home/li/rm/buff/1.mov");
int totalFrames = capture.get(CAP_PROP_FRAME_COUNT);
int currentFrames = 0;

    while(1)
    {

	capture >> temp;
	if (currentFrames == totalFrames - 1)
        {
            currentFrames = 0;
            capture.set(CAP_PROP_POS_FRAMES, 0);
        }
	currentFrames++;
	if(temp.empty()){
	}else{

	BUFF.setSrcImg(temp);
	BUFF.SetType(0);
	BUFF.process(0);
	}
    }
	
	return 0;
}
