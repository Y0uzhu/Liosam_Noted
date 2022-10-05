#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    Eigen::Affine3f lidarOdomAffine;// 最近一帧激光里程计的变换
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;// 最近一帧激光里程计的时间戳
    deque<nav_msgs::Odometry> imuOdomQueue;
    /**
     * @brief 构造函数, 以最近一帧激光里程计位姿为基础，计算该时刻与当前时刻间imu里程计的位姿变换，
     * 相乘得到修正后的当前时刻imu里程计位姿,之后重新发布修订后的位姿和imu的path
     * 
     */
    TransformFusion()
    {
        // 如果lidar系与baselink系不同（激光系和载体系），需要外部提供二者之间的变换关系
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                // 等待3s
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                // lidar系到baselink系的变换
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }
        // 订阅激光里程计，来自mapOptimization
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 
            5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅imuodom,来自本文件的IMUPreintegration类,里边发布的是每一帧imu的位姿
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   
            2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);
    }

    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }
    /**
     * @brief 订阅激光里程计的回调函数，来自mapOptimization,获取最近一帧激光里程计的位姿和时间戳
     * 
     * @param odomMsg 
     */
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 激光里程计对应变换矩阵
        lidarOdomAffine = odom2affine(*odomMsg);
        // 激光里程计时间戳
        lidarOdomTime = odomMsg->header.stamp.toSec();
        //这二者里面保存的都是最近的一个雷达激光里程计的变换和时间戳（不再是用一个vector之类的东西保存起来）
    }
    /**
     * @brief 订阅imu里程计，来自IMUPreintegration
     * 1、以最近一帧激光里程计位姿为基础，计算该时刻与当前时刻间imu里程计的位姿变换，相乘得到修正后的当前时刻imu里程计位姿
     * 2、发布当前时刻里程计位姿，用于rviz展示；发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
     * 
     * @param odomMsg 
     */
    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // 发布tf，map与odom系设为同一个系
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1)
            return;
        // 从imu里程计队列中删除当前（最近的一帧）激光里程计时刻之前的数据
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        // 当前imu时刻到最近的一帧激光里程计时刻的位姿变换
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        // 利用激光里程计,获得修订后的当前imu时刻的位姿
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);// 转换为欧拉角
        
        // 发布
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        // 发布tf，当前时刻odom与baselink系变换关系
        //由于之前把map和odom坐标系固定了，因此这里我认为发布的就是真正的最终位姿关系
        //map优化提供激光，预积分提供imu，imu之间变换再乘以激光里程计得到各个时刻精确位姿
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // 发布imu里程计路径，注：只是最近一帧激光里程计时刻与当前时刻之间的一段
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        // 每隔0.1s添加一个
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            // 删除最近一帧激光里程计时刻之前的imu里程计
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;

    // 这是两个预积分量和两个存储imu数据的容器他们的用途完全不同,opt是用于优化的,而imu是用于处理做lidar位姿预测的。
    //首先，imu传入的数据都会存入到两个容器之中。然后利用激光里程计发来的第i帧的激光位姿，
    //以及imuQueOpt中从i-1到i激光帧之间的imu数据填充imuIntegratorOpt_预积分项，并构建因子图做优化。
    //之后利用因子图优化获得的bias和imuQueImu中第i雷达帧之后的imu数据来构造imuIntegratorImu_这个预积分项。
    //最后利用imuIntegratorImu_、因子图优化获得的第i帧位姿和新传入的imu数据来预测每一个imu帧的位姿
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;
    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1; //odom帧索引
    
    // T_bl: tramsform points from lidar frame to imu frame 
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    // T_lb: tramsform points from imu frame to lidar frame
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
    /**
     * @brief 构造函数
     * 
     */
    IMUPreintegration()
    {
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   
            2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 
            5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }
    /**
     * @brief 重置ISAM2优化器
     * 
     */
    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }
    /**
     * @brief 利用激光里程计发来的第i帧的激光位姿，
     * 以及imuQueOpt中从i-1到i激光帧之间的imu数据填充imuIntegratorOpt_预积分项，并构建因子图做优化。
     * 之后利用因子图优化获得的bias和imuQueImu中第i雷达帧之后的imu数据来构造imuIntegratorImu_这个预积分项。
     * 
     * @param odomMsg 这部分订阅的是/mapping/odometry_incremental，这个话题是由mapOptmization.cpp发布的，
     * 可以把它理解为激光里程计。同理，也不要被incremental误导，觉得好像是两帧激光之间的变换，可不是这样的啊。
     * 它和imu里程计性质类似，就是相对世界坐标系的位姿。
     */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);// 当前odom帧的时间戳

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;
        
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
        // odom发来的雷达位姿
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        // 初始化,相当于利用odom发来的第一帧的位姿创建了一个初始的因子图
        if (systemInitialized == false)
        {
            resetOptimization();

            // pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);// 中间系的位姿
            // 新建一个位姿因子,输入参数:X(0)代表第0帧的位姿序号, 位姿的观测值?, 位姿噪声
            // X可能是固定搭配（当使用Pose时），如果是速度则是V，bias则是B
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            //通过调用总的因子图模型的add方式，添加第一个因子
            graphFactors.add(priorPose);
            // initial velocity
            // 构造并添加速度因子
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // 构造并添加bias因子
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            // 变量节点赋初值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            // 优化一次
            optimizer.update(graphFactors, graphValues);
            //图和节点均清零  为什么要清零不能继续用吗?
            //是因为节点信息保存在gtsam::ISAM2 optimizer，所以要清理后才能继续使用
            graphFactors.resize(0);
            graphValues.clear();
            //积分器重置,重置优化之后的偏置
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }


        // 每隔100帧激光里程计，重置ISAM2优化器，保证优化效率
        if (key == 100)
        {
            // get updated noise before reset
            // 前一帧的位姿、速度、偏置噪声模型
            //保存最后的噪声值
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            // 重置ISAM2优化器
            resetOptimization();
            // 添加位姿先验因子，用前一帧的值初始化
            //重置之后还有类似与初始化的过程 区别在于噪声值不同
            //prevPose_等三项，也是上一时刻得到的，
            //初始时刻是lidar里程计的pose直接用lidar2IMU变量转到imu坐标系下，而此处则是通过上一时刻，即接下来的后续优化中得到
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        // 1. 计算前一帧与当前帧之间的imu预积分量，用前一帧状态施加预积分量得到当前帧初始状态估计，
        //  添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                // imu预积分数据输入：加速度、角速度、dt
                // 加入的是这个用来因子图优化的预积分器imuIntegratorOpt_,注意加入了上一步算出的dt
                // 作者要求的9轴imu数据中欧拉角在本程序文件中没有任何用到,全在地图优化里用到的
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        //利用两帧之间的IMU数据完成了预积分后增加imu因子到因子图中,
        //注意后面容易被遮挡，imuIntegratorOpt_的值经过格式转换被传入preint_imu，
        //因此可以推测imuIntegratorOpt_中的integrateMeasurement函数应该就是一个简单的积分轮子，
        //传入数据和dt，得到一个积分量,数据会被存放在imuIntegratorOpt_中
        const gtsam::PreintegratedImuMeasurements& preint_imu = 
            dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        // 构造一个imu因子,参数：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一帧偏置，预计分量(观测值)
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        // 将imu因子添加到因子图中
        graphFactors.add(imu_factor);
        // add imu bias between factor
        // 添加imu偏置因子，前一帧偏置B(key - 1)，当前帧偏置B(key)，观测值，噪声协方差；deltaTij()是积分段的时间
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        // 构建位姿因子
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        // 向因子图中添加预测值(由之前的状态和bias和当前的预计分量预测而来)
        // 这个预测值就相当于优化的初值
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        // 获得优化的结果
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // 重置预积分器，设置新的偏置，这样下一帧激光里程计进来的时候，预积分量就是新的两帧之间的增量
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 2. 优化之后，执行重传播；优化更新了imu的偏置，
        //用最新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        // 从imu队列中删除当前激光里程计时刻之前的imu数据
        double lastImuQT = -1;
        //注意，这里是要“删除”当前帧“之前”的imu数据，是想根据当前帧“之后”的累积递推。
        //而前面imuIntegratorOpt_做的事情是，“提取”当前帧“之前”的imu数据，用两帧之间的imu数据进行积分。处理过的就弹出来。
        //因此，新到一帧激光帧里程计数据时，imuQueOpt队列变化如下：
        //当前帧之前的数据被提出来做积分，用一个删一个（这样下一帧到达后，队列中就不会有现在这帧之前的数据了）
        //那么在更新完以后，imuQueOpt队列不再变化，剩下的原始imu数据用作下一次优化时的数据。
        //而imuQueImu队列则是把当前帧之前的imu数据都给直接剔除掉，仅保留当前帧之后的imu数据，
        //然后利用前面优化获得的bias重新对后面的imu数据做预积分。
        //经过重新预积分后的imuIntegratorImu_项则会用作两帧lidar里程计到达时刻之间发布的imu增量式里程计的预测.
        //imuQueImu和imuQueOpt的区别要明确,imuIntegratorImu_和imuIntegratorOpt_的区别也要明确,见imuhandler中的注释
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        // 对剩余的imu数据计算预积分
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // 传入状态,重置预积分器和最新的偏置
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            // 遍历之后的每一个imu帧,加入预积分项中
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }
    /**
     * @brief imu因子图优化结果，速度或者偏置过大，认为失败
     * 
     * @param velCur 优化完的速度
     * @param biasCur 优化完的bias
     * @return true 
     * @return false 
     */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }
    /**
     * @brief 将imu转换到中间系,imu预积分,同时预测每一帧imu的状态,最后再变换到lidar系下
     * 把转换后的imu信息分别放入到imuQueOpt和imuQueImu
     * 最后利用imuIntegratorImu_来预测每一个imu帧的位姿，并发布为imu的odom
     * 
     * @param imu_raw 
     */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 把imu信息转换到中间坐标系下
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false)
            return;
    
        double imuTime = ROS_TIME(&thisImu);// 当前帧imu的时间戳
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);// 两帧之间的时间差
        lastImuT_imu = imuTime;

        // integrate this single imu message
        // imu预积分,直接向gtsam的PreintegratedImuMeasurements对象中输入就行了嘛
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // 预测当前帧imu的状态
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
