#include "utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;// 存放imu消息

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;// 存放imu里程计消息

    std::deque<sensor_msgs::PointCloud2> cloudQueue;// 存放激光点云
    sensor_msgs::PointCloud2 currentCloudMsg;
    // 各帧imu的时间戳,能够完全包含当前雷达帧
    double *imuTime = new double[queueLength];
    // 第一帧imu的旋转量置零,下面从第一帧开始,不停地用角速度来做累加,获得从第一帧到其余各帧的旋转量
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;// imu帧的索引(在当前雷达帧前0.01s处的为第0帧imu)
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;// imu增量开始时相对于雷达帧开始时的变换

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;// 当前帧点云
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;// ouster雷达点云处理过程的中间量
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    // 根据odom的数据得到的,从当前雷达帧结束时到开始时的位移
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;// 当前雷达帧开始的时间
    double timeScanEnd;// 当前雷达帧结束的时间
    std_msgs::Header cloudHeader;// 当前帧雷达消息的整个的头戳信息

    vector<int> columnIdnCountVec;


public:
    /**
     * @brief 构造函数
     * 
     */
    ImageProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}
    /**
     * @brief 将imu消息放入queue中
     * 
     * @param imuMsg 
     */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }
    /**
     * @brief 这个话题是imu里程计，是来自IMUPreintegration(imuPreintegration.cpp中的类IMUPreintegration)发布的里程计话题。
     *既然叫里程计，就不是两帧之间的预积分数据。不要以为它好像是一个增加的量，以为是一段中间时刻内的数据。
     *它就是一个里程计数据，通过imu预积分计算优化来得到的任意时刻在世界坐标系下的位姿。
     * 
     *另外,mapOptimization.cpp文件还会发布一个叫"lio_sam/mapping/odometry_incremental"，
     *它代表的是激光里程计，不要和这里的odometry/imu_incremental混起来。
     * 
     * @param odometryMsg 
     */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        if (!cachePointCloud(laserCloudMsg))
            return;

        if (!deskewInfo())
            return;

        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();
    }
    /**
     * @brief 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
     * 
     * @param laserCloudMsg 
     * @return true 
     * @return false 
     */
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // 取出最早的一帧点云消息
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        // 下面根据雷达类型的不同将点云消息转换为点云
        // velodyne和livox的直接转就行
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        // ouster的感觉像是只取了一部分的属性,同时更改了时间戳的量级
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }
    /**
     * @brief 当前帧起止时刻对应的imu数据、imu里程计数据处理.
     * 
     * 
     * @return true 
     * @return false 
     */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }
    /**
     * @brief 遍历激光帧，从imuqueue中找当前激光帧前0.01s开始，当前帧后0.01s结束，找到最近的一个imu数据，
     * 把它的原始角度数据，作为cloudInfo.imuRollInit等变量。cloudInfo是自定义的一个消息类型，用来之后发布去畸变的点云，
     * 具体可以看cloud_info.msg里面的定义。同时，要根据角速度信息做一个积分，保存到imuRotX等数据结构中，之后用来去畸变。
     * 
     */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;
        // 找到当前激光帧前0.01s以内的imu数据,但是这里只规定了imu开始的时间,没有限制他结束的时间啊
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }
        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];// 拿出一帧imu数据
            double currentImuTime = thisImuMsg.header.stamp.toSec();// 该帧imu的时间戳

            // get roll, pitch, and yaw estimation for this scan
            // 获得在当前雷达帧之前,且离当前雷达帧最近的旋转作为imuRPYInit量
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
            // 这里规定了imu数据的结束时间
            if (currentImuTime > timeScanEnd + 0.01)
                break;
            // 第一帧imu的旋转量置零,下面从第一帧开始,不停地用角速度来做累加,获得从第一帧到其余各帧的旋转量
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // 角速度累加,作为各帧imu的旋转量
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;// 经过累加后,最后停留在倒数第二帧

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }
    /**
     * @brief 遍历imu里程计的odomqueue队列，剔除当前0.01s之前的数据，找到第一个大于当前激光帧的数据，
     * 获得其位姿,用来初始化雷达，也是填充角度，不过这次填充的是cloudInfo.initialGuessRoll等变量，
     * 位置也会被填充到initialGuessX里。
     * 
     * 
     * cloudInfo.initialGuessRoll和cloudInfo.imuRollInit，这些东西都是角度数据，
     * 带Guess的为imu里程计提供的数据，带init的为imu原始数据。
     * 如果有合适的，分别会填充cloudInfo.odomAvailable和cloudInfo.imuAvailable变量为true，代表这块的数据可用。
     * 这些数据会被用在mapOptimization.cpp的updateInitGuess函数，
     * 给激光里程计做一个初始化，然后在这个初始化的基础上进行非线性优化。
     */
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;
        // 要求第一帧odom要比当前雷达帧早
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // 获得比当前雷达帧晚且离当前雷达帧最近的odom数据
        nav_msgs::Odometry startOdomMsg;
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // 给initialGuess量赋值
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;
        // 获得比当前雷达帧的结束时间晚晚且离当前雷达帧的结束时间最近的odom数据
        nav_msgs::Odometry endOdomMsg;
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // T_s_e，结束到开始时的位姿变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
        // 转换成xyzrpy的形式,这里只把xyz传了出去
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }
    /**
     * @brief 根据imu数据获得的旋转增量,得到每一个点相对于imu增量开始时的旋转
     * 
     * @param pointTime 当前点的时间戳
     * @param rotXCur 
     * @param rotYCur 
     * @param rotZCur 
     */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        // 找到离当前点最近的imu帧索引(imu要晚于点的时间)
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // 这里为啥还有pointTime > imuTime[imuPointerFront]的情况呢?
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        // 如果传感器移动速度较慢，例如人行走的速度，那么可以认为激光在一帧时间范围内，平移量小到可以忽略不计
        *posXCur = 0; *posYCur = 0; *posZCur = 0;
        // 否则,直接利用odom数据获得的位移,直接插值获得
        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }
    /**
     * @brief 去畸变  运动补偿 
     * 
     * @param point 
     * @param relTime 
     * @return PointType 
     */
    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;
        // 获得当前点相对于imu增量开始时的旋转和平移
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);
        // 获得imu增量开始时相对于雷达帧开始时的变换
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // 当前点相对于雷达帧开始时的变换
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // 便利当前帧点云中的每一个点
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;
            // 这里是要把点云的深度存到一张图像中,点的线号作为图像的行,
            int rowIdn = laserCloudIn->points[i].ring;// 点的线号
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;
            // 计算点在图像中的列
            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                // 当前点的水平角,范围是[-180,180]
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                // 水平角分辨率, Horizon_SCAN默认为1800,ang_res_x就是0.2
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                // horizonAngle -90 为[-270,90],-round 为[-90,270],再/ang_res_x 为[-450,1350],再+Horizon_SCAN/2为[450,2250]
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                //大于1800，则减去1800，相当于把1801～2250映射到1～450
                //先把columnIdn从horizonAngle:(-PI,PI]转换到columnIdn:[H/4,5H/4],
                //然后判断columnIdn大小，把H到5H/4的部分切下来，补到0～H/4的部分。
                //将它的范围转换到了[0,H] (H:Horizon_SCAN)。
                //这样就把扫描开始的地方角度为0与角度为360的连在了一起，非常巧妙。
                //如果前方是x，左侧是y，那么正后方左边是180，右边是-180。这里的操作就是，把它展开成一幅图:
                //                   0
                //   90                        -90
                //          180 || (-180)
                //  (-180)   -----   (-90)  ------  0  ------ 90 -------180
                //变为:  90 ----180(-180) ---- (-90)  ----- (0)    ----- 90
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }
            
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            // 点云去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            // 将点云深度存到图像中
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 按照图像的行与列储存去畸变后的点云
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }
    /**
     * @brief 把特征依次装到一个一维数组中，以便发布到别的进程里处理。
     * 同时记录每根扫描线起始第5个和倒数第5个激光点在一维数组中的索引
     * 
     */
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            //提取特征的时候，每一行的前5个和最后5个不考虑
            //记录每根扫描线起始第5个激光点在一维数组中的索引
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    // 记录激光点对应的Horizon_SCAN方向上的索引
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    // 激光点距离
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    // 加入有效激光点
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;// 创建对象
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
