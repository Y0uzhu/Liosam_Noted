#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

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

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Publisher pubSLAMInfo;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lio_sam::cloud_info cloudInfo;
    // 存储了每一个关键帧的角点和平面点
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;// 历史所有的关键帧位置
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;// 历史所有的关键帧位姿
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
    // 当前帧的特征点
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; 
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; 

    pcl::PointCloud<PointType>::Ptr laserCloudOri;// 所有参与优化的特征点
    pcl::PointCloud<PointType>::Ptr coeffSel;// 所有参与优化的残差参数

    std::vector<PointType> laserCloudOriCornerVec; // 参与优化的角点
    std::vector<PointType> coeffSelCornerVec;// 角点的残差函数的参数
    std::vector<bool> laserCloudOriCornerFlag;// 参与优化的角点标志位
    std::vector<PointType> laserCloudOriSurfVec; // 参与优化的平面点
    std::vector<PointType> coeffSelSurfVec;// 平面点的残差函数的参数
    std::vector<bool> laserCloudOriSurfFlag;// 参与优化的平面点标志位
    // 存了每一帧的角点和平面点(已经转换到了世界坐标系下),first是角点,second是平面点
    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    //最后一个关键帧的周围的关键帧中的所有的特征点(已经转换到了世界坐标系下)
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;// 当前帧时间戳

    float transformTobeMapped[6];// 待优化变量,前三项为旋转

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;
    //最后一帧关键帧周围的关键帧中的所有的特征点的数量
    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    // 当前帧的特征点数量
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;// 某种标志位
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;// 回环队列,first是老帧的索引,second为新帧的索引
    vector<gtsam::Pose3> loopPoseQueue;// 回环新帧到老帧的位姿变换
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;// 由待优化量变换而来的变换矩阵(优化前)
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;// 由优化量变换而来的变换矩阵(优化后)


    mapOptimization()
    {
        // 构建isam
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        pubSLAMInfo           = nh.advertise<lio_sam::cloud_info>("lio_sam/mapping/slam_info", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }
    /**
     * @brief 先保存时间戳，然后提取已经在featureExtraction.cpp中被提取的角点和平面点信息，
     * 保存到*laserCloudCornerLast和*laserCloudSurfLast中
     * 
     * @param msgIn 作者自定义的一个特殊的msg类型，包含了当前激光帧的角点集合，平面点集合，
     * 一维数组，每根线开始的点和结束点在一维数组中的索引……
     */
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        // 频率控制：当前时刻-上一时刻>=0.15s时，才开始循环
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }
    /**
     * @brief 将关键帧点云转换到世界坐标系下
     * 
     * @param cloudIn 关键帧的特征点云
     * @param transformIn 关键帧位姿
     * @return pcl::PointCloud<PointType>::Ptr 
     */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    













    bool saveMapService(lio_sam::save_mapRequest& req, lio_sam::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        // save corner cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        lio_sam::save_mapRequest  req;
        lio_sam::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }












    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }
    /**
     * @brief 监听外部的回环消息,没用到
     * 
     * @param loopMsg 
     */
    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }
    /**
     * @brief 闭环scan-to-map，icp优化位姿
     * 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     * 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
     * 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
     * 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
     * 
     */
    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;
        // 把关键帧复制一下,后面可能涉及到更改
        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;// 当前关键帧索引
        int loopKeyPre;// 候选闭环匹配帧索引
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());// 当前关键帧的特征点集合
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());// 闭环及其周围帧的特征点集合
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            // 发布闭环匹配关键帧局部map
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        // ICP参数设置
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        // scan-to-map，调用icp匹配
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);
        // 未收敛，或者匹配不够好
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        // 通过回环算得的当前帧位姿
        // 这里不对呀,太不对了
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        // 闭环匹配帧的位姿
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // 添加闭环因子需要的数据
        //这些内容会在函数addLoopFactor中用到
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }
    /**
     * @brief 在历史关键帧中查找与最后一个关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     * 
     * @param latestID 最后一帧关键帧
     * @param closestID 对应的回环帧
     * @return true 
     * @return false 
     */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 最后一个关键帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 当前帧已经添加过闭环对应关系，不再继续添加
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
        //配置文件中默认historyKeyframeSearchRadius=15m
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
        //配置文件中默认30s
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }
    /**
     * @brief not-used, 来自外部闭环检测程序提供的闭环匹配索引对
     * 
     * @param latestID 
     * @param closestID 
     * @return true 
     * @return false 
     */
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }
    /**
     * @brief 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合(转换到世界系下)，降采样
     * 
     * @param nearKeyframes 
     * @param key 
     * @param searchNum 
     */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }







    


    /**
     * @brief 当前帧位姿初始化,主要是向transformTobeMapped传入初值
     * 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     * 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
     * 
     */
    void updateInitialGuess()
    {
        // save current transformation before any processing
        // 前一帧的位姿，注：这里指lidar的位姿，后面都简写成位姿
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);
        // 前一帧的初始化姿态角（来自原始imu数据），用于估计第一帧的位姿（旋转部分）
        static Eigen::Affine3f lastImuTransformation;
        // initialization
        // 如果关键帧集合为空，则进行初始化
        if (cloudKeyPoses3D->points.empty())
        {
            // 当前帧位姿的旋转部分，用激光帧信息中的RPY（来自imu原始数据）初始化
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        // 用当前帧和前一帧对应的imu里程计计算相对位姿变换，
        // 再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        //odomAvailable和imuAvailable均来源于imageProjection.cpp中赋值，
        //imuAvailable是遍历激光帧前后起止时刻0.01s之内的imu数据，
        //如果都没有那就是false，因为imu频率一般比激光帧快，因此这里应该是都有的。
        //odomAvailable同理，是监听imu里程计的位姿，如果没有紧挨着激光帧的imu里程计数据，那么就是false；
        //这俩应该一般都有
        if (cloudInfo.odomAvailable == true)
        {
            // cloudInfo来自featureExtraction.cpp发布的lio_sam/feature/cloud_info,
            //而其中的initialGuessX等信息本质上来源于ImageProjection.cpp发布的deskew/cloud_info信息，
            //而deskew/cloud_info中的initialGuessX则来源于ImageProjection.cpp中的回调函数odometryHandler，
            //odometryHandler订阅的是imuPreintegration.cpp发布的odometry/imu_incremental话题，
            //该话题发布的xyz是imu在前一帧雷达基础上的增量位姿
            //纠正一个观点：增量位姿，指的绝不是预积分位姿！！是在前一帧雷达的基础上(包括该基础!!)的（基础不是0）的位姿
            //这个可以作为当前帧的初始估计位姿（来自imu里程计），后面用来计算增量位姿变换
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            // 如果是第一帧
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                // 得到到上一帧的变换
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                // 前一帧的位姿,这里应该是优化完后的量
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // 当前帧的位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 将transFinal传入，结果输出至transformTobeMapped中,作为当前帧的初值
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        // 只在第一帧调用（注意上面的return），用imu数据初始化当前帧位姿，仅初始化旋转部分
        // 也有可能是imuodom数据不足时
        if (cloudInfo.imuAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        // 存储在最后一个关键帧周围所有的关键帧
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        // 存储在最后一个关键帧周围所有的关键帧的下采样
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        // 构建关键帧kd树,搜索在最后一个关键帧距离范围内的所有的关键帧
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        for(auto& pt : surroundingKeyPosesDS->points)
        {
            //可能是认为降采样以后有误差,所以再搜索距离降采样以后的最近的一个
            //把强度替换掉，注意是从原始关键帧数据中替换的,相当于是把下采样以后的点的强度，换成是原始点强度
            //注意，这里的intensity应该不是强度，因为在函数saveKeyFramesAndFactor中:
            //thisPose3D.intensity = cloudKeyPoses3D->size();
            //就是索引，只不过这里借用intensity结构来存放
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        // 提取了一些最新的关键帧(10s内的)加入到surroundingKeyPosesDS中，以防机器人在一个位置原地旋转 
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }
        // 对降采样后的点云进行提取出边缘点和平面点对应的localmap 
        extractCloud(surroundingKeyPosesDS);
    }
    /**
     * @brief 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
     * 
     * @param cloudToExtract 关键帧的集合
     */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        // 遍历每一个关键帧
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            // 该帧的id
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }
    /**
     * @brief 提取局部角点、平面点云集合，加入局部map
     * 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     * 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
     * 
     */
    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }
    /**
     * @brief 把6维的待优化量转换成变换矩阵
     * 
     */
    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }
    /**
     * @brief 当前激光帧角点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，
     *        且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     * 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     * 
     */
    void cornerOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历当前帧的每一个角点
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));// 协方差矩阵
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;// 重心
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                    // area_012，也就是三个点组成的三角形面积*2，叉积的模|axb|=a*b*sin(theta)
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                    // line_12，底边边长
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                    // 两次叉积，得到点到直线的垂线段单位向量，x分量，下面同理
                    // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    // 得到底边上的高的方向向量[la,lb,lc]
                    // LLL=[la,lb,lc]是V1[0]这条高上的单位法向量。||LLL||=1；
 
                    //如不理解则看图：
                    //        A
                    //   B        C
                    // 这里ABxAC，代表垂直于ABC面的法向量，其模长为平行四边形面积
                    //因此BCx(ABxAC),代表了BC和（ABC平面的法向量）的叉乘，那么其实这个向量就是A到BC的垂线的方向向量
                    //那么(ABxAC)/|ABxAC|,代表着ABC平面的单位法向量
                    //BCxABC平面单位法向量，即为一个长度为|BC|的（A到BC垂线的方向向量），因此再除以|BC|，得到A到BC垂线的单位方向向量
                    // 所以下面为0点到12所组成的线段的垂线的方向向量
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
                    // 0点到线段的距离
                    float ld2 = a012 / l12;
                    // 下面涉及到一个鲁棒核函数，作者简单地设计了这个核函数。
                    // 距离越大，s越小，是个距离惩罚因子（权重）
                    float s = 1 - 0.9 * fabs(ld2);
                    // coeff代表系数的意思
                    // coff用于保存距离的方向向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                    // 程序末尾根据s的值来判断是否将点云点放入点云集合laserCloudOri以及coeffSel中。
                    // 所以就应该认为这个点是边缘点
                    // s>0.1 也就是要求点到直线的距离ld2要小于1m
                    // s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;
            // 想要求解平面方程pa*X+pb*Y+pc*Z+1=0的参数
            // 构造A0*X0=B0的方程, A0就是5个点按列排布(5*3), X0就是[pa, pb, pc], B0就是[-1, -1, -1]
            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();
            
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // 求解
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;
                // 归一化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;
                // 每个点距离平面都要小于0.2
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
                    // 点到平面距离并附上鲁棒核函数
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));
                    // 残差参数
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }
    /**
     * @brief 把参与优化的点和残差参数都放到一起
     * 
     */
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);
        // 参与优化特征点数数量
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }
        // 使用高斯牛顿法进行优化,求解 ATA * X = ATB, 所以要获得A和B矩阵, 
        // 这部分具体公式见https://blog.csdn.net/weixin_37835423/article/details/111587379
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;
        // 遍历匹配特征点，构建Jacobian矩阵
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // camera -> lidar
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // 求解X
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
        // iterCount==0 说明是第一次迭代，需要初始化
        if (iterCount == 0) {
            // 对近似的Hessian矩阵(ATA矩阵)求特征值和特征向量，
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
            // 对近似的Hessian矩阵求特征值和特征向量，
            //matE特征值,matV是特征向量
            //退化方向只与原始的约束方向  A有关，与原始约束的位置 b 无关
            //算这个的目的是要判断退化，即约束中较小的偏移会导致解所在的局部区域发生较大的变化
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            //初次优化时，特征值门限设置为100，小于这个值认为是退化了
            //系统退化与否和系统是否存在解没有必然联系，即使系统出现退化，系统也是可能存在解的，
            //因此需要将系统的解进行调整，一个策略就是将解进行投影，
            //对于退化方向，使用优化的状态估计值，对于非退化方向，依然使用方程的解。
            //另一个策略就是直接抛弃解在退化方向的分量。
            //对于退化方向，我们不考虑，直接丢弃，只考虑非退化方向解的增量。
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }
        // 关于退化问题的解法, 可见
        // https://blog.csdn.net/i_robots/article/details/108724606
        // 以及https://zhuanlan.zhihu.com/p/258159552
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        // 更新优化量
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));
        // 旋转或者平移量足够小就停止这次迭代过程
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }
    /**
     * @brief 特征点配准优化,得到优化后的当前帧位姿
     * 
     */
    void scan2MapOptimization()
    {
        //根据现有地图与最新点云数据进行配准从而更新机器人精确位姿与融合建图，
        //它分为角点优化、平面点优化、配准与更新等部分。
        // 优化的过程与里程计的计算类似，是通过计算点到直线或平面的距离，构建优化公式再用LM法求解。
        if (cloudKeyPoses3D->points.empty())
            return;

        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 用周围关键帧的所有特征点构建kdtree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);
            // 迭代30次
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }
    /**
     * @brief 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
     * 
     */
    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            // imu获得的俯仰角小于1.4
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                // roll角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // pitch角
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }
        // 更新当前帧位姿的roll, pitch, z坐标；因为是小车，roll、pitch是相对稳定的，
        // 不会有很大变动，一定程度上可以信赖imu的数据，z是进行高度约束
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
        // 转换为变换矩阵
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }
    /**
     * @brief 计算当前帧和前一帧位姿变换，如果太小不设关键帧。默认是角度<0.2，距离<1m，两者不满足其一就不保存
     * 
     * @return true 
     * @return false 
     */
    bool saveFrame()
    {
        // 没有关键帧,直接选为关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }
        // 最后一个关键帧的位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 当前帧位姿
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 相对位姿
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
        // 阈值判断
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }
    /**
     * @brief 添加激光里程计因子
     * 
     */
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);// 变量节点设置初始值
        }
    }
    /**
     * @brief 添加GPS因子
     * 
     */
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // 位姿协方差很小，没必要加入GPS数据进行校正
        //3和4我猜可能是x和y？（6维，roll，pitch，yaw，x，y，z）
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // 与当前帧时间戳对齐
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                // GPS噪声协方差太大，不能用
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                // GPS里程计位置
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                // 每隔5m添加一个GPS里程计
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                // 添加GPS因子,cloudKeyPoses3D->size()应该是当前帧的索引
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }
    /**
     * @brief 添加闭环因子.这个其实和当前的回调函数无关，因为当前回调函数监听的是/feature/cloud_info信息，
     * 回环是由其他线程监控和检测的。那么在这里，它查询回环队列，加入回环因子，就是一个顺手的事情，
     * 反正现在要更新优化，那么查一下，如果有候选的等在那里，就顺手加入优化。
     * 如果用做饭来比喻这件事，那么另外的回环检测的线程就是相当于另一个人在备菜，
     * 这里addLoopFactor相当于是在炒菜，备好了就先炒，没有备好就算了
     * 
     */
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            // 闭环边对应两帧的索引
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 闭环边的位姿变换
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        // 判断当前帧是否为关键帧
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        // 执行优化
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }
        // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
        // 优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        // 当前帧位姿结果
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // 加入新的关键帧
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 位姿协方差
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        // 更新当前帧的待优化量,为下一帧提供初值时会用到
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        // 保存关键帧点云
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // 更新里程计轨迹
        updatePath(thisPose6D);
    }
    /**
     * @brief 当有回环模块或者GPS模块介入时,历史关键帧的位姿可能发生改变,此时需要重新更新历史关键帧的位姿
     * 
     */
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                lio_sam::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudCornerLastDS;
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudCornerFromMapDS;
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
