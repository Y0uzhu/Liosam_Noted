#include "utility.h"
#include "lio_sam/cloud_info.h"

struct smoothness_t{ 
    float value;// 点的曲率
    size_t ind;// 点的索引
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;// 用于特征提取的所有的激光点,按线排列的
    pcl::PointCloud<PointType>::Ptr cornerCloud;// 提取得到的角点
    pcl::PointCloud<PointType>::Ptr surfaceCloud;// 提取得到的平面点

    pcl::VoxelGrid<PointType> downSizeFilter;

    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;// 被遮挡或者是周围有点被认作特征时置一
    int *cloudLabel;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 
            1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }
    /**
     * @brief 订阅从imageProjection中发来的cloud_info消息,提取特征并重新发布
     * 
     * @param msgIn 
     */
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }
    /**
     * @brief 计算每一个点的曲率
     * 
     */
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        // 遍历每一个点
        for (int i = 5; i < cloudSize - 5; i++)
        {
            // 计算点的曲率
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }
    /**
     * @brief 将可能被遮挡或者与光束大致平行的点标记
     * 
     */
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        // 遍历每一个点
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            // 当前点和下一个点的深度
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            // 两点的列号差
             // 两个激光点之间的一维索引差值，如果在一条扫描线上，那么值为1；
             //如果两个点之间有一些无效点被剔除了，可能会比1大，但不会特别大
            // 如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会很大
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));
            // 两个点在同一扫描线上，且距离相差大于0.3，认为存在遮挡关系
            //（也就是这两个点不在同一平面上，如果在同一平面上，距离相差不会太大）
            //  远处的点会被遮挡，标记一下该点以及相邻的5个点，后面不再进行特征提取
            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            // 用前后相邻点判断当前点所在平面是否与激光束方向平行
            //diff1和diff2是当前点距离前后两个点的距离
            //如果当前点距离左右邻点都过远，则视其为瑕点，因为入射角可能太小导致误差较大
            // 选择距离变化较大的点，并将他们标记为1
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }
    /**
     * @brief 提取特征点
     * 
     */
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());// 平面点
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());// 下采样后的平面点

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
            // 将一条扫描线扫描一周的点云数据，划分为6段，每段分开提取有限数量的特征，保证特征均匀分布
            for (int j = 0; j < 6; j++)
            {
                // 每段点云的起始、结束索引；startRingIndex为扫描线起始第5个激光点在一维数组中的索引
                //注意：所有的点云在这里都是以"一维数组"的形式保存
                //startRingIndex和 endRingIndex 在imageProjection.cpp中的 cloudExtraction函数里被填入
                //假设 当前ring在一维数组中起始点是m，结尾点为n（不包括n），那么6段的起始点分别为：
                // m + [(n-m)/6]*j   j从0～5
                // 化简为 [（6-j)*m + nj ]/6
                // 6段的终止点分别为：
                // m + (n-m)/6 + [(n-m)/6]*j -1  j从0～5,-1是因为最后一个,减去1
                // 化简为 [（5-j)*m + (j+1)*n ]/6 -1
                //这块不必细究边缘值到底是不是划分的准（例如考虑前五个点是不是都不要，还是说只不要前四个点），
                //只是尽可能的分开成六段，首位相接的地方不要。因为庞大的点云中，一两个点其实无关紧要。
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;
                //将每段上的点按照曲率从小到大排序点云
                //可以看出之前的byvalue在这里被当成了判断函数来用
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());
                
                int largestPickedNum = 0;
                // 从后向前,按照曲率从大到小遍历,提取角点
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    // 当前激光点还未被处理，且曲率大于阈值，则认为是角点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        // 每段只取20个角点，如果单条扫描线扫描一周是1800个点，则划分6段，每段300个点，从中提取20个角点
                        largestPickedNum++;
                        if (largestPickedNum <= 20){
                            // 标记为角点,加入角点点云
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }
                        // 标记已被处理
                        cloudNeighborPicked[ind] = 1;
                        // 同一条扫描线上前后各5个点标记一下，不再处理，避免特征聚集
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 从前向后,提取平面点
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    // <=不是把除了角点以外的点都当成是平面点了吗
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}