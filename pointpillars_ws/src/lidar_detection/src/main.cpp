#include <fstream>
#include <iostream>
#include <iomanip> //设置输出格式
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <dirent.h>
#include <time.h>
#include <cmath>
#include <string>
#include <string.h>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"



// PP
#include "pointpillar.h"
#include "params.h"
std::string Model_File = "/model/pointpillar.onnx";


// IA

#include "logging.h"
#include "params.h"
#include "submConv3dlayer.h"
#include "sparseConv3dlayer.h"
#include "voxelGenerator.h"
#include "sparse2Dense.h"
#include "zeroPad2d.h"
#include "generateAnchorDecode.h"
#include "filterBoxByScore.h"

// ROS
#include <ros/ros.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>

using namespace nvinfer1;


typedef std::unique_lock<std::mutex> ULK;
std::mutex pcd_mtx;
sensor_msgs::PointCloud2 pcd_buf;

std::vector<std::string> box_type;

void PointCloudCallback(const sensor_msgs::PointCloud2& msg) {
    ULK ulk(pcd_mtx);
    pcd_buf = msg;
}









int main(int argc, char** argv) {
    ros::init(argc, argv, "cuda_pp_cia_node");
    ros::NodeHandle nh("~");
    std::string Src_Path = "";
    nh.getParam("src_path", Src_Path);
    std::string lidar_topic;
    nh.getParam("lidar_topic", lidar_topic);
    ros::Subscriber pclsub = nh.subscribe(lidar_topic, 10, PointCloudCallback);
    std::string vis_topic;
    nh.getParam("vis_topic", vis_topic);
    ros::Publisher markerpub = nh.advertise<visualization_msgs::MarkerArray>(vis_topic, 10);
    std::string vis_color;
    nh.getParam("vis_color", vis_color);
    ros::Rate rate(10);
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaStream_t stream = NULL;

    
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaSetDevice(DEVICE);
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    
    Params params_;

    box_type.emplace_back("car");
    box_type.emplace_back("people");
    box_type.emplace_back("people");

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("cia-ssd-spp.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("cia-ssd-spp.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./cia-ssd-spp -s  // serialize model to plan file" << std::endl;
        std::cerr << "./cia-ssd-spp -d// deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::cout << "detection start   " << std::endl;
    IRuntime* runtime = createInferRuntime(rt_glogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    context->setOptimizationProfile(0);
    int line_num = MAX_BOX_NUM;
    int feature_map_channel = LAST_DIMS;
    unsigned int voxel_feature_byte_size = 1 * line_num * feature_map_channel * sizeof(float);
    const ICudaEngine& work_engine = context->getEngine();
    assert(work_engine.getNbBindings() == 4);
    void* buffers[4];
    const int inputIndex1 = work_engine.getBindingIndex(INPUT_POINTS);
    const int inputIndex2 = work_engine.getBindingIndex(INPUT_POINTS_SIZE);
    const int outputIndex1 = work_engine.getBindingIndex(OUTPUT_VOXELS);
    const int outputIndex3 = work_engine.getBindingIndex(OUTPUT_VOXEL_NUM);
    context->setBindingDimensions(inputIndex1, Dims3{ 1, MAX_POINTS,4 });
    Dims dims1;
    dims1.d[0] = 1;
    dims1.nbDims = 1;
    context->setBindingDimensions(inputIndex2, dims1);
    // Create GPU buffers on device
    checkCudaErrors(cudaMalloc(&buffers[inputIndex1], 1 * MAX_POINTS * 4 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&buffers[inputIndex2], 1 * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&buffers[outputIndex1], voxel_feature_byte_size));
    // checkCudaErrors(cudaMalloc(&buffers[outputIndex2],1 * output_max_voxel * 4 * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&buffers[outputIndex3], 1 * sizeof(unsigned int)));

    // Create stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    float* voxel_feature = (float*) malloc(voxel_feature_byte_size);
    std::vector<Bndbox> nms_pred;
    nms_pred.reserve(100);
    std::vector<Bndbox> res_;
    

    std::vector<Bndbox> nms_pred;
    nms_pred.reserve(100);

    PointPillar pointpillar(Src_Path + Model_File, stream);

    while (ros::ok()) {
        const clock_t begin_time = clock();
        auto st = system_clock::now();

        //for test, output to file
        int n_zero = 6;
        std::string _str = "0";
        std::string index_str = std::string(n_zero - _str.length(), '0') + _str;

        ULK ulk(pcd_mtx);
        // 计算点的数量
        size_t points_size = pcd_buf.width * pcd_buf.height;

        // 分配内存，用于存储点云数据
        float* points = new float[points_size * 4]; // 每个点对应4个float (x, y, z, intensity)

        if (pcd_buf.fields.empty()) {
            ulk.unlock();
            ROS_INFO("waiting for pointcloud");
            ros::spinOnce();
            rate.sleep();
            continue;
        }

        // 使用PointCloud2的迭代器来访问数据
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(pcd_buf, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(pcd_buf, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(pcd_buf, "z");
        sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(pcd_buf, "intensity");

        for (size_t i = 0; i < points_size; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
            points[i * 4 + 0] = *iter_x;       // x
            points[i * 4 + 1] = *iter_y;       // y
            points[i * 4 + 2] = *iter_z;       // z
            points[i * 4 + 3] = *iter_intensity; // intensity
        }
        ulk.unlock();

        std::cout << "find points num: " << points_size << std::endl;

        float* points_data = nullptr;
        unsigned int points_data_size = points_size * 4 * sizeof(float);
        checkCudaErrors(cudaMallocManaged((void**) &points_data, points_data_size));
        checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
        checkCudaErrors(cudaDeviceSynchronize());
        // delete[] points;

        checkCudaErrors(cudaMemcpy(buffers[inputIndex1], points, 1 * MAX_POINTS * 4 * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(buffers[inputIndex2], &points_size, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "find points num: " << points_size << std::endl;
        context->enqueueV2(buffers, stream, nullptr);
        delete[] points;

        checkCudaErrors(cudaMemcpy(voxel_feature, buffers[outputIndex1],
            voxel_feature_byte_size, cudaMemcpyDeviceToHost));
        int voxel_num = 0;
        checkCudaErrors(cudaMemcpy(&voxel_num, buffers[outputIndex3], 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::cout << "voxel_num: " << voxel_num << std::endl;
        
        cudaEventRecord(start, stream);


        pointpillar.doinfer(points_data, points_size, nms_pred);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "TIME: pointpillar: " << elapsedTime << " ms." << std::endl;

        checkCudaErrors(cudaFree(points_data));

        std::cout << "Bndbox objs: " << nms_pred.size() << std::endl;

        PublishBoxPred(nms_pred, markerpub, vis_color);
        nms_pred.clear();

        std::cout << ">>>>>>>>>>>" << std::endl;
        ros::spinOnce();
        rate.sleep();
    }


    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));

    return 0;

}