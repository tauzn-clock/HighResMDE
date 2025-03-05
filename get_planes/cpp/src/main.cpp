#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include "information_optimisation.h"

int main(int argc, char** argv) {
    std::cout << argv[1] << std::endl;
    
    YAML::Node config = YAML::LoadFile(argv[1]);

    //std::string img_path = config["img_path"].as<std::string>();
    //std::string depth_path = config["depth_path"].as<std::string>();

    //std::cout << "img_path: " << img_path << std::endl;
    //std::cout << "depth_path: " << depth_path << std::endl;

    for(int i=0; i<1449; i++){
        std::string depth_path = "/scratchdata/stair/depth/" + std::to_string(i) + ".png";
        cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

        std::vector<std::vector<int> > plane = information_optimisation(depth, config, 10);

        int H = depth.rows;
        int W = depth.cols;

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                depth.at<ushort>(i, j) = (int)plane[i][j];
            }
        }

        cv::imwrite("/scratchdata/stair/our/"+std::to_string(i)+".png", depth);
    }
    /*
    //cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    //cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

    std::vector<std::vector<int> > plane = information_optimisation(depth, config, 12);

    int H = img.rows;
    int W = img.cols;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            depth.at<ushort>(i, j) = (int)plane[i][j];
        }
    }
    //cv::imwrite("/scratchdata/nyu_plane/new_gt_sigma_1_full/0.png", depth);
    */
    return 0;
}
