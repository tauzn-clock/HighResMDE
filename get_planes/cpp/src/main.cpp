#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include "visualisation.cpp"
#include "get_rgb_regions.cpp"
#include "information_optimisation.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    std::cout << argv[1] << std::endl;
    
    YAML::Node config = YAML::LoadFile(argv[1]);

    std::string rgb_path = config["file_path"].as<std::string>() + "/rgb/" + std::to_string(0) + ".png";
    std::string depth_path = config["file_path"].as<std::string>() + "/depth/" + std::to_string(0) + ".png";

    cv::Mat img = cv::imread(rgb_path, cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

    cv::Mat seg_mask;
    int n_labels = get_rgb_regions(img, config, seg_mask);

    cv::Mat labelImg = visualisation(seg_mask, n_labels);

    cv::Mat plane_mask = cv::Mat::zeros(depth.rows, depth.cols, CV_16UC1);
    unsigned short plane_cnt = 0;

    for (int l = 1; l < n_labels; l++) {
        int cnt = 0;
        std::vector<int> mask(depth.rows * depth.cols);
        for (int i = 0; i < depth.rows; i++) {
            for (int j = 0; j < depth.cols; j++) {
                if (seg_mask.at<int>(i, j) == l) {
                    mask[i * depth.cols + j] = 1;
                    cnt++;
                } 
                else {
                    mask[i * depth.cols + j] = 0;
                }
            }
        }
        std::cout<<cnt<<std::endl;

        int max_plane = information_optimisation(depth, config, 8, mask);

        for (int i=0; i<plane_mask.rows; i++){
            for (int j=0; j<plane_mask.cols; j++){
                if (mask[i*plane_mask.cols+j] > 0 && mask[i*plane_mask.cols+j] <= max_plane){
                    plane_mask.at<unsigned short>(i,j) = (unsigned short) (plane_cnt + mask[i*plane_mask.cols+j]);
                }
            }
        }
        
        plane_cnt += max_plane;
    }

    unsigned int max = 0;
    unsigned int min = 10000;
    for (int i=0; i<plane_mask.rows; i++){
        for (int j=0; j<plane_mask.cols; j++){
            if (plane_mask.at<unsigned short>(i,j) > max){
                max = plane_mask.at<unsigned short>(i,j);
            }
            if (plane_mask.at<unsigned short>(i,j) < min){
                min = plane_mask.at<unsigned short>(i,j);
            }
        }
    }

    std::cout << "max: " << max << std::endl;
    std::cout << "min: " << min << std::endl;
    std::cout << "plane_cnt: " << plane_cnt << std::endl;

    plane_mask.convertTo(plane_mask, CV_32SC1);
    cv::Mat Img = visualisation(plane_mask, plane_cnt+1);

    cv::imshow("labelImg", labelImg);
    cv::imshow("plane_mask", Img);
    cv::waitKey(0);

    //std::cout << "img_path: " << img_path << std::endl;
    //std::cout << "depth_path: " << depth_path << std::endl;

    /*
    for(int i=0; i<1449; i++){
        std::string depth_path = config["file_path"].as<std::string>() + "/depth/" + std::to_string(i) + ".png";
        std::cout << "depth_path: " << depth_path << std::endl;
        cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

        std::vector<std::vector<int> > plane = information_optimisation(depth, config, 10);

        int H = depth.rows;
        int W = depth.cols;

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                depth.at<ushort>(i, j) = (int)plane[i][j];
            }
        }

        cv::imwrite(config["file_path"].as<std::string>() + "/our/" +std::to_string(i)+".png", depth);
    }
    */
    return 0;
}
