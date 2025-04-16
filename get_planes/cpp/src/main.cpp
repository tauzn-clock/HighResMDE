#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
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

    //Grayscale, Blur , Canny
    cv::Mat edge;
    cv::cvtColor(img, edge, cv::COLOR_BGR2GRAY);
    int kernel_size = config["edge_detection_params"]["guassian_blur"]["kernel_size"].as<int>();
    float sigma = config["edge_detection_params"]["guassian_blur"]["sigma"].as<float>();
    cv::GaussianBlur(edge, edge, cv::Size(kernel_size, kernel_size), sigma);
    float low_threshold = config["edge_detection_params"]["canny"]["low_threshold"].as<float>();
    float high_threshold = config["edge_detection_params"]["canny"]["high_threshold"].as<float>();
    cv::Canny(edge, edge, low_threshold, high_threshold);

    //Get mask
    cv::Mat seg_mask;
    cv::bitwise_not(edge, seg_mask);
    int n_lables = cv::connectedComponents(seg_mask, seg_mask, 4);

    std::vector<std::array<int,2> > edge_index;
    for(int i=0; i<edge.rows; i++){
        for(int j=0; j<edge.cols; j++){
            if(edge.at<unsigned char>(i,j) == 255){
                std::array<int,2> index = {i,j};
                edge_index.push_back(index);
            }
        }
    }

    cv::Mat new_seg_mask = seg_mask.clone();
    int edge_size = edge_index.size();
    int new_edge_size;

    std::array< std::array<int,2>, 8> index_array = {
        std::array<int,2>{-1,0},
        std::array<int,2>{1,0},
        std::array<int,2>{0,-1},
        std::array<int,2>{0,1},
        std::array<int,2>{-1,-1},
        std::array<int,2>{-1,1},
        std::array<int,2>{1,-1},
        std::array<int,2>{1,1}
    };

    while (edge_size > 0){
        std::cout << "edge_size: " << edge_size << std::endl;
        new_edge_size = 0;
        for (int i=0; i<edge_size; i++){
            std::array<int,2> index = edge_index[i];
            int x = index[0];
            int y = index[1];

            bool filled = false;
            for (int j=0; j<8; j++){
                int x_ = x + index_array[j][0];
                int y_ = y + index_array[j][1];

                if (x_ >= 0 && x_ < edge.rows && y_ >= 0 && y_ < edge.cols){
                    if (edge.at<unsigned char>(x_,y_) != 255){
                        filled = true;
                        new_seg_mask.at<int>(x,y) = new_seg_mask.at<int>(x_,y_);
                        break;
                    }
                }
            }

            if (not filled){
                edge_index[new_edge_size] = edge_index[i];
                new_edge_size++;
            }
        }
        edge_size = new_edge_size;
        seg_mask = new_seg_mask.clone();
    }

    cv::Mat labelImg(seg_mask.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors(n_lables);
    colors[0] = cv::Vec3b(0, 0, 0); // background
    for (int i = 1; i < n_lables; ++i)
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);

    for (int r = 0; r < seg_mask.rows; ++r) {
        for (int c = 0; c < seg_mask.cols; ++c) {
            int label = seg_mask.at<int>(r, c);
            labelImg.at<cv::Vec3b>(r, c) = colors[label];
        }
    }
    std::cout<<"Number of labels: " << n_lables << std::endl;

    std::vector<std::vector<int> > plane = information_optimisation(depth, config, 10);

    cv::imshow("edge", edge);
    cv::imshow("labelImg", labelImg);
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
