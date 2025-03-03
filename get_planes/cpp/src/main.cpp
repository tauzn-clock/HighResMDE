#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>

int main(int argc, char** argv) {

    std::cout << argv[1] << std::endl;
    
    YAML::Node config = YAML::LoadFile(argv[1]);

    std::string img_path = config["img_path"].as<std::string>();
    std::string depth_path = config["depth_path"].as<std::string>();

    float fx = config["camera_params"]["fx"].as<float>();
    float fy = config["camera_params"]["fy"].as<float>();
    float cx = config["camera_params"]["cx"].as<float>();
    float cy = config["camera_params"]["cy"].as<float>();

    std::cout << "img_path: " << img_path << std::endl;
    std::cout << "depth_path: " << depth_path << std::endl;
    std::cout << "fx: " << fx << std::endl;
    std::cout << "fy: " << fy << std::endl;
    std::cout << "cx: " << cx << std::endl;
    std::cout << "cy: " << cy << std::endl;

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

    int H = img.rows;
    int W = img.cols;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (depth.at<unsigned short>(i, j) == 0) {
                img.at<cv::Vec3b>(i, j)[0] = 0;
                img.at<cv::Vec3b>(i, j)[1] = 0;
                img.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }

    cv::imwrite("new_img.png", img);

    return 0;
}
