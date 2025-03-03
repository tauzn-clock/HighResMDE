#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "information_optimisation.h"

using namespace std;

array<float,4> get_plane(array<float,3> a, array<float,3> b, array<float,3> c) {
    array<float,3> v1, v2;

    v1[0] = b[0] - a[0];
    v1[1] = b[1] - a[1];
    v1[2] = b[2] - a[2];

    v2[0] = c[0] - a[0];
    v2[1] = c[1] - a[1];
    v2[2] = c[2] - a[2];

    array<float,4> plane;

    plane[0] = v1[1] * v2[2] - v1[2] * v2[1];
    plane[1] = v1[2] * v2[0] - v1[0] * v2[2];
    plane[2] = v1[0] * v2[1] - v1[1] * v2[0];
    plane[3] = -plane[0] * a[0] - plane[1] * a[1] - plane[2] * a[2];

    float norm = sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]) + 1e-10;
    plane[0] /= norm;
    plane[1] /= norm;
    plane[2] /= norm;

    return plane;
}

vector<vector<int> > information_optimisation(cv::Mat depth, YAML::Node config, int max_pt) {

    float fx = config["camera_params"]["fx"].as<float>();
    float fy = config["camera_params"]["fy"].as<float>();
    float cx = config["camera_params"]["cx"].as<float>();
    float cy = config["camera_params"]["cy"].as<float>();
    float scale = config["scale"].as<float>();

    float R = config["R"].as<float>();
    float eps = config["eps"].as<float>();

    float conf = config["conf"].as<float>();
    float inlier_th = config["inlier_th"].as<float>();

    float ITERATION = log(1 - conf) / log(1 - pow(inlier_th, 3));
    cout<<ITERATION<<endl;

    int H = depth.rows;
    int W = depth.cols;

    vector<array<float,3> > points(H*W);
    vector<int> mask(H*W, 0);
    int total_points = 0;

    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            points[i*W + j][0] = (j - cx) * depth.at<unsigned short>(i, j) / fx / scale;
            points[i*W + j][1] = (i - cy) * depth.at<unsigned short>(i, j) / fy / scale;
            points[i*W + j][2] = depth.at<unsigned short>(i, j) / scale;
            if (depth.at<unsigned short>(i, j) == 0) {
                mask[i*W + j] = -1;
            }
            else{
                total_points++;
            }
        }
    }

    vector<float> information(max_pt, 0);
    vector<array<float,4> > plane(max_pt);
    vector<int> available_points(total_points, 0);
    int available_points_cnt = 0;

    information[0] = total_points * log(R/eps);

    array<float,4> trial_plane;

    for (int plane_cnt = 1; plane_cnt < max_pt; plane_cnt++) {
        int available_points_cnt = 0;
        for (int i = 0; i < total_points; i++) {
            if (mask[i] == 0) {
                available_points[available_points_cnt] = i;
                available_points_cnt++;
            }
        }
        
        information[plane_cnt] = information[plane_cnt-1] + total_points * log((plane_cnt+1)/plane_cnt) + 3 * log(R/eps);

        for (int trial=0; trial<ITERATION; trial++){

            int index_a = rand() % available_points_cnt;
            int index_b = rand() % available_points_cnt;
            int index_c = rand() % available_points_cnt;

            if (index_a == index_b || index_a == index_c || index_b == index_c) {
                continue;
            }

            trial_plane = get_plane(points[available_points[index_a]], points[available_points[index_b]], points[available_points[index_c]]);
        }

        plane[plane_cnt] = trial_plane;
    }

    return vector<vector<int> >();
}