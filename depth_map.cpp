// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Shlok Sabarwal
// =============================================================================
//
// Chrono demonstration of a camera sensor.
// Generates a mesh object and rotates camera sensor around the mesh to calculate
// a depth map in real-time of the environment.
//
// =============================================================================

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace chrono;

// --------------------------
// Defining openCV parameters
// --------------------------

int numDisparities = 8;
int blockSize = 5;
int preFilterType = 1;
int preFilterSize = 1;
int preFilterCap = 31;
int minDisparity = 2;
int textureThreshold = 10;
int uniquenessRatio = 15;
int speckleRange = 0;
int speckleWindowSize = 0;
int disp12MaxDiff = -1;
int dispType = CV_16S;

// Creating an object of StereoSGBM algorithm
cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();

std::string pathL = "data/SENSOR_INPUT/CAM_DEMO/left/*.png";
std::string pathR = "SENSOR_INPUT/CAM_DEMO/right/*.png";
std::string pathOutput = "data/SENSOR_OUTPUT/";
std::vector<std::string> imagesL;
std::vector<std::string> imagesR;

cv::Mat frameL, frameR, grayL, grayR, imgL, imgR, img_grayL, img_grayR;

// Defining callback functions for the trackbars to update parameter values

static void on_trackbar1(int, void *)
{
    stereo->setNumDisparities(numDisparities * 16);
    numDisparities = numDisparities * 16;
}

static void on_trackbar2(int, void *)
{
    stereo->setBlockSize(blockSize * 2 + 5);
    blockSize = blockSize * 2 + 5;
}

static void on_trackbar3(int, void *)
{
    stereo->setPreFilterType(preFilterType);
}

static void on_trackbar4(int, void *)
{
    stereo->setPreFilterSize(preFilterSize * 2 + 5);
    preFilterSize = preFilterSize * 2 + 5;
}

static void on_trackbar5(int, void *)
{
    stereo->setPreFilterCap(preFilterCap);
}

static void on_trackbar6(int, void *)
{
    stereo->setTextureThreshold(textureThreshold);
}

static void on_trackbar7(int, void *)
{
    stereo->setUniquenessRatio(uniquenessRatio);
}

static void on_trackbar8(int, void *)
{
    stereo->setSpeckleRange(speckleRange);
}

static void on_trackbar9(int, void *)
{
    stereo->setSpeckleWindowSize(speckleWindowSize * 2);
    speckleWindowSize = speckleWindowSize * 2;
}

static void on_trackbar10(int, void *)
{
    stereo->setDisp12MaxDiff(disp12MaxDiff);
}

static void on_trackbar11(int, void *)
{
    stereo->setMinDisparity(minDisparity);
}

// -----------------------------------------------------------------------------
// Camera parameters
// -----------------------------------------------------------------------------

// Noise model attached to the sensor
enum NoiseModel
{
    CONST_NORMAL,    // Gaussian noise with constant mean and standard deviation
    PIXEL_DEPENDENT, // Pixel dependent gaussian noise
    NONE             // No noise model
};

NoiseModel noise_model = NONE;

// Camera lens model
// Either PINHOLE or FOV_LENS
CameraLensModelType lens_model = CameraLensModelType::RADIAL;

// Update rate in Hz
float update_rate = 30.f;

// Image width and height
unsigned int image_width = 1280;
unsigned int image_height = 720;

// Camera's horizontal field of view
float fov = 1.431f; //(float)CH_C_PI / 3.;

// Lag (in seconds) between sensing and when data becomes accessible
float lag = 0.0f;

// Exposure (in seconds) of each image
float exposure_time = 0.0f;

int alias_factor = 1;

bool use_gi = false; // whether cameras should use global illumination

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Simulation step size
double step_size = 1e-2;

// Simulation end time
float end_time = 5.0f;

// Save camera images
bool save = true;

// Render camera images
bool vis = true;

// Output directory
const std::string out_dir = "SENSOR_OUTPUT/CAM_DEMO/";

int main(int argc, char *argv[])
{
    GetLog() << "Copyright (c) 2020 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // ----------------------
    // Creating openCV window
    // ----------------------

    cv::Mat disp, disparity;
    cv::FileStorage cv_file2 = cv::FileStorage("data/rectified_parameters.xml", cv::FileStorage::READ);

    cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
    cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

    cv_file2["Left_Stereo_Map_x"] >> Left_Stereo_Map1;
    cv_file2["Left_Stereo_Map_y"] >> Left_Stereo_Map2;
    cv_file2["Right_Stereo_Map_x"] >> Right_Stereo_Map1;
    cv_file2["Right_Stereo_Map_y"] >> Right_Stereo_Map2;

    cv_file2.release();

    cv::glob(pathR, imagesR);
    cv::glob(pathL, imagesL);

    // Looping over all the images in the directory
    int i = 0;

    while (true)
    {
        frameL = cv::imread(imagesL[i]);
        frameR = cv::imread(imagesR[i]);

        // Converting images to grayscale
        cv::cvtColor(frameL, img_grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frameR, img_grayR, cv::COLOR_BGR2GRAY);

        // Initialize matrix for rectified stereo images
        cv::Mat Left_nice, Right_nice;

        // Applying stereo image rectification on the left image
        cv::remap(img_grayL, Left_nice, Left_Stereo_Map1, Left_Stereo_Map2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);

        // Applying stereo image rectification on the right image
        cv::remap(img_grayR, Right_nice, Right_Stereo_Map1, Right_Stereo_Map2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT,
                  0);

        // cv::imshow("Left Image", Left_nice);
        // cv::imshow("Right Image", Right_nice);

        // Calculating disparity using the StereoBM algorithm
        stereo->compute(img_grayL, img_grayR, disp);

        // Converting disparity values to CV_32F from CV_16S
        disp.convertTo(disparity, CV_32F, 1.0);

        // Scaling down the disparity values and normalizing them
        disparity = (disparity / 16.0f - (float)minDisparity) / ((float)numDisparities);

        // Displaying the disparity map

        if (save)
            cv::imwrite(pathOutput + std::to_string(i) + ".png", disparity);

        // cv::imshow("disparity", disparity);

        i++;
    }

    return 0;
}