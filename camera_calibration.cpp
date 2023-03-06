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
// Chrono demonstration of stereo camera sensor calibration using openCV.
//
// Generates a chessboard object and displaces its location randomly and
// uses generated images to calibrate stereo camera using openCV.
//
// =============================================================================

#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/assets/ChVisualMaterial.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_thirdparty/filesystem/path.h"

#include "chrono_sensor/sensors/ChCameraSensor.h"
#include "chrono_sensor/ChSensorManager.h"
#include "chrono_sensor/filters/ChFilterAccess.h"
#include "chrono_sensor/filters/ChFilterGrayscale.h"
#include "chrono_sensor/filters/ChFilterSave.h"
#include "chrono_sensor/filters/ChFilterVisualize.h"
#include "chrono_sensor/filters/ChFilterCameraNoise.h"
#include "chrono_sensor/filters/ChFilterImageOps.h"

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <unistd.h>

using namespace chrono;
using namespace chrono::geometry;
using namespace chrono::sensor;

// -----------------------------------------------------------------------------
// Camera parameters
// -----------------------------------------------------------------------------

// Camera lens model
CameraLensModelType lens_model = CameraLensModelType::PINHOLE;

// Update rate in Hz
float update_rate = 30;

// Image width and height
unsigned int image_width = 1280;
unsigned int image_height = 720;

// Camera's horizontal field of view
float fov = 1.431f;

// Lag (in seconds) between sensing and when data becomes accessible
float lag = 0.f;

// Exposure (in seconds) of each image
float exposure_time = 0.0f;

int alias_factor = 1;

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Simulation step size
double step_size = 1e-2;

// Simulation end time
float end_time = 100.0f;

// Save camera images
bool save = true;

// Render camera images
bool vis = true;

// Output directory
const std::string out_dir = "SENSOR_OUTPUT/";

// --------------------------------------
// OpenCV parameters
// --------------------------------------

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

// Checkerboard parameters
int CHECKERBOARD[2]{6, 9};

// Path to image folders and string vectors to store individual image locations
std::string pathL = "SENSOR_OUTPUT/camera_rad/left/*.png";
std::string pathR = "SENSOR_OUTPUT/camera_rad/right/*.png";

std::vector<std::string> imagesL;
std::vector<std::string> imagesR;

// Defining cv Mat objects
cv::Mat frameL, frameR, grayL, grayR, imgL, imgR, img_grayL, img_grayR;

// Defining vector to store pixel coordinates of checkerboard corners
std::vector<cv::Point2f> corner_ptsL, corner_ptsR;
bool successL, successR;

// Defining the world coordinates for 3D points
std::vector<cv::Point3f> objp;

// Creating vector to store vectors of 3D points for each checkerboard image
std::vector<std::vector<cv::Point3f>> objpoints;

// Creating vector to store vectors of 2D points for each checkerboard image
std::vector<std::vector<cv::Point2f>> img_pointsL, img_pointsR;

int main(int argc, char *argv[])
{

    // Delete previously existent camera output files
    // rm("/home/shlok/chrono_branches/chrono_lunar_sensor/build/bin/SENSOR_OUTPUT/camera_rad");

    GetLog() << "Copyright (c) 2020 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // -----------------
    // Create the system
    // -----------------
    ChSystemNSC mphysicalSystem;

    // -----------------------------------------
    // add box with checkerboard pattern to the scene
    // -----------------------------------------

    // Defining world coordinates of checkerboard

    for (int i{0}; i < CHECKERBOARD[1]; i++)
    {

        for (int j{0}; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }

    auto box_body = chrono_types::make_shared<ChBodyEasyBox>(0.001, 10 * .023, 7 * .023, 1000, true, false);
    box_body->SetPos({1.9, 0, 0});
    box_body->SetBodyFixed(true);
    mphysicalSystem.Add(box_body);

    auto floor = chrono_types::make_shared<ChBodyEasyBox>(1, 1, 1, 1000, false, false);
    floor->SetPos({0, 0, 0});
    floor->SetBodyFixed(true);
    mphysicalSystem.Add(floor);

    auto checkerboard = chrono_types::make_shared<ChVisualMaterial>();

    checkerboard->SetKdTexture(GetChronoDataFile("sensor/textures/checkerboard.png"));
    checkerboard->SetRoughness(0.8);
    box_body->GetVisualModel()->GetShape(0)->SetMaterial(0, checkerboard);

    // -----------------------
    // Create a sensor manager
    // -----------------------
    auto manager = chrono_types::make_shared<ChSensorManager>(&mphysicalSystem);
    manager->scene->AddPointLight({-10.f, 0.0f, 0.f}, {2.0f / 2, 1.8902f / 2, 1.7568f / 2}, 50.0f);

    // -------------------------------------------------------
    // Create the left camera and add it to the sensor manager
    // -------------------------------------------------------

    chrono::ChFrame<double> offset_pose({1.5, 0, 0}, QUNIT);
    auto cam_l = chrono_types::make_shared<ChCameraSensor>(floor,        // body camera is attached to
                                                           update_rate,  // update rate in Hz
                                                           offset_pose,  // offset pose
                                                           image_width,  // image width
                                                           image_height, // image height
                                                           fov,          // camera's horizontal field of view
                                                           alias_factor, // supersample factor for antialiasing
                                                           lens_model,   // FOV
                                                           false);       // use global illumination or not
    cam_l->SetRadialLensParameters({-.369, .1257, -.0194});
    cam_l->SetName("Left Camera");

    if (vis)
        cam_l->PushFilter(chrono_types::make_shared<ChFilterVisualize>(image_width, image_height, ""));
    if (save)
        cam_l->PushFilter(chrono_types::make_shared<ChFilterSave>(out_dir + "camera_rad/left/"));
    manager->AddSensor(cam_l);

    // --------------------------------------------------------
    // Create the right camera and add it to the sensor manager
    // --------------------------------------------------------

    chrono::ChFrame<double> offset_pose2({1.5, 0.5, 0}, QUNIT);
    auto cam_r = chrono_types::make_shared<ChCameraSensor>(floor,        // body camera is attached to
                                                           update_rate,  // update rate in Hz
                                                           offset_pose2, // offset pose
                                                           image_width,  // image width
                                                           image_height, // image height
                                                           fov,          // camera's horizontal field of view
                                                           alias_factor, // supersample factor for antialiasing
                                                           lens_model,   // FOV
                                                           false);       // use global illumination or not

    cam_r->SetRadialLensParameters({-.369, .1257, -.0194});
    cam_r->SetName("Right Camera");

    if (vis)
        cam_r->PushFilter(chrono_types::make_shared<ChFilterVisualize>(image_width, image_height, ""));
    if (save)
        cam_r->PushFilter(chrono_types::make_shared<ChFilterSave>(out_dir + "camera_rad/right/"));

    manager->AddSensor(cam_r);

    // ---------------
    // Simulate system
    // ---------------
    float ch_time = 0.0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    float x_min = 1.5;
    float x_max = 2.5;
    float y_min = -.5;
    float y_max = .5;
    float z_min = -.4;
    float z_max = .4;

    float ax_min = -CH_C_PI / 6.0;
    float ax_max = CH_C_PI / 6.0;
    float ay_min = -CH_C_PI / 6.0;
    float ay_max = CH_C_PI / 6.0;
    float az_min = -CH_C_PI / 6.0;
    float az_max = CH_C_PI / 6.0;

    // ----------------------------
    // Generating chessboard images
    // ----------------------------

    while (ch_time < end_time)
    {
        // Update sensor manager
        // Will render/save/filter automatically

        // Set object pose randomly to generate calibration images
        box_body->SetPos({x_min + ChRandom() * (x_max - x_min), y_min + ChRandom() * (y_max - y_min),
                          z_min + ChRandom() * (z_max - z_min)});
        box_body->SetRot(
            Q_from_Euler123({ax_min + ChRandom() * (ax_max - ax_min), ay_min + ChRandom() * (ay_max - ay_min),
                             az_min + ChRandom() * (az_max - az_min)}));

        manager->Update();

        // Perform step of dynamics
        mphysicalSystem.DoStepDynamics(step_size);

        // Get the current time of the simulation
        ch_time = (float)mphysicalSystem.GetChTime();
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wall_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    std::cout << "Simulation time: " << ch_time << "\nWall time: " << wall_time.count() << "s.\n";

    // ------------------------------------------------------------
    // Generating corner points of chessboard from generated images
    // ------------------------------------------------------------

    /* Find path for images */

    cv::glob(pathR, imagesR);
    cv::glob(pathL, imagesL);

    // Looping over all the images in the directory
    for (int i{0}; i < imagesL.size(); i++)
    {

        /* Read in frame and store gray_scale into gray */
        frameL = cv::imread(imagesL[i]);
        frameR = cv::imread(imagesR[i]);

        cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true

        successL = findChessboardCorners(grayL, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsL);
        successR = findChessboardCorners(grayR, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsR);

        if (successL && successR)
        {

            std::cout << "Success!" << std::endl;

            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

            // refining pixel coordinates for given 2d points.
            cornerSubPix(grayL, corner_ptsL, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cornerSubPix(grayR, corner_ptsR, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(frameL, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsL, successL);
            cv::drawChessboardCorners(frameR, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsR, successR);

            objpoints.push_back(objp);
            img_pointsL.push_back(corner_ptsL);
            img_pointsR.push_back(corner_ptsR);
        }

        cv::imshow("ImageL", frameL);
        cv::imshow("ImageR", frameR);
        cv::waitKey(1);
    }

    printf("Total objects: %d\n", objpoints.size());
    cv::destroyAllWindows();

    // --------------------------------
    // Calibrating cameras individually
    // --------------------------------

    cv::Mat mtxL, distL, RotL, TransL;
    cv::Mat mtxR, distR, RotR, TransR;
    cv::Mat Rot, Trns, Emat, Fmat;
    cv::Mat new_mtxL, new_mtxR;

    // Calibrating left camera
    cv::calibrateCamera(objpoints, img_pointsL, grayL.size(), mtxL,
                        distL, RotL, TransL);

    new_mtxL = cv::getOptimalNewCameraMatrix(mtxL, distL, grayL.size(),
                                             1, grayL.size(), 0);

    // Calibrating right camera
    cv::calibrateCamera(objpoints, img_pointsR, grayR.size(), mtxR,
                        distR, RotR, TransR);

    new_mtxR = cv::getOptimalNewCameraMatrix(mtxR, distR, grayR.size(),
                                             1, grayR.size(), 0);

    int flag = 0;
    flag |= cv::CALIB_FIX_INTRINSIC;

    // ----------------------------------------
    // Calculating image rectification matrices
    // ----------------------------------------

    // This step is performed to transformation between the two cameras and calculate Essential and
    // Fundamenatl matrix

    cv::stereoCalibrate(objpoints, img_pointsL, img_pointsR, new_mtxL, distL,
                        new_mtxR, distR, grayR.size(), Rot, Trns, Emat, Fmat,
                        flag, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 1e-6));

    cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;

    // Once we know the transformation between the two cameras we can perform
    // stereo rectification

    cv::stereoRectify(new_mtxL, distL, new_mtxR, distR, grayR.size(), Rot, Trns,
                      rect_l, rect_r, proj_mat_l, proj_mat_r, Q, 1);

    cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
    cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

    cv::initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, grayL.size(),
                                CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);

    cv::initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, grayR.size(),
                                CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);

    cv::FileStorage cv_file = cv::FileStorage("data/rectified_parameters.xml", cv::FileStorage::WRITE);

    cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map1);
    cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map2);
    cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map1);
    cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map2);

    return 0;
}