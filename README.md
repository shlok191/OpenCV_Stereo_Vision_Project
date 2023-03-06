# OpenCV Stereo Vision & 3-Dimensional Perception Project

This project is an implementation of 3-Dimensional environment perception and navigation by stereo camera sensors used for accurate real-time generation of depth images for vehicles with multiple cameras and/or LIDAR sensors (`active stereo vision commits underway!`. Real-Time depth maps assist with navigating environments for self-driving automobiles by providing a more hollistic description of the surrounding environment and the distance of detected obstacles from instance segmentation to write more appropriate software for self-navigation of automobiles!

This project was implemented to test practical uses of depth maps for RTSL applications for self-driving lunar rovers being worked on at the Simulations Based Engineering Lab (SBEL) organization as part of a planned NASA lunar exploration mission scheduled for 2024. This mission will be conducted by autonomous lunar rovers, for which SBEL has been contracted to develop the software stack!

To demonstrate the initial capacity of the implementation, below are 2 stereo images and the generated depth map from the [POLAR dataset](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/) published by NASA in 2017 containing stereo images captured from incredibly realistic lunar terrain re-creations with realistic obstacles and craters one can expect to find on the Moon for the purpose of training computer vision neural networks to detect obstacles, calculate their distances in real-time, and use the calculated information to navigate the rover safely!

<p align="center"><img width=37.5% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/images/CamL_259659217_1024.png"/>    <img width=37.5% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/images/CamR_259659217_1024.png"/></p>  

Generated Real Time Depth Image:  

<p align="center"><img width=50% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/depth_map.png"/><p>  

We can view from the demonstrated depth map that even with particularly dark input images as one may expect to see on the Moon, proper data augmentation with OpenCV with proper implementation of stereo sensors' calibration in `camera_calibration.cpp` allows us to generate up to 97% accurate depth maps that are pivotal for the planned Real Time Location System (RTLS) being implemented by SBEL.
