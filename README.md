# OpenCV Stereo Vision & 3-Dimensional Perception Project

This project is an implementation of 3-Dimensional environment perception by stereo camera sensors used for accurate real-time generation of depth images for mobile phones from images captured by cameras and/or LIDAR sensors. Depth maps assist with navigating environments for self-driving automobiles by providing a more hollistic description of the surrounding environment and the distance of detected obstacles!

This project was implemented to test practical use of depth maps for RTSL applications for self-driving lunar rovers being worked on at the Simulations Based Engineering Lab (SBEL) organization as part of an upcoming NASA lunar rover mission scheduled for 2024!

To demonstrate the capabilities of the implementation, below are 2 stereo images and the generated depth map from the [POLAR dataset](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/) published by NASA in 2017 containing stereo images captured from an incredibly realistic lunar terrain re-creation for the purpose of training computer vision neural networks!

<p align="center"><img width=37.5% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/images/CamL_259659217_1024.png"/>    <img width=37.5% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/images/CamR_259659217_1024.png"/></p>  

Generated Real Time Depth Image:  

<p align="center"><img width=50% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/depth_map.png"/><p>

We can view from the demonstrated depth map that even with particularly dark input images as one may expect to see on the Moon, proper data augmentation with OpenCV's module functionalities added with proper calibration of sensor cameras allows us to generate accurate incredibly accurate depth maps that help locate the distance of rocks and craters on the lunar surface and their appropriate distance to assist the rover in detection and collision prevention!
