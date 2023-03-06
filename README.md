# OpenCV Stereo Sensing Project

This project is an implementation of Stereo Camera Sensors used for accurate real-time generation of depth-maps of automobile/phone environments from images captured by cameras and/or LIDAR sensors. Depth maps assist with navigating environments for self-driving automobiles by providing a more hollistic description of the surrounding environment and the distance of detected obstacles!

This project was implemented to test practical use of depth maps for RTSL applications for self-driving lunar rovers being worked on at the Simulations Based Engineering Lab (SBEL) organization as part of an upcoming NASA lunar rover mission scheduled for 2024!

To demonstrate the capabilities of the implementation, below are 2 stereo images and the generated depth map from the [POLAR dataset](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/) published by NASA in 2017 containing stereo images captured from an incredibly realistic lunar terrain re-creation for the purpose of training computer vision neural networks:

<p align="center"><img width=37.5% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/images/CamL_259659217_1024.png"/>    <img width=37.5% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/images/CamR_259659217_1024.png"/></p>  

Generated Real Time Depth Image:  

<p align="center"><img width=50% src="https://github.com/shlok191/OpenCV_Stereo_Sensing_Project/blob/main/data/depth_map.png"/><p>

We can view from the demonstrated depth map that even with particularly dark input images, proper data augmentation using OpenCV added with enhanced calculated of sensor extrinsic and intrinsic parameters allows us to generate accurate depth maps!
