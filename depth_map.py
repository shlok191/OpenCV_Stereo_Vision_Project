import cv2
import numpy as np
from matplotlib import pyplot as plt


def create_point_cloud_file(vertices, colors, filename):

    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = """ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """

    with open(filename, "w") as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, "%f %f %f %d %d %d")


# Storing the internal matrix values
matL = np.array([[2068.44, 0, 964.405], [0, 2064.29, 593.512], [0, 0, 1]])

matR = np.array([[2065.6, 0, 952.098], [0, 2061.9, 606.559], [0, 0, 1]])

distL = np.array([[-0.108441, 0.158389, 0.000537106, -0.00127904]])
distR = np.array([[-0.114041, 0.178219, -0.000146877, -0.00112736]])

wL = 1936
hL = 1216

wR = 1936
hR = 1216

new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(matL, distL, (wL, hL), 1, (wL, hL))
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(matR, distR, (wR, hR), 1, (wR, hR))

rot = np.array(
    [
        [0.999997, 0.00172602, 0.0019164],
        [-0.00172807, 0.999998, 0.00106908],
        [-0.00191455, -0.00107239, 0.999998],
    ]
)

trans = np.array([[-0.301556], [0.000599], [0.001489]])

# Defining the left and the right images for processing
imgL = cv2.imread("images/CamL_259659217_1024.png", 0)
imgR = cv2.imread("images/CamR_259659217_1024.png", 0)

# Defining the stereo characteristics necessary for generation of a disparity map
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

# Extrapolation of disparity and depth maps
disparity = stereo.compute(imgL, imgR)
disparity_map = np.float32(np.divide(disparity, 16.0))

plt.imshow(disparity)
plt.show()

rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
    new_mtxL, distL, new_mtxR, distR, imgL.shape[::-1], rot, trans, 1, (0, 0)
)

image = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
colors = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

mask_map = disparity_map > disparity_map.min()

output_points = image[mask_map]
output_colors = colors[mask_map]

output_file = "pointCloud.ply"

# Experimentation with creation of point cloud of environment!
create_point_cloud_file(output_points, output_colors, output_file)
