## **SDCND Project 4: Advanced Lane Finding**

### Objective
On the first project of the Self-Driving Car Nanodegree we developed a rudimentary algorithm to detect lane lines on roads. This algorithm was only able to detect straight lines under constant lighting and road conditions. In addition, the algorithm was not able to extract valuable road information such as road curvature which we would need to steer the vehicle correctlty. A more robust algorithm is necessary to tackle these challenges. The objective of this project was to develop a more robust approach to tackle the following unresolved challenges from project one: 

1. Detecting lanes on roads that can accurately identify road curvature 
2. Detect lanes under varying lighting conditions (shadows, rapid changes of brightness, and pavement color)

### Algorithm Overview
The following are the steps to detect road lane lines robustly:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("bird's-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./readme_images/calib.png "Undistorted"
[image2]: ./test_images/undist.png "Road Transformed"
[image3]: ./readme_images/final_thresh.png "Binary Example"
[image4]: ./readme_images/warp.png "Warp Example"
[image5]: ./readme_images/hist.png "Fit Visual"
[image6]: ./readme_images/slide.png "Output"
[image7]: ./readme_images/formula.png "Curvature Equation"
[image8]: ./readme_images/lane_binary.png "Projected Lane"
[image9]: ./readme_images/lane_projection_final.png "Projected Lane"
[video1]: ./project_video.mp4 "Video"

## Camera Calibration
---
In order to perform the camera calibration, A seperate script named `camera_calibration.py` was created to save the camera matrix `mtx` and the distortion coefficients `dist` in a pickle file named `dist_pickle.p`. The benefit of doing this is that you only have to compute and load the coefficients once before passing them to the pipeline. The loading operation looks likes this:
```
with open('dist_pickle.p', 'rb') as f:
    dist_params = pickle.load(f)
    mtx = dist_params['mtx']
    dist = dist_params['dist']
```
The script starts by setting the chessboard size to 9 by 6 (# inside corners in x, # inside corners in y). Then the "object points" (`objp`), points are prepared, which will be the (x, y, z) coordinates of the chessboard corners in the world. The assumption is that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time "image points" are successfully detected in an image using the OpenCV function **`cv2.FindChessboardCorners`**. "image points" are 2-D points in the image plane for each chessboard corner. `img_points` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then  `obj_points` and `img_points` are used to compute the camera calibration and distortion coefficients using the **`cv2.calibrateCamera()`** function. To apply the  distortion correction the function **`cv2.undistort()`** function. Below is an example of distortion correction on a chessboard image. 

![alt text][image1]

#### Pipeline Distortion Correction
The following wrapper function was made to apply distortion correction in the final video processing pipeline and can be found in **code cell 3 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.**
```
def undistort_image(image, mtx, dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist 
```
An example of a distortion-corrected image can be found below.

![alt text][image2]

## Binary Thresholding
---
To reliably detect lane lines on roads a combination of color and gradient thresholds were used to generate generate a binary image. Five different tresholding methods were tested and combined to find the optimal binary image representaion. They are the following:

* `abs_sobel_tresh()`: Method applies the Sobel operation in the x or y direction and returns a binary image
* `mag thresh()`: Method finds the magnitude of the Sobel operation in th x and y direction and returns binary image
* `dir_thresh()`: Method finds the direction of gradient of the Sobel operation and returns a binary image
* `hls_tresh()`: Methods converts the image into HLS color splace and uses the S-channel (saturation) for thresholding and returns a binary image

After experimenting with these, I determined that the best results were obtained by combining the Sobel threshold in the x direction and the color threshold of the S-channel in the HLS color space with a bitwise OR operation. The code for the method called `gradient_color_threshold` looks like this:
```
def gradient_color_thresh(image):
    # Threshold values (min, max)
    thresh_x = (20, 200)
    thresh_hls = (150, 255)
    
    # Apply Sobel gradient in the x-direction and color threshold 
    # on the S-channel of the HLS color space
    binary_x = abs_sobel_thresh(image, orient='x', 
                sobel_kernel=5, thresh=thresh_x)
    binary_hls = hls_thresh(image, thresh=thresh_hls)
    
    # Combined binary output
    combined_binary = np.zeros_like(binary_x)
    combined_binary[(binary_hls==1) | (binary_x==1)] = 1

    return combined_binary
```
The code and the testing results can be found in **code cells 5 through 11 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.**  An example of the output for this step.

![alt text][image3]

## Perspective Transform (Bird's-Eye View)
---
In order to compute the perspective transform, points of a given image have to be mapped to different points in the same image. The perspective transform of interest for the lane finding algorithm is the bird's-eye (top-down) view of the lane ahead. This is extremely useful to calculate the curvature of a given lane. The steps to compute the perspective transform to achieve a bird's-eye view are the following:

1. Determine source points, coordinates of the region of the image you want to transform
2. Determine destination points, coordinates of the region you want to map your source points to
3. Use the OpenCV function **`cv2.getPerspectiveTransform`** to get M, the transform matrix
4. Use the OpenCV function **`cv2.warpPerspective`**  to apply M and warp image to a top-down view

The source and destination points selected were hardcoded and are the following:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 592, 450      | 240, 0        | 
| 180, 720      | 240, 720      |
| 1130, 720     | 1040, 720     |
| 687, 460      | 1040, 0       |

To verify he perspective transform was working as expected the source and destinations points were drawn onto a test image and its warped counterpart to verify that the lines appear parallel and straight.

![alt text][image4]

The code for the perspective transform includes a function called `perspective transform()` that warps the image given a transform matrix, M. The code and the testing results for the perspective transform can be found in **code cells 12 through 16 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.**  

## Curve Fitting
---
Once the color and gradient thresholds and the perspective transform have been applied we can utilize the resulting binary image to locate position of the left and right lines that compose a lane. This process can be summerized in following steps:

1. Identify the initial location of each line by computing the histogram of the lower half of the binary image. The two peaks of the histogram correspond to the initial location of the left and right lines of the lane in the x-axis
 
![alt text][image5]

2. Divide image into 10 horizontal slices
3. Draw a window centered at the initial location of the left and right line and store the x and y coordinates that are within the bounds of the window
4. Repeat Step 3 for each window this time recentering the window to the average of the x values of the pixels that are within the bounds of the window

With the collected x and y coordinates a second degree polynomial was fitted using the function **`numpy.polyfit()`**. Below is a visual representation of this operation that includes the final fitted lines.

![alt text][image6]

The code for detecting the lane lines can be found in **code cells 17 through 22 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.**  

## Radius of Curvature and Lane Center Offset
---
#### Radius of Curvature
With the fitted polynomials it is possible to derive the radius of curvature of the lines. The derivation is shown below.

![alt text][image7]

But before we can calculate this measurement, the x and y pixel coordinates have to be converted to real-world coordinates. In this case meters. The lines were refitted using appropriate scaling factors and the radius of curvature of each line was calculated. The final radius of curvature for the lane is the average values of the left and right line. Below is the code used to obtain this measurement.
```
def get_curvature_radius(left_fit, right_fit, image_size):
    # Convert pixel space to real world space in meters under the assumption
    # that a lane spans 30 meters and it's 3.7 meters wide
    ym_per_pix = 30/720    # meters per pixel in y dimension
    xm_per_pix = 3.7/700   # meters per pixled in x dimension
    
    # Generate  and x, y plotting points along the height of the entire frame
    plot_y = np.linspace(0, 719, num=720)
    left_fitx = np.polyval(left_fit, plot_y)
    right_fitx = np.polyval(right_fit, plot_y)
    
    # Fit new polynomial in real world space (meters)
    left_fit_m = np.polyfit(plot_y*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_m = np.polyfit(plot_y*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Closes point to the front of the car will be current 
    # response of the car for the lane curvature
    y_eval = image_size[0]
    
    # Calculate the new radii of curvature
    left_curve_rad = ((1 + (2*left_fit_m[0]*y_eval*ym_per_pix + left_fit_m[1])**2)**1.5) /                          np.absolute(2*left_fit_m[0])
    right_curve_rad = ((1 + (2*right_fit_m[0]*y_eval*ym_per_pix + right_fit_m[1])**2)**1.5) /                       np.absolute(2*right_fit_m[0])
    lane_curvature = np.mean([left_curve_rad, right_curve_rad])
    return lane_curvature
```

This code can also be found in  **code cells 23 and 24 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.**  

#### Lane Center Offset
To compute the offset of the car from the center of the lane, the left and right lines' polynomial fits were used to evaluate the postion of the line at the bottom of each frame (the position closest to the car). Under the assumption that a lane is 3.7 meters wide, according to the US Department of Transportation, the distance between the left and right line was scaled to a real-world value. To find the offset, the difference between the center of the image and the center of the lane was computed. The code below was used to obtain this measurement:
```
def get_camera_offset(left_fit, right_fit, image_size):
    lane_width = 3.7    # meters
    height = image_size[0]  
    width = image_size[1]
    
    # Obtain the x position of each line
    # at the bottom of each frame
    left_x = np.polyval(left_fit, height)
    right_x = np.polyval(right_fit, height)
    
    # Obtain camera location: center of the frame
    midpoint = width/2
    
    # Scale lane width to meters
    diff = np.absolute(right_x - left_x)
    scale = lane_width/diff
    offset = (midpoint -np.mean([left_x, right_x]))*scale
    return offset
```
This code can also be found in  **code cells 25 and 26 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.**  

### Mapping Detected Lane Back on the Road
---
To visualize the detection back on the road, plotting points were obtained from each line fit (`left_fitx`, `right_fitx` and `ploty`). Using these, the resulting polygon was drawn on an empty image like the example below.

![alt text][image8]

This resulting figure is still in bird's-eye view perspective. To convert this image into it's original perspective the OpenCV function **`cv2.warpPerspective`** was used to apply the inverse of the previously calculated transform matrix, M. The resulting warped image combined with the orignal image frame with the embedded calculations for lane curvature and center lane offset look like the image below.

![alt text][image9]

This code can also be found in  **code cells 27 and 28 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.** 

### Final Output (video)
--- 
The function **`process_frame`** returns the original image with the detected lane drawn on the road along with the curvature and lane center offset measurements. The function keeps track of three global variables to apply a first order filter on the left and right fit polynomial coefficients and the radius of curvature to eliminate noise and smoother performance.

#### Project Video
<a href="https://www.youtube.com/embed/j9AjvjqNx4Q target="_blank"><img src="http://img.youtube.com/vi/j9AjvjqNx4Q/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

### Discussion
---
**Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  