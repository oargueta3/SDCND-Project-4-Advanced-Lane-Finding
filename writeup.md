## **SDCND Project 4: Advanced Lane Finding**

#### Objective
On the first project of the Self-Driving Car Nanodegree we developed a rudimentary algorithm to detect lane lines on roads. This algorithm was only able to detect straight lines under constant lighting and road conditions. In addition, the algorithm was not able to extract valuable road information such as road curvature which we would need to steer the vehicle correctlty. A more robust algorithm is necessary to tackle these challenges. The objective of this project was to develop a  more robust algorithm to tackle the unresolved challenges from project one: 
1. Detecting lanes on roads that can accurately identify road curvature 
2. Detect lanes under varying lighting conditions (shadows, rapid changes of brightness, and pavement color)

### Algorithm Overview
The following are the steps to find road lane lines that overcome the previously discussed challenges:
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./readme_images/calib.png "Undistorted"
[image2]: ./test_images/undist.jpg "Road Transformed"
[image3]: ./readme_images/final_thresh.jpg "Binary Example"
[image4]: ./readme_images/warp.jpg "Warp Example"
[image5]: ./readme_images/hist.jpg "Fit Visual"
[image6]: ./readme_images/slide.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration
---
In order to perform the camera calibration, A seperate script named `camera_calibration.py` was created to save the camera matrix `mtx` and the distortion coefficients `dist` in a pickle file named `dist_pickle.p`. The benefit of doing this is that you only have to compute and load the coefficients once before passing them to the pipeline. The loading operation looks likes this:
```
with open('dist_pickle.p', 'rb') as f:
    dist_params = pickle.load(f)
    mtx = dist_params['mtx']
    dist = dist_params['dist']
```
The script starts by setting the size of the chessboard to 9 by 6 (# inside corners in x, # inside corners in y). Then the "object points" (`objp`), points are prepared, which will be the (x, y, z) coordinates of the chessboard corners in the world. The assumption is that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time "image points" are successfully detected in an image using the OpenCV function **`cv2.FindChessboardCorners`**. "image points" are 2-D points in the image plane for each chessboard corner. `img_points` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then  `obj_points` and `img_points` are used to compute the camera calibration and distortion coefficients using the **`cv2.calibrateCamera()`** function. To apply the  distortion correction the function **`cv2.undistort()`** function. Below is an example of distortion correction on a chessboard image. 

![alt text][image1]

#### Pipeline Distortion Correction
The following wrapper function was made to apply distortion correction in the final video processing pipeline and can be found in **code cell 3 in the `Advanced Lane Detection - Oscar Argueta.ipynb` IPython notebook.**
```
def undistort_image(image, mtx, dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist 
```
An example of a distortion-corrected image can be found below:

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
In order to compute the perspective transform we have to map the points of a given image to  different points in the same image. The perspective transform of interest for the lane finding algorithm is the bird's-eye view because it lets us see the the lane from above. This is extremely useful to calculate the curvature of a given lane. The steps to compute the perspective transform to achieve a bird's-eye view are the following:

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
Once the color and gradient thresholds and the perspective transform have been applied we can utilize the resulting binary image image to locate position of the left and right lines that compose a lane. This process can be summerized in these steps.
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
####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`


### Mapping Detected Lane Back on the Road
---
####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Final Output (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

### Discussion
---
**Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  