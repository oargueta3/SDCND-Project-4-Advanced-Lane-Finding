# Self-Driving Car Nanodegree
# Advanced Lane Detection with OpenCV

### Overview
On the first project of the Self-Driving Car Nanodegree we developed a rudimentary algorithm to detect lane lines on roads. This algorithm was only able to detect straight lines under constant lighting and road conditions. In addition, the algorithm was not able to extract valuable road information such as road curvature which we would need to steer the vehicle correctlty. A more robust algorithm is necessary to tackle these challenges. The objective of this project was to develop a more robust approach to tackle the following unresolved challenges from project one: 

1. Detecting lanes on roads that can accurately identify road curvature 
2. Detect lanes under varying lighting conditions (shadows, rapid changes of brightness, and pavement color)

### Included Files

This project was written in Python. The follwing files were used to create an test the model.

1. `Advanced Lane Finding - Oscar Argueta.ipynb`: Used to develop and tune the model. Detailed explanations and figures can be found in this jupyter notebook 
2. `camera_calibration.py`: File that loads the neural net model and communicates with the simulator to excecute predicted steering angles
3. `dist_pickle.py`: File that was used to make and train a KERAS model for a modified version of the LeNet Architecture. Saves model weights 
4. `result.mp4`: File that contains the trained weights
5. `writeup.md`: File contains a JSON representation of the neural net that can be used to predict steering angles in realtime

### Algorithm
The following are the steps to detect road lane lines robustly:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("bird's-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The diagram below summarizes the pipeline to process images.

<img src="readme_images/flow_chart.png">

