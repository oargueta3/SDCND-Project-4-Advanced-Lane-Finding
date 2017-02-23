import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def pickle_save(dist, mtx, PATH):
	""" Save the camera calibration results for camera matrix
	and distortion coefficients """
	dist_pickle = {}
	dist_pickle["mtx"] = mtx
	dist_pickle["dist"] = dist
	pickle.dump( dist_pickle, open(PATH + "dist_pickle.p", "wb" ))

	print()
	print ('Distortion Coefficients and Camera Matrix Saved....')
	print()

if __name__ == "__main__":
	
	debug = True # flag: plots undistorted test image

	# Chessboard size - (inside corners in x, inside corners in y)
	board_size = (9, 6) 
	
	# Read calibration images
	image_files = glob.glob("camera_cal/calibration*.jpg")

    # Generate object points (0,0,0), (1,2,0)...(x,y,z)
	objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
	objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2)
	
	# Store object and image points for all calibration images
	img_points = [] # 2D points in image plane - distorted
	obj_points = [] # 3D points in the real world space - undistorted 

	for file_name in image_files:
		image = cv2.imread(file_name)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# Find the chessboard corners - image points
		ret, corners = cv2.findChessboardCorners(gray, board_size, None)	
		if ret == True:
			img_points.append(corners)
			obj_points.append(objp)

	# Calibrate Camera
	image_size = gray.shape[::-1]
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, 
														image_size, None, None)
	
	# Save distortion coefficients and Camera Matrix in a pickle file
	save_path = "" 
	pickle_save(dist, mtx, save_path)
	
	# Test calibration
	if debug:
		# Read in the saved objpoints and imgpoints
		dist_pickle = pickle.load(open("dist_pickle.p", "rb"))
		bjpoints = dist_pickle["mtx"]
		imgpoints = dist_pickle["dist"]

		# Undistort test image
		img = cv2.imread('camera_cal/test_images.jpg')
		undist = cv2.undistort(img, mtx, dist, None, mtx)

		# plot  original and undistorted images
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
		ax1.imshow(img)
		ax1.set_title('Original Distorted Image')
		ax2.imshow(undist)
		ax2.set_title('Undistorted Image')
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()


