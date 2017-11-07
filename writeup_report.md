## Project Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist_calibration1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/threshold_test1.jpg "Binary Example"
[image4]: ./output_images/pts_test2.jpg "Warp Example"
[image5]: ./output_images/lane_lines_image.jpg "Fit Visual"
[image6]: ./output_images/final_image.jpg "Output"
[video1]: ./project_video_solution.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first two code cells of the IPython notebook located in `./solutions/Testbed_1.ipynb` and some portions of it exist in `./solutions/testbed_misc.py` in lines 74 - 89.

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of `objp` every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` as parameters into my `cal_undistort` function which uses the `cv2.calibrateCamera()` to compute the camera calibration and distortion coefficients.  I applied this distortion correction to the test image using the `cv2.undistort()` function in my `cal_undistort` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 42 - 73 in `./solutions/testbed_misc.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 92 - 109 in the file `./solutions/testbed_misc.py` (Its usage can be observed in the 5th code cell of the IPython notebook located in "./solutions/Testbed_1.ipynb").  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I initially chose to hardcode the source and destination points, but I later moved them to a tunable json file which is located in `./solutions/pipeline_settings.ini`:

This values I used for my source and destination points were:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 577, 460      | 320, 0        | 
| 705, 460      | 960, 0        |
| 190, 720      | 320, 720      |
| 1120, 460     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find the lane lines, I started out by getting the histogram of the bottom half of the warped image. Then I used the histogram to identify the point that had the highest value from the mid points on the left and right side. After identifying the base points, I then used a sliding window from the bottom of the binary warped image to the top identifying non zero pixel points as I moved the window up. After doing this I would have values for the lane line pixel points for the left and right lanes.
After I had the values for the left and right lane pixels, I used the polyfit function with a 2nd order polynomial to find a line (line equation with coefficients) that would fit the lane line points.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature of each lane line in lines 293 - 296 in my code in `./solutions/testbed_misc.py`.
I calculated the x-intercept of each lane line in lines 298 - 300 in my code in `./solutions/testbed_misc.py`.
Then I used the average of both lane line radii to calculate the curve radius displayed over the processed video.
To calculate the distance from center, I calculated the midpoint of the x-intercept of the left and right lane lines.
I then subtracted the midpoint from the center of the image to get the distance from the center in pixels, then I scaled it to meters by multplying the distance in pixels by `xm_per_pix`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 261 - 271 in my code in `./solutions/testbed_misc.py` in the function `draw_polyline_on_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_solution.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In general, I tried to follow the approaches and techniques suggested/introduced in the video tutorials.
I also applied some general principles I learnt from my past experiences developing video games.
For example, I tried to move hard coded values to ini files so I can easily tune for better performance.
Also, visualizing the image processing steps was a huge deal, having a debug view that let me see what the algorithm was doing per frame really helped me understand when and where my algorithm was failing (rather than having random guesses).
I also, tried to stick with simple things that work and go complex when staying simple does not solve the problem well enough.
For example, I used the sliding window technique over the convolution technique because it was fairly simple I got it and it worked!
I am not too happy with how I detect parallel lines, I feel there might be a better way, but studying the pattern my algorithm works with measuring the difference in the coefficient of the line fits.
Overall, I think my algorithm does a decent job of detecting the lane lines and compensating for difficult areas, I wonder how fast it would be though if used in real-time (hopefully no humans or animals get injured in Tunde's learning of SDCs!)
