import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import os.path
import json


def abs_sobel_thresh(single_channel_image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(single_channel_image, cv2.CV_64F, 1, 0, sobel_kernel))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(single_channel_image, cv2.CV_64F, 0, 1, sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(single_channel_image, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(single_channel_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(single_channel_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 

    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(single_channel_image, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(single_channel_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(single_channel_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def threshold_pipeline(img, thresholds, ksize=3, debug=False):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Gradient Thesholds
    single_ic = s_channel
    grad_x = abs_sobel_thresh(single_ic, orient='x', sobel_kernel=ksize, thresh=thresholds['grad_x'])
    grad_y = abs_sobel_thresh(single_ic, orient='y', sobel_kernel=ksize, thresh=thresholds['grad_y'])
    grad_mag = mag_thresh(single_ic, sobel_kernel=ksize, mag_thresh=thresholds['grad_mag'])
    grad_dir = dir_threshold(single_ic, sobel_kernel=ksize, thresh=thresholds['grad_dir'])
    grad_threshold = ((grad_x == 1) & (grad_y == 1)) & ((grad_mag == 1) & (grad_dir == 1))
    grad_combined = np.zeros_like(grad_dir)
    grad_combined[grad_threshold] = 1

    col_threshold = (s_channel >= thresholds['grad_col'][0]) & (s_channel <= thresholds['grad_col'][1])
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[col_threshold] = 1
    
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(grad_x), grad_combined, s_binary))
    
    combo_binary = np.zeros_like(s_channel)
    combo_binary[grad_threshold | col_threshold] = 255
    
    if debug:
        return grad_combined, grad_x, grad_y, grad_mag, grad_dir, single_ic
    else:
        return combo_binary

def cal_undistort(img, objpoints=None, imgpoints=None):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if objpoints is not None and imgpoints is not None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        camera_calibration = {}
        camera_calibration["mtx"] = mtx
        camera_calibration["dist"] = dist
        pickle.dump(camera_calibration, open("./camera_calibration.p", "wb"))
    else:
        camera_calibration = pickle.load( open( "camera_calibration.p", "rb" ) )
        mtx = camera_calibration["mtx"]
        dist = camera_calibration["dist"]
        
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def warp_image(img, src=None, dst=None):
    img_size = img.shape[:2][::-1]
    
    if src is not None and dst is not None:
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
    
        warp_transform = {}
        warp_transform["M"] = M
        warp_transform["Minv"] = Minv
        pickle.dump(warp_transform, open("./warp_transform.p", "wb"))
    else:
        warp_transform = pickle.load( open( "warp_transform.p", "rb" ) )
        M = warp_transform["M"]
        Minv = warp_transform["Minv"]
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def sliding_window_line_search(binary_warped, margin, nwindows, minpix):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    out_img = out_img.astype(np.uint8)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []
    
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(binary_warped.shape[0]//nwindows)

    leftx_current = leftx_base
    rightx_current = rightx_base

    # Step through the windows one by one
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 6) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 6) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    # left_lane_inds == all the good left inds
    left_lane_inds = np.concatenate(left_lane_inds)
    # right_lane_inds == all the good right inds
    right_lane_inds = np.concatenate(right_lane_inds)
    output_image, left_fit, right_fit, left_fitx, right_fitx, ploty, x_pts, y_pts = \
        update_lane_line_image(out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
    return output_image, left_fitx, right_fitx, ploty, left_fit, right_fit, x_pts, y_pts

def bounded_window_line_search(binary_warped, margin, left_fit, right_fit):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    out_img = out_img.astype(np.uint8)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    output_image, left_fit, right_fit, left_fitx, right_fitx, ploty, x_pts, y_pts = \
        update_lane_line_image(out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
    window_img = np.zeros_like(output_image)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    blended_image = cv2.addWeighted(output_image, 1, window_img, 0.3, 0)
    return blended_image, left_fitx, right_fitx, ploty, left_fit, right_fit, x_pts, y_pts

def update_lane_line_image(image, left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # VISUALIZE
    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    all_x_pts = {'left': leftx, 'right': rightx}
    all_y_pts = {'left': lefty, 'right': righty}
    return image, left_fit, right_fit, left_fitx, right_fitx, ploty, all_x_pts, all_y_pts

def visualize_images(image1, image2, image3):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 9))
    #f.subplots_adjust(hspace=0.3)
    f.tight_layout()
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Threshold Image', fontsize=15)
    ax3.imshow(image3, cmap='gray')
    ax3.set_title('Warped Image', fontsize=15)    
    return ax1, ax2, ax3

def draw_polyline_on_image(binary_warped, undistorted_image, Minv, l_line, r_line, ploty):
    # warp line back unto undistorted image
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([l_line.best_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([r_line.best_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    l_line.plot_line(color_warp, l_line.best_fitx, ploty, (255,0,0))
    r_line.plot_line(color_warp, r_line.best_fitx, ploty, (0,0,255))
    new_warp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    out_image = cv2.addWeighted(undistorted_image, 1, new_warp, 0.3, 0)
    return out_image

class Settings():
    if not os.path.isfile('pipeline_settings.ini'):
        raise Exception('Settings file does not exist, create one.')
        
    with open('pipeline_settings.ini', 'r') as configfile:
        settings_str = configfile.read()
    config_settings = json.loads(settings_str)
    thresholds = config_settings['thresholds']
    src = np.float32(config_settings['src'])
    dst = np.float32(config_settings['dst'])
    nwindows = config_settings['nwindows']
    cache_size = config_settings['cache_size']
    max_diffs = config_settings['max_diffs']
    max_para_diffs = config_settings['max_para_diffs']
    
class Line():
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
    def __init__(self):
        self.detected = False  
        self.recent_fits = []  
        self.recent_fitxs = []   
        self.best_fit = None
        self.best_fitx = None
        self.line_base_pos = None 
        self.diffs = np.array([0,0,0], dtype='float')
        self.curve_rad = None
        self.x_intercept = None
        self.is_parallel = False
        
    def update(self, image_shape, fit, fitx, x, y, ploty):
        if fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(fit - self.best_fit)
                
            # Check if there is a huge difference between this fit and the previous best fit
            huge_diff = (self.diffs[0] > Settings.max_diffs[0] or self.diffs[1] > Settings.max_diffs[1])
            #if (huge_diff and not self.is_parallel) and len(self.recent_fits) > 0:
            if (not self.is_parallel) and len(self.recent_fits) > 0:
                self.detected = False
            else:
                self.detected = True
                
                # update curve radius (radius of curvature)
                y_eval = np.max(ploty)
                fit_cr = np.polyfit(y*Line.ym_per_pix, x*Line.xm_per_pix, 2)
                self.curve_rad = ((1 + (2*fit_cr[0]*y_eval*Line.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

                # update x-intercept
                h = image_shape[0]
                self.x_intercept = fit[0]*h**2 + fit[1]*h + fit[2]
                
                if self.is_parallel:
                    # update recent_fits and recent_fitxs
                    self.recent_fits.append(fit)
                    self.recent_fitxs.append(fitx)
                    if len(self.recent_fits) > Settings.cache_size:
                        self.recent_fits = self.recent_fits[len(self.recent_fits)-Settings.cache_size:]
                        self.recent_fitxs = self.recent_fitxs[len(self.recent_fitxs)-Settings.cache_size:]
        else:
            self.detected = False
            if len(self.recent_fits) > 0:
                self.recent_fits = self.recent_fits[:len(self.recent_fits)-1]
                self.recent_fitxs = self.recent_fitxs[:len(self.recent_fitxs)-1]
                
        if len(self.recent_fits) > 0:
            self.best_fit = np.average(self.recent_fits, axis=0)
            self.best_fitx = np.average(self.recent_fitxs, axis=0)
        else:
            self.best_fit = fit
            self.best_fitx = fitx
        
    def plot_line(self, image, fitx, ploty, color=(255,255,0), line_size=10):
        line_pts = np.column_stack((fitx, ploty))
        line_pts = line_pts.astype(np.int32)
        line_pts = line_pts.reshape((-1,1,2))
        cv2.polylines(image, [line_pts], False, color, line_size)
        
