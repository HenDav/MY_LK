import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from cv2 import resize, INTER_NEAREST
import matplotlib.pyplot as plt

def LucasKanade(It, It1, xy, wh, p0 = np.zeros(2)):
	# Input:
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the object
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
    #   o____x
    #   |
    #   |
    #   y image(y, x) opencv convention

    rect = np.concatenate([xy - wh * 0.5, xy + wh * 0.5])

    threshold = 0.01
    rows_img, cols_img = It.shape
    rows_rect, cols_rect = int(wh[0]), int(wh[1])
    dp = np.array([[cols_img], [rows_img]]) #just an intial value to enforce the loop

    # template-related can be precomputed
    Iy, Ix = np.gradient(It1)
    y = np.arange(0, rows_img, 1)
    x = np.arange(0, cols_img, 1)
    spline = RectBivariateSpline(y, x, It)
    spline_gx = RectBivariateSpline(y, x, Ix)
    spline_gy = RectBivariateSpline(y, x, Iy)
    spline1 = RectBivariateSpline(y, x, It1)

    # in translation model jacobian is not related to coordinates
    jac = np.array([[1,0],[0,1]])

    while np.square(dp).sum() > threshold:

        # warp images
        x1_w, y1_w, x2_w, y2_w = xy[0]+(p0[0]-wh[0])/2.0, xy[1]+(p0[1]-wh[1])/2.0, xy[0]+(wh[0]+p0[0])/2.0, xy[1]+(wh[1]+p0[1])/2.0
        x1_t, y1_t, x2_t, y2_t = xy[0]-(wh[0]+p0[0])/2.0, xy[1]-(p0[1]+wh[1])/2.0, xy[0]+(wh[0]-p0[0])/2.0, xy[1]+(wh[1]-p0[1])/2.0

        cw = np.linspace(x1_w, x2_w, cols_rect)
        rw = np.linspace(y1_w, y2_w, rows_rect)
        ccw, rrw = np.meshgrid(cw, rw)

        warpImg = spline1.ev(rrw, ccw)

        ct = np.linspace(x1_t, x2_t, cols_rect)
        rt = np.linspace(y1_t, y2_t, rows_rect)
        cct, rrt = np.meshgrid(ct, rt)

        T = spline.ev(rrt, cct)

        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1)

        #compute gradient
        Ix_w = spline_gx.ev(rrw, ccw)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T

        #computer Hessian
        delta = I @ jac
        #H is (2,2)
        H = delta.T @ delta

        #compute dp
        #dp is (2,2)@(2,n)@(n,1) = (2,1)
        dp = np.linalg.inv(H) @ (delta.T) @ errImg

        #update parameters
        p0[0] += dp[0,0]
        p0[1] += dp[1,0]

    return p0, np.sum(np.abs(errImg))

def PyrLK(It, It1, xy, wh, height = 1, p0 = np.zeros(2)):
    It_levels = [It]
    It1_levels = [It1]

    for l in range(1, height):
        # compute down sampled images
        new_size = (np.array((It_levels[-1].shape[1], It_levels[-1].shape[0])) / 2.0**l).astype(int)
        It_levels.append(resize(gaussian_filter(It_levels[-1], 0.65), new_size, interpolation=INTER_NEAREST)) # std as cited in the paper
        It1_levels.append(resize(gaussian_filter(It1_levels[-1], 0.65), new_size, interpolation=INTER_NEAREST)) # std as cited in the paper

    # start p at an appropriate starting value
    p = p0 / (2.0**(height - 1))

    for l in range(height-1, -1, -1):
        curr_xy = xy / (2.0**l)
        curr_wh = wh / (2.0**l)
        print(f"p at height {l}: {p}")
        p, err = LucasKanade(It_levels[l], It1_levels[l], curr_xy, curr_wh, p)
        if l != 0:
            p = p*2.0

    return p, err