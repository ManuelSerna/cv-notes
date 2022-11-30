

import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np


def get_latest_pts(track):
    """ Given list of lists of tuples, get latest point (last element)"""
    latest = []
    for i in range(len(track)):
        # print(track[i][-1])
        latest.append(track[i][-1])
    return latest


def find_closest_pt(pt, track_pts):
    """ Find closest point via Euclidean distance given a point and other points"""
    track_pts = np.asarray(track_pts.copy())
    linear_dist = np.sqrt(np.sum((track_pts - pt) ** 2, axis=1))
    return np.argmin(linear_dist)


def add_points_to_tracks(tracks=None, new_pts=None):
    if len(tracks) == 0:
        for i in range(new_pts.shape[0]):
            tracks.append([(new_pts[i][0], new_pts[i][1])])
        return tracks
    elif len(tracks) > 0:
        # Find matches
        matches = [None] * len(tracks) # initial matches to tracks are None, will be filled in with the points themselves
        last_pts = get_latest_pts(tracks)
        for pt in new_pts:
            idx = find_closest_pt(pt, last_pts)
            matches[idx] = pt

        # Assign matches to tracks
        for idx, match in enumerate(matches):
            if match is not None:
                new_pt = (match[0], match[1])
                tracks[idx].append(new_pt)

        return tracks
    else:
        raise Exception('[Add points to tracks function] Length of tracks not 0 or positive!')


def shi_tomasi_corners(img, return_corners=True):
    """ Get shi-tomasi corners via OpenCV (implemented by goodFeaturesToTrack)
    given an image

    """
    color = (0, 255, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Array of corners
    corners = cv2.goodFeaturesToTrack(
        image=gray,
        maxCorners=255,
        qualityLevel=.01,
        minDistance=10
    )
    corners = np.int0(corners)
    
    if return_corners:
        return corners
    else:
        for corner in corners:
            x,y = corner[0].ravel()
            cv2.circle(img, (x,y),3,color,-1)
        return img


def video_corners_harris(vid_filename:str = None):
    """ Use Harris corner detection on a video"""
    vid_cap = cv2.VideoCapture(vid_filename)
    
    tot_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while vid_cap.isOpened():
        print("frame: [{}/{}]".format(frame_count, tot_frames))
        frame_count += 1
        
        vid_ret, frame = vid_cap.read()
        
        if vid_ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute Harris corners
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            
            #result is dilated for marking the corners, not important
            dst = cv2.dilate(dst,None)
            
            # Threshold for an optimal value, it may vary depending on the image.
            frame[dst>0.01*dst.max()]=[0,0,255]
            
            cv2.imshow("Harris Corner Detect Result", frame)
            cv2.waitKey(30)
        else:
            break
        
    vid_cap.release()


def video_corners_st(vfilename:str = None):
    """ Use Shi-Tomasi corner detection on a video."""
    vid_cap = cv2.VideoCapture(vfilename)
    
    tot_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while vid_cap.isOpened():
        print("frame: [{}/{}]".format(frame_count, tot_frames))
        frame_count += 1
        
        vid_ret, frame = vid_cap.read()
        
        if vid_ret:
            res = shi_tomasi_corners(frame, return_corners=False)
            
            cv2.imshow("S-T Corner Detect Result", res)
            cv2.waitKey(30)
        else:
            break
    
    vid_cap.release()


def lk_track_features(
    vid_filename:str = None,
    max_corners:int = 512,
    show_tracks:bool = True,
    #start_frame:int = 0,
    #n_skip:int = 1,
    show_n_frames:int = 10
    ) -> None:
    """ Use LK method for tracking shi-tomasi corners

    Can also:
        - give how many frames to show until we kill the video feed

    Args:
        vid_filename:
        max_corners:
        show_tracks:
        show_n_frames:
    
    Base code taken from OpenCV demo
    """
    # ShiTomasi corner detection params
    min_dist = 10
    feature_params = dict(
        maxCorners = max_corners,
        qualityLevel = 0.01,
        minDistance = min_dist,
        blockSize = 7
    )

    # LK tracking parameters
    lk_params = dict(
        winSize  = (15, 15),
        maxLevel = 2,
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2 .TERM_CRITERIA_COUNT, 
            10, 
            0.03
        )
    )

    # Track-related variables
    n_colors = max_corners
    color = np.random.randint(0, 255, (n_colors, 3))
    tracks = [] # list of lists

    # Video capture
    vid_cap = cv2.VideoCapture(vid_filename)
    
    tot_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 1
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter("track_out.mp4", fourcc, 30, (1400, 1400))
    print('[INFO] Showing {} frames until video ends!'.format(show_n_frames))
    
    # Get first frame
    ret, old_frame = vid_cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #tracks = add_points_to_tracks(tracks=tracks, new_pts=p0)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    # Process rest of video
    while vid_cap.isOpened():
        print("frame: [{}/{}]".format(frame_count, tot_frames))
        frame_count += 1
        
        vid_ret, frame = vid_cap.read()
        if not vid_ret:
            print('No frames retrieved!')
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #res = shi_tomasi_corners(frame, return_corners=False)
        #import pdb;pdb.set_trace()
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

            if frame_count == 2:
                tracks = add_points_to_tracks(tracks, good_old)
            elif frame_count > 2:
                tracks = add_points_to_tracks(tracks, good_old)

        # draw the tracks
        # NOTE: we do not enforce a track has a certain color,
        # tracks are assumed to be in the same order for all frames
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            color_idx = i % n_colors
            
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[color_idx].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 2, color[color_idx].tolist(), -1)
        
        if show_tracks:
            res = cv2.add(frame, mask)
        else:
            res = frame
        
        vid_out.write(res)
        
        cv2.imshow("S-T Corner Detect Result", res)
        cv2.waitKey(30)
        
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if frame_count > show_n_frames:
            break

    if True:
        for track in tracks:
            #print(track)
            print("Track:")
            for pt in track:
                print(f'...{pt}')

    vid_cap.release()
    vid_out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tracking Demos")
    parser.add_argument('--video', type=str, help="path to video file")
    parser.add_argument('--nfeat', type=int, help="number of features (corners) to track")
    
    args = parser.parse_args()

    #video_corners_st(args.video)
    
    lk_track_features(
        vid_filename=args.video, 
        max_corners=args.nfeat,
        show_tracks=False
    )
    '''
    min_dist = 10.0
    T = [[(101.0, 84.9)], [(62.0, 86.1)], [(75.0, 65.0)]] # track already built
    P = np.array([[100.99156, 85.06581],[61.92603, 85.95519],[74.055504, 65.11824]]) # cand pts

    #T = [[(), (62.7, 84.9)], [(), (75.0, 65.0)], [(), (62.0, 86.1)]]
    #last_pts = [(62.7, 84.9), (75.0, 65.0), (62.0, 86.1)]
    #P = np.array([[62.3, 83.1], [61.8, 85.5], [76.3, 65.1]]) # candidate pts, very close

    def get_latest_pts(track):
        """ Given list of lists of tuples, get latest points"""
        latest = []
        for i in range(len(track)):
            #print(track[i][-1])
            latest.append(track[i][-1])
        return latest

    #test = [[(1, 1), (2, 2)], [(3, 3), (4, 4)], [(5, 5), (6, 6)]]
    last_pts = get_latest_pts(T)
    #print(test_last)
    print("latest tracks' pts:",last_pts)

    #print(last_pts)
    print(P)

    for pt in P:
        i = find_closest_pt(pt, last_pts)
        print(f'candidate pt:{pt}...closest track={last_pts[i]}')
    #'''