"""
This is a video utilities module. Video-related functionality should be
written and called from here.
"""

import cv2
import numpy as np
import os



def combine_frames(
    vid1:np.array = None, 
    vid2:np.array = None,
    weight1:float = 0.3,
    weight2:float = 0.7,
    bias:float = 0.0,
    channel:int = 1) -> np.array:
    """ Combine two frames via a weighted sum.
    
    Input:
        vid1: (numpy array) frame from first video or first image
        vid2: (numpy array) frame from second video or second image
        weight1: (float) weight in the sum for first image
        weight2: (float) weight in the sum for second image
        bias: (float) additional term to add to sum
        channel: (int) color channel to add for both images
    
    Return:
        modified numpy array vid1
    """
    vid1[:,:,channel] = (weight1*vid1[:,:,channel] + weight2*vid2[:,:,channel] + bias).astype(np.uint8)
    
    return vid1



def draw_bboxes(
    img:np.array = None, 
    bboxes:list = None, 
    color:tuple = (0,255,0)
    ) -> np.array:
    """Draw bounding boxes onto an image.
    
    Input:
        img: (numpy array) image/video frame
        bboxes: (list) list of custom Box objects
        color: (tuple) color tuple for coloring boxes
    
    Return: Modified image numpy array
    
    The list bboxes is a list of custom Box objects.
    """
    # Draw bbox and center point
    for box in bboxes:
        tl_pt = (box.box[0], box.box[1])
        br_pt = (box.box[0]+box.box[2], box.box[1]+box.box[3])
        cv2.rectangle(img, tl_pt, br_pt, color, thickness=1)
        #cv2.circle(img, (box.box[0]+box.box[2]//2, box.box[1]+box.box[3]//2), 1, (0,0,255), 3)
    
    return img



def draw_tracks(
    img:np.array = None, 
    tracks:list = None#, color:tuple = None
    ) -> np.array:
    """ Draw latest boxes for given frame image.
    
    Input:
        img: (numpy array) image/video frame
        tracks: (list) list of custom Track objects
        ###color: (tuple) color tuple for coloring boxes
        ###    NOTE: id None is given, then the tracks' element's color 3-tuple will be used
        ###    (recall, OpenCV uses the BGR ordering for color channels)
    
    Return: Modified image numpy array
    """
    for t_idx, track in enumerate(tracks):
        if track.active:
            # Only draw if track is active!
            box = track.track[-1].box
            
            # Draw box
            tl_pt = (box[0], box[1]) # top-left pt
            br_pt = (box[0]+box[2], box[1]+box[3]) # bottom-right pt

            color = track.drawing_color
            cv2.rectangle(img, tl_pt, br_pt, color, thickness=1)
            #cv2.circle(img, (box[0]+box[2]//2, box[1]+box[3]//2), 1, color, 2)
            
            # Add track label
            label = '{}'.format(track.id)
            cv2.putText(
                img=img, 
                text=label, 
                org=(box[0], box[1]-5), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5,
                color=color, 
                thickness=1
            )
    
    return img



def show_video_dir(dirname: str = None) -> None:
    """Show video given directory of frames image files.
    
    Input:
        dirname: (str) name of directory
        
    Return: NA
    """
    img_paths = sorted(os.listdir(dirname))
    
    # View a window of original video
    #tl_x, tl_y = 0, 0
    #k = 256 # patch size
    
    for i, name in enumerate(img_paths):
        full_path = os.path.join(dirname, name)
        frame = cv2.imread(full_path, 0) # phase contrast images gray anyway
        
        # Crop frame...maybe upscale
        #frame = frame[tl_y:k, tl_x:k]
        
        cv2.imshow('Video Feed', frame)
        cv2.waitKey(10) # millisecs



if __name__ == "__main__":
    print('Video 1 frames:', get_n_frames("data/video1.mp4"))
    print('Video 2 frames:', get_n_frames("data/video2.mp4"))
