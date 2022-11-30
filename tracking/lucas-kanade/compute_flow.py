"""
Optical Flow Methods
- Lucas Kanade flow for all pixels, but this method is meant for "good" features we want to track
- Horn-Schunck dense flow, which is meant for all pixels
"""
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def get_dt(f1:np.array = None, f2:np.array = None) -> np.array:
    """Compute derivative of image with respect to time 
    by computing forward difference """
    #return (((f1.astype(np.float64) - f2.astype(np.float64)) + 255.0)/2.0)
    return f1.astype(np.float64) - f2.astype(np.float64)


def get_Ix(img:np.array = None) -> np.array:
    """Get gradients for x-direction using Sobel filter """
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    k = 3
    
    blur = cv2.GaussianBlur(img, (k,k), 0)
    
    grad_x = cv2.Sobel(blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.convertScaleAbs(grad_x).astype(np.float64)
    
    return grad_x


def get_Iy(img:np.array = None) -> np.array:
    """Get gradients for x-direction using Sobel filter """
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    k = 3
    
    blur = cv2.GaussianBlur(img, (k,k), 0)
    
    grad_y = cv2.Sobel(blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.convertScaleAbs(grad_y).astype(np.float64)
    
    return grad_y


def get_local_avg(M:np.array, i=None, j=None) -> float:
    """ Get local average for either U (x-direction) or V (y-direction)

    Args:
        M: Matrix for flow, just generally denoted as M
        i: index for x coord
        j: index for y coord

    Returns:
        local average
    """
    H = M.shape[0]
    W = M.shape[1]

    # point (x,y), images sliced y-first
    top_pt = None  # (i,j-1)
    left_pt = None  # (i-1,j)
    bottom_pt = None  # (i,j+1)
    right_pt = None # (i+1,j)

    if j == 0:
        top_pt = M[j+1,i] # reflexive
    else:
        top_pt = M[j-1,i]

    if i == 0:
        left_pt = M[j,i+1] # reflexive
    else:
        left_pt = M[j,i-1]

    if j >= H-1:
        bottom_pt = M[j-1,i] # reflexive
    else:
        bottom_pt = M[j+1,i]

    if i >= W-1:
        right_pt = M[j,i-1] # reflexive
    else:
        right_pt = M[j,i+1]

    return 0.25 * (top_pt + left_pt + bottom_pt + right_pt)


def hs_flow(vid_filename:str = None, n_iters:int = 10, write_n_frames:int = None, write_array=False, write_path:str = None):
    """ Compute Horn-Schunck (dense) optical flow

    Args:
        vid_filename: path to video file
        n_iters: number of iterations to converge
        write_n_frames: max number of frames to write to (numpy) array file,
            default is None (will be set to number of frames in input video)
        write_array: (bool) write to numpy .npy file?
            NOTE: arrays will be float32 files (originally they were float64)
        write_path: (str) path to write numpy array to (if toggled)

    Returns: NA
    """
    # Video capture
    vid_cap = cv2.VideoCapture(vid_filename)
    
    W = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if write_n_frames is None:
        tot_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        tot_frames = write_n_frames
    print('[INFO] Writing {}/{} frames to array file!'.format(tot_frames-1, tot_frames))
    frame_count = 1
    frame_idx = 0
    
    # Video writing
    out_data = None
    if write_array:
        out_data = np.zeros((tot_frames-1, H, W, 3)) # (frames-1, H, W, channels[r: img, g: vel x, b: vel y])
    
    # Get current_frame
    ret, current_frame = vid_cap.read()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # HS flow-related params
    brightness_weight = 1.0 # lambda term in HS flow equation
    n_iters = n_iters # number of times to iterate
    
    # Compute flow
    while vid_cap.isOpened() and frame_count < tot_frames:
        #print("frame: [{}/{}]".format(frame_count, tot_frames))
        frame_count += 1
        
        # Get next_frame
        vid_ret, next_frame = vid_cap.read()
        if not vid_ret:
            print('No frames retrieved! Breaking...')
            break

        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        #k = 3
        #current_frame = cv2.medianBlur(current_frame, k)
    
        # Compute current image gradients Ix, Iy, and temporal gradients It
        Ix = get_Ix(current_frame)
        Iy = get_Iy(current_frame)
        It = get_dt(f1=current_frame, f2=next_frame)
        
        # Initialize flow fields
        U = np.zeros((H, W)) # x-component
        U_avg = np.zeros_like(U)  # local neighborhood averages

        V = np.zeros((H, W)) # y-component
        V_avg = np.zeros_like(V)

        #bottom = (1 / brightness_weight) + Ix @ Ix + Iy @ Iy
        # TODO: do element-wise squaring of Ix Iy
        bottom = (1/brightness_weight) + (Ix*Ix) + (Iy*Iy)

        # Loop to converge for each pixel
        for i in range(n_iters):
            # Compute local averages for U,V
            for iy in range(H):
                for ix in range(W):
                    U_avg[iy,ix] = get_local_avg(M=U, i=ix, j=iy) # local avg u(k,l) (x)
                    V_avg[iy,ix] = get_local_avg(M=V, i=ix, j=iy) # local avg v(k,l) (y)

            # Update directional velocities
            top = Ix*U_avg + Iy*V_avg + It

            temp = top / bottom

            U = U_avg - temp * Ix
            V = V_avg - temp * Iy

        # Write to output data array
        if write_array:
            out_data[frame_idx, :, :, 0] = current_frame # grayscale img
            out_data[frame_idx, :, :, 1] = U # velocity in x-dir
            out_data[frame_idx, :, :, 2] = V # velocity in y-dir

        # Update
        #res = current_frame
        #cv2.imshow("H-S Flow Result", res)
        #cv2.waitKey(30)
        
        current_frame = next_frame
        frame_idx += 1

    # Cleanup
    out_data = out_data.astype(np.float32) # reduce float accuracy to save memory
    #out_data = out_data.astype(np.float16)  # reduce float accuracy to save memory

    vid_cap.release()
    if write_array:
        print('hs flow: min=',out_data.min())
        print('hs flow: max=',out_data.max())

        #import pdb;pdb.set_trace()
        #if out_data.min() == -np.inf or out_data.max() == np.inf:
        #    import pdb;pdb.set_trace()

        np.save(write_path, out_data)


def video_lk_flow_pixels(vid_filename:str = None):
    """ Compute LK optical flow for each pixel for a given video"""
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    subregion_size = 15 // 2 # size of region to compute flow from for each pixel
    
    # Get consecutive frames from video
    vid_cap = cv2.VideoCapture(vid_filename)
    
    tot_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 1
    
    vid_ret1, img1 = vid_cap.read() # get first frame
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    H, W = img1.shape

    while vid_cap.isOpened() and frame_count < tot_frames:
        print("frame: [{}/{}]".format(frame_count, tot_frames))
        frame_count += 1
        
        vid_ret2, img2 = vid_cap.read()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # acts as future frame

        Vx = np.zeros((H, W))
        Vy = np.zeros((H, W))

        # Compute spatial derivative: sobel x and y to get gradients for x and y
        img1_blur = cv2.GaussianBlur(img1, (3, 3), 0)
        img2_blur = cv2.GaussianBlur(img2, (3, 3), 0)
        
        grad_x = cv2.Sobel(img1_blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_x = cv2.convertScaleAbs(grad_x).astype(np.float64)

        grad_y = cv2.Sobel(img1_blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.convertScaleAbs(grad_y).astype(np.float64)

        # Compute temporal derivative: frame differencing
        #dt = get_dt(img1, img2)
        dt = get_dt(img1_blur, img2_blur) # blur to reduce effect of noise

        # Pad current frame derivatives and compute velocities between it and the next frame
        grad_x_pad = np.pad(array=grad_x, pad_width=subregion_size, mode='reflect')
        grad_y_pad = np.pad(array=grad_y, pad_width=subregion_size, mode='reflect')
        dt_pad = np.pad(array=dt, pad_width=subregion_size, mode='reflect')

        for iy in range(subregion_size, H + subregion_size):
            for ix in range(subregion_size, W + subregion_size):
                grad_x_pad_sub = grad_x_pad[iy - subregion_size:iy + subregion_size + 1, ix - subregion_size: ix + subregion_size + 1]
                grad_y_pad_sub = grad_y_pad[iy - subregion_size:iy + subregion_size + 1, ix - subregion_size: ix + subregion_size + 1]
                dt_pad_sub = dt_pad[iy - subregion_size:iy + subregion_size + 1, ix - subregion_size: ix + subregion_size + 1]

                ax = grad_x_pad_sub.reshape(((subregion_size*2+1)**2, 1))
                ay = grad_y_pad_sub.reshape(((subregion_size*2+1)**2, 1))

                A = np.hstack((ax, ay))
                b = -1 * dt_pad_sub.reshape(((subregion_size*2+1)**2, 1))

                pixel_vel = np.linalg.lstsq(A, b, rcond=None)[0]

                Vx[iy - subregion_size, ix - subregion_size] = pixel_vel[0]
                Vy[iy - subregion_size, ix - subregion_size] = pixel_vel[1]
                
        #Vx_disp = cv2.convertScaleAbs(Vx)
        Vx_disp = cv2.normalize(Vx, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

        cv2.imshow("Vx", Vx_disp)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
        img1 = img2 # t <- t+1
    
    vid_cap.release()


def video_cleaning_comparison(vid_filename:str = None):
    """ Visualize video cleaning with simple techniques

    Args:
        vid_filename:

    Returns: NA
    """
    vid_cap = cv2.VideoCapture(vid_filename)

    tot_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 1

    while vid_cap.isOpened():
        # Get next_frame
        vid_ret, frame = vid_cap.read()

        if not vid_ret:
            print('No frames retrieved! Breaking...')
            break

        print("frame: [{}/{}]".format(frame_count, tot_frames))
        frame_count += 1

        # Process
        k = 3
        res = cv2.medianBlur(np.copy(frame), k)

        # Show result
        cv2.imshow("Result", res)
        cv2.imshow("Original", frame)
        cv2.waitKey(60)

    vid_cap.release()


def read_flow_data_test(filepath:str = None):
    """ Test reading numpy array .npy file and playback image component
    (output from hs_flow())

    Args:
        filepath: path to .npy file, array is in shape
            (frames, H, W, channels)
            where first channel: grayscale image
                  second channel: x velocities
                  third channel: y velocities

    Returns: NA
    """
    data = np.load(filepath)

    for i in range(data.shape[0]):
        frame = data[i, :, :, 0].astype(np.uint8)
        Vx = data[i, :, :, 1]
        Vy = data[i, :, :, 2]

        print(Vx.min())
        print(Vy.min())
        print(Vx.max())
        print(Vy.max())

        #if Vx.min() < -100.0 or Vy.min() < -100.0 or Vx.max() > 1000.0 or Vy.max() > 1000.0:
        import pdb;pdb.set_trace()

        cv2.imshow("Frame", frame)
        cv2.waitKey(30)


def read_gif(gif_path:str = None):
    '''
    from PIL import Image
    frame = Image.open(gif_path)
    frame_idx = 1

    try:
        while True:
            print('[INFO] Frame {}'.format(frame_idx))
            frame.seek(frame.tell() + 1)

            import pdb;pdb.set_trace()

            # TODO: do something

            frame_idx += 1
    except EOFError:
        pass # reached last frame already
    '''
    from PIL import Image
    from PIL import GifImagePlugin

    img_object = Image.open(gif_path)
    print(img_object.is_animated)
    print(img_object.n_frames)

    for frame in range(0, img_object.n_frames):
        img_object.seek(frame)
        img_object.show()


def create_hs_dataset(input_dir:str = None, write_dir:str = None):
    """ Create dataset with grayscale image in first channel and
    x- and y-velocity information in the second and third channels

    Args:
        input_dir: directory to read .mp4 files from
        write_dir: directory to write .npy files to
            NOTE: if directory does not exist, it will be made in current work directory

    Returns: NA
    """
    hs_iters = 10
    write_frames = 33 # need very last for time derivative

    # Directory checkig
    in_files = os.listdir(input_dir)

    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
        print('[INFO] Created path: {}'.format(write_dir))

    # Process file-by-file
    for f in in_files:
        filepath = os.path.join(input_dir, f)
        out_name = os.path.join(write_dir, f[:-4] + ".npy")

        print('[INFO] Input file: {}'.format(filepath))
        print('[INFO] Writing to: {}'.format(out_name))

        hs_flow(
            vid_filename=filepath,
            n_iters=hs_iters,
            write_n_frames=write_frames,
            write_array=True,
            write_path=out_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="path to file to read")
    parser.add_argument("--rdir", help="path to *read* data from")
    parser.add_argument("--wdir", help="path to *write* data to")
    args = parser.parse_args()

    filepath = args.filepath
    read_dir = args.rdir
    write_dir = args.wdir

    #video_cleaning_comparison(filepath)

    ##video_lk_flow_pixels(vid_filename="")

    #hs_flow(vid_filename=filepath, n_iters=10, write_n_frames=32, write_array=True, write_path="test.npy")
    #read_flow_data_test(filepath=filepath)

    create_hs_dataset(input_dir=read_dir, write_dir=write_dir)

    #read_gif(gif_path=filepath)
