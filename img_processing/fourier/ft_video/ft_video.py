import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_img_dft(x=None, use_shift=True, use_mag=False):
    # Get DFT of image
    #dft = cv2.dft(np.float32(x), flags = cv2.DFT_COMPLEX_OUTPUT)
    #dft_shift = np.fft.fftshift(dft)
    #mag_spectrum = 10*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    #f = np.fft.fft2(x)
    f = cv2.dft(np.float32(x), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(f) # take magnitude before filtering out zeros
    
    #import pdb;pdb.set_trace()
    
    if np.min(np.abs(fshift)) == 0.0:
        temp = np.abs(np.where(fshift > 0.0, fshift, 0.001))
        mag_spectrum = 20*np.log(cv2.magnitude(temp[:,:,0], temp[:,:,1]))
        return temp, mag_spectrum
    else:
        mag_spectrum = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
        return fshift, mag_spectrum


if __name__ == "__main__":
    filename = "./video1.mp4"
    ms = 30
    
    delta = 256
    
    x1 = 512
    y1 = 512
    x2 = x1 + delta
    y2 = y1 + delta
    
    vid_cap = cv2.VideoCapture(filename)
    n_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 1
    
    while True:    
        vid_ret, vid_frame = vid_cap.read() # np array: (h, w, channels)

        if vid_ret is True:
            print(f'Frame {frame_count}')
            frame_count += 1
            
            vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
            #vid_frame = np.expand_dims(vid_frame, axis=-1)
            
            frame = vid_frame[y1:y2,x1:x2]
            fshift, mag_spec = get_img_dft(x=frame, use_shift=True, use_mag=True)
            
            #fshift, mag_spec = np.rint(ft_frame).astype(np.uint8) # round float -> uint8
                        
            # now, pretend we load integer ft_frame, reconstruct image
            #f_ishift = np.fft.ifftshift(ft_frame)
            #img_back = np.fft.ifft2(f_ishift)
            #img_back = np.real(img_back)
            
            rec = cv2.idft(np.fft.ifftshift(fshift))
            rec = cv2.magnitude(rec[:,:,0], rec[:,:,1])
            
            rec = (rec - np.min(rec))/np.ptp(rec) 
            
            import pdb;pdb.set_trace()
            
            #cv2.imshow(winname="FT Video", mat=ft_frame)
            cv2.imshow(winname="Reconstructed from Int FT", mat=rec)
            cv2.waitKey(ms)
        else:
            break
    
    # Cleanup
    cv2.destroyAllWindows()
    vid_cap.release()
    
