import cv2
import numpy as np

from Box import Box
import color
from geometry import infty


class Track():
    def __init__(
        self, 
        track_id:int = 9999,
        inactive_limit:int = 5
        ) -> None:
        """ Track class
        
        The track class maintains a list of Box objects that represent 
        bounding boxes in order to track an object. In order to pick 
        the best box candidate, this track class can use a variety of 
        methods such as:
            1) Kalman filter to predict the next frame's centroid.
            
        Input:
            track_id: (int) Unique integer ID for the track.
            inactive_limit: (int) Number of frames track can have no
                boxes added to it before it will be considered "inactive".
        
        Return: NA
        """
        # Track-related attributes
        self.id:int = track_id
        rng = np.random.default_rng(seed=track_id)
        #color = rng.integers(low=100, high=255, size=3).astype(np.uint8)
        #self.drawing_color:tuple = (int(color[0]), int(color[1]), int(color[2]))
        
        self.drawing_color = color.hsv2rgb(
            color.get_random_bright_hsv(track_id)
        )
        
        self.active:bool = True # decides whether to add more boxes or not
        self.track:list = [] # Box objects will be stored here
        self.frame_id:list = [] # frame box was added to
        
        self.INACTIVE_FRAMES:int = inactive_limit
        self.inactive_count:int = 0
        
        # Kalman filter
        state_params = 4 # thus, prediction is [x, y, vel_x, vel_y]
        measure_params = 2 # [center_x, center_y]
        control_params = 0
        
        self.kalman_filter = cv2.KalmanFilter(
            state_params, 
            measure_params, 
            control_params
        )
        
        self.kalman_filter.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        
        self.kalman_filter.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        
        self.kalman_filter.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03
        
        self.kf_pred = (0,0) # can't make a prediction yet...
        
    
    def add_box(
        self,
        box:Box = None, 
        frame_idx:int = None,
        ) -> None:
        """ Add box to the track.
        
        Note that this does several things
            1) Clearly, adds 'box' to the track list
            2) Resets inactive frame counter back to zero
            3) Updates frame ID so we know which frame box was added in
            4) Correct Kalman filter
            5) Store the Kalman filter's prediction (that's just how
                the OpenCV implementation seems to work)
        
        Input:
            box: (Box) custom Box object
            frame: (int) frame index box was added in (or could not provide)
        
        Return: NA
        """
        if self.active:
            # 1)
            self.track.append(box)
            
            # 2)
            self.inactive_count = 0
            
            # 3)
            self.frame_id.append(frame_idx)
            
            # 4) Correct Kalman filter
            centroid = box.get_centroid() # tuple of floats (x,y)
            #print('centroid:',centroid)
            new_pt = np.array(
                [[centroid[0]], [centroid[1]]], 
                np.float32
            )
            
            self.kalman_filter.correct(new_pt)
            
            # 5) After correction, get next prediction and store
            self.kf_pred = self.kalman_filter.predict()
        else:
            pass
    
    
    def add_nothing(self) -> None:
        """ Add nothing to the track for the current frame.
        
        This means
            1) Number of inactive frames will be incremented by one.
            2) *IF* the inactivity limit is reached, then the track will
                be considered inactive and will no longer be updated.
        
        Input: NA
        
        Return: NA
        """
        self.inactive_count += 1
        
        # Possibly mark track as inactive
        if self.inactive_count == self.INACTIVE_FRAMES:
            self.active = False
    
    
    def predict_next_centroid(self) -> tuple:
        """ Return most recent prediction made by Kalman filter.
        
        Input: NA
        
        Return: Tuple of ints in the form (x,y).
        """
        #pred = self.kalman_filter.predict()
        return (int(self.kf_pred[0][0]), int(self.kf_pred[1][0]))



if __name__ == "__main__":
    t = Track()
    
    # Test: add boxes and get kalman filter predictions
    t.add_box(Box(4, 3, 10, 10))
    print('kf pred:', t.predict_next_centroid())
    t.add_box(Box(4, 3, 10, 10))
    print('kf pred:', t.predict_next_centroid())
    t.add_box(Box(4, 3, 11, 10))
    print('kf pred:', t.predict_next_centroid())
    t.add_box(Box(4, 3, 12, 10))
    print('kf pred:', t.predict_next_centroid())
    t.add_box(Box(4, 3, 9, 10))
    print('kf pred:', t.predict_next_centroid())
    t.add_box(Box(3, 6, 11, 9))
    print('kf pred:', t.predict_next_centroid())
    t.add_box(Box(6, 5, 12, 12))
    print('kf pred:', t.predict_next_centroid())
    
    # Add many nothings and see results
    t.add_nothing()
    t.add_nothing()
    t.add_nothing()
    t.add_nothing()
    t.add_nothing()
    print(t.inactive_count)
    print(t.active)
    
    print(t.add_box(Box(6, 5, 12, 12)))
    
    # Test: get most recently-added box's centroid
    # (call method of Box object)
    print('centroid:',t.track[-1].get_centroid())
    
