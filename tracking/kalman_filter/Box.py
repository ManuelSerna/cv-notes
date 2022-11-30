""" Custom Box class. 

This was originally implemented for tracking applications.
"""



class Box():
    def __init__(
        self, 
        tl_x:int = None,
        tl_y:int = None,
        w:int = None,
        h:int = None
        ) -> None:
        """ Box class
        
        Input:
            tl_x: (int) Top-left x coordinate of box.
            tl_y: (int) Top-left y coordinate of box.
            w: (int) Width of box.
            h: (int) Height of box.
        
        Return: NA
        """
        self.box:tuple = (tl_x, tl_y, w, h) 
        self.centroid:tuple = (
            self.box[0] + self.box[2]/2.0, 
            self.box[1] + self.box[3]/2.0
        )
        
        self.neighbor_box:Box = None
        self.nearest_box:float = None
        self.neighbor_id:int = None
        
    
    def get_centroid(self) -> tuple:
        """Return centroid, which is a tuple in the form (x,y),
        where both x and y are floats. """
        return self.centroid
    
    
    def set_nearest_neighbor(
        self, 
        box = None, 
        dist:float = None,
        uid:int = None
    ) -> None:
        """ Set box's closest neighbor, which is another box which
        may be associated with unique id.
        """
        self.neighbor_box = box
        self.nearest_box = dist
        self.neighbor_id = uid
    
    
    def __str__(self) -> str:
        return f'Box{self.box}'




if __name__ == "__main__":
    b = Box(1, 2, 10, 10)
    c = b.get_centroid()
    print(b)
    print(c)
