import EasyPySpin

class SpinCamera:
    def __init__(self, camera_id=0):
        self.cam = EasyPySpin.VideoCapture(camera_id)

    def get_img(self):
        ret, frame = self.cam.read()

        return frame

    def read(self):
        return self.cam.read()
    
    def release(self):
        self.cam.release()
        