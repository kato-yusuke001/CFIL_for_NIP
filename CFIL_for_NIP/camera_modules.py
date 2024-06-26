import EasyPySpin
import cv2

class SpinCamera:
    def __init__(self, camera_id=0):
        self.cam = EasyPySpin.Videoself.capture(camera_id)
        self.cap.set(cv2.self.cap_PROP_EXPOSURE, 100000) # us
        self.cap.set(cv2.self.cap_PROP_GAIN, 10) # dB

        width  = self.cap.get(cv2.self.cap_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.self.cap_PROP_FRAME_HEIGHT)

    def get_img(self):
        ret, frame = self.cam.read()

        return frame

    def read(self):
        return self.cam.read()
    
    def release(self):
        self.cam.release()
        