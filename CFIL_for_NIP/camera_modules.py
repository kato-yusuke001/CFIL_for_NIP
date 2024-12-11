from typing import Any
import PySpin
import EasyPySpin
import cv2

class CameraModules:
    def __init__(self, module="spin", camera_id=0):
        self.cam=None
        if module == "spin":
            self.cam = SpinCamera(camera_id)
        elif module == "cv2":
            self.cam = Cv2Camera(camera_id)
        else:
            NotImplementedError
    
    def read(self):
        return self.cam.cap.read()
    
    def get_img(self):
        ret, frame = self.read()
        return frame

    def __del__(self):
        if self.cam is not None:
            self.cam.cap.release()

class Camera:
    def __init__(self):
        self.cap = None
    
    def get_img(self):
        ret, frame = self.cap.read()

        return frame

    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()

class Cv2Camera(Camera):
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

class SpinCamera(Camera):
    def __init__(self, camera_id=0):
        self.cap = EasyPySpin.Videoself.capture(camera_id)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 100000) # us
        self.cap.set(cv2.CAP_PROP_GAIN, 10) # dB

        width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.cap.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12)
        self.cap.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)

        self.cap.get_pyspin_value("GammaEnable")
        self.cap.get_pyspin_value("DeviceModelName")

    # def get_img(self):
    #     ret, frame = self.cap.read()

    #     return frame

    # def read(self):
    #     return self.cap.read()
    
    # def release(self):
    #     self.cap.release()
        