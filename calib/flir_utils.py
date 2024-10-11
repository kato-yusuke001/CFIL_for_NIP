import cv2
import EasyPySpin

class Flir:
    def __init__(self, width=640, height=480, fps=30, crop_settings={}, **kwargs):
        self.cap = EasyPySpin.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_TRIGGER_DELAY, 208) # us
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 100000) # us
        self.cap.set(cv2.CAP_PROP_GAMMA, 0.8)
        self.cap.set(cv2.CAP_PROP_GAIN, 0) # dB
        self.width = width
        self.height = height
        self.fps = fps

        if crop_settings:
            self.crop_settings = crop_settings
        else:
            self.crop_settings = [{
                "crop_size": min(width, height),
                "crop_center_x": int(width / 2),
                "crop_center_y": int(height / 2)}]
            
    def get_image(self, crop=False, get_mask=False):
        color_images, depth_images, mask_images, max_contours = [], [], [], []
        ret, frame = self.cap.read()
        if crop:
            x = self.crop_settings[0]["crop_center_x"]
            y = self.crop_settings[0]["crop_center_y"]
            s = int(self.crop_settings[0]["crop_size"] / 2)
            frame = frame[y-s:y+s, x-s:x+s]
        color_images.append(frame)
        
        return color_images, depth_images, mask_images, max_contours

if __name__ == "__main__":
    flir = Flir()
    img, _, _ ,_ = flir.get_image()
    cv2.imwrite("flir.jpg", img[0])