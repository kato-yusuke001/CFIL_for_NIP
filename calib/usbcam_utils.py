import cv2

class UsbCam:
    def __init__(self, width=640, height=480, fps=30, crop_settings={}, **kwargs):
        self.cap = cv2.VideoCapture(0)
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
    flir = UsbCam()
    img, _, _ ,_ = flir.get_image()
    cv2.imwrite("usbcam.jpg", img[0])