import os, time, json
import pyrealsense2 as rs
import numpy as np
import cv2

class RealSense:
    def __init__(self, width=640, height=480, fps=30, clipping_distance=0.2, crop_settings={}):
        # detect devices
        ctx = rs.context()
        devices = ctx.query_devices()
        self.rs_num = devices.size()
        self.pipeline, self.config, self.profile = [], [], []
        self.color_sensors, self.depth_sensors = [], []
        self.color_intrinsics, self.depth_intrinsics = [], []
        self.cameraMatrix, self.cameraMatrix_woCrop, self.distCoeffs = [], [], []
        self.device_name, self.serial_number = [], []
        if crop_settings:
            self.crop_settings = crop_settings
        else:
            self.crop_settings = [{
                "crop_size": min(width, height),
                "crop_center_x": int(width / 2),
                "crop_center_y": int(height / 2)}] * self.rs_num
            print("##################################################")
        for i in range(self.rs_num):
            self.pipeline.append(rs.pipeline())
            # initialize streaming
            self.config.append(rs.config())
            self.config[i].enable_device(devices[i].get_info(rs.camera_info.serial_number))
            self.config[i].enable_stream(rs.stream.color, 640,  480, rs.format.bgr8, fps)
            # self.config[i].enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config[i].enable_stream(rs.stream.depth,640 , 480, rs.format.z16, fps)
            # self.config[i].enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            # start streaming
            self.profile.append(self.pipeline[i].start(self.config[i]))
            # register sensors
            self.color_sensors.append(devices[i].query_sensors()[0])
            # Store camera info
            pipeline_profile = self.pipeline[i].get_active_profile()
            self.device_name.append(devices[i].get_info(rs.camera_info.name))
            self.serial_number.append(devices[i].get_info(rs.camera_info.serial_number))
            # intrinsics
            self.color_intrinsics.append(rs.video_stream_profile(self.profile[i].get_stream(rs.stream.color)).get_intrinsics())
            self.depth_intrinsics.append(rs.video_stream_profile(self.profile[i].get_stream(rs.stream.depth)).get_intrinsics())
            print("color_intrinsics:", self.color_intrinsics[-1])
            offset_x = self.crop_settings[i]["crop_center_x"] - self.crop_settings[i]["crop_size"] / 2
            offset_y = self.crop_settings[i]["crop_center_y"] - self.crop_settings[i]["crop_size"] / 2
            self.cameraMatrix.append(
                np.array([[self.color_intrinsics[-1].fx, 0., self.color_intrinsics[-1].ppx - offset_x],
                          [0., self.color_intrinsics[-1].fy, self.color_intrinsics[-1].ppy - offset_y],
                          [0., 0., 1.]]))
            self.cameraMatrix_woCrop.append(
                np.array([[self.color_intrinsics[-1].fx, 0., self.color_intrinsics[-1].ppx],
                          [0., self.color_intrinsics[-1].fy, self.color_intrinsics[-1].ppy],
                          [0., 0., 1.]]))
            self.distCoeffs.append(np.array(self.color_intrinsics[-1].coeffs))
        print(self.device_name, self.serial_number)
        # clipping threshold
        depth_sensor = self.profile[0].get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.clipping_distance = clipping_distance
        # generate align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def save_camera_param(self, dir_name):
        if len(self.cameraMatrix) > 0 and len(self.distCoeffs) > 0:
            np.save(os.path.join(dir_name, "mtx.npy"), np.array(self.cameraMatrix_woCrop))
            np.save(os.path.join(dir_name, "dist.npy"), np.array(self.distCoeffs))

    def set_exposure(self, exposure):
        for i in range(self.rs_num):
            self.color_sensors[i].set_option(rs.option.enable_auto_exposure, 0)
            self.color_sensors[i].set_option(rs.option.exposure, exposure)

    def get_image(self, crop=False, get_mask=False):
        color_images, depth_images, mask_images, max_contours = [], [], [], []
        for i in range(self.rs_num):
            ret, frames = self.pipeline[i].try_wait_for_frames()
            # D405ではalignするとdepth画像が少し拡大されカラー画像と位置が合わない
            if not "D405" in self.device_name[i]:
                frames = self.align.process(frames)
            if ret:
                color_img = np.asanyarray(frames.get_color_frame().get_data())
                depth_img = np.asanyarray(frames.get_depth_frame().get_data())
                # print("depth_scale:", self.depth_scale)
                # print("depth:", depth_img.shape, "max:", depth_img.max(), depth_img.dtype)
                depth_img = depth_img.astype(np.float64) * self.depth_scale
                # print("depth:", depth_img.shape, "max:", depth_img.max(), depth_img.dtype)
            else:
                raise Exception("Failed to get image from realsense.")
            if crop:
                x = self.crop_settings[i]["crop_center_x"]
                y = self.crop_settings[i]["crop_center_y"]
                s = int(self.crop_settings[i]["crop_size"] / 2)
                color_img = color_img[y-s:y+s, x-s:x+s]
                depth_img = depth_img[y-s:y+s, x-s:x+s]
            if get_mask:
                mask_img, max_cnt = self.make_mask(depth_img)
                color_images.append(color_img)
                depth_images.append(depth_img)
                mask_images.append(mask_img)
                max_contours.append(max_cnt)
            else:
                color_images.append(color_img)
                depth_images.append(depth_img)
                mask_images.append(None)
                max_contours.append(None)
        return color_images, depth_images, mask_images, max_contours, frames

    def make_mask(self, depth_img):
        clip_img = (depth_img < self.clipping_distance) * depth_img
        clip_img_norm = clip_img / self.clipping_distance  # 0-1
        clip_img_255 = (clip_img_norm * 255.).astype(np.uint8)
        blur_image = cv2.blur(clip_img_255, (3, 3))
        contours, _ = cv2.findContours(blur_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_img = np.zeros_like(blur_image)
        if len(contours) > 0:
            max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
            # https://pystyle.info/opencv-structural-analysis-and-shape-descriptors/
            arclen = cv2.arcLength(max_cnt, True)
            max_cnt = cv2.approxPolyDP(max_cnt, epsilon=0.005 * arclen, closed=True)
            # https://pystyle.info/opencv-mask-image/
            cv2.drawContours(mask_img, [max_cnt], -1, color=255, thickness=-1)
        else:
            max_cnt = None
        return mask_img, max_cnt

    def show_image(self, crop=False, get_mask=False, scale=1.0):
        while True:
            before = time.time()
            color_images, depth_images, mask_images, max_contours = self.get_image(crop, get_mask)
            images = []
            for i in range(self.rs_num):
                color_img = color_images[i]
                depth_img = depth_images[i] / self.clipping_distance
                depth_img = np.clip(depth_img, 0, 1)
                depth_img = (depth_img * 255.).astype(np.uint8)
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
                original_height, original_width, _ = color_img.shape
                original_height = int(original_height * scale / 2.0)
                original_width = int(original_width * scale / 2.0)
                if not crop:
                    # Draw crop area
                    x = self.crop_settings[i]["crop_center_x"]
                    y = self.crop_settings[i]["crop_center_y"]
                    s = int(self.crop_settings[i]["crop_size"] / 2)
                    cv2.rectangle(color_img, (x-s, y-s), (x+s, y+s), (0, 0, 255))
                    cv2.rectangle(depth_img, (x-s, y-s), (x+s, y+s), (0, 0, 255))
                if get_mask:
                    mask_img = cv2.cvtColor(mask_images[i], cv2.COLOR_GRAY2BGR)
                    if not (max_contours[i] is None):
                        depth_img = cv2.drawContours(depth_img, max_contours[i], -1, (0, 255, 0), 3)
                    images.append(np.hstack([color_img, depth_img, mask_img]))
                else:
                    images.append(np.hstack([color_img, depth_img]))
            im = np.vstack(images)
            if scale != 1.0:
                height, width, _ = im.shape
                im = cv2.resize(im, (int(width * scale), int(height * scale)))
            cv2.imshow('RealSense', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            dt = time.time() - before
            print("FPS:", 1/dt)
        cv2.destroyAllWindows()
    
    def get_point_from_pixel(self, x, y, depth, camera_id=0):
        x, y, z = rs.rs2_deproject_pixel_to_point(self.color_intrinsics[camera_id], [x, y], depth)
        return x, y, z

    def close(self):
        # stop streaming
        for i in range(self.rs_num):
            self.pipeline[i].stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # D405
    rs_settings = {
        "width": 640, "height": 480, "fps": 90, "clipping_distance": 1.0,
        "crop_settings": [{"crop_size": 224, "crop_center_x": 320, "crop_center_y": 240}]}

    # D435
    # rs_settings = {
    #     "width": 640, "height": 480, "fps": 60, "clipping_distance": 1.0,
    #     "crop_settings": [{"crop_size": 224, "crop_center_x": 320, "crop_center_y": 240}]}

    # For D455
    # rs_settings = {
    #     "width": 640, "height": 360, "fps": 90, "clipping_distance": 1.0,
    #     "crop_settings": [{"crop_size": 224, "crop_center_x": 320, "crop_center_y": 180}]}


    # Start realsense pipeline
    rs = RealSense(**rs_settings)
    rs.show_image(crop=False, get_mask=False, scale=1.0)
    # rs.show_image(crop=True, get_mask=True, scale=1.0)