import cv2


class CameraCapture:

    def __init__(self):
        self._capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        self._min_dimension = None  # Will be initialised on first frame capture

    def __del__(self):
        self._capture.release()

    def crop_frame_to_square(self, frame):
        if self._min_dimension is None:
            self._min_dimension = min(frame.shape[0], frame.shape[1])
        return frame[0:self._min_dimension, 0:self._min_dimension]

    def get_frame(self):
        ret, frame = self._capture.read()
        cropped_frame = self.crop_frame_to_square(frame)
        return cropped_frame
