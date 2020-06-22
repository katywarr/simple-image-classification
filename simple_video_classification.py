import numpy as np
import cv2
import classifier
from camera_capture import CameraCapture
import sys
import argparse
import os


def crop_image_to_square(self, frame):
    min_dimension = min(frame.shape[0], frame.shape[1])
    return frame[0:min_dimension, 0:self.min_dimension]


class DisplayWindow:

    def __init__(self, frame_size):

        self._offset_x = 10
        self._offset_y = 10

        # Assume that the image has been cropped to a square
        # Still frame retains the same resolution as the original
        self._still_frame_shape_x = frame_size
        self._still_frame_shape_y = frame_size
        # Moving frame is shrunk to fit in the corner of the still one
        self._moving_frame_shape_x = int(frame_size/4)
        self._moving_frame_shape_y = int(frame_size/4)
        # Plot has same height as still frame, but is wider
        self._plot_shape_x = int(self._still_frame_shape_x * 1.5)
        self._plot_shape_y = self._still_frame_shape_y

        # Moving frame is positioned top let
        self._moving_frame_offset_x = self._offset_x
        self._moving_frame_offset_y = self._offset_y

        # Still frame beside it positioned top Left
        self._still_frame_offset_x = self._moving_frame_offset_x + self._moving_frame_shape_x + self._offset_x
        self._still_frame_offset_y = self._offset_y

        # Plot is positioned right of the still frame
        self._plot_offset_x = self._still_frame_offset_x + self._still_frame_shape_x + self._offset_x
        self._plot_offset_y = self._offset_y

        # Total window size
        total_window_size_x = self._plot_offset_x + self._plot_shape_x + self._offset_x
        total_window_size_y = self._plot_offset_y + self._plot_shape_y + self._offset_y

        # Initialise the image
        self._window_image = np.zeros([total_window_size_y, total_window_size_x, 3], dtype=np.uint8)
        self._window_image[:] = (225, 225, 225)
        # Placeholder for the camera image (is over the still one)
        self.overlay_image((225, 225, 0), self._still_frame_offset_x, self._still_frame_offset_y,
                           self._still_frame_shape_x, self._still_frame_shape_y)
        # Placeholder for the camera image (is over the still one)
        self.overlay_image((0, 225, 225), self._moving_frame_offset_x, self._moving_frame_offset_y,
                           self._moving_frame_shape_x, self._moving_frame_shape_y)
        # Placeholder for the prediction plot
        self.overlay_image((225, 0, 225),  self._plot_offset_x, self._plot_offset_y, self._plot_shape_x,
                           self._plot_shape_y)

    def overlay_image(self, image, x_offset, y_offset, x_size, y_size):
        self._window_image[y_offset:y_offset + y_size,
                           x_offset:x_offset + x_size] = image

    def get_display_window(self, frame=None, plot=None):

        if frame is not None:

            if plot is not None:
                # A new image has been captured, refresh the still frame and the prediction plot
                still_frame = cv2.resize(frame, (self._still_frame_shape_y, self._still_frame_shape_x),
                                         interpolation=cv2.INTER_AREA)
                self.overlay_image(still_frame, self._still_frame_offset_x, self._still_frame_offset_y,
                                   self._still_frame_shape_x, self._still_frame_shape_y)

                plot = cv2.resize(plot, (self._plot_shape_x, self._plot_shape_y),
                                  interpolation=cv2.INTER_AREA)

                self.overlay_image(plot, self._plot_offset_x, self._plot_offset_y, self._plot_shape_x,
                                   self._plot_shape_y)

            # Always update the moving frame. This is an overlay so occurs last
            moving_frame = cv2.resize(frame, (self._moving_frame_shape_y, self._moving_frame_shape_x),
                                      interpolation=cv2.INTER_AREA)
            self.overlay_image(moving_frame, self._moving_frame_offset_x, self._moving_frame_offset_y,
                               self._moving_frame_shape_x, self._moving_frame_shape_y)

        return self._window_image


def live_camera_classifier():

    camera = CameraCapture()

    cv2.namedWindow('Live Video Camera Classifier', cv2.WINDOW_NORMAL)

    display_window = DisplayWindow(frame_size=480)

    while True:
        frame = camera.get_frame()
        # Could flip the frame to show mirror image, if front facing
        # cv2.imshow('Streaming Video', np.flip(frame, axis=1))
        image = display_window.get_display_window(frame=frame)
        cv2.imshow('Live Video Camera Classifier', image)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            predictions, plot = classifier.classify(frame)
            print(predictions)
            image = display_window.get_display_window(frame=frame, plot=plot)
            cv2.imshow('Live Video Camera Classifier', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    del camera
    cv2.destroyAllWindows()


def main():
    live_camera_classifier()


if __name__ == "__main__":
    main()


