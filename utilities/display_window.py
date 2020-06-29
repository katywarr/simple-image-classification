import numpy as np
import cv2


class DisplayWindow:
    """
    DisplayWindow represents the window being displayed.
    """

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
        """
        Overlays one the display window with an image at the specified offset.
        :param image: Image to be overlayed
        :param x_offset: x offset of overlay
        :param y_offset: y offset of overlay
        :param x_size: x size of image
        :param y_size: y size of image
        :return: None
        """
        self._window_image[y_offset:y_offset + y_size,
                           x_offset:x_offset + x_size] = image

    def get_display_window(self, frame, plot=None):
        """
        Returns a display window containing the live camera feed, the current frame, and the current classification
        plot.
        :param frame: Frame to be classified (an image)
        :param plot: Classification plot
        :return: The updated display window
        """

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
