import cv2
from utilities import image_classification_utils
from utilities.camera_capture import CameraCapture
from utilities.display_window import DisplayWindow


def crop_image_to_square(self, frame):
    min_dimension = min(frame.shape[0], frame.shape[1])
    return frame[0:min_dimension, 0:self.min_dimension]


def live_camera_classifier():
    """
    Captures and displays a live camera feed.
    When 'p' is pressed on the keyboard, the current frame is displayed classified by ResNet50.
    Press 'q' to quit.
    """

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
            predictions, plot = image_classification_utils.classify(frame)
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
