import numpy as np
import cv2
import classifier
from camera_capture import CameraCapture


def main():

    camera = CameraCapture()

    cv2.namedWindow('Streaming Video', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Streaming Video', 70, 50)
    cv2.resizeWindow('Streaming Video', 500, 500)
    cv2.namedWindow('Prediction Frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Prediction Frame', 580, 50)
    cv2.resizeWindow('Prediction Frame', 500, 500)
    cv2.namedWindow('Predictions', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Predictions', 1090, 50)
    cv2.resizeWindow('Predictions', 800, 500)

    while True:
        frame = camera.get_frame()
        cv2.imshow('Streaming Video', np.flip(frame, axis=1))  # NB flip the frame to show mirror image

        if cv2.waitKey(1) & 0xFF == ord('p'):
            predictions, plot = classifier.classify(frame)
            print(predictions)
            cv2.imshow('Prediction Frame', np.flip(frame, axis=1)) # Show the frame being predicted
            cv2.imshow('Predictions', plot) # Show the plot

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    del camera
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


