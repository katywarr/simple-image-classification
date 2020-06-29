import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# Static values
MODEL = ResNet50(weights='imagenet')
DIMENSIONS = (224, 224)    # Dimensions of image required for ResNet50


def get_predictions(img, num_predictions=5):
    """
    Get top predictions for an image
    :param img: image
    :param num_predictions: top number of predictions to return
    :return: vector of predictions for the image
    """
    img_resize = cv2.resize(img, DIMENSIONS, interpolation=cv2.INTER_AREA)
    img_for_classifier = np.expand_dims(img_resize, axis=0)
    img_for_classifier = preprocess_input(img_for_classifier)
    # decode the results into a list of tuples (class, description, probability)
    return decode_predictions(MODEL.predict(img_for_classifier), top=num_predictions)[0]


def classify(img, num_predictions=5):
    """
    Get top predictions for an image and an associated bar chart display
    :param img: image
    :param num_predictions: top number of predictions to return
    :return: predictions, predictions bar chart
    """
    predictions = get_predictions(img, num_predictions)
    transpose_predictions = np.transpose(predictions)
    classifications = transpose_predictions[1]
    # values = np.around((transpose_predictions[2] * 100), decimals=2)
    values = transpose_predictions[2].astype(np.float)
    print('Predictions: ', predictions)
    fig, ax = plt.subplots()
    y_pos = np.arange(len(classifications))
    ax.barh(y_pos, values, align='center')
    ax.set_xlabel('Prediction')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classifications)
    ax.invert_yaxis()
    plt.xlim(0, 1.0)
    plt.tight_layout()
    # redraw the canvas
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return predictions, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
