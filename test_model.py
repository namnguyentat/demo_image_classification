# USAGE
# python test_model.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from utils.imagehelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='dataset',
                help="path to input dataset")
ap.add_argument("-m", "--model", default='output/minivggnet.h5',
                help="path to input model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])


def main():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Detect Image")

    print('Press ESC to quit')

    while True:
        ret, frame = cam.read()
        if not ret:
            print('Can not connect to camera')
            break
        cv2.imshow("Detect Image", frame)
        k = cv2.waitKey(30)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit - closing...")
            break
        else:
            # pre-process the frame then classify it
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = preprocess(image, 100, 100)
            image = np.expand_dims(img_to_array(image), axis=0) / 255.0
            result = model.predict(image)
            pred = result.argmax(axis=1)[0]
            score = result[0][pred]
            if score >= 0.6:
                detected_class = cv2.imread(args['dataset'] + '/' + str(pred) +
                                            '/' + 'opencv_frame_0.png')
                cv2.putText(detected_class, str(score), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow('Detected class', detected_class)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
