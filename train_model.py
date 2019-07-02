# USAGE
# python train_model.py --dataset dataset --model output/lenet.hdf5

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from utils.lenet import LeNet
from utils.minivggnet import MiniVGGNet
from utils.imagehelper import preprocess
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='dataset',
                help="path to input dataset")
ap.add_argument("-o", "--output", default='output/lenet.hdf5',
                help="path to output model")
ap.add_argument("-m", "--model", default='lenet',
                help="train model")
ap.add_argument("-r", "--reset", default='1',
                help="reset dataset")
args = vars(ap.parse_args())


def main():
    class_counter = 0
    img_counter = 0

    if args['reset'] == '1':
        if os.path.exists(args['dataset']):
            shutil.rmtree(args['dataset'])
        os.mkdir(args['dataset'])
        os.mkdir(args['dataset'] + "/" + str(class_counter))

    if args['reset'] == '0':
        train()
        return

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Image")

    print('Press SPACE to capture image')
    print('Press SHIFT to move to next class')
    print('Press ENTER to train')
    print('Press ESC to quit')

    while True:
        ret, frame = cam.read()
        if not ret:
            print('Can not capture image')
            break
        cv2.imshow("Capture Image", frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit - closing...")
            cam.release()
            cv2.destroyAllWindows()
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_path = "{}/{}/opencv_frame_{}.png".format(
                str(args['dataset']), str(class_counter), str(img_counter))
            print(img_path)
            cv2.imwrite(img_path, frame)
            img_counter += 1
            print(
                "SPACE hit - capture {} images for class {}!".format(img_counter, class_counter + 1))
        elif k % 256 == 226:
            # SHIFT pressed
            print('SHIFT hit - next class')
            class_counter += 1
            img_counter = 0
            os.mkdir(args['dataset'] + "/" + str(class_counter))
        elif k % 256 == 13:
            # ENTER pressed
            print('ENTER hit - start training')
            cam.release()
            cv2.destroyAllWindows()
            train()
            break


def train():
    # initialize the data and labels
    data = []
    labels = []

    CNN = MiniVGGNet if args['model'] == 'minivggnet' else LeNet

    # loop over the input images
    for imagePath in paths.list_images(args["dataset"]):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess(image, 224, 224)
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    classes_count = len(np.unique(labels))

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    lb = LabelBinarizer().fit(trainY)
    trainY = lb.transform(trainY)
    testY = lb.transform(testY)

    # initialize the model
    print("[INFO] compiling model...")
    model = CNN.build(width=224, height=224, depth=3, classes=classes_count)
    epochs = 10 if args['model'] == 'minivggnet' else 20
    opt = SGD(lr=0.01, decay=0.01 / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # construct the callback to save only the *best* model to disk
    # based on the validation loss
    checkpoint = ModelCheckpoint(args["output"], monitor="val_loss",
                                 save_best_only=True, verbose=1)
    callbacks = [checkpoint]

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY,  validation_data=(testX, testY),
                  batch_size=32, epochs=epochs, callbacks=callbacks, verbose=1)

if __name__ == '__main__':
    main()
