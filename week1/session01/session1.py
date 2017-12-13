import cv2
import numpy as np
import cPickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit


def GetLabel(Label):
    switcher = {
        1: "coast",
        2: "forest",
        3: "highway",
        4: "inside_city",
        5: "mountain",
        6: "Opencountry",
        7: "street",
        8: "tallbuilding",
    }
    for key, value in switcher.iteritems():
        if Label == value:
            return key

def inputImagesLabels():
    train_images_filenames = cPickle.load(open('train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('train_labels.dat', 'r'))
    test_labels = cPickle.load(open('test_labels.dat', 'r'))

    # print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
    # print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)
    return train_images_filenames, test_images_filenames, train_labels, test_labels


def featureExtraction(train_images_filenames, train_labels):
    # create the SIFT detector object

    SIFTdetector = cv2.SIFT(nfeatures=100)

    # read the just 30 train images per class
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays

    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(train_images_filenames)):
        filename = train_images_filenames[i]
        if Train_label_per_descriptor.count(train_labels[i]) < 30:
            # print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            Train_descriptors.append(des)
            Train_label_per_descriptor.append(train_labels[i])
        # print str(len(kpt)) + ' extracted keypoints and descriptors'
    print 'Features extracted'
    # Transform everything to numpy arrays

    D = Train_descriptors[0]
    L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

    for i in range(1, len(Train_descriptors)):
        D = np.vstack((D, Train_descriptors[i]))
        L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))
    return D, L,SIFTdetector


def trainClassifier(D, L):
    # Train a k-nn classifier
    myknn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    myknn.fit(D, L)
    #for Kfold cross validation
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myknn, D, L, cv=cv)
    #Accuracy for Kfold, scores.mean()
    print scores, scores.mean()

    print 'Training the knn classifier...'
    #myknn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    #myknn.fit(D, L)
    print 'Done!'
    return myknn


def predictAndTest(test_images_filenames, test_labels, myknn,SIFTdetector):
    # get all the test data and predict their labels

    numtestimages = 0
    numcorrect = 0
    PredictList = []
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        predictions = myknn.predict(des)
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        #print 'image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass
        numtestimages += 1
        PredictList.append(GetLabel(predictedclass))
        if predictedclass == test_labels[i]:
            numcorrect += 1
    npPredictList = np.array(PredictList)
    return numcorrect * 100.0 / numtestimages,npPredictList

def evaluation(L,PredictList,scores):
    precision = []
    recall = []

    cm = confusion_matrix(L, PredictList)
    precision.append(float(cm[1, 1]) / (cm[1, 1] + cm[0, 1]))
    recall.append(float(cm[1, 1]) / (cm[1, 1] + cm[1, 0]))

    x1 = np.array(precision)
    y1 = np.array(recall)
    plt.plot(x1, y1)  # line plot
    plt.title('Precision/Recall curve')
    plt.show()

    #ROC CURVE
    #CREATE LABEL LIST

    #VALUES FOR EACH LABEL

    #fpr, tpr, thresholds = metrics.roc_curve(, , pos_label=2)


def __main__():
    start = time.time()
    train_images_filenames, test_images_filenames, train_labels, test_labels = inputImagesLabels()
    D,L,SIFTdetector = featureExtraction(train_images_filenames, train_labels)
    myknn = trainClassifier(D, L)
    accuracy,PredictList = predictAndTest(test_images_filenames, test_labels, myknn,SIFTdetector)
    #evaluation(L,PredictList)
    print 'Final accuracy: ' + str(accuracy)

    end = time.time()
    print 'Done in ' + str(end - start) + ' secs.'

__main__()