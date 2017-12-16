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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB

features           = "SIFT" #SIFT/hist
Globalclassifier   = "GNB"  #KNN/RandomForest/GNB
agregate_sift_desc = True
nfeatures          = 100
loadimages         = 30

def GetKey(Label):
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
    return Label

def inputImagesLabels():
    train_images_filenames = cPickle.load(open('train_images_filenames.dat', 'r'))
    test_images_filenames  = cPickle.load(open('test_images_filenames.dat', 'r'))
    train_labels           = cPickle.load(open('train_labels.dat', 'r'))
    test_labels            = cPickle.load(open('test_labels.dat', 'r'))

    # print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
    # print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)
    return train_images_filenames, test_images_filenames, train_labels, test_labels

def SIFTfeatures(filenames,labels):
    descriptors = []
    label_per_descriptor = []

    for i in range(len(filenames)):
        filename = filenames[i]
        if label_per_descriptor.count(labels[i]) < loadimages:
            # print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            SIFTdetector = cv2.SIFT(nfeatures=nfeatures)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            if agregate_sift_desc:
                d = np.zeros([1, nfeatures * 128])
                d[0, 0:des.size] = des[0:nfeatures, 0:128].reshape(1, -1)
                descriptors.append(list(d))
            else:
                descriptors.append(des)
            label_per_descriptor.append(labels[i])

    return descriptors,label_per_descriptor

def histfeatures(filenames,labels):
    descriptors = []
    label_per_descriptor = []

    for i in range(len(filenames)):
        filename = filenames[i]
        if label_per_descriptor.count(labels[i]) < loadimages:
            # print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            hist, bin_edges = np.histogram(gray,10)
            descriptors.append(hist)
            label_per_descriptor.append(labels[i])
    return descriptors,label_per_descriptor

def getImageDescriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if features == "hist":
        hist, bin_edges = np.histogram(gray, 10)
        return hist
    elif features == "SIFT":
        SIFTdetector = cv2.SIFT(nfeatures=nfeatures)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)

        if agregate_sift_desc:
            d = np.zeros([1, nfeatures * 128])
            d[0, 0:des.size] = des[0:nfeatures, 0:128].reshape(1, -1)
            return list(d)

        return des



def featureExtraction(filenames, labels):
    descriptors = []
    label_per_descriptor = []

    # Load images and get its descriptors
    for i in range(len(filenames)):
        filename = filenames[i]
        if label_per_descriptor.count(labels[i]) < loadimages:
            # print 'Reading image ' + filename
            image = cv2.imread(filename)
            descriptors.append(getImageDescriptors(image))
            label_per_descriptor.append(labels[i])

    # Transform everything to numpy arrays
    if features == "SIFT" and not agregate_sift_desc:
        D = descriptors[0]
        L = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])

        for i in range(1, len(descriptors)):
            D = np.vstack((D, descriptors[i]))
            L = np.hstack((L, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
    else:
        D = np.array(descriptors[0])
        L = np.array(label_per_descriptor[0])

        for i in range(1, len(descriptors)):
            D = np.vstack((D, np.array(descriptors[i])))
            L = np.hstack((L, np.array(label_per_descriptor[i])))
    return D, L

def printScores(Scrores):
    print scores, scores.mean()
    print 'Done!'

def trainKNNClassifier(D, L, k=5):
    # Train a k-nn classifier
    print 'Training the knn classifier...'
    myknn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    myknn.fit(D, L)
    #VALIDATION: Kfold cross validation
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myknn, D, L, cv=cv)
    #Accuracy for Kfold, scores.mean()
    print(scores)
    return myknn

def trainRFClassifier(D, L,depth=2):
    # Train a RandomForest classifier
    print 'Training the RandomForest classifier...'
    myRF = RandomForestRegressor(max_depth=depth, random_state=0)
    LRF  = []
    for i in range(0, L.shape[0]):
        LRF.append(int(GetKey(L[i])))
    LRF = np.array(LRF)
    myRF.fit(D, LRF)
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myRF, D, LRF, cv=cv)
    #printScores(scores)
    return myRF

def trainBayesClassifier(D, L,depth=2):
    # Train a RandomForest classifier
    print 'Training the Bayes classifier...'
    myGNB = GaussianNB()
    myGNB.fit(D,L)
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myGNB, D, L, cv=cv)
    #printScores(scores)
    return myGNB



def predictAndTest( classifier,descriptors,label_per_descriptor):
    # get all the test data and predict their labels

    numtestimages = 0
    numcorrect = 0
    PredictList = []

    for i in range(len(descriptors)):
        predictions    = classifier.predict(descriptors[i].reshape(1, -1))
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        #print 'image ' + test_images_filenames[i] + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass
        numtestimages += 1
        PredictList.append(predictedclass)
        if Globalclassifier == "RandomForest":
            if round(predictedclass) == int(GetKey(label_per_descriptor[i])):
                numcorrect += 1
        elif predictedclass == label_per_descriptor[i]:
            numcorrect += 1
    npPredictList = np.array(PredictList)
    return numcorrect * 100.0 / numtestimages,npPredictList

def evaluation(L,kPredictions):

    #----canviar per precision recall bien hecho
    precision = []
    recall = []
    for p in kPredictions:
        cm = confusion_matrix(L, p)
        precision.append(float(cm[1, 1]) / (cm[1, 1] + cm[0, 1]))
        recall.append(float(cm[1, 1]) / (cm[1, 1] + cm[1, 0]))
    # ----canviar per precision recall bien hecho

    x1 = np.array(precision)
    y1 = np.array(recall)
    plt.plot(x1, y1)  # line plot
    plt.title('Precision/Recall curve')
    plt.show()


def rocCurve():
    #ROC CURVE
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def __main__():
    start = time.time()
    train_images_filenames, test_images_filenames, train_labels, test_labels = inputImagesLabels()
    D,L = featureExtraction(train_images_filenames, train_labels)

    kVector=np.arange(2,9,1)
    kPredictions = []
    Test_descriptors, Test_label_per_descriptor = featureExtraction(test_images_filenames, test_labels)
    if Globalclassifier == "KNN":
        for k in kVector:
            myknn = trainKNNClassifier(D, L, k)
            accuracy,PredictList = predictAndTest(myknn,Test_descriptors,Test_label_per_descriptor)
            kPredictions.append(PredictList)
            print ('for K = '+ str(k) + ' accuracy is ' + str(accuracy))
            evaluation(L,kPredictions)
    elif Globalclassifier == "RandomForest":
            myRF = trainRFClassifier(D, L)
            accuracy,PredictList = predictAndTest(myRF,Test_descriptors,Test_label_per_descriptor)
            kPredictions.append(PredictList)
            print 'RandomForest accuracy is: ' + str(accuracy)
    elif Globalclassifier == "GNB":
            myGNB = trainBayesClassifier(D, L)
            accuracy,PredictList = predictAndTest(myGNB,Test_descriptors,Test_label_per_descriptor)
            kPredictions.append(PredictList)
            print 'Bayes accuracy is: ' + str(accuracy)

    end = time.time()
    print 'Done in ' + str(end - start) + ' secs.'


__main__()