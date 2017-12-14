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

def SIFTfeatures(filenames,labels):
    SIFTdetector = cv2.SIFT(nfeatures=100)
    descriptors = []
    label_per_descriptor = []
    for i in range(len(filenames)):
        filename = filenames[i]
        if label_per_descriptor.count(labels[i]) < 30:
            # print 'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            descriptors.append(des)
            label_per_descriptor.append(labels[i])#REDUNDANT!
    return descriptors,label_per_descriptor

def featureExtraction(train_images_filenames, train_labels):

    Train_descriptors, Train_label_per_descriptor = SIFTfeatures(train_images_filenames, train_labels)
    print 'Features extracted'
    # Transform everything to numpy arrays

    D = Train_descriptors[0]
    L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

    for i in range(1, len(Train_descriptors)):
        D = np.vstack((D, Train_descriptors[i]))
        L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))
    return D, L


def trainClassifier(D, L, k=5):
    # Train a k-nn classifier

    print 'Training the knn classifier...'
    myknn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    myknn.fit(D, L)

    #VALIDATION: Kfold cross validation
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myknn, D, L, cv=cv)
    #Accuracy for Kfold, scores.mean()
    print scores, scores.mean()

    print 'Done!'
    return myknn


def predictAndTest(test_images_filenames, test_labels, myknn):
    # get all the test data and predict their labels

    numtestimages = 0
    numcorrect = 0
    PredictList = []
    Test_descriptors, Test_label_per_descriptor = SIFTfeatures(test_images_filenames, test_labels)
    for i in range(len(Test_descriptors)):
        predictions = myknn.predict(Test_descriptors[i])
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        #print 'image ' + test_images_filenames[i] + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass
        numtestimages += 1
        PredictList.append(GetLabel(predictedclass))
        if predictedclass == Test_label_per_descriptor[i]:
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

    #ROC CURVE
    #CREATE LABEL LIST

    #VALUES FOR EACH LABEL

    #fpr, tpr, thresholds = metrics.roc_curve(, , pos_label=2)

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
    for k in kVector:
        myknn = trainClassifier(D, L, k)
        accuracy,PredictList = predictAndTest(test_images_filenames, test_labels, myknn)
        kPredictions.append(PredictList)
        print ('for K ='+ str(k) + 'accuracy is ' + str(accuracy))

    evaluation(L,kPredictions)
    #print 'Final accuracy: ' + str(accuracy)

    end = time.time()
    print 'Done in ' + str(end - start) + ' secs.'

__main__()