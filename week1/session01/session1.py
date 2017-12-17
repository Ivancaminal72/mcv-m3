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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, f1_score


features           = "SIFT" #SIFT/hist
Globalclassifier   = "RF"  #KNN/RF/GNB/LG
agregate_sift_desc = True
nfeatures          = 100
loadimages         = 30
UseROC             = True   #True/False

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

def printScores(scores,K=0):
    #print scores
    if K > 0:
        print 'Acuracy for Kfold with K = ' + str(K) + ' in training: ' + str(scores.mean())
    else:
        print 'Acuracy for Kfold in training: ' + str(scores.mean())

def trainKNNClassifier(D, L, k=5):
    # Train a k-nn classifier
    myknn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    myknn.fit(D, L)
    #VALIDATION: Kfold cross validation
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myknn, D, L, cv=cv)
    printScores(scores,k)
    return myknn

def trainRFClassifier(D, L,depth=2):
    # Train a RandomForest classifier
    myRF = RandomForestClassifier(max_depth=depth, random_state=0)
    myRF.fit(D, L)
    #VALIDATION: Kfold cross validation
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myRF, D, L, cv=cv)
    printScores(scores)
    return myRF

def trainBayesClassifier(D, L):
    # Train a RandomForest classifier
    myGNB = GaussianNB()
    myGNB.fit(D,L)
    #VALIDATION: Kfold cross validation
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
    scores = cross_val_score(myGNB, D, L, cv=cv)
    printScores(scores)
    return myGNB

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def logistic_regression( features, target, steps, alpha ):
    LRtarget  = []
    for i in range(len(features)):
        LRtarget.append( int( GetKey( target[i] ) ) )
    LRtarget = np.array( LRtarget )
    weights  = np.zeros( features.shape[1] )
    m,n   = features.shape
    y     = LRtarget.reshape( m, 1 )
    theta = np.ones( shape=( n, 1 ) ) 
    for step in range( steps ):
        h = sigmoid( np.dot( features, theta ) )
        error = ( h - y )
        gradient = np.dot( features.T , error ) / m
        theta = theta - alpha * gradient
    return theta

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
        if Globalclassifier == "RF":
            if predictedclass == int(GetKey(label_per_descriptor[i])):
                numcorrect += 1
        elif predictedclass == label_per_descriptor[i]:
            numcorrect += 1
    npPredictList = np.array(PredictList)
    return numcorrect * 100.0 / numtestimages,npPredictList

def prCurve(L, kPredictions):
    print "Plotting Precision/Recall curve"
    precision = []
    recall = []
    for p in kPredictions:
        cm = confusion_matrix(L, p)
        precision.append(float(cm[1, 1]) / (cm[1, 1] + cm[0, 1]))
        recall.append(float(cm[1, 1]) / (cm[1, 1] + cm[1, 0]))
    x1 = np.array(precision)
    y1 = np.array(recall)
    plt.plot(x1, y1)  # line plot
    plt.title('Precision/Recall curve')
    plt.show()

def rocCurve(descriptors,label_per_descriptor,classifier):
    print "Plotting ROC curve"
    tprs = []
    aucs = []
    LROC  = []
    for i in range(len(descriptors)):
        LROC.append(int(GetKey(label_per_descriptor[i])))
    LROC = np.array(LROC)
    #Binarize to convert in one vs all
    LROC = label_binarize(LROC, classes=[1,2,3,4,5,6,7,8])
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test =\
    train_test_split(descriptors, LROC, test_size=0.125, random_state=0)
    # classifier
    clf = OneVsRestClassifier(classifier)
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    print y_score.mean()
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(8):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def F1_Recall(L,predictList):
    #Weighted average of the F1 score of each class
    print 'F1 Score: '+ str(f1_score(L, predictList, average='macro'))


def evaluationForOneVsAll(D, L, classifier):
    print 'Evaluating the classifier for ...'
    # PRECISION / RECALL curve
    #prCurve(L, kPredictions)

    #ROC curve
    if UseROC == True and Globalclassifier != "LG":
        rocCurve(D,L,classifier)

def __main__():
    start = time.time()
    train_images_filenames, test_images_filenames, train_labels, test_labels = inputImagesLabels()
    D,L = featureExtraction(train_images_filenames, train_labels)

    kVector=np.arange(2,9,1)
    kaccuracy = np.zeros(len(kVector))
    kPredictions = []
    Test_descriptors, Test_label_per_descriptor = featureExtraction(test_images_filenames, test_labels)
    if Globalclassifier == "KNN":
        print 'KNN classifier...'
        for idx, k in enumerate(kVector):
            classifier = trainKNNClassifier(D, L, k)
            accuracy,PredictList = predictAndTest(classifier,Test_descriptors,Test_label_per_descriptor)
            kPredictions.append(PredictList)
            kaccuracy[idx] = accuracy
            print "KNN with K = " + str(k) + " accuracy in test is " + str(accuracy / 100.0)
            F1_Recall(L,PredictList)
        print "The best KNN is:"
        #Recompute the best classifier (based on accuracy)
        classifier = trainKNNClassifier(D, L, kVector[np.argmax(kaccuracy)])
        accuracy, PredictList = predictAndTest(classifier, Test_descriptors, Test_label_per_descriptor)
        print "KNN with K = " + str(k) + " accuracy in test is " + str(accuracy / 100.0)
        F1_Recall(L,PredictList)

    elif Globalclassifier == "RF":
            print "RandomForest classifier..."
            LRF  = []
            for i in range(0, L.shape[0]):
                LRF.append(int(GetKey(L[i])))
            LRF = np.array(LRF)
            L = LRF
            classifier = trainRFClassifier(D,L)
            accuracy,PredictList = predictAndTest(classifier,Test_descriptors,Test_label_per_descriptor)
            F1_Recall(L,PredictList)
            #kPredictions.append(PredictList)
            print "RandomForest accuracy in test is: " + str(accuracy / 100.0)
    elif Globalclassifier == "GNB":
            print "Bayes classifier..."
            classifier = trainBayesClassifier(D, L)
            accuracy,PredictList = predictAndTest(classifier,Test_descriptors,Test_label_per_descriptor)
            kPredictions.append(PredictList)
            print "Bayes accuracy in test is: " + str(accuracy / 100.0)
            F1_Recall(L,PredictList)
    elif Globalclassifier == "LG":
        print "Logistic regresion classifier..."
        values = logistic_regression(Test_descriptors,Test_label_per_descriptor,1000,10.5)
        print "Logistic regresion values are: " + str(values)

    #Evaluation for One Vs All
    evaluationForOneVsAll(D, L, PredictList, kPredictions, classifier)

    end = time.time()
    print "Finished"
    print 'Done in ' + str(end - start) + ' secs.'

__main__()