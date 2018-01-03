import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm, grid_search
from sklearn import cluster
# import matplotlib.pyplot as plt

SIFTTYPE = "spatialPyramids" #DSIFT,SIFT,spatialPyramids
USECV    = False    #True, False
KERNEL   = 'rbf'   #'rbf','poly' and 'sigmoid'

def inputImagesLabels():
    # read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('train_labels.dat', 'r'))
    test_labels = cPickle.load(open('test_labels.dat', 'r'))
    print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
    print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)
    return train_images_filenames, test_images_filenames, train_labels, test_labels


def SIFTextraction(filenames,labels=[]):
    # create the SIFT detector object
    SIFTdetector = cv2.SIFT(nfeatures=300)

    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    descriptors = []
    label_per_descriptor = []
    for i in range(len(filenames)):
        filename = filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        #SIFT DETECTOR
        if SIFTTYPE == "SIFT":
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
        #DENSE SIFT DETECTOR
        elif SIFTTYPE == "DSIFT":
            dense  = cv2.FeatureDetector_create("Dense")
            kp=dense.detect(gray)
            kpt,des=SIFTdetector.compute(gray,kp)
        descriptors.append(des)
        if len(labels)!=0:
            label_per_descriptor.append(labels[i])
        print str(len(kpt)) + ' extracted keypoints and descriptors with ' + SIFTTYPE

    # Transform everything to numpy arrays
    size_descriptors = descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint + len(descriptors[i])] = descriptors[i]
        startingpoint += len(descriptors[i])
    return D,descriptors,label_per_descriptor


def spatialPyramids(filenames,labels=[],flag=1,l=3):
    # create the SIFT detector object
    SIFTdetector = cv2.SIFT(nfeatures=300)

    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    descriptors = []
    label_per_descriptor = []
    for i in range(len(filenames)):
        filename = filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        # SIFT DETECTOR
        m,n = gray.shape
        for j in range(1,l):
            step_m=np.round(m/l)
            step_n=np.round(n/l)
            vec_m=range(0,m,step_m)
            vec_m.append(m)
            vec_n=range(0,n,step_n)
            vec_n.append(n)
            for idx in range(0,len(vec_m)-2):
                cell = gray[vec_m[idx]:vec_m[idx+1], vec_n[idx]:vec_n[idx+1]]
                kpt, des = SIFTdetector.detectAndCompute(gray, None)
                descriptors.append(des)
                if len(labels) != 0:
                    label_per_descriptor.append(labels[i])
        print ' extracted keypoints and descriptors with all grids '

    size_descriptors = descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint + len(descriptors[i])] = descriptors[i]
        startingpoint += len(descriptors[i])
    if flag==1:
        cPickle.dump(descriptors, open("train_descriptors.dat", "wb"))
        cPickle.dump(D, open("D.dat", "wb"))
        cPickle.dump(label_per_descriptor, open("label_per_descriptor.dat", "wb"))
    else:
        cPickle.dump(descriptors, open("test_descriptors.dat", "wb"))

    return D,descriptors,label_per_descriptor


def computeCodebook(D,k=512):
    # compute the codebook
    print 'Computing kmeans with ' + str(k) + ' centroids'
    init = time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                       reassignment_ratio=10 ** -4, random_state=42)
    codebook.fit(D)
    cPickle.dump(codebook, open("codebook.dat", "wb"))
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return codebook

def getWords(codebook,descriptors,k=512):
    # get train visual word encoding
    print 'Getting BoVW representation'
    init = time.time()
    visual_words = np.zeros((len(descriptors), k), dtype=np.float32)
    for i in xrange(len(descriptors)):
        words = codebook.predict(descriptors[i])
        visual_words[i, :] = np.bincount(words, minlength=k)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'

    return visual_words

def svc_param_selection(D_scaled, train_labels,C,gamma, nfolds=4):
    param_grid = {'C': C, 'gamma' : gamma}
    clf = GridSearchCV(svm.SVC(kernel=KERNEL), param_grid, cv=nfolds)
    clf.fit(D_scaled, train_labels)
    print clf.best_params_
    return clf

def histogramIntersection(M, N):
    N = np.transpose(N)
    K_int = np.zeros((M.shape[0], N.shape[1]), dtype=int)

    for y in range(M.shape[0]):
        for x in range(N.shape[1]):
            K_int[y, x] = np.sum(np.minimum(M[y, :], N[:, x]))

    return K_int

def trainSVM(visual_words,train_labels):
    # Train an SVM classifier with RBF kernel
    print 'Training the SVM classifier...'
    init = time.time()
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    if USECV:
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.002,0.001, 0.01, 0.1, 1]
        clf = svc_param_selection(D_scaled,train_labels,Cs,gammas)
    else:
        clf = svm.SVC(kernel='rbf', C=10, gamma=.002).fit(D_scaled, train_labels) #Best params for SIFT
        #clf = svm.SVC(kernel='rbf', C=?, gamma=?).fit(D_scaled, train_labels)  # Best params for DSIFT
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    return clf,stdSlr


def evaluateAccuracy(clf,stdSlr,visual_words_test,test_labels):
    # Test the classification accuracy
    print 'Testing the SVM classifier...'
    init = time.time()
    accuracy = 100 * clf.score(stdSlr.transform(visual_words_test), test_labels)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    print 'Final accuracy: ' + str(accuracy)
    #return accuracy

def __main__():
    start = time.time()  # global time

    train_images_filenames, test_images_filenames, train_labels, test_labels=inputImagesLabels() #get images sets
    if SIFTTYPE == "spatialPyramids":
        D, train_descriptors, label_per_descriptor=spatialPyramids(train_images_filenames, train_labels) #get SIFT descriptors for train set
    else:
        D, train_descriptors, label_per_descriptor=SIFTextraction(train_images_filenames, train_labels) #get SIFT descriptors for train set
    k = 512
    codebook=computeCodebook(D,k) #create codebook using train SIFT descriptors
    train_visual_words=getWords(codebook,train_descriptors,k) #assign descriptors to nearest word(features cluster) in codebook
    clf,stdSlr=trainSVM(train_visual_words,train_labels) #train SVM with with labeled visual words
    if SIFTTYPE == "spatialPyramids":
        D, test_descriptors, foo=spatialPyramids(test_images_filenames,2) #get SIFT descriptors for test set
    else:
        D, test_descriptors, foo=SIFTextraction(test_images_filenames) #get SIFT descriptors for test set
    test_visual_words=getWords(codebook,test_descriptors) #words found at test set
    evaluateAccuracy(clf, stdSlr, test_visual_words,test_labels)

    end = time.time()
    print 'Everything done in ' + str(end - start) + ' secs.'
    ### 69.02%

__main__()