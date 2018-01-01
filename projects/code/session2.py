import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
# import matplotlib.pyplot as plt

SIFTTYPE = "DSIFT"

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
        if SIFTTYPE == "DSIFT":
        	kpt, des = SIFTdetector.detectAndCompute(gray, None)
        #DENSE SIFT DETECTOR
        elif SIFTTYPE == "DSIFT":
	    	dense  = cv2.FeatureDetector_create("Dense")
	    	kp=dense.detect(gray)
	        kp,des=sift.compute(imgGray,kp)
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

    return words,visual_words


def trainSVM(visual_words,train_labels):
    # Train an SVM classifier with RBF kernel
    print 'Training the SVM classifier...'
    init = time.time()
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    clf = svm.SVC(kernel='rbf', C=1, gamma=.002).fit(D_scaled, train_labels)
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
    D, train_descriptors, label_per_descriptor=SIFTextraction(train_images_filenames, train_labels) #get SIFT descriptors for train set
    k = 512
    codebook=computeCodebook(D,k) #create codebook using train SIFT descriptors
    train_words, train_visual_words=getWords(codebook,train_descriptors,k) #assign descriptors to nearest word(features cluster) in codebook
    clf,stdSlr=trainSVM(train_visual_words,train_labels) #train SVM with with labeled visual words
    D, test_descriptors, foo=SIFTextraction(test_images_filenames) #get SIFT descriptors for test set
    test_words, test_visual_words=getWords(codebook,test_descriptors) #words found at test set
    evaluateAccuracy(clf, stdSlr, test_visual_words,test_labels)

    end = time.time()
    print 'Everything done in ' + str(end - start) + ' secs.'
    ### 69.02%

__main__()