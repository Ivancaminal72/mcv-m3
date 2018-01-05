import cv2
import numpy as np
import cPickle
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm, grid_search
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

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


def SIFTextraction(filenames, dataset, labels=[]):
    descriptors = []
    label_per_descriptor = []
    des = np.array([])
    try:
        if not os.path.exists("./data_s2"):
            os.makedirs("./data_s2")
        # Load descriptors & labels
        descriptors = cPickle.load(open("./data_s2/" + SIFTTYPE + "_" + dataset + "_descriptors.dat", "rb"))
        label_per_descriptor = cPickle.load(open("./data_s2/" + SIFTTYPE +"_" + dataset + "_label_per_descriptor.dat", "rb"))

        print "descriptors loaded!"

    except (OSError, IOError):
        # create the SIFT detector object
        SIFTdetector = cv2.SIFT(nfeatures=300)

        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
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
            #SPATIAL PYRAMIDS SIFT DETECTOR
            elif SIFTTYPE == "spatialPyramids":
                des = spatialPyramids(gray, SIFTdetector, 3)

            descriptors.append(des)
            if len(labels)!=0:
                label_per_descriptor.append(labels[i])
            print str(des.shape[0]) + ' extracted descriptors with ' + SIFTTYPE

        #Save descriptors & labels
        cPickle.dump(descriptors, open("./data_s2/" + SIFTTYPE +"_" + dataset + "_train_descriptors.dat", "wb"))
        cPickle.dump(label_per_descriptor, open("./data_s2/" + SIFTTYPE +"_" + dataset + "_train_label_per_descriptor.dat", "wb"))

    # Transform everything to numpy arrays
    size_descriptors = descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint + len(descriptors[i])] = descriptors[i]
        startingpoint += len(descriptors[i])

    return D,descriptors,label_per_descriptor


def spatialPyramids(gray, SIFTdetector, levels=3):
    descriptors = []
    m,n = gray.shape
    for level in range(1, levels):
        step_m=np.round(m/level)
        step_n=np.round(n/level)
        vec_m=range(0,m,step_m)
        vec_m.append(m)
        vec_n=range(0,n,step_n)
        vec_n.append(n)
        for idx in range(0,len(vec_m)):
            if idx + 1 > len(vec_m) -1:
                break
            cell = gray[vec_m[idx]:vec_m[idx+1], vec_n[idx]:vec_n[idx+1]]
            kpt, des = SIFTdetector.detectAndCompute(cell, None)
            try:
                des
            except NameError:
                des = None
            if des is None:
                continue
            else:
                for i in des:
                    descriptors.append(i)
        vec_m = []
        vec_n = []
    return np.array(descriptors)


def computeCodebook(D,k=512):
    try:
        codebook = cPickle.load(open("./data_s2/"+ str(k) + "_codebook.dat", "rb"))
    except(IOError, EOFError):
        # compute the codebook
        print 'Computing kmeans with ' + str(k) + ' centroids'
        init = time.time()
        codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                           reassignment_ratio=10 ** -4, random_state=42)
        codebook.fit(D)
        cPickle.dump(codebook, open("./data_s2/"+ str(k) + "_codebook.dat", "wb"))
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

def ShowGraphic(results,C_range,gamma_range):
    # We extract just the scores
    scores = [x[1] for x in results]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    # Make a nice figure
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.show()

def trainSVM(visual_words,train_labels):
    # Train an SVM classifier with RBF kernel
    print 'Training the SVM classifier...'
    init = time.time()
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    if USECV:
        #Cs = [0.001, 0.01, 0.1, 1, 10, 100]
        #gammas = [0.002,0.001, 0.01, 0.1, 1, 10]
        Cs     = 10. ** np.arange(-3, 8)
        gammas = 10. ** np.arange(-5, 4)
        clf = svc_param_selection(D_scaled,train_labels,Cs,gammas)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print means, stds
        print clf.cv_results_['params']
        ShowGraphic(clf.grid_scores_,Cs,gammas)
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
    D, train_descriptors, label_per_descriptor=SIFTextraction(train_images_filenames, "train", train_labels) #get SIFT descriptors for train set
    k = 512
    codebook=computeCodebook(D,k) #create codebook using train SIFT descriptors
    train_visual_words=getWords(codebook,train_descriptors,k) #assign descriptors to nearest word(features cluster) in codebook
    clf,stdSlr=trainSVM(train_visual_words,train_labels) #train SVM with with labeled visual words
    D, test_descriptors, foo=SIFTextraction(test_images_filenames, "test") #get SIFT descriptors for test set
    test_visual_words=getWords(codebook,test_descriptors) #words found at test set
    evaluateAccuracy(clf, stdSlr, test_visual_words,test_labels)

    end = time.time()
    print 'Everything done in ' + str(end - start) + ' secs.'
    ### 69.02%

__main__()