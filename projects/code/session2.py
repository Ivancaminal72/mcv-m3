import cv2
import numpy as np
import cPickle
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import cluster
import matplotlib.pyplot as plt
try:
    from yael import ynumpy
except ImportError:
    print "Yael library not found, you can not use FVECTORS variable"
    print ""

SIFTTYPE = "spatialPyramids"  #DSIFT/SIFT/spatialPyramids
USECV    = False   #True/False
KERNEL   = 'histogramIntersection'   #'rbf'/'poly'/'sigmoid'/'histogramIntersection'
k        = 512     #number of visual words
CODESIZE = 32      #use very short codebooks (32/64)
FVECTORS = False   #True/False (Only with DSIFT)

def inputImagesLabels():
    # read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('train_labels.dat', 'r'))
    test_labels = cPickle.load(open('test_labels.dat', 'r'))
    print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
    print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)
    print ""
    return train_images_filenames, test_images_filenames, train_labels, test_labels

def featureExtraction(filenames, dataset, codebook = None):
    descriptors = []
    des = np.array([])
    try:
        if SIFTTYPE == "spatialPyramids" and codebook is None:
            print "Loading SIFT " + dataset + " descritpors (to compute codebook)..."
            if not os.path.exists("./data_s2"):
                os.makedirs("./data_s2")
            init = time.time()
            # Load descriptors & labels
            descriptors = cPickle.load(open("./data_s2/SIFT_" + dataset + "_descriptors.dat", "rb"))
            end = time.time()
            print 'Done in ' + str(end - init) + ' secs.'
            print ""

        elif SIFTTYPE == "spatialPyramids" and codebook is not None:
            raise Exception('Compute visual words')
        else:
            print "Loading " + SIFTTYPE  + dataset + " descriptors..."
            if not os.path.exists("./data_s2"):
                os.makedirs("./data_s2")
            init = time.time()
            # Load descriptors & labels
            descriptors = cPickle.load(open("./data_s2/" + SIFTTYPE + "_" + dataset + "_descriptors.dat", "rb"))
            end = time.time()

            print 'Done in ' + str(end - init) + ' secs.'
            print ""

    except (OSError, IOError, Exception):
        # create the SIFT detector object
        SIFTdetector = cv2.SIFT(nfeatures=300)

        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        init = time.time()
        for i in range(len(filenames)):
            if SIFTTYPE == "spatialPyramids" and codebook is None:
                print "Extracting " + dataset + " descriptors using SIFT (to compute codebook)... " + str(i) + '/' + str(len(filenames))
            else:
                print "Extracting " + dataset + " descriptors using " + SIFTTYPE + "... " + str(i) + '/' + str(len(filenames))
            filename = filenames[i]
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            #SIFT DETECTOR
            if SIFTTYPE == "SIFT" or "spatialPyramids" and codebook is None:
                kpt, des = SIFTdetector.detectAndCompute(gray, None)
            #DENSE SIFT DETECTOR
            elif SIFTTYPE == "DSIFT":
                dense  = cv2.FeatureDetector_create("Dense")
                kp=dense.detect(gray)
                kpt,des=SIFTdetector.compute(gray,kp)
            #SPATIAL PYRAMIDS SIFT DETECTOR
            elif SIFTTYPE == "spatialPyramids":
                des = spatialPyramids(gray, SIFTdetector, codebook, 3)
            else:
                raise ValueError('Not valid SIFTTYPE option')

            descriptors.append(des)
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.'
        print ""
        if FVECTORS and SIFTTYPE == "DSIFT":
            print "Calculating Fisher vectors"
            init = time.time()
            gmm = ynumpy.gmm_learn(des, CODESIZE)
            fv  = ynumpy.fisher(gmm, des, include = ['mu','sigma'])
            end = time.time()
            print 'Done in ' + str(end - init) + ' secs.'
            print fv

            #Save descriptors & labels
            init = time.time()
            if SIFTTYPE == "spatialPyramids" or codebook is None:
                print "Saving SIFT descriptors..."
                cPickle.dump(descriptors, open("./data_s2/SIFT_" + dataset + "_descriptors.dat", "wb"))
            else:
                print "Saving " + dataset + " descriptors..."
                cPickle.dump(descriptors, open("./data_s2/" + SIFTTYPE + "_" + dataset + "_descriptors.dat", "wb"))
            end = time.time()
            print 'Done in ' + str(end - init) + ' secs.'
            print ""


    # Transform everything to numpy arrays
    size_descriptors = descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint + len(descriptors[i])] = descriptors[i]
        startingpoint += len(descriptors[i])

    return D,descriptors


def spatialPyramids(gray, SIFTdetector, codebook, levels=3):
    m,n = gray.shape
    visual_words = np.array([])
    for level in range(1, levels):
        step_m=int(m/level)
        step_n=int(n/level)
        vec_m=range(0,m+1,step_m)
        vec_n=range(0,n+1,step_n)
        for idx in range(0,len(vec_m)-1):
            cell = gray[vec_m[idx]:vec_m[idx+1], vec_n[idx]:vec_n[idx+1]]
            kpt, des = SIFTdetector.detectAndCompute(cell, None)
            try:
                des
            except NameError:
                des = None
            if des is None:
                vw = np.zeros((1,k))
                if visual_words.shape[0] == 0:
                    visual_words = vw
                else:
                    visual_words = np.vstack((visual_words, vw))
            else:
                vw = getWords(codebook, [des])
                if visual_words.shape[0] == 0:
                    visual_words = vw
                else:
                    visual_words = np.vstack((visual_words, vw))

    return visual_words.reshape(1,-1)

def computeCodebook(D):
    try:
        print 'Loading kmeans with ' + str(k) + ' centroids...'
        init = time.time()
        if SIFTTYPE == "spatialPyramids":
            codebook = cPickle.load(open("./data_s2/SIFT_" + str(k) + "_codebook.dat", "rb"))
        else:
            codebook = cPickle.load(open("./data_s2/"+ SIFTTYPE + "_" + str(k) + "_codebook.dat", "rb"))
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.'
        print ""
    except(IOError, EOFError):
        # compute the codebook
        print 'Computing kmeans with ' + str(k) + ' centroids...'
        init = time.time()
        codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                           reassignment_ratio=10 ** -4, random_state=42)
        codebook.fit(D)
        if SIFTTYPE == "spatialPyramids":
            cPickle.dump(codebook, open("./data_s2/SIFT_" + str(k) + "_codebook.dat", "wb"))
        else:
            cPickle.dump(codebook, open("./data_s2/"+ SIFTTYPE + "_" + str(k) + "_codebook.dat", "wb"))
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.'
        print ""
    return codebook

def getWords(codebook,descriptors):
    # get train visual word encoding
    visual_words = np.zeros((len(descriptors), k), dtype=np.uint8)
    for i in xrange(len(descriptors)):
        words = codebook.predict(descriptors[i])
        visual_words[i, :] = np.bincount(words, minlength=k)

    return visual_words

def svc_param_selection(D_scaled, train_labels,C,gamma, nfolds=4):
    param_grid = {'C': C, 'gamma' : gamma}
    if KERNEL == 'histogramIntersection':
        kernelMatrix = histogramIntersection(D_scaled, D_scaled)
        clf = GridSearchCV(svm.SVC(kernel='precomputed'), param_grid, cv=nfolds)
        clf.fit(kernelMatrix, train_labels)
    else:
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

def trainSVM(D_scaled,train_labels):
    # Train an SVM classifier with RBF kernel
    print 'Training the SVM classifier...'
    init = time.time()
    if USECV:
        Cs     = 10. ** np.arange(-3, 8)
        gammas = 10. ** np.arange(-5, 4)
        clf = svc_param_selection(D_scaled,train_labels,Cs,gammas)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        print means, stds
        print clf.cv_results_['params']
        ShowGraphic(clf.grid_scores_,Cs,gammas)
    elif KERNEL == 'histogramIntersection':
        kernelMatrix = histogramIntersection(D_scaled, D_scaled)
        clf = svm.SVC(kernel='precomputed',C=10, gamma=.002)
        clf.fit(kernelMatrix, train_labels)
    else:
        clf = svm.SVC(kernel='rbf', C=10, gamma=.002).fit(D_scaled, train_labels) #Best params for SIFT
        #clf = svm.SVC(kernel='rbf', C=?, gamma=?).fit(D_scaled, train_labels)  # Best params for DSIFT
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    print ""
    return clf


def evaluate(clf, D_test, test_labels, D_train):
    # Test the classification accuracy
    print 'Testing the SVM classifier...'
    init = time.time()
    if KERNEL == 'histogramIntersection':
        predictMatrix = histogramIntersection(D_test, D_train)
        SVMpredictions = clf.predict(predictMatrix)
        predicted =0
        images = 0
        for i in xrange(len(SVMpredictions)):
            images +=1
            if SVMpredictions[i] == test_labels[i]:
                predicted +=1
        accuracy = 100 * predicted / images
    else:
        accuracy = 100 * clf.score(D_test, test_labels)
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.'
    print ""
    print 'Final accuracy: ' + str(accuracy) + "%"
    print ""

def __main__():
    start = time.time()  # global time

    train_images_filenames, test_images_filenames, train_labels, test_labels=inputImagesLabels() #get images sets
    D, train_descriptors = featureExtraction(train_images_filenames, "train") #get SIFT descriptors for train set
    D, test_descriptors = featureExtraction(test_images_filenames, "test")  # get SIFT descriptors for test set
    codebook = computeCodebook(D)  # create codebook using train SIFT descriptors
    if SIFTTYPE == "spatialPyramids":
        train_visual_words, train_descriptors = featureExtraction(train_images_filenames, "train", codebook)  # get SIFT descriptors for train set
        test_visual_words, test_descriptors = featureExtraction(test_images_filenames, "test", codebook)  # get SIFT descriptors for test set
    else:
        train_visual_words = getWords(codebook, train_descriptors)  # assign descriptors to nearest word(features cluster) in codebook
        test_visual_words = getWords(codebook, test_descriptors)  # words found at test set

    stdSlr = StandardScaler().fit(train_visual_words.astype(float))
    D_train = stdSlr.transform(train_visual_words.astype(float))  # normalize train words
    D_test = stdSlr.transform(test_visual_words.astype(float))  # normalize test words
    clf = trainSVM(D_train,train_labels) #train SVM with with labeled visual words
    evaluate(clf, D_test,test_labels,D_train) #evaluate performance

    end = time.time()
    print 'Everything done in ' + str(end - start) + ' secs.'
    
__main__()
