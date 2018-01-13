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
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

try:
    from yael import ynumpy
except ImportError:
    print "Yael library not found, you can not use fisher vector variables\n"

DESTYPE = "MLP"  #DSIFT/SIFT/spatialPyramids/MLP
MLP_DES_DIR = "/data/MIT_split_descriptors" #Descriptors/L_train directory
DES_LEN = 2048 #Length of the descriptors
USECV    = False   #True/False
KERNEL   = 'rbf'   #'rbf'/'poly'/'sigmoid'/'histogramIntersection' (SVM KERNEL)
k        = 512     #number of visual words
CVSCORES = False    #True/False use Kfold to get cross validation accuracy mean
levels   = 4     #number of levels for spatial pyramids
#FISHER VECTORS (Only if you have Yael library installed)
CODESIZE = 3      #use very short codebooks (32/64)
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
    image_fvs = []
    try:
        if DESTYPE == "spatialPyramids" and codebook is None:
            print "Loading SIFT " + dataset + " descritpors (to compute codebook)..."
            if not os.path.exists("./data_s2"):
                os.makedirs("./data_s2")
            init = time.time()
            # Load descriptors & L_train
            descriptors = cPickle.load(open("./data_s2/SIFT_" + dataset + "_descriptors.dat", "rb"))
            end = time.time()
            print 'Done in ' + str(end - init) + ' secs.\n'

        elif DESTYPE == "spatialPyramids" and codebook is not None:
            raise Exception('Compute visual words')
        elif FVECTORS and DESTYPE == "DSIFT":
            print "Loading DSIFT with FV" + dataset + " descritpors (to compute codebook)..."
            image_fvs   = cPickle.load(open("./data_s2/DSIFT_FV_image_fvs" + dataset + "_descriptors.dat", "rb"))
            descriptors = cPickle.load(open("./data_s2/" + DESTYPE + "_" + dataset + "_descriptors.dat", "rb"))
        elif DESTYPE == "MLP":
            descriptors = cPickle.load(open(MLP_DES_DIR +"/"+dataset+"/MLP_"+str(DES_LEN)+"_descriptors.dat", "rb"))
        else:
            print "Loading " + DESTYPE + " " + dataset + " descriptors..."
            if not os.path.exists("./data_s2"):
                os.makedirs("./data_s2")
            init = time.time()
            # Load descriptors & L_train
            descriptors = cPickle.load(open("./data_s2/" + DESTYPE + "_" + dataset + "_descriptors.dat", "rb"))
            end = time.time()

            print 'Done in ' + str(end - init) + ' secs.\n'

    except (OSError, IOError, Exception):
        # create the SIFT detector object
        SIFTdetector = cv2.SIFT(nfeatures=300)
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        init = time.time()
        image_descs = []
        for i in range(len(filenames)):
            if DESTYPE == "spatialPyramids" and codebook is None:
                print "Extracting " + dataset + " descriptors using SIFT (to compute codebook)... " + str(i) + '/' + str(len(filenames))
            else:
                print "Extracting " + dataset + " descriptors using " + DESTYPE + "... " + str(i) + '/' + str(len(filenames))
            filename = filenames[i]
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            #SIFT DETECTOR
            if (DESTYPE == "SIFT" or DESTYPE== "spatialPyramids") and codebook is None:
                kpt, des = SIFTdetector.detectAndCompute(gray, None)
            #DENSE SIFT DETECTOR
            elif DESTYPE == "DSIFT":
                if FVECTORS:
                    desc, meta = ynumpy.siftgeo_read(filename)
                    if desc.size == 0: desc = np.zeros((0, 128), dtype = 'uint8')
                    # we drop the meta-information (point coordinates, orientation, etc.)
                    image_descs.append(desc)
                else:
                    dense  = cv2.FeatureDetector_create("Dense")
                    dense.setDouble('featureScaleMul', 0.1)
                    dense.setInt('featureScaleLevels', 1)
                    dense.setInt('initXyStep', 5)
                    dense.setInt('initFeatureScale', 7)
                    dense.setBool('varyImgBoundWithScale', False)
                    dense.setBool('varyXyStepWithScale', True)

                    kp=dense.detect(gray)
                    kpt,des=SIFTdetector.compute(gray,kp)
            #SPATIAL PYRAMIDS SIFT DETECTOR
            elif DESTYPE == "spatialPyramids":
                des = spatialPyramids(gray, SIFTdetector, codebook)
            #MLP DESCRIPTOR
            elif DESTYPE == 'MLP':
                raise ValueError("Cant load " + DESTYPE + "_" + str(DES_LEN) + " descriptors")
            else:
                raise ValueError('Not valid DESTYPE option')
            descriptors.append(des)

        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.\n'

        if FVECTORS and DESTYPE == "DSIFT":
            # make a big matrix with all image descriptors
            all_desc = np.vstack(image_descs)

            k = 64
            n_sample = k * 1000

            # choose n_sample descriptors at random
            sample_indices = np.random.choice(all_desc.shape[0], n_sample)
            sample = all_desc[sample_indices]

            # until now sample was in uint8. Convert to float32
            sample = sample.astype('float32')

            # compute mean and covariance matrix for the PCA
            mean = sample.mean(axis = 0)
            sample = sample - mean
            cov = np.dot(sample.T, sample)

            # compute PCA matrix and keep only 64 dimensions
            eigvals, eigvecs = np.linalg.eig(cov)
            perm = eigvals.argsort()                   # sort by increasing eigenvalue
            pca_transform = eigvecs[:, perm[64:128]]   # eigenvectors for the 64 last eigenvalues

            # transform sample with PCA (note that numpy imposes line-vectors,
            # so we right-multiply the vectors)
            sample = np.dot(sample, pca_transform)

            # train GMM
            gmm = ynumpy.gmm_learn(sample, k)
            print "Calculating Fisher vectors"
            init = time.time()
            for image_desc in image_descs:
                # apply the PCA to the image descriptor
                image_desc = np.dot(image_desc - mean, pca_transform)
                # compute the Fisher vector, using only the derivative w.r.t mu
                fv = ynumpy.fisher(gmm, image_desc, include = 'mu')
                image_fvs.append(fv)
            end = time.time()
            print 'Done in ' + str(end - init) + ' secs.\n'

        #Save descriptors & labels
        init = time.time()
        if DESTYPE == "spatialPyramids" and codebook is None:
            print "Saving SIFT descriptors..."
            cPickle.dump(descriptors, open("./data_s2/SIFT_" + dataset + "_descriptors.dat", "wb"))
        else:
            if FVECTORS and DESTYPE == "DSIFT":
                print "Saving DSIFT descriptors with fisher vectors..."
                cPickle.dump(image_fvs, open("./data_s2/DSIFT_FV_image_fvs" + dataset + "_descriptors.dat", "wb"))
                cPickle.dump(descriptors, open("./data_s2/DSIFT_FV" + dataset + "_descriptors.dat", "wb"))
            else:
                print "Saving " + dataset + " descriptors..."
                cPickle.dump(descriptors, open("./data_s2/" + DESTYPE + "_" + dataset + "_descriptors.dat", "wb"))
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.\n'

    if FVECTORS and DESTYPE == "DSIFT":
        # make one matrix with all FVs
        image_fvs = np.vstack(image_fvs)
        # normalizations are done on all descriptors at once
        # power-normalization
        image_fvs = np.sign(image_fvs) * np.abs(image_fvs) ** 0.5
        # L2 normalize
        norms = np.sqrt(np.sum(image_fvs ** 2, 1))
        image_fvs /= norms.reshape(-1, 1)

        # handle images with 0 local descriptor (100 = far away from "normal" images)
        image_fvs[np.isnan(image_fvs)] = 100
        # get the indices of the query images (the subset of images that end in "00")
        query_imnos = [i for i, name in enumerate(filenames) if name[-2:] == "00"]

        # corresponding descriptors
        descriptors = image_fvs[query_imnos]
        size_descriptors = descriptors[0].shape[1]
        D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.uint8)
        startingpoint = 0
        for i in range(len(descriptors)):
            D[startingpoint:startingpoint + len(descriptors[i])] = descriptors[i]
        startingpoint += len(descriptors[i])
        return D,descriptors

    # Transform everything to numpy arrays
    size_descriptors = descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint + len(descriptors[i])] = descriptors[i]
        startingpoint += len(descriptors[i])

    return D,descriptors


def spatialPyramids(gray, SIFTdetector, codebook):
    m,n = gray.shape
    visual_words = np.array([])
    for level in range(1, levels+1):
        step_m=int(m/level)
        step_n=int(n/level)
        vec_m=range(0,m+1,step_m)
        vec_n=range(0,n+1,step_n)
        for idx1 in range(0,len(vec_m)-1):
            for idx2 in range(0, len(vec_n) - 1):
                cell = gray[vec_m[idx1]:vec_m[idx1+1], vec_n[idx2]:vec_n[idx2+1]]
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
        if DESTYPE == "spatialPyramids":
            codebook = cPickle.load(open("./data_s2/SIFT_" + str(k) + "_codebook.dat", "rb"))
        else:
            codebook = cPickle.load(open("./data_s2/" + DESTYPE + "_" + str(k) + "_codebook.dat", "rb"))
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.\n'
    except(IOError, EOFError):
        # compute the codebook
        print 'Computing kmeans with ' + str(k) + ' centroids...'
        init = time.time()
        codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                           reassignment_ratio=10 ** -4, random_state=42)
        codebook.fit(D)
        if DESTYPE == "spatialPyramids":
            cPickle.dump(codebook, open("./data_s2/SIFT_" + str(k) + "_codebook.dat", "wb"))
        else:
            cPickle.dump(codebook, open("./data_s2/" + DESTYPE + "_" + str(k) + "_codebook.dat", "wb"))
        end = time.time()
        print 'Done in ' + str(end - init) + ' secs.\n'
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
        if CVSCORES:
            # VALIDATION: Kfold cross validation
            cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=1)
            scores = cross_val_score(clf, D_scaled, train_labels, cv=cv)
            print 'Acuracy for Kfold in training: ' + str(scores.mean())
    end = time.time()
    print 'Done in ' + str(end - init) + ' secs.\n'
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
    print 'Done in ' + str(end - init) + ' secs.\n'
    print 'Final accuracy: ' + str(accuracy) + "%\n"

def __main__():
    start = time.time()  # global time
    train_images_filenames = test_images_filenames = train_labels = test_labels = None
    if DESTYPE != 'MLP':
        train_images_filenames, test_images_filenames, train_labels, test_labels=inputImagesLabels() #get images sets
    else:
        try:
            train_labels = cPickle.load(open(MLP_DES_DIR + "/train/MLP_2048_labels.dat", "rb"))
            test_labels = cPickle.load(open(MLP_DES_DIR + "/test/MLP_2048_labels.dat", "rb"))
        except(IOError, EOFError):
            print("Cant load " + DESTYPE + "_" + str(DES_LEN) +" labels")

    D_train, train_descriptors = featureExtraction(train_images_filenames, "train") #get SIFT descriptors for train set
    D_test, test_descriptors   = featureExtraction(test_images_filenames, "test")  # get SIFT descriptors for test set
    codebook = computeCodebook(D_train)  # create codebook using train SIFT descriptors
    if DESTYPE == "spatialPyramids":
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
