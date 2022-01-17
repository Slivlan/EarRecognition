import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
from sklearn import model_selection
from sklearn.model_selection import train_test_split

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.test_images_path = config['test_images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        for i in range(len(im_list)):
            im_list[i] = im_list[i].replace('\\', '/')

        im_test_list = sorted(glob.glob(self.test_images_path + '/*.png', recursive=True))
        for i in range(len(im_test_list)):
            im_test_list[i] = im_test_list[i].replace('\\', '/')

        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        #dict: key = file name, value = class
        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()
        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()
        
        lbp_features_arr = []
        plain_features_arr = []
        # y = list - classes
        y = []
        p = Preprocess()
        n_bins = 10000
        applyHistogram = True
        applyEdgeAugmentation = False

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            t = '/'.join(im_name.split('/')[-2:])

            y.append(cla_d[t])

            # Apply some preprocessing here
            if (applyHistogram):
                img = p.histogram_equlization_rgb(img)

            # edge augmentation
            if (applyEdgeAugmentation):
                img = p.edge_augmentation(img)
            
            # Run the feature extractors            
            plain_features = pix2pix.extract(img)
            plain_features_arr.append(plain_features)
            #lbp_features = lbp.extract(img, n_bins)
            #lbp_features_arr.append(lbp_features)

        y_test = []
        lbp_features_test_arr = []
        plain_features_test_arr = []
        for im_name in im_test_list:
            # Read an image
            img = cv2.imread(im_name)

            t = '/'.join(im_name.split('/')[-2:])

            y_test.append(cla_d[t])

            # Apply some preprocessing here
            if(applyHistogram):
                img = p.histogram_equlization_rgb(img)

            # edge augmentation
            if(applyEdgeAugmentation):
                img = p.edge_augmentation(img)

            # Run the feature extractors
            plain_features = pix2pix.extract(img)
            plain_features_test_arr.append(plain_features)
            #lbp_features = lbp.extract(img, n_bins)
            #lbp_features_test_arr.append(lbp_features)

        # # spodnja zakomentirana koda je koda za odkrivanje najboljsih parametrov za random forest - 1600 dreves, 340 globine
        # from sklearn.model_selection import RandomizedSearchCV
        # # number of trees in random forest
        # n_estimators = [int(x) for x in np.linspace(start=800, stop=3000, num=10)]
        # # number of features at every split
        # max_features = ['auto', 'sqrt']
        #
        # # max depth
        # max_depth = [int(x) for x in np.linspace(100, 800, num=11)]
        # max_depth.append(None)
        # # create random grid
        # random_grid = {
        #     'n_estimators': n_estimators,
        #     'max_features': max_features,
        #     'max_depth': max_depth
        # }
        # # Random search of parameters
        # rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
        #                                 random_state=42, n_jobs=-1)
        # # Fit the model
        # rfc_random.fit(lbp_features_arr, y)
        # # print results
        # print(rfc_random.best_params_)
        #
        # return

        rfc = RandomForestClassifier()
        rfc.n_estimators = 2000
        rfc.max_depth = 100
        rfc.max_features = 'auto'
        rfc.fit(plain_features_arr, y)

        predictions = rfc.predict(plain_features_test_arr)
        correct = 0
        all = 0
        for i in range(len(plain_features_test_arr)):
            all += 1
            print(f"prediction: {predictions[i]}  --  real class: {y_test[i]}")
            if(predictions[i] == y_test[i]):
                correct += 1

        print(f"Random forest: correct: {correct}, all: {all}, percent: {correct/all*100.0}")


        # svm_model = svm.SVC()
        # svm_model.fit(lbp_features_arr, y)
        #
        # predictions = svm_model.predict(lbp_features_test_arr)
        # correct = 0
        # all = 0
        # for i in range(len(lbp_features_test_arr)):
        #     all += 1
        #     print(f"prediction: {predictions[i]}  --  real class: {y_test[i]}")
        #     if (predictions[i] == y_test[i]):
        #         correct += 1
        #
        # print(f"SVM correct: {correct}, all: {all}, percent: {correct / all * 100.0}")

        return

        Y_plain = cdist(plain_features_arr, plain_features_arr, 'jensenshannon')
        y_lbp = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')

        r1 = eval.compute_rank1_sum(Y_plain, y)
        print('Pix2Pix Rank-1 sum[%]', r1)
        r1 = eval.compute_rank1_sum(y_lbp, y)
        print('LBP Rank-1 sum[%]', r1)

        r1 = eval.compute_rank1(Y_plain, y)
        print('Pix2Pix Rank-1[%]', r1)
        r1 = eval.compute_rank1(y_lbp, y)
        print('LBP Rank-1[%]', r1)

        r1 = eval.compute_rank1_avg_top3(Y_plain, y)
        print('Pix2Pix Rank-1-avg-top3[%]', r1)
        r1 = eval.compute_rank1_avg_top3(y_lbp, y)
        print('LBP Rank-1-avg-top3[%]', r1)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()