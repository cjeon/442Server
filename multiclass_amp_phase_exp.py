import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from main import read_from_file
from main import fft_to_amplitude_band
from main import fft_to_phase_band
from itertools import chain


def to_raw_data_map(class_file_map):
    ret = {}
    for class_name, file_name in class_file_map.items():
        ret[class_name] = read_from_file(file_name)
    return ret


def to_amplitude_map(m):
    ret = {}
    for class_name, raw_data in m.items():
        ret[class_name] = list(map(fft_to_amplitude_band, raw_data))
    return ret


def to_phase_map(m):
    ret = {}
    for class_name, raw_data in m.items():
        ret[class_name] = list(map(fft_to_phase_band, raw_data))
    return ret

'''
5 classes
                   precision    recall  f1-score   support
          blanket       0.55      0.47      0.51       120
   coffee_machine       0.43      0.54      0.48       112
      glasses_box       0.50      0.52      0.51       116
plastic_container       0.48      0.39      0.43       111
         umbrella       0.58      0.62      0.60       104
         
         accuracy                           0.50       563
        macro avg       0.51      0.51      0.50       563
     weighted avg       0.51      0.50      0.50       563
'''

if __name__ == '__main__':
    # insert class name and file name here.
    # classFileMap = {
    #                 "umbrella": "umbrella_tail.txt",
    #                 "blanket": "blanket_tail.txt",
    #                 "keyboard": "keyboard_tail.txt",
    #                 "pencil_holder": "pencil_holder_tail.txt",
    #                 "stuffed_animal": "stuffed_animal_tail.txt",
    #                 "wrapper": "wrapper_tail.txt",
    #                 "pot": "pot_tail.txt",
    #                 "plastic_container": "plastic_container_tail.txt",
    #                 "power_outlet": "power_outlet_tail.txt",
    #                 "glasses_box": "glasses_box_tail.txt",
    #                 "coffee_machine": "coffee_machine_tail.txt",
    #                 "paper_cup": "paper_cup_tail.txt",
    #                 "plastic_cup": "plastic_cup_tail.txt",
    #                 "hand_wash": "hand_wash_tail.txt",
    #                 }

    classFileMap = {
        "metal_container": "mc_all.txt",
        "heatgun": "hg_all.txt",
        "paper_cup": "pc_all.txt"
    }

    # import raw data
    raw_data_map = to_raw_data_map(classFileMap)

    # change to amplitude
    amplitudeMap = to_amplitude_map(raw_data_map)

    # change to phase
    # phaseMap = to_phase_map(raw_data_map)

    # merge the amplitude and phase
    merged = {}
    for className in classFileMap.keys():
        # merged[className] = amplitudeMap[className] + phaseMap[className]
        merged[className] = amplitudeMap[className]
        # merged[className] = phaseMap[className]

    # make X and Y
    def flatmap(nested_list):
        return list(chain.from_iterable(nested_list))


    keys = merged.keys()
    X = flatmap(map(merged.get, keys))
    Y = flatmap( [key] * len(merged[key]) for key in keys)

    trainX, testX, trainY, testY = train_test_split(X, Y, shuffle=True, test_size=0.3)

    pca = PCA(100)
    trainX = pca.fit_transform(trainX)
    testX = pca.transform(testX)

    # model = MLPClassifier(solver="lbfgs", max_iter=3000)
    # model = SVC(kernel="linear")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    classifiers = [
        # KNeighborsClassifier(n_neighbors=10, n_jobs=-1, leaf_size=100),
        # LogisticRegression(n_jobs=-1, max_iter=10000, solver="saga", multi_class="multinomial"),
        # DecisionTreeClassifier(),
        RandomForestClassifier(n_jobs=-1, n_estimators=10000),
        # GradientBoostingClassifier(n_estimators=20000),
        # GaussianNB(),
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis(),
        # AdaBoostClassifier(n_estimators=2000),
        # MLPClassifier(solver="adam", max_iter=30000)
    ]

    for model in classifiers:
        print(model.__class__.__name__)
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        print(classification_report(testY, predY))

    # model = GaussianNB()
    # model.fit(trainX, trainY)
    # predY = model.predict(testX)
    #
    # print(classification_report(testY, predY))