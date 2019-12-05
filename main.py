import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


def read_from_file(filename: str):
    data_file = open(filename, "r")
    data_lines = data_file.readlines()
    data_file.close()
    raw_data = []

    # remove newline and brackets
    for line in data_lines:
        raw_data.append(line.replace("\n", "")[1:-1])

    # make values float128 and
    # put them in a np.array and
    # put them in a variable called "data"
    data = []
    for item in raw_data:
        item_as_list = item.split(",")
        string_to_float128 = list(map(np.float128, item_as_list))
        data.append(np.array(string_to_float128))

    return data


def fft_to_amplitude_band(fft: list):
    result = []

    for i in range(0, len(fft), 2):
        result.append(math.sqrt(math.pow(fft[i], 2)) + math.pow(fft[i + 1], 2))

    return np.array(list(map(np.float128, result)))


def fft_to_phase_band(fft: list):
    result = []

    for i in range(0, len(fft), 2):
        # Phase = arctan(Imaginary(F)/Real(F))
        # https://stackoverflow.com/questions/6393257/getting-fourier-transform-from-phase-and-magnitude-matlab
        result.append(math.atan2(fft[i+1], fft[i]))

    return np.array(list(map(np.float128, result)))


def euclidean_distance(x: np.array, y: np.array):
    return np.linalg.norm(x - y)


if __name__ == "__main__":
    data_a = read_from_file("data_group_a")
    data_b = read_from_file("data_group_b")

    amplitude_a = list(map(fft_to_amplitude_band, data_a))
    amplitude_b = list(map(fft_to_amplitude_band, data_b))

    # for item in amplitude_a: plt.plot(item)
    # plt.xlabel("Relative Frequency")
    # plt.ylabel("Relative Amplitude")
    # plt.title("Without a reflector")
    # # plt.savefig("fig1.png")
    # plt.show()
    # for item in amplitude_b: plt.plot(item)
    # plt.xlabel("Relative Frequency")
    # plt.ylabel("Relative Amplitude")
    # plt.title("With a reflector")
    # # plt.savefig("fig2.png")
    # plt.show()

    # ------- Reflector existence test start --------

    X = amplitude_a + amplitude_b
    Y = ["With reflector"] * len(amplitude_a) + ["Without reflector"] * len(amplitude_b)
    trainX, testX, trainY, testY = train_test_split(X, Y, shuffle=True, test_size=0.3)

    model = GaussianNB()
    model.fit(trainX, trainY)
    predY = model.predict(testX)

    print(classification_report(testY, predY))

    # ------- Reflector existence test end --------

    # ------- Strong distance test --------
    strong_close = read_from_file("strong_close")
    strong_mid = read_from_file("strong_mid")
    strong_far = read_from_file("strong_far")

    amp_strong_close = list(map(fft_to_amplitude_band, strong_close))
    amp_strong_mid = list(map(fft_to_amplitude_band, strong_mid))
    amp_strong_far = list(map(fft_to_amplitude_band, strong_far))

    # for item in amp_strong_close: plt.plot(item)
    # plt.show()
    # for item in amp_strong_mid: plt.plot(item)
    # plt.show()
    # for item in amp_strong_far: plt.plot(item)
    # plt.show()

    X = amp_strong_close + amp_strong_mid + amp_strong_far
    Y = ["close"] * len(amp_strong_close) + ["mid"] * len(amp_strong_mid) + ["far"] * len(amp_strong_far)

    trainX, testX, trainY, testY = train_test_split(X, Y, shuffle=True, test_size=0.3)

    pca = PCA(50)
    trainX = pca.fit_transform(trainX)
    testX = pca.transform(testX)

    model = MLPClassifier(solver="lbfgs", max_iter=3000)
    model.fit(trainX, trainY)
    predY = model.predict(testX)

    print(classification_report(testY, predY))
    # ------- Strong distance test end --------

    # ------- Strong distance test --------
    weak_close = read_from_file("weak_close")
    weak_mid = read_from_file("weak_mid")
    weak_far = read_from_file("weak_far")

    amp_weak_close = list(map(fft_to_amplitude_band, weak_close))
    amp_weak_mid = list(map(fft_to_amplitude_band, weak_mid))
    amp_weak_far = list(map(fft_to_amplitude_band, weak_far))

    # for item in amp_strong_close: plt.plot(item)
    # plt.show()
    # for item in amp_weak_mid: plt.plot(item)
    # plt.show()
    # for item in amp_weak_far: plt.plot(item)
    # plt.show()

    X = amp_strong_close + amp_weak_mid + amp_weak_far
    Y = ["close"] * len(amp_strong_close) + ["mid"] * len(amp_weak_mid) + ["far"] * len(amp_weak_far)

    trainX, testX, trainY, testY = train_test_split(X, Y, shuffle=True, test_size=0.3)

    pca = PCA(50)
    trainX = pca.fit_transform(trainX)
    testX = pca.transform(testX)

    model = MLPClassifier(solver="lbfgs", max_iter=3000)
    model.fit(trainX, trainY)
    predY = model.predict(testX)

    # print(classification_report(testY, predY))

    print(classification_report(testY, predY))
    # ------- Strong distance test end --------