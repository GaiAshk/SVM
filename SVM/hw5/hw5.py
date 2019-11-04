from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array, linspace
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amount of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    # cut the data by the max count given
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]
    else:
        max_count = len(labels)

    train_data = array([])
    train_labels = array([])
    test_data = array([])
    test_labels = array([])

    # merge the data and labels to a signal matrix
    complete_data = concatenate((data, labels[:, None]), axis=1)

    # shuffle the matrix, this way all labels and data will stay in the right places
    complete_data = permutation(complete_data)

    # split the data according to train ration
    train_size = int(train_ratio * max_count)
    train_data = complete_data[:train_size, :]
    test_data = complete_data[train_size:, :]

    # split back to data and labels
    test_labels = test_data[:, -1]
    train_labels = train_data[:, -1]
    train_data = train_data[:, :-1]
    test_data = test_data[:, :-1]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """

    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    # print("shape:")
    # print(labels.shape[0])
    # print("len:")
    # print(len(labels))

    size_of_data = len(labels)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(size_of_data):
        if labels[i] == 1:
            if prediction[i] == 1:
                tp = tp+1
            else:
                fn = fn+1
        else:
            if prediction[i] == 1:
                fp = fp+1
            else:
                tn = tn+1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return (tp / (tp + fn)), (fp / (tn + fp)), ((tp + tn) / size_of_data)


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    folds_array_length = len(folds_array)

    for i in range(folds_array_length):
        # create test data for this i
        test_data = folds_array[i]
        test_labels = labels_array[i]

        # create train data
        temp_data = folds_array[:i] + folds_array[i+1:]
        temp_labels = labels_array[:i] + labels_array[i+1:]

        for j in range(len(temp_data)):
            if j == 0:
                train_data = temp_data[0]
                train_labels = temp_labels[0]
            else:
                train_data = concatenate((train_data, temp_data[j]))
                train_labels = concatenate((train_labels, temp_labels[j]))

        # run the SVM algorithm on the given data
        clf.fit(train_data, train_labels)
        tpr_cur, fpr_cur, acc_cur = get_stats(clf.predict(test_data), test_labels)
        tpr.append(tpr_cur)
        fpr.append(fpr_cur)
        accuracy.append(acc_cur)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    tpr_tuple = ()
    fpr_tuple = ()
    acc_tuple = ()

    data = array_split(data_array, folds_count)
    labels = array_split(labels_array, folds_count)

    # run on all the given params
    for i in range(len(kernel_params)):
        param = list(kernel_params[i])

        gamma = SVM_DEFAULT_GAMMA
        degree = SVM_DEFAULT_DEGREE
        c = SVM_DEFAULT_C
        if 'degree' in param:
            degree = kernel_params[i].get('degree')
        if 'gamma' in param:
            gamma = kernel_params[i].get('gamma')
        if 'C' in param:
            c = kernel_params[i].get('C')

        clf = SVC(degree=degree, gamma=gamma, C=c, kernel=kernels_list[i])

        # get k fold stats
        tpr, fpr, accuracy = get_k_fold_stats(data, labels, clf)

        tpr_tuple = tpr_tuple + (tpr, )
        fpr_tuple = fpr_tuple + (fpr, )
        acc_tuple = acc_tuple + (accuracy, )

    svm_df['tpr'] = tpr_tuple
    svm_df['fpr'] = fpr_tuple
    svm_df['accuracy'] = acc_tuple

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm_df


def get_most_accurate_kernel(acc_array):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    best_kernel, value = max(enumerate(acc_array), key=lambda e: e[1])
    return best_kernel


def get_kernel_with_highest_score(score_array):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel, value = max(enumerate(score_array), key=lambda e: e[1])
    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()
    fpr_best = x[get_kernel_with_highest_score(df['score'])]
    tpr_best = y[get_kernel_with_highest_score(df['score'])]
    b = -1 * (alpha_slope * fpr_best) + tpr_best

    p = poly1d([alpha_slope, b])
    range2 = [0, 1]
    p1 = p(range2)
    plt.plot(range2, p1, color='red', lw=2)
    plt.scatter(x, y, alpha=0.5)

    z = polyfit(x, y, 3)
    p = poly1d(z)
    xp = linspace(0, 1, 100)
    _ = plt.plot(x, y, '.', xp, p(xp), '-')
    plt.ylim(0, 1.5)
    plt.show()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count, kernel_type, kernel_params):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    map2 = {}
    kernel_type_list = []
    list_of_kernel_maps = []
    list_i = [1, 0, -1, -2, -3, -4]
    list_j = [3, 2, 1]
    for i in list_i:
        for j in list_j:
            map2['C'] = (10**i)*(j/3)
            map2.update(kernel_params)
            list_of_kernel_maps.append(map2)
            kernel_type_list.append(kernel_type)
            map2 = {}

    res = compare_svms(data_array, labels_array, folds_count, kernel_type_list, list_of_kernel_maps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, kernel_type, kernel_params):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0
    
    gamma = SVM_DEFAULT_GAMMA
    degree = SVM_DEFAULT_DEGREE
    c = SVM_DEFAULT_C

    if 'degree' in kernel_params:
        degree = kernel_params.get('degree')
    if 'gamma' in kernel_params:
        gamma = kernel_params.get('gamma')
    if 'C' in kernel_params:
        c = kernel_params.get('C')

    clf = SVC(degree=degree, gamma=gamma, C=c, kernel=kernel_type, class_weight='balanced')

    clf.fit(train_data, train_labels)
    tpr, fpr, accuracy = get_stats(clf.predict(test_data), test_labels)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
