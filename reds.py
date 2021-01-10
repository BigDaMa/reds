########################################
# REDS
# Mohammad Mahdavi
# moh.mahdavi.l@gmail.com
# March 2017
# Big Data Management Group
# TU Berlin
# All Rights Reserved
########################################


########################################
import os
import sys
import re
import time
import math
import string
import random
import operator
import pickle
import numpy
import nltk
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.tree
import sklearn.svm
import sklearn.neural_network
import sklearn.kernel_ridge
import sklearn.ensemble
import dataset
import data_cleaning_tool
########################################


########################################
class REDS:
    """
    The main class.
    """

    def __init__(self):
        """
        The constructor does nothing.
        """
        pass

    def dataset_profiler(self, dataset_dictionary):
        """
        This method profiles the dataset.
        """
        d = dataset.Dataset(dataset_dictionary)
        print "Profiling dataset {}...".format(d.name)
        characters_unique_list = [0.0] * d.dataframe.shape[1]
        characters_alphabet_list = [0.0] * d.dataframe.shape[1]
        characters_numeric_list = [0.0] * d.dataframe.shape[1]
        characters_punctuation_list = [0.0] * d.dataframe.shape[1]
        characters_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        words_unique_list = [0.0] * d.dataframe.shape[1]
        words_alphabet_list = [0.0] * d.dataframe.shape[1]
        words_numeric_list = [0.0] * d.dataframe.shape[1]
        words_punctuation_list = [0.0] * d.dataframe.shape[1]
        words_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        words_length_list = [0.0] * d.dataframe.shape[1]
        cells_unique_list = [0.0] * d.dataframe.shape[1]
        cells_alphabet_list = [0.0] * d.dataframe.shape[1]
        cells_numeric_list = [0.0] * d.dataframe.shape[1]
        cells_punctuation_list = [0.0] * d.dataframe.shape[1]
        cells_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        cells_length_list = [0.0] * d.dataframe.shape[1]
        cells_null_list = [0.0] * d.dataframe.shape[1]
        top_keywords_dictionary = {a.lower(): 1.0 for a in d.dataframe.columns}
        stop_words_set = set(nltk.corpus.stopwords.words("english"))
        for column, attribute in enumerate(d.dataframe.columns):
            characters_dictionary = {}
            words_dictionary = {}
            cells_dictionary = {}
            keywords_dictionary = {}
            for cell in d.dataframe[attribute]:
                for character in cell:
                    if character not in characters_dictionary:
                        characters_dictionary[character] = 0
                        characters_unique_list[column] += 1
                    characters_dictionary[character] += 1
                    if re.findall("^[a-zA-Z]$", character):
                        characters_alphabet_list[column] += 1
                    elif re.findall("^[0-9]$", character):
                        characters_numeric_list[column] += 1
                    elif re.findall("^[{}]$".format(string.punctuation), character):
                        characters_punctuation_list[column] += 1
                    else:
                        characters_miscellaneous_list[column] += 1
                for word in nltk.word_tokenize(cell):
                    if word not in words_dictionary:
                        words_dictionary[word] = 0
                        words_unique_list[column] += 1
                    words_dictionary[word] += 1
                    if re.findall("^[a-zA-Z_-]+$", word):
                        words_alphabet_list[column] += 1
                        word = word.lower()
                        if word not in keywords_dictionary:
                            keywords_dictionary[word] = 0
                        keywords_dictionary[word] += 1
                    elif re.findall("^[0-9]+[.,][0-9]+$", word) or re.findall("^[0-9]+$", word):
                        words_numeric_list[column] += 1
                    elif re.findall("^[{}]+$".format(string.punctuation), word):
                        words_punctuation_list[column] += 1
                    else:
                        words_miscellaneous_list[column] += 1
                    words_length_list[column] += len(word)
                if cell not in cells_dictionary:
                    cells_dictionary[cell] = 0
                    cells_unique_list[column] += 1
                cells_dictionary[cell] += 1
                if re.findall("^[a-zA-Z_ -]+$", cell):
                    cells_alphabet_list[column] += 1
                elif re.findall("^[0-9]+[.,][0-9]+$", cell) or re.findall("^[0-9]+$", cell):
                    cells_numeric_list[column] += 1
                elif re.findall("^[{}]+$".format(string.punctuation), cell, re.IGNORECASE):
                    cells_punctuation_list[column] += 1
                else:
                    cells_miscellaneous_list[column] += 1
                cells_length_list[column] += len(cell)
                if cell == "":
                    cells_null_list[column] += 1
            if sum(words_dictionary.values()) > 0:
                words_length_list[column] /= sum(words_dictionary.values())
            sorted_keywords_dictionary = sorted(keywords_dictionary.items(), key=operator.itemgetter(1), reverse=True)
            for keyword, frequency in sorted_keywords_dictionary[:self.KEYWORDS_COUNT_PER_COLUMN]:
                if keyword not in stop_words_set:
                    top_keywords_dictionary[keyword] = float(frequency) / d.dataframe.shape[0]

        def f(columns_value_list):
            return numpy.mean(numpy.array(columns_value_list).astype(numpy.float) / d.dataframe.shape[0])

        def g(columns_value_list):
            return numpy.var(numpy.array(columns_value_list).astype(numpy.float) / d.dataframe.shape[0])

        dataset_profile = {
            "dataset_top_keywords": top_keywords_dictionary,
            "dataset_rules_count": len(self.DATASETS[d.name]["functions"]),
            "dataset_patterns_count": len(self.DATASETS[d.name]["patterns"]),
            "characters_unique_mean": f(characters_unique_list),
            "characters_unique_variance": g(characters_unique_list),
            "characters_alphabet_mean": f(characters_alphabet_list),
            "characters_alphabet_variance": g(characters_alphabet_list),
            "characters_numeric_mean": f(characters_numeric_list),
            "characters_numeric_variance": g(characters_numeric_list),
            "characters_punctuation_mean": f(characters_punctuation_list),
            "characters_punctuation_variance": g(characters_punctuation_list),
            "characters_miscellaneous_mean": f(characters_miscellaneous_list),
            "characters_miscellaneous_variance": g(characters_miscellaneous_list),
            "words_unique_mean": f(words_unique_list),
            "words_unique_variance": g(words_unique_list),
            "words_alphabet_mean": f(words_alphabet_list),
            "words_alphabet_variance": g(words_alphabet_list),
            "words_numeric_mean": f(words_numeric_list),
            "words_numeric_variance": g(words_numeric_list),
            "words_punctuation_mean": f(words_punctuation_list),
            "words_punctuation_variance": g(words_punctuation_list),
            "words_miscellaneous_mean": f(words_miscellaneous_list),
            "words_miscellaneous_variance": g(words_miscellaneous_list),
            "words_length_mean": f(words_length_list),
            "words_length_variance": g(words_length_list),
            "cells_unique_mean": f(cells_unique_list),
            "cells_unique_variance": g(cells_unique_list),
            "cells_alphabet_mean": f(cells_alphabet_list),
            "cells_alphabet_variance": g(cells_alphabet_list),
            "cells_numeric_mean": f(cells_numeric_list),
            "cells_numeric_variance": g(cells_numeric_list),
            "cells_punctuation_mean": f(cells_punctuation_list),
            "cells_punctuation_variance": g(cells_punctuation_list),
            "cells_miscellaneous_mean": f(cells_miscellaneous_list),
            "cells_miscellaneous_variance": g(cells_miscellaneous_list),
            "cells_length_mean": f(cells_length_list),
            "cells_length_variance": g(cells_length_list),
            "cells_null_mean": f(cells_null_list),
            "cells_null_variance": g(cells_null_list)
        }
        pickle.dump(dataset_profile, open(os.path.join(self.RESULTS_FOLDER, d.name, "dataset_profile.dictionary"), "wb"))

    def strategy_profiler(self, dataset_dictionary):
        """
        This method profiles the output of error detection strategies.
        """
        d = dataset.Dataset(dataset_dictionary)
        strategies_profile = {}
        for tn in self.ERROR_DETECTION_STRATEGIES:
            print "Running error detection strategy {}...".format(tn)
            if tn.startswith("histogram") or tn.startswith("gaussian") or tn.startswith("mixture") or \
                    tn.startswith("partitionedhistogram"):
                td = {"name": "dboost", "configuration": tn.split("_")}
            elif tn == "fd_checker":
                td = {"name": "fd_checker", "configuration": self.DATASETS[d.name]["functions"]}
            elif tn == "regex":
                td = {"name": "regex", "configuration": self.DATASETS[d.name]["patterns"]}
            elif tn == "katara":
                td = {"name": "katara", "configuration": ["tools/KATARA/dominSpecific"]}
            else:
                sys.stderr.write("I do not know the data cleaning tool!\n")
                return
            start_time = time.time()
            if not td["configuration"]:
                detected_cells_dictionary = {}
            else:
                t = data_cleaning_tool.DataCleaningTool(td)
                detected_cells_dictionary = t.run(d)
            strategies_profile["{}|output".format(tn)] = detected_cells_dictionary.keys()
            strategies_profile["{}|runtime".format(tn)] = time.time() - start_time
        pickle.dump(strategies_profile, open(os.path.join(self.RESULTS_FOLDER, d.name, "strategies_profile.dictionary"), "wb"))

    def evaluation_profiler(self, dataset_dictionary):
        """
        This method evaluates the error detection strategies on dataset.
        """
        d = dataset.Dataset(dataset_dictionary)
        actual_errors = d.get_actual_errors_dictionary()
        sp = pickle.load(open(os.path.join(self.RESULTS_FOLDER, d.name, "strategies_profile.dictionary"), "rb"))
        evaluation_profile = {}
        for tool_name in self.ERROR_DETECTION_STRATEGIES:
            print "Evaluating error detection strategy {}...".format(tool_name)
            tp = 0.0
            for cell in sp["{}|output".format(tool_name)]:
                if cell in actual_errors:
                    tp += 1.0
            precision = 0.0 if len(sp["{}|output".format(tool_name)]) == 0 else tp / len(sp["{}|output".format(tool_name)])
            recall = 0.0 if len(actual_errors) == 0 else tp / len(actual_errors)
            f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
            evaluation_profile[tool_name] = [precision, recall, f1]
        pickle.dump(evaluation_profile, open(os.path.join(self.RESULTS_FOLDER, d.name, "evaluation_profile.dictionary"), "wb"))

    def regression(self):
        """
        This method extracts feature vectors, train, and test regression models.
        """
        # --------------------Loading Dirtiness Profiles--------------------
        dp = {}
        sp = {}
        ep = {}
        keywords_dictionary = {}
        for dataset_name in self.DATASETS:
            print "Loading dirtiness profile for dataset {}...".format(dataset_name)
            dp[dataset_name] = pickle.load(open(os.path.join(self.RESULTS_FOLDER, dataset_name, "dataset_profile.dictionary"), "rb"))
            for keyword in dp[dataset_name]["dataset_top_keywords"]:
                if keyword not in keywords_dictionary:
                    keywords_dictionary[keyword] = 1
            sp[dataset_name] = pickle.load(open(os.path.join(self.RESULTS_FOLDER, dataset_name, "strategies_profile.dictionary"), "rb"))
            ep[dataset_name] = pickle.load(open(os.path.join(self.RESULTS_FOLDER, dataset_name, "evaluation_profile.dictionary"), "rb"))
        mean_squared_error_list = []
        precision_at_k_list = []
        for r in range(self.RUN_COUNT):
            print "Run {}...".format(r)
            x = []
            y = []
            for dataset_name in self.DATASETS:
                print "Extracting feature vector for dataset {}...".format(dataset_name)
                feature_vector = []
                # --------------------Extracting Content Features--------------------
                if "C" in self.SELECTED_FEATURE_GROUPS:
                    for keyword in keywords_dictionary.keys():
                        a = 0.0
                        if keyword in dp[dataset_name]["dataset_top_keywords"]:
                            a = dp[dataset_name]["dataset_top_keywords"][keyword]
                        feature_vector.append(a)
                # --------------------Extracting Structure Features--------------------
                if "S" in self.SELECTED_FEATURE_GROUPS:
                    for f, v in dp[dataset_name].items():
                        if f not in ["dataset_top_keywords"]:
                            feature_vector.append(v)
                # --------------------Extracting Dirtiness Features--------------------
                if "D" in self.SELECTED_FEATURE_GROUPS:
                    d = dataset.Dataset(self.DATASETS[dataset_name])
                    for tool_name in self.ERROR_DETECTION_STRATEGIES:
                        ndc = float(len(sp[dataset_name]["{}|output".format(tool_name)]))
                        ndc /= (d.dataframe.shape[0] * d.dataframe.shape[1])
                        feature_vector.append(ndc)
                    for tool_name_1 in self.ERROR_DETECTION_STRATEGIES:
                        a = set([tuple(z) for z in sp[dataset_name]["{}|output".format(tool_name_1)]])
                        for tool_name_2 in self.ERROR_DETECTION_STRATEGIES:
                            if tool_name_1 < tool_name_2:
                                b = set([tuple(z) for z in sp[dataset_name]["{}|output".format(tool_name_2)]])
                                no = float(len(list(a & b)))
                                de = len(list(a | b))
                                v = 0.0 if de == 0 else no / de
                                feature_vector.append(v)
                    if self.SAMPLING_RATE > 0.0:
                        d = dataset.Dataset(self.DATASETS[dataset_name])
                        actual_errors = d.get_actual_errors_dictionary()
                        selected_rows_dictionary = {r: 1 for r in random.sample(range(0, d.dataframe.shape[0]),
                                                                                int(self.SAMPLING_RATE * d.dataframe.shape[0]))}
                        for tool_name in self.ERROR_DETECTION_STRATEGIES:
                            tp = 0.0
                            total = 0.0
                            for cell in sp[d.name]["{}|output".format(tool_name)]:
                                if cell[0] in selected_rows_dictionary:
                                    total += 1.0
                                    if cell in actual_errors:
                                        tp += 1.0
                            precision = 0.0 if total == 0 else tp / total
                            feature_vector.append(precision)
                x.append(feature_vector)
                # --------------------Extracting Target Vector--------------------
                target_vector = []
                for tool_name in self.ERROR_DETECTION_STRATEGIES:
                    target_vector.append(ep[dataset_name][tool_name][2])
                y.append(target_vector)
            # --------------------Removing Useless Features--------------------
            x_matrix = numpy.array(x, dtype=numpy.float)
            non_identical_columns = numpy.any(x_matrix != x_matrix[0, :], axis=0)
            x = x_matrix[:, non_identical_columns].tolist()
            selected_rows = random.sample(range(0, len(self.DATASETS)), self.DIRTINESS_PROFILES_COUNT)
            x = numpy.array(x, dtype=numpy.float)[selected_rows, :].tolist()
            y = numpy.array(y, dtype=numpy.float)[selected_rows, :].tolist()
            # --------------------Training and Testing Regression Models--------------------
            print "Training regression models..."
            estimated_f1_list_list = []
            for i in range(len(x)):
                x_train = x[:i] + x[i + 1:]
                y_train = y[:i] + y[i + 1:]
                x_test = [x[i]]
                y_test = [y[i]]
                estimated_f1_list = []
                for j in range(len(self.ERROR_DETECTION_STRATEGIES)):
                    model = ""
                    if self.REGRESSION_MODEL == "LR":
                        model = sklearn.linear_model.LinearRegression(normalize=True)
                    elif self.REGRESSION_MODEL == "KNR":
                        model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
                    elif self.REGRESSION_MODEL == "RR":
                        model = sklearn.linear_model.Ridge(alpha=0.04, normalize=True)
                    elif self.REGRESSION_MODEL == "BRR":
                        model = sklearn.linear_model.BayesianRidge(normalize=False)
                    elif self.REGRESSION_MODEL == "DTR":
                        model = sklearn.tree.DecisionTreeRegressor(criterion="mae")
                    elif self.REGRESSION_MODEL == "SVR":
                        model = sklearn.svm.SVR(kernel="rbf")
                    elif self.REGRESSION_MODEL == "GBR":
                        model = sklearn.ensemble.GradientBoostingRegressor(loss="lad", n_estimators=100)
                    y_temp = [z[j] for z in y_train]
                    model = model.fit(x_train, y_temp)
                    y_estimated = model.predict(x_test)[0]
                    if math.isnan(y_estimated) or y_estimated < 0.0:
                        y_estimated = 0.0
                    if y_estimated > 1.0:
                        y_estimated = 1.0
                    estimated_f1_list.append(y_estimated)
                estimated_f1_list_list.append(estimated_f1_list)
            actual_f1_list_list = y
            mse = numpy.mean(numpy.power(numpy.array(actual_f1_list_list) - numpy.array(estimated_f1_list_list), 2))
            mean_squared_error_list.append(mse)
            pak = numpy.array([0.0] * self.PRECISION_AT_K)
            for i in range(len(x)):
                estimated_f1 = [round(z, 2) for z in estimated_f1_list_list[i]]
                actual_f1 = [round(z, 2) for z in actual_f1_list_list[i]]
                estimated_index_list = []
                actual_index_list = []
                precision_at_list = [0.0] * self.PRECISION_AT_K
                for k in range(self.PRECISION_AT_K):
                    estimated_max_f1 = max(estimated_f1)
                    actual_max_f1 = max(actual_f1)
                    estimated_index_list += [j for j, z in enumerate(estimated_f1) if z == estimated_max_f1]
                    actual_index_list += [j for j, z in enumerate(actual_f1) if z == actual_max_f1]
                    precision_at_list[k] += float(
                        len(list(set(estimated_index_list) & set(actual_index_list)))) / len(
                        list(set(estimated_index_list)))
                    estimated_f1 = [z if z != estimated_max_f1 else -1000.0 for z in estimated_f1]
                    actual_f1 = [z if z != actual_max_f1 else -1000.0 for z in actual_f1]
                pak += numpy.array(precision_at_list) / len(x)
            precision_at_k_list.append(pak)
        print "REDS,", self.SAMPLING_RATE
        print "MSE = {:.3f} +- {:.3f}".format(numpy.mean(mean_squared_error_list), numpy.std(mean_squared_error_list))
        for k in range(self.PRECISION_AT_K):
            temp_array = numpy.array(precision_at_k_list)[:, k]
            print "P@{} = {:.3f} +- {:.3f}".format(k + 1, numpy.mean(temp_array), numpy.std(temp_array))

    def baseline(self):
        """
        The method provide maximum entropy-based baseline.
        """
        actual_f1_list_list = []
        sp = {}
        for dataset_name in self.DATASETS:
            ep = pickle.load(open(os.path.join(self.RESULTS_FOLDER, dataset_name, "evaluation_profile.dictionary"), "rb"))
            actual_f1_list_list.append([ep[sn][2] for sn in self.ERROR_DETECTION_STRATEGIES])
            sp[dataset_name] = pickle.load(open(os.path.join(self.RESULTS_FOLDER, dataset_name, "strategies_profile.dictionary"), "rb"))
        mean_squared_error_list = []
        precision_at_k_list = []
        for r in range(self.RUN_COUNT):
            print "Run {}...".format(r)
            estimated_f1_list_list = []
            for dataset_name in self.DATASETS:
                print "Evaluating error detection strategies on a sample of dataset {}...".format(dataset_name)
                estimated_precision_list = []
                d = dataset.Dataset(self.DATASETS[dataset_name])
                actual_errors = d.get_actual_errors_dictionary()
                selected_rows_dictionary = {r: 1 for r in random.sample(range(0, d.dataframe.shape[0]),
                                                                        int(self.SAMPLING_RATE * d.dataframe.shape[0]))}
                for tool_name in self.ERROR_DETECTION_STRATEGIES:
                    tp = 0.0
                    total = 0.0
                    for cell in sp[dataset_name]["{}|output".format(tool_name)]:
                        if cell[0] in selected_rows_dictionary:
                            total += 1.0
                            if cell in actual_errors:
                                tp += 1.0
                    precision = 0.0 if total == 0 else tp / total
                    estimated_precision_list.append(precision)
                estimated_f1_list_list.append(estimated_precision_list)
            mse = numpy.mean(numpy.power(numpy.array(actual_f1_list_list) - numpy.array(estimated_f1_list_list), 2))
            mean_squared_error_list.append(mse)
            pak = numpy.array([0.0] * self.PRECISION_AT_K)
            for i in range(len(self.DATASETS)):
                estimated_f1 = [round(z, 2) for z in estimated_f1_list_list[i]]
                actual_f1 = [round(z, 2) for z in actual_f1_list_list[i]]
                estimated_index_list = []
                actual_index_list = []
                precision_at_list = [0.0] * self.PRECISION_AT_K
                for k in range(self.PRECISION_AT_K):
                    estimated_max_f1 = max(estimated_f1)
                    actual_max_f1 = max(actual_f1)
                    estimated_index_list += [j for j, z in enumerate(estimated_f1) if z == estimated_max_f1]
                    actual_index_list += [j for j, z in enumerate(actual_f1) if z == actual_max_f1]
                    precision_at_list[k] += float(
                        len(list(set(estimated_index_list) & set(actual_index_list)))) / len(
                        list(set(estimated_index_list)))
                    estimated_f1 = [z if z != estimated_max_f1 else -1000.0 for z in estimated_f1]
                    actual_f1 = [z if z != actual_max_f1 else -1000.0 for z in actual_f1]
                pak += numpy.array(precision_at_list) / len(self.DATASETS)
            precision_at_k_list.append(pak)
        print "Baseline,", self.SAMPLING_RATE
        print "MSE = {:.3f} +- {:.3f}".format(numpy.mean(mean_squared_error_list), numpy.std(mean_squared_error_list))
        for k in range(self.PRECISION_AT_K):
            temp_array = numpy.array(precision_at_k_list)[:, k]
            print "P@{} = {:.3f} +- {:.3f}".format(k + 1, numpy.mean(temp_array), numpy.std(temp_array))
########################################


########################################
if __name__ == "__main__":
    # ----------------------------------------
    application = REDS()
    application.DATASETS_FOLDER = "datasets"
    application.RESULTS_FOLDER = "results"
    application.DATASETS = {
        "hospital": {
           "name": "hospital",
           "path": os.path.join(application.DATASETS_FOLDER, "hospital", "dirty.csv"),
           "clean_path": os.path.join(application.DATASETS_FOLDER, "hospital", "clean.csv"),
           "functions": [["zip", "city"], ["zip", "state"], ["zip", "county"], ["zip", "type"],
                         ["zip", "emergency_service"],
                         ["state_average", "type"], ["state_average", "state"], ["state_average", "condition"],
                         ["state_average", "measure_code"], ["state_average", "measure_name"],
                         ["state", "type"],
                         ["score", "type"],
                         ["sample", "type"],
                         ["provider_number", "name"], ["provider_number", "address_1"], ["provider_number", "city"],
                         ["provider_number", "state"], ["provider_number", "zip"],
                         ["provider_number", "county"], ["provider_number", "phone"], ["provider_number", "type"],
                         ["provider_number", "owner"], ["provider_number", "emergency_service"],
                         ["phone", "provider_number"], ["phone", "name"], ["phone", "address_1"], ["phone", "city"],
                         ["phone", "state"], ["phone", "county"], ["phone", "zip"], ["phone", "type"],
                         ["phone", "owner"], ["phone", "emergency_service"],
                         ["measure_name", "type"], ["measure_name", "condition"], ["measure_name", "measure_code"],
                         ["measure_code", "type"], ["measure_code", "condition"], ["measure_code", "measure_name"],
                         ["owner", "type"],
                         ["name", "provider_number"], ["name", "address_1"], ["name", "city"],
                         ["name", "state"], ["name", "zip"],
                         ["name", "county"], ["name", "phone"], ["name", "type"],
                         ["name", "owner"], ["name", "emergency_service"],
                         ["emergency_service", "type"],
                         ["county", "state"], ["county", "type"], ["county", "emergency_service"],
                         ["condition", "type"],
                         ["city", "state"], ["city", "county"], ["city", "type"], ["city", "emergency_service"],
                         ["address_1", "provider_number"], ["address_1", "name"], ["address_1", "city"],
                         ["address_1", "state"], ["address_1", "zip"], ["address_1", "county"], ["address_1", "phone"],
                         ["address_1", "type"], ["address_1", "owner"], ["address_1", "emergency_service"]],
           "patterns": [["provider_number", "^[\d]+$", "ONM"], ["zip", "^[\d]{5}$", "ONM"],
                        ["state", "^[a-z]{2}$", "ONM"], ["phone", "^[\d]+$", "ONM"],
                        ["emergency_service", "^(yes|no)$", "ONM"]],
        },
        "flights": {
            "name": "flights",
            "path": os.path.join(application.DATASETS_FOLDER, "flights", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "flights", "clean.csv"),
            "functions": [["flight", "act_dep_time"], ["flight", "sched_arr_time"], ["flight", "act_arr_time"],
                          ["flight", "sched_dep_time"]],
            "patterns": [["sched_dep_time", "^[\d]{1,2}[:][\d]{1,2}[ ][ap].m.$", "ONM"],
                         ["act_dep_time", "^[\d]{1,2}[:][\d]{1,2}[ ][ap].m.$", "ONM"],
                         ["sched_arr_time", "^[\d]{1,2}[:][\d]{1,2}[ ][ap].m.$", "ONM"],
                         ["act_arr_time", "^[\d]{1,2}[:][\d]{1,2}[ ][ap].m.$", "ONM"]],
        },
        "address": {
            "name": "address",
            "path": os.path.join(application.DATASETS_FOLDER, "address", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "address", "clean.csv"),
            "functions": [["address", "state"], ["zip", "state"]],  # ["zip", "city"], ["city", "state"]
            "patterns": [["state", "^[A-Z]{2}$", "ONM"], ["zip", "^[\d]+$", "ONM"], ["ssn", "^[\d]*$", "ONM"]],
        },
        "beers": {
            "name": "beers",
            "path": os.path.join(application.DATASETS_FOLDER, "beers", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "beers", "clean.csv"),
            "functions": [["brewery_id", "brewery_name"], ["brewery_id", "city"], ["brewery_id", "state"]],
            "patterns": [["state", "^[A-Z]{2}$", "ONM"], ["brewery_id", "^[\d]+$", "ONM"]],
        },
        "rayyan": {
            "name": "rayyan",
            "path": os.path.join(application.DATASETS_FOLDER, "rayyan", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "rayyan", "clean.csv"),
            "functions": [],   # ["jounral_abbreviation", "journal_title"], ["jounral_abbreviation", "journal_issn"], ["journal_issn", "journal_title"]
            "patterns": [["article_jvolumn", "^[\d]+$|^[-][1]$", "ONM"], ["article_jissue", "^[\d]+$|^[-][1]$", "ONM"],
                         ["article_jcreated_at", "^[\d]+[/][\d]+[/][\d]+$|^$", "ONM"]],
                        # ["article_jvolumn", "^$", "OM"], ["article_jissue", "^$", "OM"],
                        # ["article_jcreated_at", "^[\d]+[/][\d]+[/][\d]+$|^$", "OM"], ["journal_issn", "^$", "OM"],
                        # ["journal_title", "^$", "OM"], ["article_language", "^$", "OM"], ["article_title", "^$", "OM"],
                        # ["jounral_abbreviation", "^$", "OM"], ["article_pagination", "^$", "OM"],
                        # ["author_list", "^$", "OM"]
        },
        "movies_1": {
            "name": "movies_1",
            "path": os.path.join(application.DATASETS_FOLDER, "movies_1", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "movies_1", "clean.csv"),
            "functions": [],
            "patterns": [["id", "^tt[\d]+$", "ONM"], ["year", "^[\d]{4}$", "ONM"], ["rating_value", "^[\d.]*$", "ONM"],
                         ["rating_count", "^[\d]*$", "ONM"], ["duration", "^([\d]+[ ]min)*$", "ONM"]],
        },
        "merck": {
            "name": "merck",
            "path": os.path.join(application.DATASETS_FOLDER, "merck", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "merck", "clean.csv"),
            "functions": [],
            "patterns": [["support_level", "^$", "OM"], ["app_status", "^$", "OM"], ["curr_status", "^$", "OM"],
                         ["tower", "^$", "OM"], ["end_users", "^$", "OM"], ["account_manager", "^$", "OM"],
                         ["decomm_dt", "^$", "OM"], ["decomm_start", "^$", "OM"], ["decomm_end", "^$", "OM"],
                         ["end_users", "^(0)$", "OM"],
                         ["retirement", "^(2010|2011|2012|2013|2014|2015|2016|2017|2018)$", "ONM"],
                         ["emp_dta", "^(n|N|y|Y|n/a|N/A|n/A|N/a)$", "ONM"],
                         ["retire_plan", "^(true|True|TRUE|false|False|FALSE|n/a|N/A|n/A|N/a)$", "ONM"],
                         ["bus_import", "^(important|n/a|IP Strategy)$", "OM"],
                         ["division", "^(Merck Research Laboratories|Merck Consumer Health Care)$", "OM"]],
        },
        "restaurants_4": {
            "name": "restaurants_4",
            "path": os.path.join(application.DATASETS_FOLDER, "restaurants_4", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "restaurants_4", "clean.csv"),
            "functions": [["zip_code", "state"]],
            "patterns": [["state", "^[A-Z]{2}$|^$", "ONM"], ["zip_code", "^[\d]*$", "ONM"],
                         ["phone", "^[(][\d]{3}[)][ ][\d]{3}[-][\d]{4}$|^$", "ONM"]],
        },
        "salaries": {
            "name": "salaries",
            "path": os.path.join(application.DATASETS_FOLDER, "salaries", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "salaries", "clean.csv"),
            "functions": [],
            "patterns": [["year", "^[\d]{4}$", "ONM"], ["base_pay", "^([\d]+)([.][\d]+)?$", "ONM"],
                         ["overtime_pay", "^([\d]+)([.][\d]+)?$", "ONM"], ["other_pay", "^([\d]+)([.][\d]+)?$", "ONM"],
                         ["benefits", "^([\d]+)([.][\d]+)?$|^$", "ONM"], ["total_pay", "^([\d]+)([.][\d]+)?$", "ONM"],
                         ["total_pay_benefits", "^([\d]+)([.][\d]+)?$", "ONM"]],
        },
        "soccer": {
            "name": "soccer",
            "path": os.path.join(application.DATASETS_FOLDER, "soccer", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "soccer", "clean.csv"),
            "functions": [["name", "surname"], ["name", "birth_year"], ["name", "birth_place"]],
            "patterns": [["birth_year", "^[\d]{4}$", "ONM"], ["season", "^[\d]{4}$", "ONM"]],
        },
        "tax": {
            "name": "tax",
            "path": os.path.join(application.DATASETS_FOLDER, "tax", "dirty.csv"),
            "clean_path": os.path.join(application.DATASETS_FOLDER, "tax", "clean.csv"),
            "functions": [["area_code", "state"], ["f_name", "gender"], ["zip", "city"], ["zip", "state"]],
            "patterns": [["gender", "^(F|M)$", "ONM"], ["area_code", "^[\d]{3}$", "ONM"],
                         ["phone", "^[\d]{3}[-][\d]{4}$", "ONM"], ["state", "^[A-Z]{2}$", "ONM"],
                         ["zip", "^[\d]+$", "ONM"], ["marital_status", "^(M|S)$", "ONM"],
                         ["has_child", "^(N|Y)$", "ONM"], ["salary", "^[\d]+$", "ONM"],
                         ["rate", "^([\d])([.][\d]+)?$", "ONM"], ["single_exemp", "^[\d]+$", "ONM"],
                         ["married_exemp", "^[\d]+$", "ONM"], ["child_exemp", "^[\d]+$", "ONM"]],
        }
        # ----------------------------------------------------
        # "toy": {
        #     "name": "toy",
        #     "path": os.path.join(application.DATASETS_FOLDER, "toy", "dirty.csv"),
        #     "clean_path": os.path.join(application.DATASETS_FOLDER, "toy", "clean.csv"),
        #     "functions": [["city", "country"]],
        #     "patterns": [["age", "^[\d]+$", "ONM"]],
        # },
    }
    application.ERROR_DETECTION_STRATEGIES = ["histogram_0.8_0.1", "histogram_0.8_0.2", "histogram_0.9_0.2",
                                              "gaussian_1.0", "gaussian_1.5", "gaussian_2.0",
                                              "mixture_2_0.005", "mixture_2_0.01", "mixture_2_0.05",
                                              "partitionedhistogram_5_0.8_0.1", "partitionedhistogram_10_0.8_0.1",
                                              "partitionedhistogram_15_0.8_0.1",
                                              "fd_checker", "regex", "katara"]
    application.RUN_COUNT = 10
    application.PRECISION_AT_K = 3
    application.KEYWORDS_COUNT_PER_COLUMN = 10
    application.SAMPLING_RATE = 0.01  # [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    application.SELECTED_FEATURE_GROUPS = ["C", "S", "D"]  # ["C", "S", "D"]
    application.REGRESSION_MODEL = "GBR"  # ["LR", "KNR", "RR", "BRR", "DTR", "SVR", "GBR"]
    application.DIRTINESS_PROFILES_COUNT = len(application.DATASETS)  # [2,..., len(application.DATASETS)]
    # ----------------------------------------
    for dd in application.DATASETS.values():
        print "===================== Dataset: {} =====================".format(dd["name"])
        if not os.path.exists(os.path.join(application.RESULTS_FOLDER, dd["name"])):
            os.mkdir(os.path.join(application.RESULTS_FOLDER, dd["name"]))
        # application.dataset_profiler(dd)
        # application.strategy_profiler(dd)
        # application.evaluation_profiler(dd)
    # ----------------------------------------
    # application.regression()
    # ----------------------------------------
    # application.baseline()
    # ----------------------------------------
########################################
