from enum import Enum
from util.data import Yatsdo
from util.data import StorageIO
import pandas as pd
import numpy as np
import math
import json
import pickle
import os

from copy import deepcopy
from multiprocessing import Process
from tsfresh.transformers.feature_selector import FeatureSelector
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
import threading

'''
Authors: Ted Yeung
Date: Nov 2020
'''

class WindowPoint(Enum):
    beginning = 0,
    middle = 1,
    end = 3


class YatsdoML(Yatsdo):
    # Yet another time series data object machine learning edition
    #
    # This object take a 2D np array and creates
    # functions for simple data manipulation
    # i.e. resampling data
    # Assume get_first column is time
    #
    # Variables added:
    # 1) window_size
    # 2) offset
    #
    # functions added:
    # 1) chunk - Chunks data in to specific windows size
    # 2)
    def __init__(self, data, col_names=[], window_size=100, offset=100):
        """
        :param data: nd.array or pd.DataFrame
        :param window_size: the size of chunk
        :param offset:  the offset between point of interest
        """
        if isinstance(data, str):
            b = StorageIO.load(data)
            if len(col_names) > 0:
                data = b.data[col_names]
            else:
                data = b.data
        super().__init__(data, col_names)
        self.window_size = window_size
        self.offset = offset
        self.chunked_list: list[pd.DataFrame] = None
        self.time_points = []
        self.last_created_chunks = None
        self.chunked_in_table_form = None

    @property
    def data_ml(self):
        return self.data

    @property
    def time(self):
        """
        Assumes Time column is the first column
        :return: time
        """
        return self.data[:, 0]

    @data_ml.setter
    def data_ml(self, x):
        self.data = x
        self.update()

    @property
    def chunks_as_numpy(self):
        return self.chunked_in_table_form

    @property
    def chunks_as_dataframe(self):
        return pd.DataFrame(data=self.chunked_in_table_form, columns=self.column_labels)

    @property
    def column_names(self):
        return self.column_labels

    @column_names.setter
    def column_names(self, x: list):
        self.column_labels = x

    def basic_chunk(self, window=None, offset=None, place=1):
        """
        This is a basic chunking method where it takes a window size and an offset
        to create the chunk
        :param place: default 1, integer indicating target time point in the frame; 0 = beginning, 1 = middle, 2 = end
        :param window: an integer of the window size i.e. number of frames
        :param offset: an integer of the offset i.e. number of frames. This overrides the previous value.
        :return: None
        """
        if window is not None:
            self.window_size = window
        if offset is not None:
            self.offset = offset
        if self.data is not None:
            chunks = []
            lots = math.ceil((self.data.shape[0]-self.window_size) / self.offset)
            end = int(lots)*self.offset
            idx = 1
            print("Basic Chunk")
            table = np.zeros([lots*self.window_size, self.data.shape[1]+1])
            for i in range(0, end, self.offset):
                df = self.data[i: i + self.window_size, :]
                new_df = np.zeros([df.shape[0], df.shape[1]+1])
                new_df[:, 0] = idx*np.ones([1, df.shape[0]])
                new_df[:, 1:] = deepcopy(df)
                table[(idx-1)*self.window_size: (idx-1)*self.window_size + self.window_size, :] = new_df
                chunks.append(new_df)
                idx += 1
                if place == 1:
                    self.time_points.append(new_df[int(df.shape[0]/2), 1])
            self.chunked_list = chunks
            self.chunked_in_table_form = table

    def chunk_to_numpy(self, data_split: float = 1, get_first=True):
        """
        This method split the data for analysis and converts the list of chunks to a numpy array
        :param data_split: a float of the split
        :param get_first: boolean to indicate if you want the first or second part of the split
        :return: np.ndarray
        """
        if get_first:
            end = int(np.round(data_split * len(self.chunked_list), 0))
            ret = np.zeros([self.window_size * int(np.round(end, 0)), self.data.shape[1] + 1])
            for i in range(0, end-1):
                ds = self.chunked_list[i]
                ds[:, 0] = ds[:, 0]*(i+1)
                ret[i*self.window_size:(i+1)*self.window_size, :] = ds
        else:
            end = int(np.round(data_split * len(self.chunked_list), 0))
            siz = int(np.round((1-data_split) * len(self.chunked_list), 0))
            ret = np.zeros([self.window_size * int(np.round(siz, 0)), self.data.shape[1] + 1])
            for i in range(end, len(self.chunked_list)):
                j = i-end
                ds = self.chunked_list[i]
                ds[:, 0] = ds[:, 0] * (j + 1)
                ret[j * self.window_size:(j + 1) * self.window_size, :] = ds

        return ret

    def chunk_to_pandas(self, lb=None):
        """
        Convert the chunk to a pandas data frame
        :param lb: labels for the columns
        :return: pd.DataFrame
        """
        if self.last_created_chunks is not None:
            d2f = self.last_created_chunks
        else:
            d2f = self.chunked_list
        if len(self.column_labels) == self.last_created_chunks.shape[1]:
            col = self.column_labels
        elif lb is not None and len(lb) == self.last_created_chunks.shape[1]:
            col = lb
        else:
            return None
        return pd.DataFrame(data=d2f, columns=col)

    def export_chunks(self, output_folder, filename, table_id):
        path1 = output_folder+filename+".h5"
        self.chunks_as_dataframe.to_hdf(path1, key=table_id, mode='w')
        chunky_data = {
            'id': table_id,
            "offset": self.offset,
            "window": self.window_size,
            "timepoints": self.time_points
        }
        path2 = output_folder + filename + ".json"
        with open(path2, 'w') as outfile:
            json.dump(chunky_data, outfile, sort_keys=True, indent=4)

    def get_samples(self, time_points, as_pandas=False):
        data = super().get_samples(time_points)
        if as_pandas:
            return pd.DataFrame(data=data, columns=self.column_labels)
        return data


class MLKeys:
    """
    This is a list of common tags
    """
    dir_x = 0
    dir_y = 1
    x_file = 2
    y_file = 3
    y_labels = 4
    x_labels = 5
    CFCParameters = 6
    MFCParameters = 7
    ModelName = 8
    AdditionalInfo = 9
    training = 10
    validation = 11
    train_and_validate = 12


class MLOperations:

    @staticmethod
    def extract_features_from_x(x, features=None, fc_parameters=MLKeys.CFCParameters):
        """
        The function uses tsfresh to extracts from X

        :param x: M x N DataFrame
        :param features: Dictionary of features
        :param fc_parameters: Set fc_parameters ComprehensiveFCParameters or MinimalFCParameters
        :return: extracted features from x
        """
        if features is None:
            if fc_parameters is MLKeys.CFCParameters:
                efx = extract_features(x,
                                       column_id="id",
                                       column_sort="time",
                                       default_fc_parameters=ComprehensiveFCParameters(),
                                       impute_function=impute)
            else:
                efx = extract_features(x,
                                       column_id="id",
                                       column_sort="time",
                                       default_fc_parameters=MinimalFCParameters(),
                                       impute_function=impute)
        else:
            efx = extract_features(x,
                                   column_id="id",
                                   column_sort="time",
                                   kind_to_fc_parameters=features,
                                   impute_function=impute)
        param = {"fc_parameters": fc_parameters,
                 "column_id": 'id',
                 "column_sort": 'time',
                 "features": [cl for cl in efx.columns]}
        return efx, param

    @staticmethod
    def train_model(efx, y):
        """
        Selects feature based on the fit of features to y

        > Using the FeatureSelector select the statistical significant features.
        The check is done by testing the hypothesis

            :math:`H_0` = the Feature is not relevant and can not be added`

                against

            :math:`H_1` = the Feature is relevant and should be kept
        > RandomForestRegressor fit to estimate the importance of the each feature

        :param efx: features from x
        :param y: target (cannot contain time)
        :return: A dictionary of feature importance and the regressor
        """
        selector = FeatureSelector()
        y.index = efx.index  # this is the line in which the issue with the length of y_pick is solved
        fc_selected = selector.fit_transform(efx, y)
        reg = RandomForestRegressor()
        reg.fit(fc_selected, y)
        fi = pd.Series(reg.feature_importances_, fc_selected.columns)
        print(reg.score(fc_selected, y))
        feat = [col for col in fc_selected.columns]
        return {"model": reg, "features": {"list": feat, "dict": from_columns(feat)},
                "fc_selected": fc_selected, "feature_importance": fi}

    @staticmethod
    def load_json(file):
        with open(file, 'r') as infile:
            data = json.load(infile)
        return data

    @staticmethod
    def sort_features_all(feature_importance):
        most_important = feature_importance.sort_values(ascending=False)
        feature_names = from_columns(most_important.index)
        return most_important, feature_names

    @staticmethod
    def export_as_h5(data, filepath, table_id):
        data.to_hdf(filepath + ".h5", table_id)

    @staticmethod
    def export_model(model, filename):
        model_file = open(filename, 'wb')
        pickle.dump(model, model_file)
        model_file.close()
        pass

    @staticmethod
    def split(data, ratio):
        boo = False
        if isinstance(data, pd.Series):
            data = data.to_frame()
            boo = True
        x = int(np.floor(data.shape[0] * ratio))
        training = data.iloc[:x, :]
        validation = data.iloc[x:, :]
        if boo:
            training = training.iloc[:, 0]
            validation = validation.iloc[:, 0]
        s = pd.Series([i for i in range(1, validation.shape[0] + 1)])
        validation.index = s.values
        return training, validation

    @staticmethod
    def import_model(filename):
        model_file = open(filename, 'rb')
        model = pickle.load(model_file)
        model_file.close()
        return model

    @staticmethod
    def dump(x, filepath):
        data_m = open(filepath, 'wb')
        pickle.dump(x, data_m)
        data_m.close()

    @staticmethod
    def features_r2(efx, y, jsfile, outfolder):
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        f_store = {}
        # this is the line in which the issue with the length of y_pick is solved
        y.index = efx.index
        count = 1
        for i in efx:
            print(i)
            selector = FeatureSelector()
            feat = efx[i].to_frame()
            if feat is None:
                print(".... error None")
                continue
            if feat.empty:
                print(".... error Empty")
                continue
            if feat.ndim < 1:
                print(".... error ndim < 1")
                continue
            print(feat.shape)
            print(y.shape)
            x = feat
            reg = RandomForestRegressor()
            try:
                reg.fit(x, y)
                sc = reg.score(x, y)
                f_store[i] = {
                    "score": float(sc)
                }

                # dic_dump = {
                #     "name": i,
                #     "model": reg,
                #     "score": sc
                # }
                # x = threading.Thread(target=MLOperations.dump, args=(dic_dump, outfolder+"feature_"+str(count),))
                # x.start()
                count += 1
            except ValueError:
                print("Why?")
            print(".... Done >> current thread count: "+str(threading.active_count()))

        score_out = {}
        for nb in f_store:
            score_out[nb] = f_store[nb]["score"]
        with open(jsfile, 'w') as outfile:
            json.dump(score_out, outfile, sort_keys=True, indent=4)
        return f_store, score_out

    @staticmethod
    def select_top_features_from_x(feature_importance, num_of_feat=100):
        """
        This function will select blindly the top x features from a series of features importance
        :param feature_importance: a panda series of features and their importance
        :param num_of_feat: The x number of top features to select
        :return: a dictionary of top features.
        """
        most_important = feature_importance.sort_values(ascending=False).head(num_of_feat)
        picked_feat = from_columns(most_important.index)
        return picked_feat

    @staticmethod
    def recursive_feature_elimination(x, y, estimator=RandomForestRegressor(), num_of_feature=7, step=1):
        """

        :param x: Extracted features
        :param y: Targets
        :param estimator: a regression or similar estimator to calculate importance i.e. RandomForestRegressor
        :param num_of_feature: the number of features you want
        :param step: how many feature to remove per iteration
        :return: the selector instance - this will change when we know how to use it :)

        The selector also have a predictor built-in
        """
        selector = RFE(estimator, n_features_to_select=num_of_feature, step=step)
        selector = selector.fit(x, y)

        return selector

    @staticmethod
    def rfe_with_cross_validation(x, y, estimator=RandomForestRegressor(), min_of_feature=5, step=1):
        """
        Recursive_feature_elimination_with_cross_validation
        <This method may take a long time to run>
        :param x: Extracted features
        :param y: Targets
        :param estimator: a regression or similar estimator to calculate importance i.e. RandomForestRegressor
        :param min_of_feature: the min features you want
        :param step: how many feature to remove per iteration
        :return: the selector instance - this will change when we know how to use it :)
        """
        selector = RFECV(estimator, min_features_to_select=min_of_feature, step=step)
        selector = selector.fit(x, y)

        return selector

    @staticmethod
    def export_features(ef: pd.DataFrame, param, output_folder, filename, table_id):
        path2 = output_folder + filename + ".json"
        with open(path2, 'w') as outfile:
            json.dump(param, outfile, sort_keys=True, indent=4)
        ef.to_hdf(output_folder+filename+".h5", table_id)
        pass

    @staticmethod
    def export_features_json(ef: dict, filepath):
        with open(filepath, 'w') as outfile:
            json.dump(ef, outfile, sort_keys=True, indent=4)


class Worker(object):
    """
    A template object for Workers
    """
    def __init__(self, thread_id, name, cache):
        self.thread_id = thread_id
        self.name = name
        self.cache = cache

    def run(self):
        return 0


class PDWriteCSV(Worker):
    def __init__(self, thread_id, name, cache):
        super().__init__(self, thread_id, name, cache)

    def run(self):
        data: pd.DataFrame = self.cache['data']
        filename = self.cache['filename']
        write_index = self.cache['write_index']
        try:
            data.to_csv(filename, index=write_index)
        except PermissionError:
            return -1
        return 0


class MLogger:
    def __init__(self):
        """
        This is a don't wait for me to write the file class.
        If you are doing a lot of writing to file this hopefully doesn't block the main program from running
        """
        self.active = []
        pass

    def save_data(self, data: Worker):
        """
        This function start a new writing process
        :param data: a Worker instance ready with the data to write to file
        :return: None
        """
        p = Process(target=MLogger.write, args=(data,))
        p.start()
        self.active.append(p)
        not_alive = []
        for x in self.active:
            if not x.is_alive():
                not_alive.append(x)
        for x in not_alive:
            self.active.remove(x)
        pass

    @staticmethod
    def write(data):
        """
        Starts the data export
        :param data: A worker
        :return: result of run
        """
        return data.run()


if __name__ == '__main__':
    x0 = np.zeros([10000, 4])
    x0[:, 0] = np.array([i for i in range(0, 10000)])
    x1 = np.ones([10000, 4])
    x1[:, 0] = np.array([i for i in range(0, 10000)])
    d = YatsdoML(x0)
    d.data_ml = x1
    d.basic_chunk()
    d.chunk_to_numpy(0.8, get_first=False)
    dx = pd.DataFrame(data=x0)
    dx.to_csv('a.csv', index=False)
    pass
