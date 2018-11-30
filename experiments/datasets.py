# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import pandas

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import csv

import gpflow
float_type = gpflow.settings.float_type


class Dataset(object):
    def __init__(self, name, N, D, type, data_path='/data/'):
        self.data_path = data_path
        self.name, self.N, self.D = name, N, D
        assert type in ['regression', 'classification', 'multiclass']
        self.type = type

    def csv_file_path(self, name):
        return '{}{}.csv'.format(self.data_path, name)

    def read_data(self):
        data = pandas.read_csv(self.csv_file_path(self.name),
                               header=None, delimiter=',').values.astype(float_type)
        return {'X':data[:, :-1], 'Y':data[:, -1, None]}

    def download_data(self):
        NotImplementedError

    def get_data(self, seed=0, split=0, prop=0.9):
        path = self.csv_file_path(self.name)
        if not os.path.isfile(path):
            self.download_data()

        full_data = self.read_data()
        split_data = self.split(full_data, seed, split, prop)
        split_data = self.normalize(split_data, 'X')

        if self.type is 'regression':
            split_data = self.normalize(split_data, 'Y')

        return split_data

    def split(self, full_data, seed, split, prop):
        ind = np.arange(self.N)

        np.random.seed(seed + split)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        X = full_data['X'][ind[:n], :]
        Xs = full_data['X'][ind[n:], :]

        Y = full_data['Y'][ind[:n], :]
        Ys = full_data['Y'][ind[n:], :]

        return {'X': X, 'Xs': Xs, 'Y': Y, 'Ys': Ys}

    def normalize(self, split_data, X_or_Y):
        m = np.average(split_data[X_or_Y], 0)[None, :]
        s = np.std(split_data[X_or_Y + 's'], 0)[None, :] + 1e-6

        split_data[X_or_Y] = ((split_data[X_or_Y] - m) / s).astype(float_type)
        split_data[X_or_Y + 's'] = ((split_data[X_or_Y + 's'] - m) / s).astype(float_type)

        split_data.update({X_or_Y + '_mean': m.flatten()})
        split_data.update({X_or_Y + '_std': s.flatten()})
        return split_data


datasets = []
uci_base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


class Boston(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'boston', 506, 12
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, 'housing/housing.data')

        data = pandas.read_fwf(url, header=None).values
        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Concrete(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'concrete', 1030, 8
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, 'concrete/compressive/Concrete_Data.xls')

        data = pandas.read_excel(url).values
        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Energy(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'energy', 768, 8
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, '00242/ENB2012_data.xlsx')

        data = pandas.read_excel(url).values
        data = data[:, :-1]

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Kin8mn(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'kin8nm', 8192, 8
        self.type = 'regression'

    def download_data(self):

        url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'

        data = pandas.read_csv(url, header=None).values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Motorcycle(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'mcycle', 133, 1
        self.type = 'regression'

    def download_data(self):
        pass


class Naval(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'naval', 11934, 12
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, '00316/UCI%20CBM%20Dataset.zip')

        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pandas.read_fwf('/tmp/UCI CBM Dataset/data.txt', header=None).values
        data = data[:, :-1]

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Power(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'power', 9568, 4
        self.type = 'regression'

    def download_data(self):
        url = '{}{}'.format(uci_base, '00294/CCPP.zip')
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pandas.read_excel('/tmp/CCPP//Folds5x2_pp.xlsx').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Protein(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'protein', 45730, 9
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, '00265/CASP.csv')

        data = pandas.read_csv(url).values

        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Sunspots(Dataset):
    """ http://www.sidc.be/silso/datafiles#total - Monthly mean total sunspot number [1/1749 - now] """
    def __init__(self):
        self.name, self.N, self.D = 'sunspots', 3235, 1
        self.type = 'regression'

    def download_data(self):
        url = 'http://www.sidc.be/silso/INFO/snmtotcsv.php'

        data = pandas.read_csv(url, sep=';').values[:, 2:4]

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class SolarRadiance(Dataset):
    """ ftp://ftp.ncdc.noaa.gov/pub/data/paleo/climate_forcing/solar_variability/lean2000_irradiance.txt """
    def __init__(self):
        self.name, self.N, self.D = 'radiance', 391, 1
        self.type = 'regression'

    def download_data(self):
        pass


class WineRed(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'wine_red', 1599, 11
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, 'wine-quality/winequality-red.csv')

        data = pandas.read_csv(url, delimiter=';').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class WineWhite(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'wine_white', 4898, 12
        self.type = 'regression'

    def download_data(self):

        url = '{}{}'.format(uci_base, 'wine-quality/winequality-white.csv')

        data = pandas.read_csv(url, delimiter=';').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Skin(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'skin', 245057, 3
        self.type = 'classification'

    def download_data(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
        data = pandas.read_table(url, header=None, names=['B', 'G', 'R', 'SKIN']).values.astype(float_type)
        data[:, -1] = data[:, -1] == 2

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)

class Mocap(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'mocap', 1504, 1
        self.type = 'regression'

    def download_data(self):
        pass


class Mocap2(Dataset):
    def __init__(self):
        self.name, self.N, self.D = 'mocap2', 6220, 1
        self.type = 'regression'

    def download_data(self):
        pass

class Datasets(object):
    def __init__(self, data_path='/data/'):
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        datasets = []

        datasets.append(Boston())
        datasets.append(Concrete())
        datasets.append(Energy())
        datasets.append(Kin8mn())
        datasets.append(Motorcycle())
        datasets.append(Naval())
        datasets.append(Power())
        datasets.append(Protein())
        datasets.append(WineRed())
        datasets.append(WineWhite())

        datasets.append(Skin())
        datasets.append(Mocap())
        datasets.append(Mocap2())
        datasets.append(Sunspots())
        datasets.append(SolarRadiance())

        self.all_datasets = {}
        for d in datasets:
            d.data_path = data_path
            self.all_datasets.update({d.name : d})


if __name__ == '__main__':
    print('Updating all datasets...')
    datasets = Datasets('./data/')
    for name, dataset in datasets.all_datasets.items():
        print('Updating %s' % name)
        dataset.download_data()
