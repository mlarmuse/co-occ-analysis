import pandas as pd
import numpy as np
import copy
import warnings

from scipy.stats import binom

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from cycler import cycler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines import CoxPHFitter


class OmicsDataSet():

    @classmethod
    def from_file(cls, path, type='Omics', sep=',', header=0, column_index=0, patient_axis='auto',
                  remove_nas=True, remove_zv=True):
        data_df = pd.read_csv(path, sep=sep, header=header, index_col=column_index)
        return cls(data_df, patient_axis=patient_axis, remove_nas=remove_nas, type=type, remove_zv=remove_zv)

    def __init__(self, dataframe, patient_axis='auto', remove_nas=True, type='Omics', remove_zv=True, verbose=True):

        if str(patient_axis).lower() == 'auto':
            if dataframe.shape[0] > dataframe.shape[1]:
                dataframe = dataframe.transpose()

        elif patient_axis == 1:
            dataframe = dataframe.transpose()

        if remove_nas:
            dataframe = dataframe.dropna(axis=1, how='any', inplace=False)

        if len(dataframe.shape) > 1:
            self.genes = dataframe.columns.values.astype('str')
        else:
            self.genes = ['Query']  # in case the dataframe is a series object, we assume it is a query.

        self.samples = np.array(list(dataframe.index)).astype('str')

        if len(self.samples) > len(set(self.samples)):
            warnings.warn('Duplicated index entries found, removed by averaging over duplicate entries.', UserWarning)
            dataframe = dataframe.groupby(self.samples).mean()
            self.samples = np.array(list(dataframe.index)).astype('str')

        self.df = dataframe
        self.df.index = self.samples
        self.df.columns = self.genes

        if remove_zv:
            self.removeZeroVarianceGenes()

        if verbose:
            print('Number of samples %d, number of genes %d' % (len(self.samples), len(self.genes)))

        self.type = type

        if (type == 'Omics') or (type is None):
            print('Please provide an identifier for the type of data stored in the dataset (EXP, MUT, CNA, PROT).')

    def keepCommonSamples(self, datasets, extra_sample_list=None, inplace=False):

        try:
            _ = (d for d in datasets)

        except TypeError:  # convert to iterable
            datasets = [datasets]

        intersecting_samples = set(self.samples)

        if extra_sample_list is not None:
            intersecting_samples = intersecting_samples.intersection(set(extra_sample_list))

        for dataset in datasets:
            intersecting_samples = intersecting_samples.intersection(set(dataset.samples))

        intersecting_samples = list(intersecting_samples)

        if inplace:
            for dataset in datasets:
                dataset.subsetSamples(intersecting_samples)

            self.subsetSamples(intersecting_samples)

        else:
            datasets = [d.getSampleSubset(intersecting_samples) for d in datasets]
            self_df = self.getSampleSubset(intersecting_samples)
            return [self_df] + datasets

    def keepCommonGenes(self, datasets, extra_gene_list=None, inplace=False):

        try:
            _ = len(datasets)

        except TypeError:  # convert to iterable
            datasets = [datasets]

        intersecting_genes = set(self.genes)

        if extra_gene_list is not None:
            intersecting_genes = intersecting_genes.intersection(set(extra_gene_list))

        for dataset in datasets:
            intersecting_genes = intersecting_genes.intersection(set(dataset.genes))

        intersecting_genes = list(intersecting_genes)
        for dataset in datasets:
            dataset.subsetGenes(intersecting_genes)

        if inplace:
            for dataset in datasets:
                dataset.subsetGenes(intersecting_genes)

            self.subsetGenes(intersecting_genes)

        else:
            datasets = [d.getGeneSubset(intersecting_genes) for d in datasets]
            self_df = self.getGeneSubset(intersecting_genes)
            return [self_df] + datasets

    def batchCorrelations(self, n_batches, dataset2=None, corr_thresh=0.5):

        if dataset2 is None:
            n_cols = self.df.shape[1]
            break_points = np.linspace(0, n_cols, num=n_batches+1, dtype=np.int)

            corrs = []

            for i in range(len(break_points) - 1):
                for j in range(i + 1):
                    correlations_ = np.corrcoef(self.df.iloc[:, break_points[i]:break_points[i+1]],
                                                self.df.iloc[:, break_points[j]:break_points[j+1]],
                                                rowvar=False)
                    print(correlations_.shape)
                    np.fill_diagonal(correlations_, 0)
                    rows, cols = np.where(correlations_ > corr_thresh)
                    corrs_ = pd.DataFrame({'Gene_A': self.genes[rows],
                                           'Gene_B': self.genes[cols],
                                           'Correlation': correlations_[(rows, cols)]})
                    corrs += [corrs_]

        else:
            n_cols1, n_cols2 = self.df.shape[1], dataset2.df.shape[1]
            break_points1 = np.linspace(0, n_cols1, num=n_batches+1, dtype=np.int)
            break_points2 = np.linspace(0, n_cols2, num=n_batches+1, dtype=np.int)

            corrs = []

            for i in range(len(break_points1) - 1):
                for j in range(len(break_points2) - 1):
                    correlations_ = np.corrcoef(self.df.iloc[:, break_points1[i]:break_points2[i+1]],
                                                self.df.iloc[:, break_points2[j]:break_points2[j+1]],
                                                rowvar=False)

                    rows, cols = np.where(correlations_ > corr_thresh)
                    corrs_ = pd.DataFrame({'Gene_A': self.genes[rows],
                                           'Gene_B': self.genes[cols],
                                           'Correlation': correlations_[(rows, cols)]})
                    corrs += [corrs_]

        corrs = pd.concat(corrs, axis=0).sort_values(by='Correlation', ascending=False)

        return corrs

    def changeGeneIDs(self, gene_map,  with_loss=False):

        if with_loss:  # throw away all interactions of which at least one gene can not be mapped
            old_N_genes = len(self.genes)
            mapped_genes = list(set(gene_map.keys()).intersection(set(self.genes)))

            self.df = self.df[mapped_genes]
            self.genes = [gene_map[g] for g in mapped_genes]
            self.df.columns = self.genes

            print(str(old_N_genes - len(self.genes)) + ' nodes have been removed')

        else:
            try:
                self.genes = np.array(list(map(lambda x: gene_map[x], self.genes))).astype('str')
                self.df.columns = self.genes
            except KeyError:
                print('Not all old gene IDs are mapped onto new IDs. Specify with_loss=True to allow for lossy mapping.')

    def changeSampleIDs(self, patient_map, with_loss=False):

        if with_loss:  # throw away all interactions of which at least one gene can not be mapped
            old_N_samples = len(self.samples)
            mapped_samples = list(set(patient_map.keys()).intersection(set(self.samples)))

            self.df = self.df.loc[mapped_samples]
            self.samples = np.array(mapped_samples)
            print(str(old_N_samples - len(self.samples)) + ' nodes have been removed')

        else:
            try:
                self.samples = np.array(list(map(lambda x: patient_map[x], self.samples))).astype('str')
                self.df.index = self.samples
            except KeyError:
                print('Not all old sample IDs are mapped onto new IDs. Specify with_loss=True to allow for lossy mapping.')

    def mean(self, axis=0):
        return self.df.mean(axis=axis)

    def max(self, axis=0):
        return self.df.max(axis=axis)

    def min(self, axis=0):
        return self.df.min(axis=axis)

    def sum(self, axis=0):
        return self.df.sum(axis=axis)

    def std(self, axis=0):
        return self.df.std(axis=axis)

    def print(self, ntop=5):
        print(self.df.head(ntop))

    def to_nparray(self):
        return self.df.values

    def unique(self):
        return pd.unique(self.df.values.ravel('K'))

    def equals(self, dataset2):
        return self.df.equals(dataset2.df)

    def subsetGenes(self, genes):
        self.df = self.df[genes]
        self.genes = np.array(genes).astype('str')

    def subsetSamples(self, samples):
        self.df = self.df.loc[samples]
        self.samples = np.array(samples).astype('str')

    def getSampleSubset(self, samples):
        df = copy.deepcopy(self.df)
        df = df.loc[samples]
        return df

    def getGeneSubset(self, genes):
        df = copy.deepcopy(self.df)
        df = df[genes]
        return df

    def get_all_pairs(self, include_type=False, colnames=('Samples', 'Genes')):
        Genes, Samples = self.df.columns.values, np.array(self.df.index)
        row_ids, col_ids = np.where(self.df.values != 0)

        if include_type:
            Genes = np.array([gene + ' ' + self.type for gene in Genes])
            Samples = np.array([sample + ' ' + 'PAT' for sample in Samples])

        return pd.DataFrame({colnames[0]: Samples[row_ids], colnames[1]: Genes[col_ids]})

    def removeZeroVarianceGenes(self):
        mask = self.df.std(axis=0).values > 1e-15
        self.genes = self.df.columns.values[mask]
        self.df = self.df[self.genes]


    def concatDatasets(self, datasets, axis=1, include_types=True):
        try:
            _ = (d for d in datasets)

        except TypeError:  # convert to iterable
            datasets = [datasets]

        datasets = datasets + [self]
        datasets2 = [copy.deepcopy(dataset) for dataset in datasets]

        if include_types:

            for dataset in datasets2:
                dataset.changeGeneIDs({gene: gene + ' ' + dataset.type for gene in dataset.genes})

        DF = pd.concat([dataset.df for dataset in datasets2], axis=axis)

        return DF

class DiscreteOmicsDataSet(OmicsDataSet):
    def __init__(self, dataframe, patient_axis='auto', remove_nas=True, attrs=None, type='Omics', remove_zv=True, verbose=True):
        super().__init__(dataframe, patient_axis=patient_axis, remove_nas=remove_nas, type=type, remove_zv=remove_zv, verbose=verbose)
        if attrs is None:
            unique_levels = pd.unique(dataframe.values.flatten())
            self.attrs = {i: ' ' + self.type + str(i) for i in unique_levels}
        else:
            self.attrs = attrs

        self.df = self.df.astype(np.uint16)

    def __getitem__(self, item):
        return DiscreteOmicsDataSet(self.df.iloc[item], type=self.type, remove_zv=False, patient_axis=0)

    def getSampleSubset(self, samples):
        return DiscreteOmicsDataSet(super().getSampleSubset(samples), patient_axis=0, attrs=self.attrs, type=self.type)

    def getGeneSubset(self, genes):
        return DiscreteOmicsDataSet(super().getGeneSubset(genes), patient_axis=0, attrs=self.attrs, type=self.type)

    def calculateHazardRatios(self, event_labels, os, age=None, shuffle=False):
        if shuffle:
            samples = self.samples
            df = self.df.sample(frac=1)
            df.index = samples
        else:
            df = self.df

        if len(self.unique()) > 2:
            HR_data = [getHazardRatio((df[col] == value).astype(np.int), os, event_labels, col, value,
                                      age=age, return_sign=True)
                       for col in df for value in np.unique(df[col].values)]
        else:
            print('Binary data, only considering one regime per gene profile.')
            HR_data = [getHazardRatio((df[col] == value).astype(np.int), os, event_labels, col, value, age=age,
                                      binary=True, return_sign=True)
                       for col in df for value in np.unique(df[col].values)[:-1]]

        return pd.DataFrame(HR_data, columns=['Gene', 'Regime', 'Hazard Ratio', 'N_samples'])

    def getSignificantGenePairs(self, dataset2=None, testtype='right', count_thresh=20,
                                pvals_thresh=np.log(1e-10)):

        self_vals = np.unique(self.df.values)
        N_vals = len(self_vals)

        if (dataset2 is None) or self.equals(dataset2):
            pval_df = [get_pval_two_mat((self.df == self_vals[v1]).astype(np.uint16),
                                        (self.df == self_vals[v2]).astype(np.uint16),
                                        pvals_thresh=pvals_thresh, count_thresh=count_thresh,
                                        testtype=testtype, attr=(' ' + str(self_vals[v1]), ' ' + str(self_vals[v2])))
                      for v1 in range(N_vals) for v2 in range(v1 + 1)]

            pval_df = pd.concat(pval_df, axis=0)
            pval_df = pval_df.sort_values(by='p-value')
            return pval_df

        else:
            other_vals = dataset2.unique()

            pval_df = [get_pval_two_mat((self.df == v1).astype(np.uint16),
                                        (dataset2.df == v2).astype(np.uint16),
                                        pvals_thresh=pvals_thresh, count_thresh=count_thresh,
                                        testtype=testtype, attr=(' ' + str(v1), ' ' + str(v2)))
                      for v1 in self_vals for v2 in other_vals]

            pval_df = pd.concat(pval_df, axis=0)
            pval_df = pval_df.sort_values(by='p-value')

            return pval_df

    def batchGenepairs(self, n_batches, dataset2=None, **kwds):

        if dataset2 is None:
            n_cols = self.df.shape[1]
            break_points = np.linspace(0, n_cols, num=n_batches+1, dtype=np.int)

            pvals = []
            dataset2 = self

            for i in range(len(break_points) - 1):
                print(i)
                for j in range(i + 1):
                    print(j)
                    pvals_ = self[:, break_points[i]:break_points[i + 1]]\
                                .getSignificantGenePairs(dataset2[:, break_points[j]:break_points[j + 1]], **kwds)

                    pvals += [pvals_]

        else:
            n_cols1, n_cols2 = self.df.shape[1], dataset2.df.shape[1]
            break_points1 = np.linspace(0, n_cols1, num=n_batches+1, dtype=np.int)
            break_points2 = np.linspace(0, n_cols2, num=n_batches+1, dtype=np.int)

            pvals = []

            for i in range(len(break_points1) - 1):
                for j in range(len(break_points2) - 1):
                    pvals_ = self[:, break_points1[i]:(break_points1[i + 1])]\
                                .getSignificantGenePairs(dataset2[:, break_points2[j]:break_points2[j + 1]], **kwds)

                    pvals += [pvals_]

        pvals = pd.concat(pvals, axis=0).sort_values(by='Mutual Information')

        return pvals

    def getMutualInformation(self, dataset2=None, MI_thresh=None):
        MIs = []

        if (dataset2 is None) or self.equals(dataset2):
            dataset2 = self

            for v1 in range(self.df.shape[1]):
                for v2 in range(v1 + 1):
                    MI = mutual_info_score(self.df.iloc[:, v1], dataset2.df.iloc[:, v2])
                    if MI > MI_thresh:
                        MIs += [(self.genes[v1], self.genes[v2], MI)]
        else:
            for v1 in range(self.df.shape[1]):
                for v2 in range(dataset2.df.shape[1]):
                    MI = mutual_info_score(self.df.iloc[:, v1], dataset2.df.iloc[:, v2])
                    if MI > MI_thresh:
                        MIs += [(self.genes[v1], dataset2.genes[v2], MI)]

        MI_df = pd.DataFrame(MIs, columns=['Gene_A', 'Gene_B', 'Mutual Information'])
        return MI_df

    def compareSampleProfiles(self, patient_list, sort=True):
        plot_patients = list(set(patient_list).intersection(set(self.samples)))
        plot_df = self.df.loc[plot_patients].transpose()

        if sort:
            nrows, ncols = plot_df.shape

            plot_df = plot_df.append(plot_df.sum(axis=0), ignore_index=True)
            plot_df = plot_df.sort_values(by=nrows, axis=1, ascending=False)
            plot_df = plot_df.drop(nrows, axis=0)
            plot_df['Rowsum'] = np.matmul(plot_df, 2.**np.arange(0, -ncols, -1))

            plot_df = plot_df.sort_values(by='Rowsum', ascending=False)
            plot_df = plot_df.drop('Rowsum', axis=1)

        plt.figure()
        plt.pcolor(plot_df)
        #plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.colorbar()
        plt.xticks(np.arange(0.5, len(plot_df.columns), 1), plot_df.columns)
        plt.xticks(rotation=45)
        plt.show()

    def compareGeneProfiles(self, gene_list, sort=True):
        '''
        :param gene_list: list of genes that are plotted in the heatmap
        :param sort: does the x axis (containing the genes) have to be sorted
        :return: a heatmap showing the genes in genelist across all samples
        '''

        plot_genes = list(set(gene_list).intersection(set(self.genes)))
        plot_df = self.df[plot_genes]

        if sort:
            nrows, ncols = plot_df.shape

            plot_df = plot_df.append(plot_df.sum(axis=0), ignore_index=True)
            plot_df = plot_df.sort_values(by=nrows, axis=1, ascending=False)
            plot_df = plot_df.drop(nrows, axis=0)
            plot_df['Rowsum'] = np.matmul(plot_df, 2.**np.arange(0, -ncols, -1))

            plot_df = plot_df.sort_values(by='Rowsum', ascending=False)
            plot_df = plot_df.drop('Rowsum', axis=1)

        plt.figure()
        plt.pcolor(plot_df)
        plt.colorbar()
        plt.xticks(np.arange(0.5, len(plot_df.columns), 1), plot_df.columns)
        plt.xticks(rotation=45)
        plt.show()

    def concatDatasets(self, datasets, axis=1, include_types=True):
        DF = super().concatDatasets(datasets, axis, include_types)

        return DiscreteOmicsDataSet(DF, patient_axis=0)

    def GetPatientDistances(self, norm):
        pass

    def GetGeneDistances(self, norm):
        pass

    def invert(self):
        self.df = 1 - self.df

    def compareAllPairs(self, count_thresh):
        '''
        :param count_thresh: a threshold, all regimes with less than this amount of observations are thrown out
        :return: a pval_df containing all significant pairs for all possible combinations
        '''
        pass

def get_pval_two_mat(df1, df2, testtype='right', pvals_thresh=1e-10, count_thresh=20, attr=None):
    '''
    :param df1: the dataframe containing the binary data
    :param testtype: whether left- or rightsided tests needs to be done
    :param pvals_thresh: float consider only pairs that have a p-value below this threshold
    :param count_thresh: integer, only co-occurrence above this threshold is kept
    :param attr: a tuple of strings, can be used to indicate the regime of the data
    :return: a pval df containing the most significant pairs and their p-values
    '''

    df1 = df1.loc[:, df1.sum(axis=0) > count_thresh]
    df2 = df2.loc[:, df2.sum(axis=0) > count_thresh]

    if (df1.shape[1] == 0) or (df2.shape[1] == 0):
        return pd.DataFrame({'Gene_A': [], 'Gene_B': [], 'Count': [], 'p-value': []})
    else:
        N_samples = df2.shape[0]

        if len(df1.shape) == 1:
            df1 = pd.DataFrame({'Query': df1, 'Dummy': [0 for i in range(df1.shape[0])]})
        elif len(df2.shape) == 1:
            df2 = pd.DataFrame({'Query': df2, 'Dummy': [0 for i in range(df2.shape[0])]})

        gene_ids1 = df1.columns.values
        gene_ids2 = df2.columns.values

        cooc_mat = df1.transpose().dot(df2).values
        if df1.equals(df2):
            cooc_mat = np.triu(cooc_mat, 1)

        P_mat = np.outer(df1.sum(axis=0).values, df2.sum(axis=0).values)

        if testtype.lower() == 'left':
            ids = np.where(cooc_mat < count_thresh)
            cooc_mat, P_mat = cooc_mat[ids], P_mat[ids]
            gene_ids1, gene_ids2 = gene_ids1[ids[0]], gene_ids2[ids[1]]
            pvals_mat = binom.logcdf(cooc_mat, N_samples, P_mat/N_samples**2)
        else:
            ids = np.where(cooc_mat > count_thresh)
            cooc_mat, P_mat = cooc_mat[ids], P_mat[ids]
            gene_ids1, gene_ids2 = gene_ids1[ids[0]], gene_ids2[ids[1]]
            pvals_mat = binom.logsf(cooc_mat, N_samples, P_mat/N_samples**2)

        pvals_mat[np.isinf(pvals_mat)] = -500
        mask = np.where(pvals_mat < pvals_thresh)
        gene_ids1 = gene_ids1[mask]
        gene_ids2 = gene_ids2[mask]
        counts = cooc_mat[mask]
        pvals = pvals_mat[mask]

        if attr is not None:
            gene_ids1 = np.core.defchararray.add(gene_ids1.astype('str'), attr[0])
            gene_ids2 = np.core.defchararray.add(gene_ids2.astype('str'), attr[1])

        pval_df = pd.DataFrame({'Gene_A': gene_ids1, 'Gene_B': gene_ids2, 'Count': counts, 'p-value': pvals})
        pval_df.sort_values(by='p-value', inplace=True)

        return pval_df

def PMI(df1, df2):
    '''
    :param df1: the dataframe containing the binary data
    :return: a pval df containing the most significant pairs and their pointwise mutual information
    '''

    epsilon = 1e-15

    if (df1.shape[1] == 0) or (df2.shape[1] == 0):
        return pd.DataFrame({'Gene_A': [], 'Gene_B': [], 'Count': [], 'p-value': []})
    else:
        N_samples = df2.shape[0]

        if len(df1.shape) == 1:
            df1 = pd.DataFrame({'Query': df1, 'Dummy': [0 for i in range(df1.shape[0])]})
        elif len(df2.shape) == 1:
            df2 = pd.DataFrame({'Query': df2, 'Dummy': [0 for i in range(df2.shape[0])]})

        cooc_mat = df1.transpose().dot(df2).values
        P_mat = np.outer(df1.sum(axis=0).values, df2.sum(axis=0).values)

        PMI = cooc_mat/N_samples * np.log(cooc_mat * N_samples/(P_mat + epsilon) + epsilon)

        return PMI


class ContinuousOmicsDataSet(OmicsDataSet):
    def __init__(self, dataframe, patient_axis='auto', remove_nas=True, type='Omics', remove_zv=True, verbose=True):
        super().__init__(dataframe, patient_axis, remove_nas, type=type,  remove_zv=remove_zv, verbose=verbose)

    def __getitem__(self, item):
        return ContinuousOmicsDataSet(self.df.iloc[item], type=self.type, remove_zv=False, patient_axis=0)

    def getSampleSubset(self, samples):
        return ContinuousOmicsDataSet(super().getSampleSubset(samples), patient_axis=0)

    def getGeneSubset(self, genes):
        return ContinuousOmicsDataSet(super().getGeneSubset(genes), patient_axis=0)

    def compareSampleProfiles(self, kind='density', randomseed=None, Npatients=4):
        np.random.seed(randomseed)

        random_patients = list(np.random.permutation(self.samples)[:Npatients])
        plot_df = self.df.loc[random_patients]
        plot_df = plot_df.transpose()
        if kind == 'density':
            plot_df.plot(kind='density', subplots=True, use_index=False)
        else:
            plot_df.hist(bins=100)
        plt.show()

    def compareGeneProfiles(self, gene_list=None, Ngenes=4, kind='histogram'):
        if gene_list is None:
            random_genes = list(np.random.permutation(self.genes)[:Ngenes])
            print(random_genes)
            plot_df = self.df[random_genes]
        else:
            plot_genes = list(set(gene_list).intersection(set(self.genes)))
            plot_df = self.df[plot_genes]

        if plot_df.shape[1] > 0:
            if kind == 'density':
                plot_df.plot(kind='density', subplots=True, use_index=False)
            else:
                plot_df.hist(bins=100)

            plt.xlabel('Normalized expression', fontsize=14)
            plt.show()
        else:
            print('None of the genes are found in the dataset...')

    def normalizeProfiles(self, method='Quantile', axis=1):
        '''
        :param method: defines the method by which the (expression) profiles are scaled.
        currently supported is standardscaling, quantile normalization and centering.
        :param axis: the axis along which to scale (0 scales the genes profiles, 1 scales the patient profiles)
        :return: None (self.df is normalized)
        '''
        if axis == 1:
            self.df = self.df.transpose()

        if method.lower() == 'standardscaling':
            self.df = (self.df - self.mean(axis=0))/self.std(axis=0)
        elif method.lower() == 'quantile':
            rank_mean = self.df.stack().groupby(self.df.rank(method='first').stack().astype(int)).mean()
            self.df = self.df.rank(method='min').stack().astype(int).map(rank_mean).unstack()
        elif method.lower() == 'center':
            self.df = self.df - self.mean(axis=0)
        else:
            raise Exception('NotImplementedError')

        if axis == 1:
            self.df = self.df.transpose()

    def concatDatasets(self, datasets, axis=1, include_types=True):
        DF = super().concatDatasets(datasets, axis, include_types)

        return ContinuousOmicsDataSet(DF, patient_axis=0)

    def __getitem__(self, item):
        if item > len(self.samples):
            raise IndexError
        return ContinuousOmicsDataSet(self.df.iloc[item], type=self.type, remove_zv=False, patient_axis=0)

    def GetPatientDistances(self, norm):
        pass

    def GetGeneDistances(self, norm):
        pass

    def applyGMMBinarization(self, save_path=None, max_regimes=2, criterion='bic', remove_zv=False):
        '''
        :param save_path: the path to which to save the binarized dataframe
        :param max_regimes: the max number of regimes
        :return: a DiscreteOmicsDataSet containing the discretized data
        '''

        np.random.seed(42)

        bin_gx = np.zeros(self.df.shape, dtype=np.uint16)
        id = 0

        for gene in self.genes:
            temp = self.df[gene]

            temp = np.reshape(temp, (-1, 1))
            max_val = np.max(temp)
            print(id)
            gm_best, BIC_min, n_regimes = get_optimal_regimes(temp, max_regimes=max_regimes, criterion=criterion)

            if n_regimes == 2:
                labels = gm_best.predict(temp)
                labels = 1*(gm_best.predict(max_val) == labels)
                bin_gx[:, id] = labels.astype(np.uint16)
            else:
                labels = gm_best.predict(temp)
                bin_gx[:, id] = labels.astype(np.uint16)

            id += 1

        bin_gx = pd.DataFrame(bin_gx, index=self.samples, columns=self.genes)

        if save_path is not None:
            print('data is being saved to:' + save_path)
            bin_gx.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(bin_gx, type=self.type, patient_axis=0, remove_zv=False)

    def applyGMMBinarization_new(self, save_path=None, max_regimes=2, criterion='bic', remove_zv=False):
        '''
        :param save_path: the path to which to save the binarized dataframe
        :param max_regimes: the max number of regimes
        :return: a DiscreteOmicsDataSet containing the discretized data
        '''
        np.random.seed(42)

        bin_gx = self.df.apply(lambda x: getGMMRegimes(x, max_regimes=max_regimes, criterion=criterion), axis=0)

        if save_path is not None:
            print('data is being saved to:' + save_path)
            bin_gx.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(bin_gx, type=self.type, patient_axis=0, remove_zv=remove_zv)

    def applySTDBinning(self, std_thresh=1.5, save_path=None):
        scaled_df = (self.df - self.mean(axis=0))/self.std(axis=0)
        std_bin_gx = 1*(scaled_df > std_thresh) - 1*(scaled_df < - std_thresh)

        if save_path is not None:
            print('data is being saved to:' + save_path)
            std_bin_gx.to_csv(save_path, sep='\t', index=True, header=True)

        return DiscreteOmicsDataSet(std_bin_gx, type=self.type, patient_axis=0)

    def plotExpressionRegime(self, gene, insert_title=True, savepath=None, remove_frame=False, criterion='bic',
                             annotated_patients=None, annotation_labels=None, max_regimes=2, method='GMM'):

        plotmat = self.df[gene].values
        max_val = np.max(plotmat)
        mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                                                'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                                                'tab:olive', 'tab:cyan'])

        if method.lower() == 'std':
            std_thresh = 1.5
            labels = (plotmat - np.mean(plotmat))/np.std(plotmat)
            labels = 1*(labels > std_thresh) - 1*(labels < - std_thresh)

        else:
            plotmat = plotmat.reshape((-1, 1))
            gm_best, BIC_min, n_regimes = get_optimal_regimes(plotmat, max_regimes=max_regimes, criterion=criterion)
            labels = gm_best.predict(plotmat)

            if n_regimes == 2:
                labels = 1*(gm_best.predict(max_val) == labels)

        n_regimes = len(np.unique(labels))

        if n_regimes == 1:

            plot_df = pd.DataFrame({'Expression': plotmat.flatten(), 'Label': labels}, index=self.samples)
            bin_seq = np.linspace(plot_df.Expression.min(), plot_df.Expression.max(), num=200)
            label = ['Basal']
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_df.groupby('Label').hist(ax=ax, bins=bin_seq, label=label)

            if annotated_patients is not None:
                print('Adding annotation lines')
                r = plot_df.loc[annotated_patients]
                r.Expression.hist(ax=ax, bins=bin_seq, label=annotation_labels)
                label.extend(annotation_labels)
                print(label)

            plt.legend(label, fontsize=16)

        elif n_regimes == 2:

            plot_df = pd.DataFrame({'Expression': plotmat.flatten(), 'Label': labels}, index=self.samples)
            plot_df.sort_values('Expression', ascending=True, inplace=True)

            labels = ['Regime 0', 'Regime 1']
            fig, ax = plt.subplots(figsize=(8, 6))
            bin_seq = np.linspace(plot_df.Expression.min(), plot_df.Expression.max(), num=200)
            plot_df.groupby('Label').hist(ax=ax, bins=bin_seq, label=labels)

            if annotated_patients is not None:
                print('Adding annotation lines')
                r = plot_df.loc[annotated_patients]
                r.Expression.hist(ax=ax, bins=bin_seq, label=annotation_labels)
                labels.extend(annotation_labels)
                print(labels)

            plt.legend(labels, fontsize=18)

        else:
            plot_df = pd.DataFrame({'Expression': plotmat.flatten(), 'Label': labels}, index=self.samples)
            plot_df.sort_values('Expression', ascending=True, inplace=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            bin_seq = np.linspace(plot_df.Expression.min(), plot_df.Expression.max(), num=200)
            plot_df.groupby('Label').hist(ax=ax, bins=bin_seq)

            if annotated_patients is not None:
                print('Adding annotation lines')
                r = plot_df.loc[annotated_patients]
                r.Expression.hist(ax=ax, bins=bin_seq)

        if insert_title:
            plt.title('Expression profile for ' + str(gene), fontsize=18)
        else:
            plt.title('')

        plt.xlabel('Normalized expression', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.grid(False)

        if remove_frame:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if savepath is not None:
            print('Saving figure to: ' + savepath)
            plt.savefig(savepath, dpi=1000, bbox_inches="tight")

        plt.show()

    def binData(self, method='GMM', n_regimes=3, criterion='bic'):

        if 'gmm' in method.lower():
            binned_data = self.applyGMMBinarization_new(max_regimes=n_regimes, criterion=criterion)

        else:
            binned_data = self.applySTDBinning()

        return binned_data

    def Benchmarkbinning(self, labels, nregimes, nsplits=5, test_ratio=0.3, save_path=None):
        '''
        :param labels: provides class labels for each of the methods
        :param params: a nested dict containing the parameters for each binning method
        :return:
        '''

        sss = StratifiedShuffleSplit(n_splits=nsplits, test_size=test_ratio, random_state=0)
        sss.get_n_splits(self.df, labels)

        if nregimes is None:
            binning_methods = ['STD', 'GMM']
            nregimes = {'STD': 3, 'GMM': 3}
        else:
            binning_methods = list(nregimes.keys())

        scores_train, scores_val = pd.DataFrame(0, index=np.arange(nsplits), columns=binning_methods + ['Continuous']),\
                                   pd.DataFrame(0, index=np.arange(nsplits), columns=binning_methods + ['Continuous'])
        split_id = 0
        n_trees = 1500

        for train_index, test_index in sss.split(self.df, labels):

            X_train, X_val = self.df.iloc[train_index], self.df.iloc[test_index]
            Y_train, Y_val = labels[train_index], labels[test_index]
            print(self.df.shape)
            rf = RandomForestClassifier(n_estimators=n_trees)
            rf.fit(X_train, Y_train)

            scores_train.loc[split_id, 'Continuous'] = accuracy_score(Y_train, rf.predict(X_train))
            scores_val.loc[split_id, 'Continuous'] = accuracy_score(Y_val, rf.predict(X_val))

            for binning_method in binning_methods:
                bin_data = self.binData(method=binning_method, n_regimes=nregimes[binning_method])  #ZV features are automatically removed
                print(bin_data.df.shape)
                X_train, X_val = bin_data.df.iloc[train_index], bin_data.df.iloc[test_index]

                rf = RandomForestClassifier(n_estimators=n_trees)
                rf.fit(X_train, Y_train)

                scores_train.loc[split_id, binning_method] = accuracy_score(Y_train, rf.predict(X_train))
                scores_val.loc[split_id, binning_method] = accuracy_score(Y_val, rf.predict(X_val))

            split_id += 1

        ax = scores_val.boxplot(boxprops={'linewidth': 2}, flierprops={'linewidth': 2},
                                medianprops={'linewidth': 2, 'color': 'darkgoldenrod'})
        plt.xticks(fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path is not None:
            plt.savefig(save_path + 'boxplot_binning_benchmark3.pdf', dpi=1000, bbox_inches="tight")
            plt.savefig(save_path + 'boxplot_binning_benchmark3.png', dpi=1000, bbox_inches="tight")

            scores_train.to_csv(save_path + 'training_scoresGMM3.csv', header=True, index=False)
            scores_val.to_csv(save_path + 'val_scoresGMM3.csv',  header=True, index=False)

            '''
            if nregimes is not None:
                with open(save_path + 'params.json', 'w') as outfile:
                    json.dump(nregimes, outfile)
            '''

        plt.show()

def get_optimal_regimes(data1d, max_regimes=2, criterion='bic', penalty=5):
    BIC_min, n_regimes = 1.e20, 1
    gm_best = None

    for regimes in range(1, max_regimes+1):
        gm = GaussianMixture(n_components=regimes, random_state=0) #42
        gm.fit(data1d)
        if criterion.lower() == 'aic':
            bic = gm.aic(data1d)
        elif criterion.lower() == 'rbic':
            bic = rbic(gm, data1d, penalty=penalty)
        else:
            bic = gm.bic(data1d)

        if bic < BIC_min:
            gm_best = gm
            BIC_min = bic
            n_regimes = regimes

    return gm_best, BIC_min, n_regimes

def getGMMRegimes(v, max_regimes, criterion='bic'):
    temp = np.reshape(v, (-1, 1))
    max_val = np.max(temp)
    gm_best, BIC_min, n_regimes = get_optimal_regimes(temp, max_regimes=max_regimes, criterion=criterion)

    if n_regimes == 2:
        labels = gm_best.predict(temp)
        labels = 1*(gm_best.predict(max_val) == labels)
        v_out = labels.astype(np.uint16)
    else:
        labels = gm_best.predict(temp)
        v_out = labels.astype(np.uint16)

    return v_out

def getHazardRatio(df_col, os, event, genename, value, binary=False, age=None, return_sign=False):
    cph = CoxPHFitter()
    os_data = pd.DataFrame({'Gene': df_col,
                            'Duration': os,
                            'Flag': event})
    if age is not None:
        os_data['Age'] = age

    try:
        cph.fit(os_data, 'Duration', 'Flag', show_progress=False)
    except ValueError:
        print('Not working, returning nans')
        return genename, value, np.nan, df_col.sum()

    hazard_ratio = np.exp(cph.hazards_['Gene'].values)

    if binary:
        if hazard_ratio < 1:
            hazard_ratio = 1/hazard_ratio
            value = 1

    if return_sign:
        return genename, value, hazard_ratio[0], df_col.sum()
    else:
        return hazard_ratio

def rbic(GMMobject, X, penalty=5):
    """Bayesian information criterion for the current model on the input X.
    Parameters
    ----------
    X : array of shape (n_samples, n_dimensions)
    Returns
    -------
    bic : float
        The lower the better.
    """
    return (-2 * GMMobject.score(X) * X.shape[0] +
            penalty * GMMobject._n_parameters() * np.log(X.shape[0]))