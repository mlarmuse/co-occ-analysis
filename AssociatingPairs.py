import pandas as pd
import numpy as np
from OmicsData import DiscreteOmicsDataSet

DATA_PATH = '/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/data_metabric/'
DATA_PATH_TCGA = '/home/bioinformatics/mlarmuse/PycharmProjects/PathwaysGDC/'
SAVE_PATH ='/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/Paper/Second_Submission/'
genomic_datapath_TCGA = '/home/bioinformatics/mlarmuse/Documents/GDCdata/'
DATAPATH = '/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/Paper/Second_Submission/'

# Read in the data
n_reg = '2'
count_thresh = 70
reg_META = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_META.csv',
                                            header=0, index_col=0), type='')
reg_TCGA = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_TCGA.csv',
                                            header=0,  index_col=0), type='')

cna_data_TCGA = pd.read_csv(genomic_datapath_TCGA + 'ProcessedFiles/GISTIC2/CNA_data_Breast', index_col=0, header=0, sep='\t')

cna_data_TCGA = 1*(cna_data_TCGA > 0) + 2*(cna_data_TCGA < 0)
cna_data_TCGA = cna_data_TCGA[((cna_data_TCGA == 1).sum(axis=1) > count_thresh) &
                              ((cna_data_TCGA == 2).sum(axis=1) > count_thresh)]
cna_data_TCGA = DiscreteOmicsDataSet(cna_data_TCGA, type='CNA')
cna_data_TCGA.changeSampleIDs({s: s+'A' for s in cna_data_TCGA.samples})

mut_data_TCGA = pd.read_csv(genomic_datapath_TCGA + 'ProcessedFiles/MuTect2/mut_data_Breast.csv',
                            index_col=0, header=0)
mut_data_TCGA = DiscreteOmicsDataSet(mut_data_TCGA)

mapid_dfs = pd.read_csv('/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/DataTCGA/ENSG2name.txt',
                        sep='\t', header=0, index_col=0)[['Gene name']]
mapper_dict = dict(zip(list(mapid_dfs.index), mapid_dfs['Gene name']))
mut_data_TCGA.changeGeneIDs(mapper_dict, with_loss=True)
mut_data_TCGA = mut_data_TCGA.df.transpose().groupby(mut_data_TCGA.genes).max().transpose()
mut_data_TCGA = mut_data_TCGA.loc[[s != 'nan' for s in mut_data_TCGA.index]]
mut_data_TCGA = mut_data_TCGA[((mut_data_TCGA == 1).sum(axis=1) > count_thresh)]

mut_data_TCGA = DiscreteOmicsDataSet(mut_data_TCGA)

reg_META = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'bin_data_META' + str(n_reg) + '_regimes_allgenes.csv',
                                            header=0, index_col=0), type='', remove_zv=False)
reg_TCGA = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'bin_data_TCGA' + str(n_reg) + '_regimes_allgenes.csv',
                                            header=0,  index_col=0), type='', remove_zv=False)

mut_data_META = DiscreteOmicsDataSet(pd.read_csv(DATA_PATH + 'mut_dataframe.csv', header=0, index_col=0),
                                type='MUT', attrs=(' 0', ' 1'), patient_axis=0)

cna_data_META = pd.read_csv(DATA_PATH + 'cna_dataframe.csv', header=0, index_col=0)
cna_data_META = 1*(cna_data_META > 0) + 2*(cna_data_META < 0)
cna_data_META = cna_data_META[((cna_data_META == 1).sum(axis=1) > count_thresh) &
                              ((cna_data_META == 2).sum(axis=1) > count_thresh)]

cna_data_META = DiscreteOmicsDataSet(cna_data_META, type='MUT', patient_axis=0)

pval_gx_TCGA = pd.read_csv(DATAPATH + 'pvals_gx_' + n_reg + '_regimes_TCGA.csv', header=0)
pval_gx_META = pd.read_csv(DATAPATH + 'pvals_gx_' + n_reg + '_regimes_META.csv', header=0)

pval_gx_META['Regime A'] = [s[-1] for s in pval_gx_META.Gene_A]
pval_gx_META['Regime B'] = [s[-1] for s in pval_gx_META.Gene_B]
pval_gx_TCGA['Regime A'] = [s[-1] for s in pval_gx_TCGA.Gene_A]
pval_gx_TCGA['Regime B'] = [s[-1] for s in pval_gx_TCGA.Gene_B]

pval_gx_META.Gene_A = [s.split(' ')[0] for s in pval_gx_META.Gene_A]
pval_gx_META.Gene_B = [s.split(' ')[0] for s in pval_gx_META.Gene_B]
pval_gx_TCGA.Gene_A = [s.split(' ')[0] for s in pval_gx_TCGA.Gene_A]
pval_gx_TCGA.Gene_B = [s.split(' ')[0] for s in pval_gx_TCGA.Gene_B]

topn = 1000

def getPairDF(df, dataset, topn):
    pairs, pairnames = [], []

    for i in range(topn):
        genea = df.Gene_A.iloc[i]
        geneb = df.Gene_B.iloc[i]

        rega = np.int(df['Regime A'].iloc[i])
        regb = np.int(df['Regime B'].iloc[i])

        pairs += [(dataset.df[genea] == rega) & (dataset.df[geneb] == regb)]
        pairnames += [genea + ' ' + str(rega) + '--' + geneb + ' ' + str(regb)]

    df_out = 1 * pd.concat(pairs, axis=1)
    df_out.columns = pairnames

    return df_out

pair_df_META = getPairDF(pval_gx_META, reg_META, 1000)
pair_df_TCGA = getPairDF(pval_gx_TCGA, reg_TCGA, 1000)


print('Starting significance calculations...')
pairs_META_, mut_META_ = DiscreteOmicsDataSet(pair_df_META, patient_axis=0).keepCommonSamples([mut_data_META])
pval_pair_mut_META = pairs_META_.getSignificantGenePairs(mut_META_, pvals_thresh=np.log(0.001/(4 * 1000 * 170)))

pairs_TCGA_, mut_TCGA_ = DiscreteOmicsDataSet(pair_df_TCGA, patient_axis=0).keepCommonSamples([mut_data_TCGA])
pval_pair_mut_TCGA = pairs_TCGA_.getSignificantGenePairs(mut_TCGA_, pvals_thresh=np.log(0.001/(4 * 1000 * mut_TCGA_.df.shape[1])))

print('Starting significance calculations for CNA data')
pairs_META_, cna_META_ = DiscreteOmicsDataSet(pair_df_META, patient_axis=0).keepCommonSamples([cna_data_META])
pval_pair_cna_META = pairs_META_.getSignificantGenePairs(cna_META_, pvals_thresh=np.log(0.001/(6 * 1000 * cna_META_.df.shape[1])))

pairs_TCGA_, cna_TCGA_ = DiscreteOmicsDataSet(pair_df_TCGA, patient_axis=0).keepCommonSamples([cna_data_TCGA])
pval_pair_cna_TCGA = pairs_TCGA_.getSignificantGenePairs(cna_TCGA_, pvals_thresh=np.log(0.001/(6 * 1000 * cna_TCGA_.df.shape[1])))

pairs_TCGA_mut = set([s[:-2] for s in pval_pair_mut_TCGA.Gene_A])
pairs_META_mut = set([s[:-2] for s in pval_pair_mut_META.Gene_A])

print('Number of pairs explained by mutation in TCGA: %i' % len(pairs_TCGA_mut))
print('Number of pairs explained by mutation in META: %i' % len(pairs_META_mut))


pairs_TCGA_cna = set([s[:-2] for s in pval_pair_cna_TCGA.Gene_A])
pairs_META_cna = set([s[:-2] for s in pval_pair_cna_META.Gene_A])
print('Number of pairs explained by copy number in TCGA: %i' % len(pairs_TCGA_cna))
print('Number of pairs explained by copy number in META: %i' % len(pairs_META_cna))


pairs_TCGA_all = set([s[:-2] for s in pval_pair_cna_TCGA.Gene_A]).union(set([s[:-2] for s in pval_pair_mut_TCGA.Gene_A]))
pairs_META_all = set([s[:-2] for s in pval_pair_cna_META.Gene_A]).union(set([s[:-2] for s in pval_pair_mut_META.Gene_A]))

print('Number of pairs explained in TCGA: %i' % len(pairs_TCGA_all))
print('Number of pairs explained in META: %i' % len(pairs_META_all))
