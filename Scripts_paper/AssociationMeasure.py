import pandas as pd
import numpy as np
from OmicsData import ContinuousOmicsDataSet, DiscreteOmicsDataSet
import matplotlib.pyplot as plt

DATA_PATH = <path to metabric data>
DATA_PATH_TCGA = <path to TCGA transcriptome data>
SAVE_PATH = <path to save results>
genomic_datapath_TCGA = <path to genomic TCGA data>
DATAPATH = <path to network>

# First we read in the necessary data
gx_df = pd.read_csv(DATA_PATH + 'data_expression.txt', sep='\t', index_col=0)
gx_df = gx_df.drop(['Entrez_Gene_Id'], axis=1)

gx_META = ContinuousOmicsDataSet(gx_df)
gx_TCGA = ContinuousOmicsDataSet(pd.read_csv(DATA_PATH_TCGA + 'Expression_data_proc.csv', header=0, index_col=0))

gx_META.keepCommonGenes(gx_TCGA)

print(gx_META.df.shape)
print(gx_TCGA.df.shape)

reg_META = gx_META.applyGMMBinarization_new(max_regimes=2)
reg_TCGA = gx_TCGA.applyGMMBinarization_new(max_regimes=2)

reg_META.df.to_csv(SAVE_PATH + 'regimes_gx_2_regimes_META.csv', header=True, index=True)
reg_TCGA.df.to_csv(SAVE_PATH + 'regimes_gx_2_regimes_TCGA.csv', header=True, index=True)

pval_gx_META = reg_META.getSignificantGenePairs(count_thresh=50)
pval_gx_META.to_csv(SAVE_PATH + 'pvals_gx_2_regimes_META.csv', header=True)

pval_gx_TCGA = reg_TCGA.getSignificantGenePairs(count_thresh=50)
pval_gx_TCGA.to_csv(SAVE_PATH + 'pvals_gx_2_regimes_TCGA.csv', header=True)

# Next we calculate the correlation between the two different datasets
#
def getCorrelationsDf(dataset, corr_thresh):
    corrs_ = np.corrcoef(dataset.df.values, rowvar=False)

    corrs_ = np.triu(corrs_, 1)
    rows, cols = np.where(np.abs(corrs_) > corr_thresh)
    corrs_df = pd.DataFrame({'Gene_A': dataset.genes[rows],
                           'Gene_B': dataset.genes[cols],
                           'Correlation': corrs_[(rows, cols)]}).sort_values(by='Correlation', ascending=False)
    return corrs_df


corr_thresh = 0.5

corr_df_META = getCorrelationsDf(gx_META, corr_thresh=corr_thresh)
corr_df_TCGA = getCorrelationsDf(gx_TCGA, corr_thresh=corr_thresh)

corr_df_META.to_csv(SAVE_PATH + 'correlations_gx_META.csv', header=True)
corr_df_TCGA.to_csv(SAVE_PATH + 'correlations_gx_TCGA.csv', header=True)

## We also have a look at mutual information
n_reg = '2'
reg_META = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_META.csv',
                                            header=0, index_col=0), type='')
reg_TCGA = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_TCGA.csv',
                                            header=0,  index_col=0), type='')

reg_META.keepCommonGenes(reg_TCGA)

MI_TCGA = reg_TCGA.getMutualInformation(MI_thresh=0.01)
MI_META = reg_META.getMutualInformation(MI_thresh=0.01)

MI_TCGA.to_csv(SAVE_PATH + 'MI_gx_TCGA_' + n_reg + 'reg.csv', header=True)
MI_META.to_csv(SAVE_PATH + 'MI_gx_META_' + n_reg + 'reg.csv', header=True)

MI_TCGA.loc[MI_TCGA.Gene_A != MI_TCGA.Gene_B].sort_values(by='Mutual Information', ascending=False)
MI_META.loc[MI_META.Gene_A != MI_META.Gene_B].sort_values(by='Mutual Information', ascending=False)


### Next we identify other pairs
MI_TCGA_filtered = MI_TCGA.loc[MI_TCGA.Gene_A.isin(reg_META.genes) &
                               MI_TCGA.Gene_B.isin(reg_META.genes)]
MI_META_filtered = MI_META.loc[MI_META.Gene_A.isin(reg_TCGA.genes) &
                               MI_META.Gene_B.isin(reg_TCGA.genes)]
topn = 10000

MI_TCGA_filtered.to_csv(SAVE_PATH + 'MI_gx_TCGA_' + n_reg + 'reg_filtered.csv', header=True)
MI_META_filtered.to_csv(SAVE_PATH + 'MI_gx_META_' + n_reg + 'reg_filtered.csv', header=True)

# Next we quantify the overlap between the two lists
#
corr_df_META = pd.read_csv(SAVE_PATH + 'correlations_gx_META.csv', header=0)
corr_df_TCGA = pd.read_csv(SAVE_PATH + 'correlations_gx_TCGA.csv', header=0)
pval_gx_TCGA = pd.read_csv(SAVE_PATH + 'pvals_gx_' + n_reg + '_regimes_TCGA.csv', header=0)
pval_gx_META = pd.read_csv(SAVE_PATH + 'pvals_gx_' + n_reg + '_regimes_META.csv', header=0)
MI_META_filtered = pd.read_csv(SAVE_PATH + 'MI_gx_META_' + n_reg + 'reg_filtered.csv', header=0)
MI_TCGA_filtered = pd.read_csv(SAVE_PATH + 'MI_gx_TCGA_' + n_reg + 'reg_filtered.csv', header=0)


pval_gx_META.Gene_A = [s.split(' ')[0] for s in pval_gx_META.Gene_A]
pval_gx_META.Gene_B = [s.split(' ')[0] for s in pval_gx_META.Gene_B]

pval_gx_TCGA.Gene_A = [s.split(' ')[0] for s in pval_gx_TCGA.Gene_A]
pval_gx_TCGA.Gene_B = [s.split(' ')[0] for s in pval_gx_TCGA.Gene_B]

pval_list_TCGA = pval_gx_TCGA[['Gene_A', 'Gene_B']]
pval_list_TCGA.values.sort(axis=1)

pval_list_META = pval_gx_META[['Gene_A', 'Gene_B']]
pval_list_META.values.sort(axis=1)

pval_list_TCGA_filtered = pval_list_TCGA.loc[pval_list_TCGA.Gene_A.isin(reg_META.genes) &
                                    pval_list_TCGA.Gene_B.isin(reg_META.genes)]

pval_list_META_filtered = pval_list_META.loc[pval_list_META.Gene_A.isin(reg_TCGA.genes) &
                                    pval_list_META.Gene_B.isin(reg_TCGA.genes)]

pval_list_TCGA_filtered.to_csv(SAVE_PATH + 'pvals_TCGA_' + n_reg + 'reg_filtered.csv')
pval_list_META_filtered.to_csv(SAVE_PATH + 'pvals_META_' + n_reg + 'reg_filtered.csv')

pval_list_TCGA.to_csv(SAVE_PATH + 'pvals_TCGA_' + n_reg + 'reg.csv')
pval_list_META.to_csv(SAVE_PATH + 'pvals_META_' + n_reg + 'reg.csv')

def getOverlap(rankedlist1, rankedlist2, max_n=None, step=1):
    if max_n is None:
        max_n = rankedlist1.shape[0]
    else:
        max_n = np.minimum(max_n, rankedlist2.shape[0])
        print(max_n)

    steps = np.arange(1, max_n, step=np.int(step), dtype=np.int)
    overlap = np.array([len(set(zip(rankedlist2.iloc[:topn].Gene_A, rankedlist2.iloc[:topn].Gene_B))
                        .intersection(zip(rankedlist1.Gene_A, rankedlist1.Gene_B)))
                        for topn in steps])
    #id = np.min(np.where(overlap == np.max(overlap))[0])
    #overlap = overlap[:id]
    return overlap

topn = 1000
max_n = 1e4
step = 1

overlap = {}
overlap['Correlation'] = getOverlap(corr_df_META.iloc[:topn, :], corr_df_TCGA, max_n=max_n, step=step)
overlap['Co-occurrence'] = getOverlap(pval_list_META.iloc[:topn, :], pval_list_TCGA, max_n=max_n, step=step)
overlap['Co-occurrence filtered'] = getOverlap(pval_list_META_filtered.iloc[:topn, :],
                                               pval_list_TCGA_filtered, max_n=max_n, step=step)
overlap['MI filtered'] = getOverlap(MI_META_filtered.iloc[:topn, :],
                                    MI_TCGA_filtered, max_n=max_n, step=step)

print(len(set(zip(corr_df_META.Gene_A, corr_df_META.Gene_B))
                        .intersection(zip(corr_df_TCGA.Gene_A, corr_df_TCGA.Gene_B))))

print(len(set(zip(pval_list_TCGA.Gene_A, pval_list_TCGA.Gene_B))
                  .intersection(zip(pval_list_META.Gene_A, pval_list_META.Gene_B))))

plt.style.use('ggplot')
fig = plt.figure(figsize=(15, 8))
tab_colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

cs = {model: tab_colors[i] for i, model in enumerate(overlap.keys())}
max_ = 0

for i, method in enumerate(overlap.keys()):
    N_pairs = len(overlap[method])
    steps = np.arange(1, N_pairs+1)

    #ax = fig.add_subplot(131 + i)
    #ax.set_title(method)
    plt.step(steps, overlap[method], c=cs[method], label=method)
    max_ = np.maximum(overlap[method][-1], max_)

ax = plt.gca()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.plot(np.arange(max_), np.arange(max_), 'k--', label='Perfect overlap')
plt.xlabel('Rank TCGA', fontsize=16)
plt.ylabel('Overlap with METABRIC', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=14)
plt.show()

# We also investigate the overlap between the two measures on each dataset
topn = 1000

overlap = {}
overlap['Metabric'] = getOverlap(pval_list_META_filtered.iloc[:topn, :], corr_df_META, max_n=1e4, step=1)
overlap['TCGA'] = getOverlap(pval_list_TCGA_filtered.iloc[:topn, :], corr_df_TCGA, max_n=1e4, step=1)

cs = {model: tab_colors[i] for i, model in enumerate(overlap.keys())}
max_ = 0

fig = plt.figure(figsize=(15, 8))

for i, method in enumerate(overlap.keys()):
    N_pairs = len(overlap[method])
    steps = np.arange(1, N_pairs+1)

    #ax = fig.add_subplot(131 + i)
    #ax.set_title(method)
    plt.step(steps, overlap[method], c=cs[method], label=method)
    max_ = np.maximum(overlap[method][-1], max_)

ax = plt.gca()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.plot(np.arange(max_), np.arange(max_), 'k--', label='Perfect overlap')
plt.xlabel('Rank co-occurrence', fontsize=16)
plt.ylabel('Overlap with Correlation', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=14)
plt.show()

# Test on biogrid
def checkInteractionOverlap(df1, df2, unsorted=True):
    if unsorted:
        df1.values.sort(axis=1)
        df2.values.sort(axis=1)

    pairs1 = set(zip(df1.Gene_A, df1.Gene_B))
    pairs2 = set(zip(df2.Gene_A, df2.Gene_B))

    overlap = pairs1.intersection(pairs2)
    print('Overlap: %i' % len(overlap))
    print('Relative overlap %f' % (len(overlap)/len(pairs2)))
    return overlap

Results_path = '/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/Paper/Second_Submission/'
corr_df_META = pd.read_csv(Results_path + 'correlations_gx_META.csv', header=0)
corr_df_TCGA = pd.read_csv(Results_path + 'correlations_gx_TCGA.csv', header=0)
pval_gx_TCGA = pd.read_csv(Results_path + 'pvals_gx_' + n_reg + '_regimes_TCGA.csv', header=0)
pval_gx_META = pd.read_csv(Results_path + 'pvals_gx_' + n_reg + '_regimes_META.csv', header=0)

MI_META_filtered = pd.read_csv(Results_path + 'MI_gx_META_' + n_reg + 'reg_filtered.csv', header=0)
MI_TCGA_filtered = pd.read_csv(Results_path + 'MI_gx_TCGA_' + n_reg + 'reg_filtered.csv', header=0)

pval_list_TCGA_filtered = pd.read_csv(Results_path + 'pvals_TCGA_' + n_reg + 'reg_filtered.csv')
pval_list_META_filtered = pd.read_csv(Results_path + 'pvals_META_' + n_reg + 'reg_filtered.csv')

pval_list_TCGA = pd.read_csv(Results_path + 'pvals_TCGA_' + n_reg + 'reg.csv')
pval_list_META = pd.read_csv(Results_path + 'pvals_META_' + n_reg + 'reg.csv')

biogrid_net = pd.read_csv(DATA_PATH + 'BIOGRID-ORGANISM-Homo_sapiens-3.4.161.tab2.txt', sep='\t')[['Official Symbol Interactor A', 'Official Symbol Interactor B']]
biogrid_net = biogrid_net[biogrid_net['Official Symbol Interactor A'] != biogrid_net['Official Symbol Interactor B']]
biogrid_net.columns = ['Gene_A', 'Gene_B']
biogrid_net.values.sort(axis=1)
tester = biogrid_net.drop_duplicates()

topn = 10000

overlaps_META = [checkInteractionOverlap(tester, l) for l in [corr_df_META[['Gene_A', 'Gene_B']].iloc[:topn, :],
                                                         pval_list_META_filtered[['Gene_A', 'Gene_B']].iloc[:topn, :],
                                                         pval_list_META[['Gene_A', 'Gene_B']].iloc[:topn, :],
                                                         MI_META_filtered[['Gene_A', 'Gene_B']].iloc[:topn, :]]]

overlaps_TCGA = [checkInteractionOverlap(tester, l) for l in [corr_df_TCGA[['Gene_A', 'Gene_B']].iloc[:topn, :],
                                                         pval_list_TCGA_filtered[['Gene_A', 'Gene_B']].iloc[:topn, :],
                                                         pval_list_TCGA[['Gene_A', 'Gene_B']].iloc[:topn, :],
                                                         MI_TCGA_filtered[['Gene_A', 'Gene_B']].iloc[:topn, :]]]

# Find the relation between the pairs and mutations and copy number data for TCGA
n_reg = '2'
genomic_datapath_TCGA = '/home/bioinformatics/mlarmuse/Documents/GDCdata/'
cna_data_TCGA = pd.read_csv(genomic_datapath_TCGA + 'ProcessedFiles/GISTIC2/CNA_data_Breast',
                            index_col=0, header=0, sep='\t')

cna_data_TCGA = 1*(cna_data_TCGA > 0) + 2*(cna_data_TCGA < 0)
cna_data_TCGA = cna_data_TCGA[((cna_data_TCGA == 1).sum(axis=1) > 50) & ((cna_data_TCGA == 2).sum(axis=1) > 50)]
cna_dataset_TCGA = DiscreteOmicsDataSet(cna_data_TCGA, type='CNA')

mut_data_TCGA = pd.read_csv(genomic_datapath_TCGA + 'ProcessedFiles/MuTect2/mut_data_Breast.csv',
                            index_col=0, header=0)

mut_data_TCGA = mut_data_TCGA.loc[:, (mut_data_TCGA.sum(axis=0) > 50)]
mut_data_TCGA = DiscreteOmicsDataSet(mut_data_TCGA, patient_axis=0)

reg_META = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_META.csv',
                                            header=0, index_col=0), type='')
reg_TCGA = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_TCGA.csv',
                                            header=0,  index_col=0), type='')

cna_dataset_TCGA.changeSampleIDs({s: s+'A' for s in cna_dataset_TCGA.samples})
len(set(list(reg_TCGA.samples)).intersection(set(list(mut_data_TCGA.samples))))
len(set(list(reg_TCGA.samples)).intersection(set(list(cna_dataset_TCGA.samples))))

for l in mut_data_TCGA.samples:
    print(l)

for l in cna_dataset_TCGA.samples:
    print(l)

reg_TCGA_, cna_dataset_TCGA_, mut_data_TCGA_ = reg_TCGA.keepCommonSamples([cna_dataset_TCGA, mut_data_TCGA])

pval_gx_TCGA = pd.read_csv(SAVE_PATH + 'pvals_gx_' + n_reg + '_regimes_TCGA.csv', header=0)

topn = 1000
pvals_ = pval_gx_TCGA.iloc[:topn, :]
pair_coocs, colnames = [], []

for geneA, geneB in zip(pvals_.Gene_A, pvals_.Gene_B):
    gene_nameA = geneA.split(' ')[0]
    gene_nameB = geneB.split(' ')[0]

    regimeA = int(geneA[-1])
    regimeB = int(geneB[-1])

    pair_coocs += [(reg_TCGA_.df[gene_nameA] == regimeA) & (reg_TCGA_.df[gene_nameB] == regimeB)]
    colnames += [gene_nameA + ' ' + str(regimeA) + '--' + gene_nameB + ' ' + str(regimeB)]

cooc_pairs = 1*pd.concat(pair_coocs, axis=1)
cooc_pairs.columns = colnames

pvals_pairs_gx_mut = DiscreteOmicsDataSet(cooc_pairs, patient_axis=0).getSignificantGenePairs(mut_data_TCGA_)
pvals_pairs_gx_cna = DiscreteOmicsDataSet(cooc_pairs, patient_axis=0).getSignificantGenePairs(cna_dataset_TCGA_)
pvals_pairs_gx_gx = DiscreteOmicsDataSet(cooc_pairs, patient_axis=0).getSignificantGenePairs()

pvals_pairs_gx_mut.to_csv(SAVE_PATH + 'pvals_pairs_gx_mut_' + n_reg +'reg.csv', index=None, header=True)
pvals_pairs_gx_cna.to_csv(SAVE_PATH + 'pvals_pairs_gx_cna_' + n_reg +'reg.csv', index=None, header=True)
pvals_pairs_gx_gx.to_csv(SAVE_PATH + 'pvals_pairs_gx_gx_' + n_reg +'reg.csv', index=None, header=True)

len(pd.unique(pvals_pairs_gx_mut.Gene_A))
len(pd.unique(pvals_pairs_gx_cna.Gene_A))
len(pd.unique(pd.concat([pvals_pairs_gx_cna.Gene_A, pvals_pairs_gx_mut.Gene_A])))
print(len(pd.unique(pd.concat([pvals_pairs_gx_gx.Gene_A.apply(lambda x: x[:-7]),
                               pvals_pairs_gx_gx.Gene_B.apply(lambda x: x[:-7]),
                               pvals_pairs_gx_cna.Gene_A.apply(lambda x: x[:-7]),
                               pvals_pairs_gx_mut.Gene_A.apply(lambda x: x[:-7])]))))

# Next we focus on comparing MI and co-occurrence between gx and mutations and copy number
n_reg = str(2)
reg_META = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_META.csv',
                                            header=0, index_col=0), type='')
reg_TCGA = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'regimes_gx_' + n_reg + '_regimes_TCGA.csv',
                                            header=0,  index_col=0), type='')

cna_data_TCGA = pd.read_csv(genomic_datapath_TCGA + 'ProcessedFiles/GISTIC2/CNA_data_Breast', index_col=0, header=0, sep='\t')

cna_data_TCGA = 1*(cna_data_TCGA > 0) + 2*(cna_data_TCGA < 0)
cna_data_TCGA = cna_data_TCGA[((cna_data_TCGA == 1).sum(axis=1) > 50) & ((cna_data_TCGA == 2).sum(axis=1) > 50)]
cna_data_TCGA = DiscreteOmicsDataSet(cna_data_TCGA, type='CNA')
cna_data_TCGA.changeSampleIDs({s: s+'A' for s in cna_data_TCGA.samples})

mut_data_TCGA = pd.read_csv(genomic_datapath_TCGA + 'ProcessedFiles/MuTect2/mut_data_Breast.csv',
                            index_col=0, header=0)
mut_data_TCGA = DiscreteOmicsDataSet(mut_data_TCGA)

#mut_data_TCGA = mut_data_TCGA.loc[:, (mut_data_TCGA.sum(axis=0) > 50)]
mapid_dfs = pd.read_csv('/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/DataTCGA/ENSG2name.txt',
                        sep='\t', header=0, index_col=0)[['Gene name']]
mapper_dict = dict(zip(list(mapid_dfs.index), mapid_dfs['Gene name']))
mut_data_TCGA.changeGeneIDs(mapper_dict, with_loss=True)
mut_data_TCGA = mut_data_TCGA.df.transpose().groupby(mut_data_TCGA.genes).max().transpose()
mut_data_TCGA = mut_data_TCGA.loc[[s != 'nan' for s in mut_data_TCGA.index]]
mut_data_TCGA = DiscreteOmicsDataSet(mut_data_TCGA)

reg_META = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'bin_data_META' + str(n_reg) + '_regimes_allgenes.csv',
                                            header=0, index_col=0), type='', remove_zv=False)
reg_TCGA = DiscreteOmicsDataSet(pd.read_csv(SAVE_PATH + 'bin_data_TCGA' + str(n_reg) + '_regimes_allgenes.csv',
                                            header=0,  index_col=0), type='', remove_zv=False)

mut_data_META = DiscreteOmicsDataSet(pd.read_csv(DATA_PATH + 'mut_dataframe.csv', header=0, index_col=0),
                                type='MUT', attrs=(' 0', ' 1'), patient_axis=0)

cna_data_META = pd.read_csv(DATA_PATH + 'cna_dataframe.csv', header=0, index_col=0)
cna_data_META = 1*(cna_data_META > 0) + 2*(cna_data_META < 0)
cna_data_META = DiscreteOmicsDataSet(cna_data_META, type='MUT', patient_axis=0)

cna_data_META = pd.read_csv(DATA_PATH + 'cna_dataframe.csv', header=0, index_col=0)
cna_data_META = 1*(cna_data_META > 0) + 2*(cna_data_META < 0)

cna_data_META = DiscreteOmicsDataSet(cna_data_META, type='MUT', patient_axis=0)

cna_data_TCGA.keepCommonGenes(cna_data_META, inplace=True)
reg_META.keepCommonGenes(reg_TCGA, inplace=True)
mut_data_TCGA.keepCommonGenes(mut_data_META, inplace=True)

mut_data_META_, reg_META_ = reg_META.keepCommonSamples([mut_data_META])
mut_data_TCGA_, reg_TCGA_ = reg_TCGA.keepCommonSamples([mut_data_TCGA])

pvals_gx_mut_META = mut_data_META_.getSignificantGenePairs(reg_META_, pvals_thresh=np.log(1e-5))
pvals_gx_mut_TCGA = mut_data_TCGA_.getSignificantGenePairs(reg_TCGA_, pvals_thresh=np.log(1e-5))

bon_thresh = 3.9967067136679376e-10
thresh_mut_pvals_META = pvals_gx_mut_META.loc[pvals_gx_mut_META['p-value'] < np.log(bon_thresh)]
thresh_mut_pvals_TCGA = pvals_gx_mut_TCGA.loc[pvals_gx_mut_TCGA['p-value'] < np.log(bon_thresh)]

# How many genes can we find back

pvals_gx_mut_META.to_csv(SAVE_PATH + 'pvals_gx_mut_META_' + n_reg + 'reg_unfiltered.csv', header=True)
pvals_gx_mut_TCGA.to_csv(SAVE_PATH + 'pvals_gx_mut_TCGA_' + n_reg + 'reg_unfiltered.csv', header=True)

cna_data_META_, reg_META_ = reg_META.keepCommonSamples([cna_data_META])
cna_data_TCGA_, reg_TCGA_ = reg_TCGA.keepCommonSamples([cna_data_TCGA])

pvals_gx_cna_META = cna_data_META_.getSignificantGenePairs(reg_META_)
pvals_gx_cna_TCGA = cna_data_TCGA_.getSignificantGenePairs(reg_TCGA_)

pvals_gx_cna_META.to_csv(SAVE_PATH + 'pvals_gx_cna_META_' + n_reg + 'reg.csv', header=True)
pvals_gx_cna_TCGA.to_csv(SAVE_PATH + 'pvals_gx_cna_TCGA_' + n_reg + 'reg.csv', header=True)

# We also calculate the MI for this data

MI_TCGA_gx_mut = mut_data_TCGA_.getMutualInformation(reg_TCGA_, MI_thresh=0.01)
MI_META_gx_mut = mut_data_META_.getMutualInformation(reg_META_, MI_thresh=0.01)

MI_TCGA_gx_mut = MI_TCGA_gx_mut.sort_values(by='Mutual Information', ascending=False)
MI_TCGA_gx_mut.to_csv(SAVE_PATH + 'MI_TCGA_gx_mut_' + n_reg + 'reg.csv', header=True)
MI_META_gx_mut = MI_META_gx_mut.sort_values(by='Mutual Information', ascending=False)
MI_META_gx_mut.to_csv(SAVE_PATH + 'MI_META_gx_mut_' + n_reg + 'reg.csv', header=True)

MI_TCGA_gx_cna = cna_data_TCGA_.getMutualInformation(reg_TCGA_, MI_thresh=0.01)
MI_META_gx_cna = cna_data_META_.getMutualInformation(reg_META_, MI_thresh=0.01)

MI_TCGA_gx_cna = MI_TCGA_gx_cna.sort_values(by='Mutual Information', ascending=False)
MI_TCGA_gx_cna.to_csv(SAVE_PATH + 'MI_TCGA_gx_cna_' + n_reg + 'reg.csv', header=True)
MI_META_gx_cna = MI_META_gx_cna.sort_values(by='Mutual Information', ascending=False)
MI_META_gx_cna.to_csv(SAVE_PATH + 'MI_META_gx_cna_' + n_reg + 'reg.csv', header=True)
