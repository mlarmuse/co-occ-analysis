import pandas as pd
import numpy as np
from OmicsData import ContinuousOmicsDataSet, DiscreteOmicsDataSet
import matplotlib.pyplot as plt

DATA_PATH = <path to metabric data>
DATA_PATH_TCGA = <path to TCGA transcriptome data>
SAVE_PATH = <path to save results>
genomic_datapath_TCGA = <path to genomic TCGA data>
DATAPATH = <path to network>

# Read in the data
n_reg = '2'
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

# Load the Biogrid network
biogrid_net = pd.read_csv(DATA_PATH + 'BIOGRID-ORGANISM-Homo_sapiens-3.4.161.tab2.txt', sep='\t')[['Official Symbol Interactor A', 'Official Symbol Interactor B']]
biogrid_net = biogrid_net[biogrid_net['Official Symbol Interactor A'] != biogrid_net['Official Symbol Interactor B']]
biogrid_net.columns = ['Gene_A', 'Gene_B']
biogrid_net.values.sort(axis=1)
tester = biogrid_net.drop_duplicates()

def checkInteractionList(l, tester_df):
    l = set(l)
    overlap_df = tester_df[tester_df.Gene_A.isin(l) & tester_df.Gene_B.isin(l)]
    return overlap_df

def splitGeneCols(df, name=''):
    df = df.drop('Count', axis=1)
    mask = [np.int(s.split(' ')[1]) == 1 for s in df.Gene_A]
    df['Regime_B'] = [np.int(s.split(' ')[1]) for s in df.Gene_B]

    df['Gene_A'] = [s.split(' ')[0] for s in df.Gene_A]
    df['Gene_B'] = [s.split(' ')[0] for s in df.Gene_B]

    return df.loc[mask]

def makeEdgesDirected(interactions, gx_reg_df, gx2reg_dict, p_thresh=0.5):
    new_data = []
    rejects = 0

    for gene_a, gene_b in zip(interactions.Gene_A, interactions.Gene_B):

        bool_a, bool_b = gene_a in gx2reg_dict.keys(), gene_b in gx2reg_dict.keys()

        if bool_a and bool_b:
            genea_regs = (gx_reg_df[gene_a] == gx2reg_dict[gene_a]).astype(np.int)
            geneb_regs = (gx_reg_df[gene_b] == gx2reg_dict[gene_b]).astype(np.int)
            P_a_given_b = genea_regs.dot(geneb_regs)/geneb_regs.sum()
            P_b_given_a = genea_regs.dot(geneb_regs)/genea_regs.sum()

            if (P_a_given_b > P_b_given_a) and (P_a_given_b > p_thresh):
                new_data += [(gene_a, gene_b, gx2reg_dict[gene_a], gx2reg_dict[gene_b], np.NaN, np.NaN, 'Known')]
            elif (P_b_given_a > P_a_given_b) and (P_b_given_a > p_thresh):
                new_data += [(gene_b, gene_a, gx2reg_dict[gene_b], gx2reg_dict[gene_a], np.NaN, np.NaN, 'Known')]
            else:
                rejects += 1

        elif bool_a: # means b is from a mutations
            new_data += [(gene_b, gene_a, np.NaN, gx2reg_dict[gene_a], 1, np.NaN, 'Known')]

        elif bool_b:
            new_data += [(gene_a, gene_b, np.NaN, gx2reg_dict[gene_b], 1, np.NaN, 'Known')]
        else: # Means both genes are from mutations
            new_data += [(gene_a, gene_b, np.NaN, np.NaN, 1, 1, 'Known')]

    return pd.DataFrame(new_data, columns=['Gene_A', 'Gene_B', 'Regime_A_gx',
                                           'Regime_B_gx', 'Regime_A_mut', 'Regime_B_mut', 'Interactions']), rejects

# Define a Query
META_mut_, META_cna_, META_gx_ = mut_data_META.keepCommonSamples([cna_data_META, reg_META])
Query = 'MLPH_FOXA1'
Genes = Query.split('-')

Query_META = pd.DataFrame({Query: (META_gx_.df['FOXA1'].values == 0) & (META_gx_.df['MLPH'].values == 0)},
                          index=list(META_mut_.df.index))

Query_META = DiscreteOmicsDataSet(1 * Query_META, patient_axis=0, remove_zv=False)

# Query META data
Query_META_gx = Query_META.getSignificantGenePairs(META_gx_, pvals_thresh=np.log(0.001/(14341 * 14341 * 4)))
Query_META_mut = Query_META.getSignificantGenePairs(META_mut_, pvals_thresh=np.log(0.001/(14341 * 170 * 4)))
Query_META_cna = Query_META.getSignificantGenePairs(META_cna_, pvals_thresh=np.log(0.001/(14341 * 14341 * 4)))

Query_META_gx = splitGeneCols(Query_META_gx, name='')
Query_META_mut = splitGeneCols(Query_META_mut, name='')
Query_META_cna = splitGeneCols(Query_META_cna, name='')

cna_regime_map = dict(zip(Query_META_cna.Gene_B, Query_META_cna.Regime_B))
gx_regime_map = dict(zip(Query_META_gx.Gene_B, Query_META_gx.Regime_B))
all_data = pd.merge(Query_META_gx, Query_META_mut, on=['Gene_A', 'Gene_B'], how='outer', suffixes=('_gx', '_mut'))

all_data['Regime_B_cna'] = [2 * cna_regime_map[gene] if gene in cna_regime_map.keys() else np.NaN for gene in all_data.Gene_B]

Interactions = checkInteractionList(all_data.Gene_B, tester)
Interactions, n_rejects = makeEdgesDirected(Interactions, META_gx_.df, gx_regime_map, p_thresh=0.5)

all_data['Interactions'] = ['Derived' for g in all_data.Gene_B]
all_data = pd.concat([all_data, Interactions], axis=0)

all_data['Regime_A_cna'] = [2*cna_regime_map[gene] if gene in cna_regime_map.keys() else np.NaN for gene in all_data.Gene_A]
all_data['p_val'] = all_data[['p-value_gx', 'p-value_mut']].min(axis=1)

mapint2name = {0.: '', 1.: 'mut', 2.: 'amp', 3.: 'amp-mut', 4: 'del', 5: 'del-mut'}
all_data['Genomic Regime A'] = all_data[['Regime_A_cna', 'Regime_A_mut']].sum(axis=1, skipna=True).apply(lambda x: mapint2name[x])
all_data['Genomic Regime B'] = all_data[['Regime_B_cna', 'Regime_B_mut']].sum(axis=1, skipna=True).apply(lambda x: mapint2name[x])

all_data = all_data.drop(['Regime_A_cna', 'Regime_A_mut', 'Regime_B_cna', 'Regime_B_mut', 'p-value_mut', 'p-value_gx'], axis=1)
all_data['Regime_A_gx'].loc[all_data['Regime_A_gx'].isnull()] = 'Query'
all_data['p_val'].loc[all_data['p_val'].isnull()] = all_data['p_val'].iloc[:30].median()

all_data.to_csv(SAVE_PATH + Query + '_cyto.csv', header=True, index=False)

print(len(pd.unique(all_data[['Gene_A', 'Gene_B']].values.ravel())))
print(len(pd.unique(all_data.loc[all_data.Interactions == 'Known'][['Gene_A', 'Gene_B']].values.ravel())))


# How we treat the NaNs is different for each gene type
# First all genes that have only copy number (but no expression) are removed)
test = all_data.loc[~(np.isnan(all_data.Regime_gx_A.values.astype(np.float)) &
                     np.isnan(all_data.Regime_mut_A.values.astype(np.float)))]

TCGA_gx_, TCGA_mut_, TCGA_cna_ = reg_TCGA.keepCommonSamples([mut_data_TCGA, cna_data_TCGA])
Query = 'TP53-ESR1'
Genes = Query.split('-')
Query_TCGA = pd.DataFrame({Query: (TCGA_mut_.df['TP53'].values == 1) & (TCGA_gx_.df['ESR1'].values == 0)},
                          index=list(TCGA_mut_.df.index))

(TCGA_mut_.df == 1).astype(np.uint16)
Query_TCGA = DiscreteOmicsDataSet(1 * Query_TCGA, patient_axis=0, remove_zv=False)

Query_TCGA_gx = Query_TCGA.getSignificantGenePairs(TCGA_gx_)
Query_TCGA_mut = Query_TCGA.getSignificantGenePairs(TCGA_mut_)
Query_TCGA_cna = Query_TCGA.getSignificantGenePairs(TCGA_cna_)

print(len(set(Query_TCGA_gx.Gene_A).intersection(set(Query_META_gx.Gene_A))))

Query_TCGA_gx = splitGeneCols(Query_TCGA_gx, name='gx')
Query_TCGA_mut = splitGeneCols(Query_TCGA_mut, name='mut')
Query_TCGA_cna = splitGeneCols(Query_TCGA_cna, name='cna')

all_data = pd.concat([Query_TCGA_gx, Query_TCGA_mut, Query_TCGA_cna], axis=0)


# How we treat the NaNs is different for each gene type
# First all genes that have only copy number (but no expression) are removed)
test = all_data.loc[~(np.isnan(all_data.Regime_gx_A.values.astype(np.float)) &
                     np.isnan(all_data.Regime_mut_A.values.astype(np.float)))]

# Then we remove all association with the negative of the condition
