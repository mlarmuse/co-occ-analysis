import pandas as pd
import numpy as np
from OmicsData import ContinuousOmicsDataSet, DiscreteOmicsDataSet
from sklearn.preprocessing import StandardScaler

DATA_PATH = '/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/data_metabric/'
SAVE_PATH ='/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/Paper/Second_Submission/'
DATA_PATH_TCGA = '/home/bioinformatics/mlarmuse/PycharmProjects/PathwaysGDC/'

gx_df = pd.read_csv(DATA_PATH + 'data_expression.txt', sep='\t', index_col=0)
gx_df = gx_df.drop(['Entrez_Gene_Id'], axis=1)

patient_df = pd.read_csv(DATA_PATH + 'data_clinical_supp_patient.txt', sep='\t')
patient_df.CLAUDIN_SUBTYPE.value_counts()
patient_df.index = np.array(patient_df.PATIENT_ID)
patient_df = patient_df[patient_df.CLAUDIN_SUBTYPE != 'NC']

common_samples = np.intersect1d(patient_df.PATIENT_ID, gx_df.columns.values)
patient_df = patient_df.loc[common_samples]

patient_df.CLAUDIN_SUBTYPE.value_counts()

gx_test = gx_df.iloc[:10, :]
# Next we do the survival analysis
# PART 1: Calculate the co-occurrence between expression data
# Plot the RF model

ContinuousOmicsDataSet(gx_df[common_samples], patient_axis=1).Benchmarkbinning(patient_df.CLAUDIN_SUBTYPE,
                                                                             nregimes={'GMM 3': 3, 'STD': 3},
                                                                             nsplits=20, test_ratio=0.3,
                                                                             save_path=SAVE_PATH + 'Figures/')
gx_dataset = ContinuousOmicsDataSet(gx_df)

gx_dataset.plotExpressionRegime('FOXA1', max_regimes=2, remove_frame=True)
gx_dataset.plotExpressionRegime('FOXA1', max_regimes=3, remove_frame=True, savepath=SAVE_PATH + 'Figures/example_3reg__FOXA1.png')

bin_data = gx_dataset.applyGMMBinarization_new(max_regimes=2, remove_zv=False)
bin_data.df.to_csv('bin_data_2_regimes_allgenes_META.csv', header=True, index=True)

gx_TCGA = ContinuousOmicsDataSet(pd.read_csv(DATA_PATH_TCGA + 'Expression_data_proc.csv', header=0, index_col=0))
bin_data = gx_TCGA.applyGMMBinarization_new(max_regimes=2, remove_zv=False)
bin_data.df.to_csv('bin_data_2_regimes_allgenes_TCGA.csv', header=True, index=True)

bin_META = pd.read_csv('bin_data_2_regimes_allgenes_META.csv', header=0, index_col=0)
bin_TCGA = pd.read_csv('bin_data_2_regimes_allgenes_TCGA.csv', header=0, index_col=0)
(bin_TCGA.max(axis=0) == 1).sum()/bin_TCGA.shape[0]
print(len(set(bin_META.columns.values).intersection(set(bin_TCGA.columns.values))))
M_, T_ = DiscreteOmicsDataSet(bin_META, remove_zv=True).keepCommonGenes([DiscreteOmicsDataSet(bin_TCGA, remove_zv=True)])


len(set(T_.genes))

nregs = [str(3), str(6)]
for nreg in nregs:
    print('########################## Regime %s ##################################' % nreg)

    bin_data = gx_dataset.applyGMMBinarization_new(max_regimes=np.int(nreg), remove_zv=False)
    bin_data.df.to_csv('bin_data_' + nreg +'_regimes_allgenes_META.csv', header=True, index=True)
    print('Reading in TCGA data.')
    bin_data = gx_TCGA.applyGMMBinarization_new(max_regimes=np.int(nreg), remove_zv=False)
    bin_data.df.to_csv('bin_data_' + nreg +'_regimes_allgenes_TCGA.csv', header=True, index=True)

KEGG_list = ['ESR1', 'ERBB2', 'NOTCH1', 'EGFR',  'KIT', 'FZD7', 'LRP6']

bin_data.df[KEGG_list].max()

nreg = '6'
df_TCGA = pd.read_csv('bin_data_' + nreg +'_regimes_allgenes_TCGA.csv', header=0, index_col=0)[KEGG_list].max()
df_META = pd.read_csv('bin_data_' + nreg +'_regimes_allgenes_META.csv', header=0, index_col=0)
df_META = (df_META.max(axis=0) > 0).sum()/df_META.shape[1]

# Next we do the survival analysis
nregs = [str(2)]
N_thresh = 190  # 10%
sd = StandardScaler()
patient_df[['OS_MONTHS', 'AGE_AT_DIAGNOSIS']] = sd.fit_transform(patient_df[['OS_MONTHS', 'AGE_AT_DIAGNOSIS']])

for nreg in nregs:
    print('########################## Regime %s ##################################' % nreg)
    pd_df = pd.read_csv('bin_data_' + nreg + '_regimes.csv', header=0, index_col=0)
    pd_df = pd_df.loc[:, (pd_df.sum(axis=0) > N_thresh) & ((pd_df.shape[0] - pd_df.sum(axis=0)) > N_thresh)]
    bin_data = DiscreteOmicsDataSet(pd_df, type='')

    print('Data read in, calculating random Hazard ratios.')
    HZ_df_random = bin_data.getSampleSubset(common_samples)\
                    .calculateHazardRatios(event_labels=patient_df.VITAL_STATUS == 'Died of Disease',
                                           os=patient_df.OS_MONTHS, age=None, shuffle=True)
    print('Saving the data')
    HZ_df_random.to_csv(SAVE_PATH + 'Results/HazardRatios' + nreg + '_random.csv', index=True, header=True)

    print('Data read in, calculating Hazard ratios.')
    HZ_df = bin_data.getSampleSubset(common_samples)\
                    .calculateHazardRatios(event_labels=patient_df.VITAL_STATUS == 'Died of Disease',
                                           os=patient_df.OS_MONTHS, age=patient_df.AGE_AT_DIAGNOSIS, shuffle=False)
    print('Saving the data')
    HZ_df.to_csv(SAVE_PATH + 'Results/HazardRatios_' + nreg + '.csv', index=True, header=True)


# Next we compare the regimes with the data
bin_data2 = DiscreteOmicsDataSet(pd.read_csv('bin_data_2_regimes.csv', header=0, index_col=0), type='gx')
bin_data3 = DiscreteOmicsDataSet(pd.read_csv('bin_data_3_regimes.csv', header=0, index_col=0), type='gx')
bin_data6 = DiscreteOmicsDataSet(pd.read_csv('bin_data_all_regimes.csv', header=0, index_col=0), type='gx')

mut_data = DiscreteOmicsDataSet(pd.read_csv(DATA_PATH + 'mut_dataframe.csv', header=0, index_col=0),
                                type='MUT', attrs=(' 0', ' 1'), patient_axis=0)

mut_, exp_ = mut_data.keepCommonSamples([bin_data2], inplace=False)
pval_mut_gx2 = mut_.getSignificantGenePairs(exp_)

mut_, exp_ = mut_data.keepCommonSamples([bin_data3], inplace=False)
pval_mut_gx3 = mut_.getSignificantGenePairs(exp_)

mut_, exp_ = mut_data.keepCommonSamples([bin_data6], inplace=False)
pval_mut_gx6 = mut_.getSignificantGenePairs(exp_)


pval_mut_all_reg = pd.read_csv('pvals__mut_gx_all_regimes.csv')

topn = 30
all_pairs2 = set(zip([s.split(' ')[0] for s in pval_mut_gx2.iloc[:topn, 1]],
                            [s.split(' ')[0] for s in pval_mut_gx2.iloc[:topn, 2]]))

all_pairs3 = set(zip([s.split(' ')[0] for s in pval_mut_gx3.iloc[:, 1]],
                     [s.split(' ')[0] for s in pval_mut_gx3.iloc[:, 2]]))

all_pairs6 = set(zip([s.split(' ')[0] for s in pval_mut_gx6.iloc[:topn, 1]],
                     [s.split(' ')[0] for s in pval_mut_gx6.iloc[:topn, 2]]))

print(len(all_pairs2.intersection(all_pairs3)))
print(len(all_pairs2.intersection(all_pairs6)))
print(len(all_pairs6.intersection(all_pairs3)))



# PART 2: Calculate the co-occurrence between expression data

bin_data = DiscreteOmicsDataSet(pd.read_csv('bin_data_temp.csv', header=0, index_col=0), type='')
c, v = np.unique(bin_data.max(axis=0), return_counts=True)
count_df = pd.DataFrame({'Value': c, 'Counts': v})

maxreg_genes = bin_data.genes[bin_data.max(axis=0) > 3]

small_test = bin_data.getGeneSubset(maxreg_genes)
pval_test = small_test.getSignificantGenePairs(count_thresh=50)

pval_gx = bin_data.getSignificantGenePairs(count_thresh=50)
pval_gx.to_csv('pvals_gx_all_regimes.csv', header=0, index=0)

cna_data = pd.read_csv(DATA_PATH + 'cna_dataframe.csv', header=0, index_col=0)
cna_data = DiscreteOmicsDataSet(-1*(cna_data < 0) + 1*(cna_data > 0), type='CNA', attrs=(' d', ' n', ' u'))



pval_mut_gx.to_csv('pvals__mut_gx_all_regimes.csv', header=0, index=0)

cna_, exp_ = cna_data.keepCommonSamples([bin_data])
pvals_cna_gx = cna_.batchGenepairs(dataset2=bin_data, n_batches=3)
pvals_cna_gx.to_csv('pvals__cna_gx_all_regimes.csv', header=0, index=0)


# Load the Biogrid network
biogrid_net = pd.read_csv(DATA_PATH + 'BIOGRID-ORGANISM-Homo_sapiens-3.4.161.tab2.txt', sep='\t')[['Official Symbol Interactor A', 'Official Symbol Interactor B']]
biogrid_net = biogrid_net[biogrid_net['Official Symbol Interactor A'] != biogrid_net['Official Symbol Interactor B']]
biogrid_net.columns = ['Gene_A', 'Gene_B']
biogrid_net.values.sort(axis=1)
tester = biogrid_net.drop_duplicates()