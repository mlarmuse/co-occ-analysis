import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from OmicsData import ContinuousOmicsDataSet
from OmicsData import DiscreteOmicsDataSet

DATAPATH = <path where intermediate results are saved >
SAVEPATH = <path to save figures>
DATA_PATH = <path to processed metabric expression data>
DATA_PATH_TCGA = <path to processed TCGA transcriptome data>



# Plotting the Expression regimes for ERBB2
gene = 'ERBB2'
gx_META = pd.read_csv(DATA_PATH +'/data_expression.txt',
                    sep='\t', index_col=0)
gx_META = gx_META.drop(['Entrez_Gene_Id'], axis=1)

gx_TCGA = pd.read_csv(DATA_PATH_TCGA + '/Expression_data_proc.csv',
                      header=0, index_col=0)
gx_TCGA = ContinuousOmicsDataSet(gx_TCGA)
gx_META = ContinuousOmicsDataSet(gx_META)

gx_META.plotExpressionRegime(gene, insert_title=False, savepath=SAVEPATH + 'ERBB2_regimes_META.pdf', remove_frame=True)
gx_TCGA.plotExpressionRegime(gene, insert_title=False, savepath=SAVEPATH + 'ERBB2_regimes_TCGA.pdf', remove_frame=True)


# Plot1: The RF plot
val_data1 = pd.read_csv(DATAPATH + 'Results/val_scoresGMM3.csv',  header=0)
val_data2 = pd.read_csv(DATAPATH + 'Results/val_scoresGMM.csv',  header=0)
val_data2.drop(['STD', 'Continuous'], inplace=True, axis=1)

val_data = pd.concat([val_data1, val_data2], axis=1)
val_data = val_data[['Continuous', 'GMM 2', 'GMM 3', 'GMM 6', 'STD']]

plt.figure(figsize=(15, 8))
ax = val_data.boxplot(boxprops={'linewidth': 2}, flierprops={'linewidth': 2},
                        medianprops={'linewidth': 2, 'color': 'darkgoldenrod'})
plt.xticks(fontsize=14)
plt.ylabel('Validation accuracy', fontsize=14)
plt.ylim([0, 0.85])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(SAVEPATH + 'boxplot_binning_benchmark.pdf', dpi=1000, bbox_inches="tight")
plt.savefig(SAVEPATH + 'boxplot_binning_benchmark.png', dpi=1000, bbox_inches="tight")
plt.show()

# Plot the Hazard Ratios for random and non-Random data
HZ_df = pd.read_csv(DATAPATH + 'Results/HazardRatios_2.csv', index_col=0, header=0).sort_values(by='Gene',
                                                                                      ascending=False)#.dropna(axis=0)
HZ_df = HZ_df.loc[(HZ_df['N_samples'] > 30) & (HZ_df['N_samples'] < 1868)]
HZ_df_random = pd.read_csv(DATAPATH + 'Results/HazardRatios2_random.csv', index_col=0, header=0).sort_values(by='Hazard Ratio',
                                                                                                    ascending=False)
HZ_df_random = HZ_df_random.loc[(HZ_df_random['N_samples'] > 30) &
                                (HZ_df_random['N_samples'] < 1868)]


bin_seq = np.linspace(np.min(HZ_df), np.max(HZ_df), num=200)
fig, ax = plt.subplots(figsize=(8, 6))
HZ_df['Hazard Ratio'].hist(ax=ax, bins=bin_seq, label='Three Regimes', c='tab:orange')
HZ_df_random['Hazard Ratio'].hist(ax=ax, bins=bin_seq, label='Random', c='tab:green')

plt.xlabel('Hazard Ratio', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=14)
plt.show()

# Plot for comparing the association rules
SAVE = False
n_reg = str(2)
topn = 1000
max_n = 2e4
step = 1

corr_df_META = pd.read_csv(DATAPATH + 'correlations_gx_META.csv', header=0)
corr_df_TCGA = pd.read_csv(DATAPATH + 'correlations_gx_TCGA.csv', header=0)
pval_gx_TCGA = pd.read_csv(DATAPATH + 'pvals_gx_' + n_reg + '_regimes_TCGA.csv', header=0)
pval_gx_META = pd.read_csv(DATAPATH + 'pvals_gx_' + n_reg + '_regimes_META.csv', header=0)

MI_META_filtered = pd.read_csv(DATAPATH + 'MI_gx_META_' + n_reg + 'reg_filtered.csv', header=0)
MI_TCGA_filtered = pd.read_csv(DATAPATH + 'MI_gx_TCGA_' + n_reg + 'reg_filtered.csv', header=0)

pval_list_TCGA_filtered = pd.read_csv(DATAPATH + 'pvals_TCGA_' + n_reg + 'reg_filtered.csv')
pval_list_META_filtered = pd.read_csv(DATAPATH + 'pvals_META_' + n_reg + 'reg_filtered.csv')

pval_list_TCGA = pd.read_csv(DATAPATH + 'pvals_TCGA_' + n_reg + 'reg.csv')
pval_list_META = pd.read_csv(DATAPATH + 'pvals_META_' + n_reg + 'reg.csv')

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
plt.figure(figsize=(15, 8))
tab_colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

max_ = 0
cs = {model: tab_colors[i] for i, model in enumerate(overlap.keys())}

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

if SAVE:
    print('Saving association figures.')
    plt.savefig(SAVEPATH + 'gx_associations_' + n_reg + 'reg.pdf', dpi=1000, bbox_inches="tight")
    plt.savefig(SAVEPATH + 'gx_associations_' + n_reg + 'reg.png', dpi=1000, bbox_inches="tight")

plt.show()

# Script for doing absolutely nothing
SAVE = False
n_reg = str(2)
topn = 1000
max_n = 2e4
step = 1

MI_TCGA_gx_mut = pd.read_csv(DATAPATH + 'MI_TCGA_gx_mut_' + n_reg + 'reg.csv', header=0)
MI_META_gx_mut = pd.read_csv(DATAPATH + 'MI_META_gx_mut_' + n_reg + 'reg.csv', header=0)

pvals_gx_mut_META = pd.read_csv(DATAPATH + 'pvals_gx_mut_META_' + n_reg + 'reg.csv', header=0)
pvals_gx_mut_TCGA = pd.read_csv(DATAPATH + 'pvals_gx_mut_TCGA_' + n_reg + 'reg.csv', header=0)

pvals_gx_mut_META.Gene_A = [s[:-2] for s in pvals_gx_mut_META.Gene_A]
pvals_gx_mut_META.Gene_B = [s[:-2] for s in pvals_gx_mut_META.Gene_B]
pvals_gx_mut_TCGA.Gene_A = [s[:-2] for s in pvals_gx_mut_TCGA.Gene_A]
pvals_gx_mut_TCGA.Gene_B = [s[:-2] for s in pvals_gx_mut_TCGA.Gene_B]

pvals_gx_mut_META_unfiltered = pd.read_csv(DATAPATH + 'pvals_gx_mut_META_' + n_reg + 'reg_unfiltered.csv', header=0)
pvals_gx_mut_TCGA_unfiltered = pd.read_csv(DATAPATH + 'pvals_gx_mut_TCGA_' + n_reg + 'reg_unfiltered.csv', header=0)

pvals_gx_mut_META_unfiltered.Gene_A = [s[:-2] for s in pvals_gx_mut_META_unfiltered.Gene_A]
pvals_gx_mut_META_unfiltered.Gene_B = [s[:-2] for s in pvals_gx_mut_META_unfiltered.Gene_B]
pvals_gx_mut_TCGA_unfiltered.Gene_A = [s[:-2] for s in pvals_gx_mut_TCGA_unfiltered.Gene_A]
pvals_gx_mut_TCGA_unfiltered.Gene_B = [s[:-2] for s in pvals_gx_mut_TCGA_unfiltered.Gene_B]

overlap = {}
overlap['Co-occurrence'] = getOverlap(pvals_gx_mut_META_unfiltered.iloc[:topn, :],
                                               pvals_gx_mut_TCGA_unfiltered, max_n=max_n, step=step)

overlap['Co-occurrence filtered'] = getOverlap(pvals_gx_mut_META.iloc[:topn, :],
                                               pvals_gx_mut_TCGA, max_n=max_n, step=step)

overlap['MI filtered'] = getOverlap(MI_META_gx_mut.iloc[:topn, :],
                                    MI_TCGA_gx_mut, max_n=max_n, step=step)

plt.style.use('ggplot')
plt.figure(figsize=(15, 8))
tab_colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

max_ = 0
cs = {model: tab_colors[i] for i, model in enumerate(overlap.keys())}

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

if SAVE:
    print('Saving association figures.')
    plt.savefig(SAVEPATH + 'gx_associations_' + n_reg + 'reg.pdf', dpi=1000, bbox_inches="tight")
    plt.savefig(SAVEPATH + 'gx_associations_' + n_reg + 'reg.png', dpi=1000, bbox_inches="tight")

plt.show()

## Plotting the copy numbers versus the gene expression
SAVE = False
n_reg = str(3)
topn = 1000
max_n = 2e4
step = 1

#MI_TCGA_gx_cna = pd.read_csv(DATAPATH + 'MI_TCGA_gx_cna_' + n_reg + 'reg.csv', header=0)
#MI_META_gx_cna = pd.read_csv(DATAPATH + 'MI_META_gx_cna_' + n_reg + 'reg.csv', header=0)
pvals_gx_cna_META = pd.read_csv(DATAPATH + 'pvals_gx_cna_META_' + n_reg + 'reg.csv', header=0)
pvals_gx_cna_TCGA = pd.read_csv(DATAPATH + 'pvals_gx_cna_TCGA_' + n_reg + 'reg.csv', header=0)

pvals_gx_cna_META.Gene_A = [s[:-2] for s in pvals_gx_cna_META.Gene_A]
pvals_gx_cna_META.Gene_B = [s[:-2] for s in pvals_gx_cna_META.Gene_B]
pvals_gx_cna_TCGA.Gene_A = [s[:-2] for s in pvals_gx_cna_TCGA.Gene_A]
pvals_gx_cna_TCGA.Gene_B = [s[:-2] for s in pvals_gx_cna_TCGA.Gene_B]

pvals_gx_cna_META = pvals_gx_cna_META.loc[pvals_gx_cna_META.Gene_A == pvals_gx_cna_META.Gene_B]
pvals_gx_cna_TCGA = pvals_gx_cna_TCGA.loc[pvals_gx_cna_TCGA.Gene_A == pvals_gx_cna_TCGA.Gene_B]

overlap = {}
overlap['Co-occurrence filtered'] = getOverlap(pvals_gx_cna_META.iloc[:topn, :],
                                               pvals_gx_cna_TCGA, max_n=max_n, step=step)

#overlap['MI filtered'] = getOverlap(MI_META_gx_cna.iloc[:topn, :], MI_TCGA_gx_cna, max_n=max_n, step=step)

plt.style.use('ggplot')
plt.figure(figsize=(15, 8))
tab_colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

max_ = 0
cs = {model: tab_colors[i] for i, model in enumerate(overlap.keys())}

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

if SAVE:
    print('Saving association figures.')
    plt.savefig(SAVEPATH + 'gx_associations_' + n_reg + 'reg.pdf', dpi=1000, bbox_inches="tight")
    plt.savefig(SAVEPATH + 'gx_associations_' + n_reg + 'reg.png', dpi=1000, bbox_inches="tight")

plt.show()

DATAPATH = '/home/bioinformatics/mlarmuse/Documents/CAMDA_challenge/Paper/Second_Submission/'
