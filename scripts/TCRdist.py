import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import os

from tcrdist.public import _neighbors_fixed_radius
from tcrdist.repertoire import TCRrep
# from tcrdist.tree import TCRtree

from sklearn.manifold import TSNE, MDS
import umap
# import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial import distance


# ## data from '../data/output_data/m5d_donoragg_aaTCRdist3.csv'
m5d_donoragg_aaTCRdist3=pd.read_csv('../data/output_data/m5d_donoragg_aaTCRdist3.csv')

# provide input dataframe as named parameters will make the it converted to numpy.ndarray
# so better use the input dataframe as the first default parameter

# input data required columns (refer to https://tcrdist3.readthedocs.io/en/latest/inputs.html): 'subject', 'cdr3_d_aa', 'v_d_gene', 'j_d_gene', 'count', 'donor_category', 'donor_category3', 'vj_d_gene'

def cdr3_tcrdist_clustering(df, v_gene=None, j_gene=None, length=None, donor_category=None,
                            shuffle=False, method='complete', n_clusters=20, criterion='maxclust', n_neighbors=15,
                            min_dist=0.1, n_components=2, reduction='umap', metric='euclidean', title='m4d',
                            n_jobs=128, random_state=1):
    """Clustering and 2D visualization

    Using different parameters for clustering and dimension reduction, then make the scatter plot colored by cluster index.

    Args:
        df (dataframe): dataframe as input for applying tcrdist3 to calculate distances.
        v_geng (str): default None, can be 'TRDV2' to specifically select cdr3 sequence with TRDV2 segment.
        j_gene (str): default None, can be 'TRDJ1', 'TRDJ2', 'TRDJ3' or 'TRDJ4'.
        length (list): default None, can be a list of numbers indicating the length of cdr3 sequence. i.e. [18] or [18, 19]
        donor_category (str): columns indicates the donor category. Can be 'donor_category' or 'donor_category3' here.
        shuffle (Boolean): default to False, shuffle the dataframe rows or not.
        method (str): default 'complete', from scipy 
                      (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), 
                      can also be 'single', 'average', 'weighted', 'centroid', 'median' or 'ward'.
        n_clusters (int): default 20, number of clusters to assign for all the samples.
        criterion (str): default 'maxclust', from scipy 
                         (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html?highlight=fcluster#scipy.cluster.hierarchy.fcluster), 
                         can also be 'inconsistent', 'distance', 'monocrit' or 'maxclust_momocrit'.
        n_neighbors (int): parameter for umap dimensional reduction.
        min_dist (float): parameter for umap dimensional reduction.
        n_components (int): default 2, parameter for umap dimensional reduction.
        reduction (str): reduction method used for 2d visualization. Can be 'umap', 'tsne' or 'mds'.
        metric (str): default 'euclidean', parameter for umap dimensional reduction.
        title (str): title used to denote the parameters used in the figure.

    Return:
        (tr.clone_df, Z, distance_matrix): df_final (dataframe): dataframe with first two components of the dimension reduction results together with the clustering information in the dataframe; Z: linkage(compressed_dmat, method=method); distance_matrix: distance_matrix = tr.pw_delta.

    """

    # only using trdv2 sequence
    if v_gene:
        v_gene_toselect = v_gene+'*01'
        df = df[df['v_d_gene'] == v_gene_toselect]
    # select if j1 or j3 gene or all the j genes
    if j_gene:
        j_gene_toselect = [j+'*01' for j in j_gene]
        df = df[df['j_d_gene'].isin(j_gene_toselect)]
    # if only using cdr3 sequence with a certain length, df[df['A'].isin([3, 6])]
    if length:
        df = df[df['aa_length'].isin(length)]

    # shuffle your dataframe in-place and reset the index
    if shuffle:
        # print(data.head())
        df_here = df.sample(frac=1, random_state=random_state)
        print(df_here.shape)
    else:
        df_here = df.copy()

# do not use sampling with replacing since it makes no sense when you have a small data, use them all?        
#     if donor_category == 'donor_category':
#         count_df = df_here[donor_category].value_counts().rename_axis(
#             'unique_values').reset_index(name='counts')
#         if np.all(count_df['counts'] > 5000):
#             replace = False
#         else:
#             replace = True

#         sampledf = pd.DataFrame().append([df_here[df_here[donor_category] == 'infant'].sample(5000, random_state=1, replace=replace),
#             df_here[df_here[donor_category] == 'adult'].sample(5000, random_state=1, replace=replace)])

#     elif donor_category == 'donor_category3':
#         count_df = df_here[donor_category].value_counts().rename_axis(
#             'unique_values').reset_index(name='counts')
#         if np.all(count_df['counts'] > 5000):
#             replace = False
#         else:
#             replace = True

#         sampledf = pd.DataFrame().append([df_here[df_here[donor_category] == 'CB'].sample(1500, random_state=1, replace=replace),
#                                           df_here[df_here[donor_category] == 'infant'].sample(
#                                           4000, random_state=1, replace=replace),
#                                           df_here[df_here[donor_category] == 'adult'].sample(4500, random_state=1, replace=replace)])
#     else:
#         if df_here.shape[0] > 10000:
#             replace = False
#         else:
#             replace = True
#         sampledf = df_here.sample(10000, random_state=1, replace=replace)
#     print(sampledf.shape)
    sample_n=df['donor_category3'].value_counts().min() # find out the donor category that contains the least number of records, sample this number of records from each donor category.
    print(sample_n)
    sampledf = pd.DataFrame().append([df_here[df_here['donor_category3'] == 'CB'].sample(sample_n, random_state=random_state), df_here[df_here['donor_category3'] == 'infant'].sample(sample_n, random_state=random_state), df_here[df_here['donor_category3'] == 'adult'].sample(sample_n, random_state=random_state)])

    cell_df = sampledf.copy()[['subject', 'cdr3_d_aa', 'v_d_gene', 'j_d_gene', 'count',
                               'donor_category', 'donor_category3', 'vj_d_gene', 'aa_length',
                               'cdr3aa_publicity', 'publicity', 'publicity3c', 'publicity2', 'publicity3', 'publicity5', 
                               'publicity10', 'publicity20', 'publicity30', 'publicity42', 'cut_quantile_10', 
                               'cut_quantile_15', 'cut_quantile_20', 'cut_quantile_25', 'cut_quantile_50', 
                               'cut_quantile_75', 'cut_quantile_80', 'cut_quantile_85', 'cut_quantile_90', 
                               'cut_quantile_95', 'cut_quantile_8', 'donor_sum', 'aa_freq']]

    tr = TCRrep(cell_df=cell_df,
                organism='human',
                chains=['delta'],
                db_file='alphabeta_gammadelta_db.tsv')
    distance_matrix = tr.pw_delta

    # Hier. Cluster TCR-Distances
    compressed_dmat = distance.squareform(distance_matrix, force="vector")
    Z = linkage(compressed_dmat, method=method)
    print('len(Z): ', len(Z))
    den = dendrogram(Z, color_threshold=np.inf, no_plot=True)
    cluster_index = fcluster(Z, t=n_clusters, criterion=criterion)
    assert len(cluster_index) == tr.clone_df.shape[0]
    assert len(cluster_index) == tr.pw_delta.shape[0]
    tr.clone_df['cluster_index'] = cluster_index
    # dimension reduction:
    if reduction == 'umap':
        X_embedded = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist,
                               metric=metric, n_jobs=n_jobs, random_state=random_state).fit_transform(distance_matrix)

        reduction_df = pd.DataFrame(X_embedded, columns=["umap1", "umap2"])
        print('len(cluster_index): ', len(cluster_index))
    elif reduction == 'tsne':
        tsne_n_neighbors = 30
        X_embedded = TSNE(n_components=2, perplexity=tsne_n_neighbors, random_state=random_state).fit_transform(
            distance_matrix)
        reduction_df = pd.DataFrame(X_embedded, columns=["tsne1", "tsne2"])
    else:
        X_embedded_mds = MDS(n_components=2, dissimilarity='precomputed',
                             n_jobs=n_jobs, random_state=random_state).fit_transform(distance_matrix)
        reduction_df = pd.DataFrame(X_embedded_mds, columns=["mds1", "mds2"])

    # tr.clone_df
    tr.clone_df['ci']=cluster_index
    tr.clone_df['ci'] = tr.clone_df['ci'].astype('category')
    tr.clone_df[reduction+'1'], tr.clone_df[reduction+'2']=reduction_df.umap1, reduction_df.umap2

    # save the output dataframe with the clustering information
    # df_final.to_csv('/home/lihua/Rcode/sequence_pattern/m4dTCRdist_df_final.csv')
    if title:
        if v_gene:
            title = title + v_gene + str(n_clusters) + criterion + str(n_neighbors) + reduction + str(min_dist) + metric
        else:
            title=title + str(n_clusters) + criterion + str(n_neighbors) + reduction + str(min_dist) + metric
    else:
        title=v_gene + str(n_clusters) + criterion + str(n_neighbors) + reduction + str(min_dist) + metric
    fig_name=str(n_clusters) + criterion + str(n_neighbors) + reduction + str(min_dist) + metric + '.pdf'
    # tr.clone_df.to_csv(f'../data/output_data/TCRdist{title}_df_final.csv')

    # plot colored by cluster index
    fig = plt.figure(figsize=(8, 6))
    # plt.figure()
    x = reduction+'1'
    y = reduction+'2'
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="ci", legend="full")
    # place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by cluster index, {title}')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="v_d_gene", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by v_d_gene, {title}')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="j_d_gene", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by j_d_gene, {title}')
    plt.tight_layout()
    plt.show()    
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="vj_d_gene", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by vj_d_gene, {title}')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="donor_category", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by donor_category, {title}')
    plt.tight_layout()
    # plt.savefig(os.path.join(script_path, 'figs', 'donor_category'+fig_name), bbox_inches='tight')
    plt.show()
 
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="donor_category3", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by donor_category3, {title}')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="cdr3aa_publicity", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by cdr3aa_publicity, {title}')
    plt.tight_layout()
    plt.show()
 
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="cut_quantile_8", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by cut_quantile_8, {title}')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tr.clone_df, x=x, y=y, s=10, hue="aa_length", legend="full")
    #place legend in upper left of plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.title(f'colored by aa_length, {title}')
    plt.tight_layout()
    plt.show()
    
    # plt.title(title, fontsize=18)
    return (tr.clone_df, Z, distance_matrix)
    

