# visualization_utils.py
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_tsne_visualization(data):
    # Using TSNE for Dimensionality reduction for 15 Features to 2D
    number_of_obs = 5000
    dfp_subsampled = data[0:number_of_obs]
    X = MinMaxScaler().fit_transform(dfp_subsampled[['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
                                                     'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
                                                     'token_set_ratio', 'token_sort_ratio', 'fuzz_ratio',
                                                     'fuzz_partial_ratio', 'longest_substr_ratio']])
    y = dfp_subsampled['is_duplicate'].values

    tsne2d = TSNE(
        n_components=2,
        init='random',  # pca
        random_state=101,
        method='barnes_hut',
        n_iter=1000,
        verbose=2,
        angle=0.5
    ).fit_transform(X)

    df_tsne = pd.DataFrame({'x': tsne2d[:, 0], 'y': tsne2d[:, 1], 'label': y})

    # Draw the plot
    sns.lmplot(data=df_tsne, x='x', y='y', hue='label', fit_reg=False, size=8, palette="Set1", markers=['s', 'o'])
    plt.title("TSNE 2D Embedding for Engineered Features")
    plt.show()

    # Using TSNE for Dimensionality reduction for 15 Features to 3D
    tsne3d = TSNE(
        n_components=3,
        init='random',  # pca
        random_state=101,
        method='barnes_hut',
        n_iter=1000,
        verbose=2,
        angle=0.5
    ).fit_transform(X)

    trace1 = go.Scatter3d(
        x=tsne3d[:, 0],
        y=tsne3d[:, 1],
        z=tsne3d[:, 2],
        mode='markers',
        marker=dict(
            sizemode='diameter',
            color=y,
            colorscale='Portland',
            colorbar=dict(title='duplicate'),
            line=dict(color='rgb(255, 255, 255)'),
            opacity=0.75
        )
    )

    data_tsne3d = [trace1]
    layout_tsne3d = dict(height=800, width=800, title='3D Embedding with Engineered Features')
    fig_tsne3d = dict(data=data_tsne3d, layout=layout_tsne3d)
    py.iplot(fig_tsne3d, filename='3D_scatter_TSNE')
