import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM

class MacroPCA:
    def __init__(self, data, n_components=None, frequency='ME'):
        """
        Initialse the PCA wrapper.
        param: data - dataframe of macroeconomic indicators
        param: n_components - number of principal components to keep (default: all)
        param: frequency - resampling frequency ('ME' = monthly, 'QE' = quarterly, 'YE' = yearly)
        """
        self.original_data = data
        self.n_components = n_components
        self.frequency = frequency.upper()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.fitted = False

    def standardise(self):
        if self.frequency not in ['ME', 'QE', 'YE']:
            raise ValueError("Frequency must be 'ME', 'QE', or 'YE'")
        self.data = self.original_data.resample(self.frequency).mean()

        self.data_scaled = self.scaler.fit_transform(self.data)
        return self.data_scaled

    def run_pca(self):
        if not hasattr(self, 'data_scaled'):
            raise ValueError("You must call standardize() before run_pca().")

        self.components = self.pca.fit_transform(self.data_scaled)
        self.explained_variance = self.pca.explained_variance_ratio_
        self.loadings = pd.DataFrame(self.pca.components_.T,
                                     index=self.data.columns,
                                     columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])
        self.fitted = True
        return pd.DataFrame(self.components,
                            columns=self.loadings.columns,
                            index=self.data.index)

    def get_explained_variance(self):
        if not self.fitted:
            raise RuntimeError("PCA has not been run yet.")
        return self.explained_variance

    def get_loadings(self):
        if not self.fitted:
            raise RuntimeError("PCA has not been run yet.")
        return self.loadings


class RegimeHMM():
    def __init__(self, pca_output, n_regimes=4, n_iterations=10000, covariance_type="full"):
        """
        Initialise HMM model.
        param: pca_output - dataframe of the PCA output on macro indicators.
        param: n_regimes - number of latent regimes.
        param: n_iterations - maximum number of iterations to perform (default: 10000).
        param: covariance_type - {'spherical', 'diag', 'full', 'tied'} The type of covariance parameters to use:
                                    - spherical: each state uses a single variane value that applies to all features
                                    - diag: each state uses a diagonal covariance matrix
                                    - full: each state uses a full (i.e. unrestricted covariance matrix) (default)
                                    - tied: all mixture components of each state uses the same full covariance matrix

        """

        self.pca_output = pca_output
        self.n_regimes = n_regimes
        self.iterations = n_iterations
        self.covariance_type = covariance_type
        self.hmm_model = GaussianHMM(n_components=self.n_regimes, 
                                     covariance_type=self.covariance_type, 
                                    n_iter=self.iterations, 
                                    random_state=42)
        
    def fit(self):
        "Fit the HMM model to the PCA output."
        self.hmm_model.fit(self.pca_output)
        regime_labels = self.hmm_model.predict(self.pca_output) # Predict regime labels
        self.pca_output['Regime'] = regime_labels # Add regime labels to DataFrame
        return self.pca_output
    
    def get_transition_matrix(self):
        "Get the transition matrix of the HMM model."
        print("HMM Transition Matrix:")
        return print(np.round(self.hmm_model.transmat_, 3))
    
    def plot_pc_with_regimes(self, title):
        """
        Plot the principal components with shaded regimes.
        param: title - title of the plot.
        """
        plt.figure(figsize=(16, 7))
        regime_colors = plt.cm.Set1(np.arange(self.n_regimes))

        # Plot each principal component
        for i in range(1, self.n_regimes + 1):
            plt.plot(self.pca_output.index, self.pca_output[f'PC{i}'], label=f'PC{i}', linewidth=1)

        # Shade regimes
        prev_regime = self.pca_output['Regime'].iloc[0]
        start_date = self.pca_output.index[0]
        used_patches = {}

        for i in range(1, len(self.pca_output)):
            current_regime = self.pca_output['Regime'].iloc[i]
            if current_regime != prev_regime:
                color = regime_colors[prev_regime]
                plt.axvspan(start_date, self.pca_output.index[i], color=color, alpha=0.2)
                if prev_regime not in used_patches:
                    used_patches[prev_regime] = mpatches.Patch(color=color, label=f'Regime {prev_regime}')
                start_date = self.pca_output.index[i]
                prev_regime = current_regime

        # Final span
        color = regime_colors[prev_regime]
        plt.axvspan(start_date, self.pca_output.index[-1], color=color, alpha=0.2)
        if prev_regime not in used_patches:
            used_patches[prev_regime] = mpatches.Patch(color=color, label=f'Regime {prev_regime}')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Principal Component Value')
        plt.legend(handles=list(used_patches.values()))
        plt.grid(True)
        plt.tight_layout()

        plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gcf().autofmt_xdate()

        pc_legend = plt.legend(loc='upper left')
        plt.gca().add_artist(pc_legend)  # Show both legends
        plt.legend(handles=list(used_patches.values()), loc='upper right', title='Regimes')

        # Save the plot
        plt.savefig('pca_with_regimes.png', dpi=300)
