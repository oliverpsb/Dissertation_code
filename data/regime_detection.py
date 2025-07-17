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

    def plot_scree(self, save_path=None, cumulative=False):
        """
        Plot a scree plot showing explained variance ratio and optionally cumulative variance.
        """
        if not hasattr(self, 'data_scaled'):
            raise RuntimeError("You must call standardize() before plotting scree plot.")

        full_pca = PCA()
        full_pca.fit(self.data_scaled)
        explained = full_pca.explained_variance_ratio_
        cumulative_var = explained.cumsum()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained) + 1), explained, marker='o', label='Explained Variance')
        if cumulative:
            plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='x', linestyle='--', label='Cumulative Variance')

        plt.title('Scree Plot of Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(range(1, len(explained) + 1))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()



class RegimeHMM():
    def __init__(self, pca_output, n_regimes=4, n_iterations=10000, covariance_type="full", simulate=False):
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
        self.simulate = simulate
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
        return np.round(self.hmm_model.transmat_, 3)
    
    def plot_pc_with_regimes(self, title, n_pca_components):
        """
        Plot the principal components with shaded regimes.
        param: title - title of the plot.
        """
        if self.simulate:
            saved_place_name = f"strategy_output/pc{n_pca_components}_with_regimes.png"
        else:
            saved_place_name = f"analysis_output/pc{n_pca_components}_with_regimes.png"

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
        plt.savefig(saved_place_name, dpi=300)
        # Save the plot


    def compare_bic(self, pca_input=None, min_regimes=2, max_regimes=6, output_path="analysis_output/hmm_bic_comparison.csv"):
        """
        Compare BIC scores for different regime counts
        """
        if pca_input is None:
            pca_input = self.pca_output.drop(columns='Regime', errors='ignore')

        n_obs = len(pca_input)
        n_features = pca_input.shape[1]
        bic_scores = {}

        for n in range(min_regimes, max_regimes + 1):
            hmm = GaussianHMM(n_components=n, covariance_type=self.covariance_type,
                            n_iter=self.iterations, random_state=42)
            hmm.fit(pca_input)
            log_likelihood = hmm.score(pca_input)

            # Parameter count
            n_transition = n * (n - 1)
            n_startprob = n - 1
            n_means = n * n_features
            n_covars = n * (n_features * (n_features + 1)) / 2
            n_params = n_transition + n_startprob + n_means + n_covars

            bic = -2 * log_likelihood + n_params * np.log(n_obs)
            bic_scores[n] = bic

        pd.Series(bic_scores, name='BIC').to_csv(output_path)
        print(f"BIC comparison saved to {output_path}")

