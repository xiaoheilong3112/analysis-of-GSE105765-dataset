import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

# Load the data
expression_data = pd.read_csv("/mnt/data/GSE105765_miRNA_count.txt", sep="\t")

# Check the basic information of the data
expression_data.info()

# Check the distribution of the data
expression_data.describe()

# Generate a histogram of the miRNA expression values
plt.hist(expression_data.drop('miRNA_Name', axis=1).values.flatten(), bins=100)
plt.show()

# Generate a box plot of the miRNA expression values
plt.boxplot(expression_data.drop('miRNA_Name', axis=1).values.flatten())
plt.show()

# Normalize the data using TPM method
expression_data_normalized = expression_data.copy()
expression_data_normalized.iloc[:, 1:] = expression_data.iloc[:, 1:].apply(lambda x: x / x.sum() * 10**6)

# Generate a histogram of the normalized miRNA expression values
plt.hist(expression_data_normalized.drop('miRNA_Name', axis=1).values.flatten(), bins=100)
plt.show()

# Perform hierarchical clustering
linked = linkage(expression_data_normalized.drop('miRNA_Name', axis=1).transpose(), 'ward')

# Generate a heatmap of the clustered data
sns.clustermap(expression_data_normalized.drop('miRNA_Name', axis=1).transpose(), row_linkage=linked)
plt.show()

# Calculate the Pearson correlation coefficients
corr = expression_data_normalized.drop('miRNA_Name', axis=1).transpose().corr()

# Generate a heatmap of the correlation matrix
sns.heatmap(corr)
plt.show()

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(expression_data_normalized.drop('miRNA_Name', axis=1).transpose())

# Generate a scatter plot of the first two principal components
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.show()

# Perform t-test for differential expression analysis
p_values = ttest_ind(expression_data_normalized.filter(regex='^EC', axis=1),
                     expression_data_normalized.filter(regex='^EU', axis=1),
                     axis=1, equal_var=False, nan_policy='omit')[1]

# Correct for multiple testing (Bonferroni correction)
corrected_p_values = p_values * len(p_values)

# Add the p-values to the DataFrame
expression_data_normalized['p_value'] = p_values
expression_data_normalized['corrected_p_value'] = corrected_p_values

# Display the miRNAs with significant differences
expression_data_normalized[expression_data_normalized['corrected_p_value'] < 0.05]
