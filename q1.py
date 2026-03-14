from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from static import FEATURES
from utils import read_dataframe

SCALED_FEATURES = [f'{c}_n' for c in FEATURES]

def read_standarized_data() -> pd.DataFrame:
    iris_df = read_dataframe()
    # TODO: Normalize the features
    scaler = StandardScaler()
    iris_df[SCALED_FEATURES] = scaler.fit_transform(iris_df[FEATURES])
    return iris_df

def run_elbow_method():
    standardized_df = read_standarized_data()
    mse_data = []
    # TODO: Run KMeans for k = 1 to 10 and calculate the MSE (inertia) for each k
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=5963, n_init='auto')
        kmeans.fit(standardized_df[SCALED_FEATURES])
        mse_data.append(kmeans.inertia_)

    # TODO: Visualize the result with a line plot (k on x-axis and MSE on y-axis)
    elbow_df = pd.DataFrame({'K': range(1, 11), 'MSE': mse_data})
    fig = px.line(elbow_df, x='K', y='MSE',
                  title='Elbow Method for KMeans Clustering',
                  labels={'K': 'Number of Clusters', 'MSE': 'Mean Squared Error (Inertia)'},
                  template='simple_white')
    fig.show()


def run_kmeans(k: int = 3):
    standardized_df = read_standarized_data()
    # TODO: Run KMeans with k clusters and get the cluster labels for each data point
    kmeans = KMeans(n_clusters=k, random_state=5963, n_init='auto')
    standardized_df['cluster'] = kmeans.fit_predict(standardized_df[SCALED_FEATURES])
    standardized_df['cluster'] = standardized_df['cluster'].apply(lambda x: f'cluster #{x + 1}')

    # TODO: Visualize the result with a scatter plot (Petal_length on x-axis and Sepal_length on y-axis, color by cluster)
    fig = px.scatter(standardized_df,
                     x='Petal_length', y='Sepal_length',
                     color='cluster',
                     title='KMeans Clustering with k=3',
                     labels={'Petal_length': 'Petal Length', 'Sepal_length': 'Sepal Length'},
                     template='simple_white')
    fig.show()


if __name__ == '__main__':
    print('[Q1][Part 1] The normalized dataframe looks like this:')
    print(read_standarized_data().head())
    print('[Q1][Part 2] Plot a line chart to show how to find the best K using the Elbow method')
    run_elbow_method()
    print('[Q1][Part 3] Visualize K-means with k=3')
    run_kmeans(k=3)
