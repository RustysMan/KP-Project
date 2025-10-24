import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define analysis functions
def load_and_clean_data(file_path):
    """
    Loads data from an Excel file and performs initial cleaning.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = pd.read_excel(file_path)

    # Remove the header row within the data and handle missing values
    df = df[df['HARGA'].astype(str) != 'HARGA'].copy()
    df['HARGA'] = df['HARGA'].astype(str).str.replace('Rp', '', regex=False).str.replace('.', '', regex=False)
    df['HARGA'] = pd.to_numeric(df['HARGA'], errors='coerce')
    df = df.dropna(subset=['HARGA']) # Drop rows where HARGA could not be converted

    # Format date column
    df['TANGGAL '] = pd.to_datetime(df['TANGGAL '])

    # Add 'Hari' column
    df['Hari'] = df['TANGGAL '].dt.day_name()

    # Rename columns first
    df = df.rename(columns={'MESS': 'Pagi', 'Unnamed: 3': 'Siang', 'Unnamed: 4': 'Malam', 'JUMLAH': 'Jumlah', 'TANGGAL ': 'Tanggal'})

    # Ensure quantity columns are numeric AFTER renaming
    columns_to_numeric = ['Pagi', 'Siang', 'Malam', 'Jumlah']
    for col in columns_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any remaining completely empty rows and duplicates
    df.dropna(how='all', inplace=True)
    df.drop_duplicates(inplace=True)


    return df

def create_features(df):
    """
    Creates features for clustering from the cleaned DataFrame.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    df_features = pd.DataFrame()

    df_features['Avg_Pagi'] = df.groupby('Tanggal')['Pagi'].mean()
    df_features['Avg_Siang'] = df.groupby('Tanggal')['Siang'].mean()
    df_features['Avg_Malam'] = df.groupby('Tanggal')['Malam'].mean()
    df_features['Total_Pesanan'] = df.groupby('Tanggal')['Jumlah'].sum()
    df_features['Total_Pendapatan'] = df.groupby('Tanggal')['TOTAL HARGA RUPIAH'].sum()
    df_features['Hari'] = df.groupby('Tanggal')['Hari'].first()


    return df_features

def scale_and_impute_features(df_features):
    """
    Scales and imputes the features for clustering.

    Args:
        df_features (pd.DataFrame): DataFrame with engineered features.

    Returns:
        pd.DataFrame: Scaled and imputed DataFrame.
    """
    columns_to_normalize = ['Avg_Pagi', 'Avg_Siang', 'Avg_Malam', 'Total_Pesanan', 'Total_Pendapatan']

    scaler = StandardScaler()
    df_features_scaled = scaler.fit_transform(df_features[columns_to_normalize])

    imputer = SimpleImputer(strategy='mean')
    df_features_scaled_imputed = imputer.fit_transform(df_features_scaled)

    df_features_scaled_imputed = pd.DataFrame(df_features_scaled_imputed, columns=columns_to_normalize, index=df_features.index)

    return df_features_scaled_imputed

def perform_clustering(df_scaled_imputed, n_clusters):
    """
    Performs K-Means clustering on the scaled and imputed data.

    Args:
        df_scaled_imputed (pd.DataFrame): The scaled and imputed DataFrame.
        n_clusters (int): The number of clusters to use.

    Returns:
        np.ndarray: The cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_scaled_imputed)
    clusters = kmeans.labels_
    return clusters

def plot_daily_revenue_trend(df):
    """
    Plots the daily revenue trend.

    Args:
        df (pd.DataFrame): The cleaned DataFrame with 'Tanggal' and 'TOTAL HARGA RUPIAH'.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Tanggal'], df['TOTAL HARGA RUPIAH'])
    ax.set_title('Daily Revenue Trend')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Total Harga Rupiah')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_average_orders_per_day(df):
    """
    Plots the average orders per day of the week.

    Args:
        df (pd.DataFrame): The cleaned DataFrame with 'Hari' and 'Jumlah'.
    """
    average_orders_per_day = df.groupby('Hari')['Jumlah'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    fig, ax = plt.subplots(figsize=(10, 6))
    average_orders_per_day.plot(kind='bar', ax=ax)
    ax.set_title('Average Orders per Day of the Week')
    ax.set_xlabel('Hari')
    ax.set_ylabel('Rata-rata Jumlah Pesanan')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_mealtime_orders_heatmap(df):
    """
    Plots a heatmap of total orders per mealtime and day of the week.

    Args:
        df (pd.DataFrame): The cleaned DataFrame with 'Hari', 'Pagi', 'Siang', 'Malam'.
    """
    mealtime_orders_per_day = df.groupby('Hari')[['Pagi', 'Siang', 'Malam']].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(mealtime_orders_per_day, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title('Total Orders per Mealtime and Day of the Week')
    ax.set_xlabel('Waktu Makan')
    ax.set_ylabel('Hari')
    plt.tight_layout()
    return fig

def plot_cluster_scatter(df_clustered_filtered):
    """
    Plots a scatter plot of clusters based on Total Orders and Total Revenue for filtered data.

    Args:
        df_clustered_filtered (pd.DataFrame): Filtered DataFrame with original and scaled/imputed features and cluster labels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Total_Pesanan_original', y='Total_Pendapatan_original', hue='Cluster', data=df_clustered_filtered, palette='viridis', ax=ax)
    ax.set_title('Cluster Distribution based on Total Orders and Total Revenue')
    ax.set_xlabel('Total Orders')
    ax.set_ylabel('Total Revenue')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_cluster_revenue_boxplot(df_clustered_filtered):
    """
    Plots boxplot of Total Revenue per Cluster for filtered data.

    Args:
        df_clustered_filtered (pd.DataFrame): Filtered DataFrame with original features and cluster labels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Total_Pendapatan_original', data=df_clustered_filtered, ax=ax)
    ax.set_title('Distribution of Total Revenue per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Total Revenue')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_cluster_mealtime_boxplot(df_clustered_filtered):
    """
    Plots boxplots of average mealtime orders per cluster for filtered data.

    Args:
        df_clustered_filtered (pd.DataFrame): Filtered DataFrame with original features and cluster labels.
    """
    df_mealtime_melted = df_clustered_filtered.melt(id_vars=['Cluster'], value_vars=['Avg_Pagi_original', 'Avg_Siang_original', 'Avg_Malam_original'], var_name='Mealtime', value_name='Average Orders')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Cluster', y='Average Orders', hue='Mealtime', data=df_mealtime_melted, ax=ax)
    ax.set_title('Distribution of Average Mealtime Orders per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Average Orders')
    ax.grid(True)
    plt.tight_layout()
    return fig


# Streamlit App Layout
st.set_page_config(layout="wide")

st.title("Dashboard Analisis Data Transaksi Catering")

st.markdown("""
Dashboard ini menyajikan analisis data transaksi harian perusahaan catering.
Meliputi statistik deskriptif, tren waktu, dan segmentasi pelanggan berdasarkan pola pesanan.
""")

# Load and clean data
# Use a relative path to the data file
file_path = "Rekap Data Catering.xlsx"
df_cleaned = load_and_clean_data(file_path)

# Add date range slider
st.sidebar.header("Filter Data")
min_date = df_cleaned['Tanggal'].min().date()
max_date = df_cleaned['Tanggal'].max().date()
date_range = st.sidebar.slider(
    "Pilih Rentang Tanggal",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Filter data by date range
start_date, end_date = date_range
df_filtered_date = df_cleaned[(df_cleaned['Tanggal'].dt.date >= start_date) & (df_cleaned['Tanggal'].dt.date <= end_date)].copy()


# Display Descriptive Statistics
st.header("Statistik Deskriptif")
st.write("Ringkasan statistik data transaksi (difilter berdasarkan tanggal):")
st.dataframe(df_filtered_date.describe())

# Display Time Series Visualizations
st.header("Visualisasi Tren Waktu")

st.write("Tren Total Pendapatan Harian (difilter berdasarkan tanggal):")
st.pyplot(plot_daily_revenue_trend(df_filtered_date))

st.write("Rata-rata Jumlah Pesanan per Hari (difilter berdasarkan tanggal):")
st.pyplot(plot_average_orders_per_day(df_filtered_date))

st.write("Total Pesanan per Waktu Makan dan Hari (difilter berdasarkan tanggal):")
st.pyplot(plot_mealtime_orders_heatmap(df_filtered_date))


# Perform Clustering
st.header("Hasil Clustering K-Means")

st.write("Melakukan K-Means Clustering untuk segmentasi hari berdasarkan pola transaksi.")

# Create features from date-filtered data
df_features = create_features(df_filtered_date)

# Scale and impute features
df_features_scaled_imputed = scale_and_impute_features(df_features)

# Determine optimal clusters (assuming 3 based on previous analysis)
optimal_clusters = 3

# Perform clustering
cluster_labels = perform_clustering(df_features_scaled_imputed, optimal_clusters)

# Add cluster labels to the original features dataframe
df_features['Cluster'] = cluster_labels

# Merge original and scaled/imputed features with cluster labels for plotting
df_clustered = pd.merge(df_features, df_features_scaled_imputed, left_index=True, right_index=True, suffixes=('_original', '_scaled_imputed'))


# Add cluster selection
all_clusters = sorted(df_clustered['Cluster'].unique())
selected_clusters = st.sidebar.multiselect(
    "Pilih Cluster untuk Ditampilkan",
    options=all_clusters,
    default=all_clusters
)

# Filter clustered data by selected clusters
df_clustered_filtered = df_clustered[df_clustered['Cluster'].isin(selected_clusters)].copy()


# Display Clustering Visualizations
st.write("Distribusi Cluster berdasarkan Total Pesanan dan Total Pendapatan (difilter berdasarkan cluster):")
st.pyplot(plot_cluster_scatter(df_clustered_filtered))

st.write("Distribusi Total Pendapatan per Cluster (difilter berdasarkan cluster):")
st.pyplot(plot_cluster_revenue_boxplot(df_clustered_filtered))

st.write("Distribusi Rata-rata Pesanan per Waktu Makan per Cluster (difilter berdasarkan cluster):")
st.pyplot(plot_cluster_mealtime_boxplot(df_clustered_filtered))

# Interpretation and Business Recommendations
st.header("Interpretasi dan Rekomendasi Bisnis")

st.markdown("""
Berdasarkan analisis data dan hasil clustering, ditemukan beberapa pola menarik:

**Interpretasi Cluster:**
- **Cluster 0:** Merepresentasikan hari-hari dengan volume pesanan dan pendapatan harian yang rendah. Hari-hari dalam cluster ini mungkin adalah hari kerja biasa dengan pesanan yang lebih sedikit atau periode dengan permintaan rendah.
- **Cluster 1:** Merepresentasikan hari-hari dengan volume pesanan dan pendapatan harian yang tinggi. Hari-hari dalam cluster ini kemungkinan adalah akhir pekan atau hari-hari dengan acara khusus yang meningkatkan pesanan.
- **Cluster 2:** Merepresentasikan hari-hari dengan volume pesanan dan pendapatan harian moderat. Ini mungkin adalah hari-hari kerja dengan pesanan standar.

**Rekomendasi Bisnis:**
- **Untuk Cluster 0 (Hari dengan Pesanan Rendah):** Pertimbangkan promosi khusus atau diskon pada hari-hari ini untuk meningkatkan volume pesanan. Analisis lebih lanjut dapat dilakukan untuk memahami mengapa hari-hari ini memiliki pesanan rendah.
- **Untuk Cluster 1 (Hari dengan Pesanan Tinggi):** Pastikan ketersediaan stok dan staf yang memadai untuk memenuhi permintaan yang tinggi pada hari-hari ini. Pertimbangkan strategi harga premium atau paket khusus untuk memaksimalkan pendapatan.
- **Untuk Cluster 2 (Hari dengan Pesanan Moderat):** Pertahankan operasi standar dan pantau tren untuk mengidentifikasi peluang peningkatan pesanan.

Analisis pola pesanan berdasarkan waktu makan (Pagi, Siang, Malam) dalam setiap cluster juga dapat memberikan wawasan tambahan untuk mengoptimalkan menu dan jadwal produksi.
""")
