import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

def perform_eda(df):
    """
    Generates exploratory data analysis plots to identify trends
    and saves them to the output folder.
    """
    print("Starting Exploratory Data Analysis & Visualizations...")
    
    out_dir = "output_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Accident Severity Distribution
    plt.figure(figsize=(8, 6))
    severity_counts = df['accident_severity'].value_counts().sort_index()
    # Map index from codes to labels for the plot
    severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    severity_counts.index = severity_counts.index.map(severity_labels)
    
    sns.barplot(x=severity_counts.index, y=severity_counts.values, palette='viridis')
    plt.title('Distribution of Accident Severities')
    plt.xlabel('Severity')
    plt.ylabel('Number of Accidents')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'severity_distribution.png'))
    plt.close()
    
    # 2. Accidents by Weather Conditions
    plt.figure(figsize=(10, 6))
    sns.countplot(x='weather_conditions', hue='accident_severity', data=df, palette='Set2')
    plt.title('Accident Severity vs Weather Conditions')
    plt.xlabel('Weather Condition Code')
    plt.ylabel('Count')
    plt.legend(title='Severity', labels=['Fatal (1)', 'Serious (2)', 'Slight (3)'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weather_vs_severity.png'))
    plt.close()
    
    # 3. Accidents by Day of Week
    plt.figure(figsize=(10, 6))
    sns.countplot(x='day_of_week', data=df, palette='coolwarm')
    plt.title('Total Accidents by Day of Week')
    plt.xlabel('Day of Week (1=Sunday, 7=Saturday)')
    plt.ylabel('Total Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'day_of_week_counts.png'))
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='RdBu', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 5. Covariance Matrix
    plt.figure(figsize=(10, 8))
    cov = df.cov()
    sns.heatmap(cov, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Feature Covariance Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'covariance_matrix.png'))
    plt.close()
    
    # 6. Outlier Detection (Boxplots)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['speed_limit'], color='skyblue')
    plt.title('Outlier Detection: Speed Limit Boxplot')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'speed_limit_boxplot.png'))
    plt.close()

    # 7. Bell Curve (Normal Distribution)
    plt.figure(figsize=(8, 6))
    data = df['speed_limit'].dropna()
    sns.histplot(data, bins=10, stat='density', alpha=0.4, color='dodgerblue')
    mu, std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit\n(mu={mu:.1f}, std={std:.1f})')
    plt.title('Bell Curve (Normal Distribution) of Speed Limits')
    plt.xlabel('Speed Limit')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'speed_limit_bell_curve.png'))
    plt.close()

    print(f"EDA complete. Visualizations saved into the '{out_dir}' folder.\n")

if __name__ == "__main__":
    # Test block
    import pandas as pd
    sample_df = pd.read_csv("dataset/road_accidents.csv")
    if 'accident_index' in sample_df.columns:
        sample_df = sample_df.drop(columns=['accident_index'])
    perform_eda(sample_df)
