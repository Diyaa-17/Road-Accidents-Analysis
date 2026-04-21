import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

def perform_statistical_analysis(df):
    """
    Computes rigorous descriptive and inferential statistics 
    including T-tests, Chi-Squared, Shapiro-Wilk, and VIF.
    """
    print("\n[Descriptive Statistics]")
    print(df.describe().T)

    # 2. Shapiro-Wilk Test (Normality of 'speed_limit')
    # Use a sample of max 5000 as shapiro might struggle with large N
    print("\n[Shapiro-Wilk Test for Normality]")
    sample_data = df['speed_limit'].dropna().sample(min(5000, len(df.dropna())))
    stat, p = stats.shapiro(sample_data)
    print(f"Statistics={stat:.3f}, p-value={p:.3f}")
    if p > 0.05:
        print("Sample looks Gaussian (fail to reject H0)")
    else:
        print("Sample does not look Gaussian (reject H0)")

    # 3. T-Test (Difference in speed limits between Fatal(1) and Slight(3) accidents)
    print("\n[Independent T-Test]")
    fatal_speeds = df[df['accident_severity'] == 1]['speed_limit'].dropna()
    slight_speeds = df[df['accident_severity'] == 3]['speed_limit'].dropna()
    # If samples exist, compute t-test
    if len(fatal_speeds) > 1 and len(slight_speeds) > 1:
        t_stat, t_p = stats.ttest_ind(fatal_speeds, slight_speeds)
        print(f"T-statistic={t_stat:.3f}, p-value={t_p:.3f}")
    
    # 4. Chi-Squared Test (Relationship between weather and severity)
    print("\n[Chi-Squared Test]")
    contingency = pd.crosstab(df['weather_conditions'], df['accident_severity'])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    print(f"Chi2 Stat={chi2:.3f}, p-value={p_chi:.3f}")

    # 5. Variance Inflation Factor (VIF for multicollinearity)
    print("\n[Variance Inflation Factor (VIF)]")
    # Calculate VIF to identify any multicollinearity among predictor variables
    X_vif = df.drop(columns=['accident_severity']).dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    print(vif_data)
    
    print("\nStatistical Analysis Complete.")
