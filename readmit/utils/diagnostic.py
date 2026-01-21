import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def compare_real_vs_synthetic(real_df: pd.DataFrame,
                              synthetic_df: pd.DataFrame,
                              categorical_cols: list,
                              numeric_cols: list,
                              target_col: str = "readmitted"):
    
    report = {}
    
    # ----------------- 1️⃣ Categorical Comparison -----------------
    print("\n=== Categorical Feature Comparison ===")
    cat_metrics = {}
    for col in categorical_cols:
        real_dist = real_df[col].value_counts(normalize=True)
        synth_dist = synthetic_df[col].value_counts(normalize=True)
        tvd = sum(abs(real_dist.get(cat, 0) - synth_dist.get(cat, 0)) for cat in real_dist.index) / 2
        cat_metrics[col] = tvd
        print(f"{col:20s} TVD: {tvd:.4f}")
    report['categorical_tvd'] = cat_metrics
    
    # ----------------- 2️⃣ Numerical Comparison -----------------
    print("\n=== Numerical Feature Comparison (KS Test) ===")
    num_metrics = {}
    for col in numeric_cols:
        ks_stat, p_val = ks_2samp(real_df[col].dropna(), synthetic_df[col].dropna())
        num_metrics[col] = {'ks_stat': ks_stat, 'p_value': p_val}
        print(f"{col:25s} KS stat: {ks_stat:.4f} | p-value: {p_val:.4f}")
    report['numerical_ks'] = num_metrics
    
    # ----------------- 3️⃣ Correlation with Target -----------------
    print("\n=== Correlation with Target ===")
    real_df = real_df.copy()
    synthetic_df = synthetic_df.copy()
    
    if real_df[target_col].dtype == object:
        real_df[target_col + "_binary"] = (real_df[target_col] == "<30").astype(int)
        synthetic_df[target_col + "_binary"] = (synthetic_df[target_col] == "<30").astype(int)
        target_col_bin = target_col + "_binary"
    else:
        target_col_bin = target_col
    
    corr_metrics = {}
    for col in numeric_cols:
        real_corr = real_df[col].corr(real_df[target_col_bin])
        synth_corr = synthetic_df[col].corr(synthetic_df[target_col_bin])
        diff = abs(real_corr - synth_corr)
        corr_metrics[col] = {'real_corr': real_corr, 'synthetic_corr': synth_corr, 'diff': diff}
        print(f"{col:25s} Real: {real_corr:.3f} | Synth: {synth_corr:.3f} | Diff: {diff:.3f}")
    report['feature_target_corr'] = corr_metrics
    
    # ----------------- 4️⃣ Pairwise Numeric Correlations -----------------
    print("\n=== Pairwise Numeric Feature Correlations ===")
    real_corr_matrix = real_df[numeric_cols].corr()
    synth_corr_matrix = synthetic_df[numeric_cols].corr()
    corr_diff_matrix = (real_corr_matrix - synth_corr_matrix).abs()
    mean_corr_diff = corr_diff_matrix.values.mean()
    print(f"Mean absolute correlation difference: {mean_corr_diff:.4f}")
    report['pairwise_corr_diff'] = mean_corr_diff
    
    # ----------------- 5️⃣ TSTR Model-Based Validation -----------------
    print("\n=== Model-Based Validation (TSTR) ===")
    X_real = pd.get_dummies(real_df[numeric_cols + categorical_cols], drop_first=True)
    X_synth = pd.get_dummies(synthetic_df[numeric_cols + categorical_cols], drop_first=True)
    
    # Align columns
    X_real, X_synth = X_real.align(X_synth, join='left', axis=1, fill_value=0)
    
    y_real = real_df[target_col_bin]
    y_synth = synthetic_df[target_col_bin]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_synth, y_synth)
    y_pred = model.predict_proba(X_real)[:, 1]
    auc = roc_auc_score(y_real, y_pred)
    print(f"TSTR AUC (train synthetic, test real): {auc:.3f}")
    report['tstr_auc'] = auc
    
    # ----------------- 6️⃣ Summary -----------------
    avg_tvd = np.mean(list(cat_metrics.values()))
    avg_ks = np.mean([v['ks_stat'] for v in num_metrics.values()])
    avg_corr_diff = np.mean([v['diff'] for v in corr_metrics.values()])
    
    print("\n=== SUMMARY ===")
    print(f"Average TVD (categorical): {avg_tvd:.4f}")
    print(f"Average KS statistic (numerical): {avg_ks:.4f}")
    print(f"Average correlation difference (target): {avg_corr_diff:.4f}")
    print(f"TSTR AUC: {auc:.3f}")
    
    report['summary'] = {
        'avg_tvd': avg_tvd,
        'avg_ks': avg_ks,
        'avg_corr_diff': avg_corr_diff,
        'tstr_auc': auc
    }
    
    return report



CATEGORICAL_COLS = [
    'age', 'gender', 'race', 'A1Cresult', 'insulin', 'diabetesMed', 
    'change', 'readmitted'
]

INT_COLS = [
    'time_in_hospital', 'num_medications', 'number_diagnoses',
    'num_lab_procedures', 'num_procedures', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id'
]

report = compare_real_vs_synthetic(
    real_df=pd.read_csv(r"data\local\diabetic_data.csv"),
    synthetic_df=pd.read_csv(r"artifacts\raw\synthetic_diabetic_tvae2.csv"),
    categorical_cols=CATEGORICAL_COLS,
    numeric_cols=INT_COLS,
    target_col="readmitted"
)














# # ============================================================================
# # readmit/utils/diagnose.py
# # ============================================================================

# import pandas as pd
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# from readmit.configuration import paths_config
# from readmit.components.logger import logging


# def diagnose_synthetic_quality():
#     """
#     Comprehensive diagnostic of synthetic data quality
#     """
#     print("="*70)
#     print("SYNTHETIC DATA QUALITY DIAGNOSTIC")
#     print("="*70)
    
#     # Load processed data
#     real_train = pd.read_csv(r"data\local\diabetic_data.csv")
#     syn_train = pd.read_csv(r"artifacts\raw\synthetic_diabetic_tvae.csv")
    
#     print(f"\n✓ Loaded data")
#     print(f"  Real: {real_train.shape}")
#     print(f"  Synthetic: {syn_train.shape}")
    
#     # 1. TARGET DISTRIBUTION CHECK
#     print("\n" + "="*70)
#     print("1. TARGET DISTRIBUTION")
#     print("="*70)
    
#     real_target_dist = real_train['readmitted'].value_counts(normalize=True)
#     syn_target_dist = syn_train['readmitted'].value_counts(normalize=True)
    
#     print("\nReal:")
#     print(f"  Class 0: {real_target_dist.get(0, 0):.4f}")
#     print(f"  Class 1: {real_target_dist.get(1, 0):.4f}")
    
#     print("\nSynthetic:")
#     print(f"  Class 0: {syn_target_dist.get(0, 0):.4f}")
#     print(f"  Class 1: {syn_target_dist.get(1, 0):.4f}")
    
#     target_diff = abs(real_target_dist.get(1, 0) - syn_target_dist.get(1, 0))
#     print(f"\nTarget distribution difference: {target_diff:.4f}")
    
#     if target_diff > 0.05:
#         print("⚠️  WARNING: Target distributions differ by >5%!")
#         print("   This will cause poor model performance.")
#     else:
#         print("✓ Target distributions are similar")
    
#     # 2. FEATURE DISTRIBUTIONS
#     print("\n" + "="*70)
#     print("2. FEATURE DISTRIBUTIONS")
#     print("="*70)
    
#     # Exclude target
#     feature_cols = [col for col in real_train.columns if col != 'readmitted']
    
#     # KS test for each feature
#     ks_results = []
#     for col in feature_cols:
#         try:
#             ks_stat, ks_pval = stats.ks_2samp(
#                 real_train[col].dropna(),
#                 syn_train[col].dropna()
#             )
#             ks_results.append({
#                 'feature': col,
#                 'ks_statistic': ks_stat,
#                 'ks_pvalue': ks_pval,
#                 'similar': ks_pval > 0.05
#             })
#         except:
#             pass
    
#     ks_df = pd.DataFrame(ks_results)
    
#     similar_features = ks_df[ks_df['similar']].shape[0]
#     total_features = ks_df.shape[0]
    
#     print(f"\nFeatures with similar distributions: {similar_features}/{total_features}")
#     print(f"Percentage: {similar_features/total_features*100:.1f}%")
    
#     if similar_features / total_features < 0.7:
#         print("\n⚠️  WARNING: Less than 70% of features have similar distributions!")
#         print("   Synthetic data is NOT preserving feature patterns well.")
#     else:
#         print("\n✓ Feature distributions are reasonably preserved")
    
#     # Show worst offenders
#     print("\nWorst 5 features (lowest p-values):")
#     worst_features = ks_df.nsmallest(5, 'ks_pvalue')
#     for _, row in worst_features.iterrows():
#         print(f"  {row['feature']}: p-value={row['ks_pvalue']:.4f}")
    
#     # 3. CORRELATION STRUCTURE
#     print("\n" + "="*70)
#     print("3. CORRELATION STRUCTURE")
#     print("="*70)

#     # Select numeric features only
#     numeric_features = real_train.select_dtypes(include=np.number).columns.tolist()

#     # Ensure target is numeric; if not, create numeric target
#     if 'readmitted' not in numeric_features:
#         # Map target to numeric if it's categorical
#         target_mapping = {'<30': 1, '>30': 0, 'NO': 0}  # adjust mapping based on your dataset
#         real_train['readmitted_num'] = real_train['readmitted'].map(target_mapping)
#         syn_train['readmitted_num'] = syn_train['readmitted'].map(target_mapping)
#         numeric_features.append('readmitted_num')
#         target_col = 'readmitted_num'
#     else:
#         target_col = 'readmitted'

#     # Sample up to 10 numeric features + target
#     sample_features = numeric_features[:10] + [target_col]

#     # Compute correlations
#     real_corr = real_train[sample_features].corr()
#     syn_corr = syn_train[sample_features].corr()

#     # Correlation difference
#     corr_diff = np.abs(real_corr - syn_corr)
#     mean_corr_diff = corr_diff.mean().mean()

#     print(f"\nMean absolute correlation difference: {mean_corr_diff:.4f}")

#     if mean_corr_diff > 0.15:
#         print("⚠️  WARNING: Correlation structure differs significantly!")
#         print("   Synthetic data is NOT preserving feature relationships.")
#     else:
#         print("✓ Correlation structure is reasonably preserved")
    
    
#     # 4. TARGET CORRELATION (MOST IMPORTANT!)
#     print("\n" + "="*70)
#     print("4. TARGET CORRELATIONS (CRITICAL!)")
#     print("="*70)
    
#     # Correlations with target
#     real_target_corr = real_train[feature_cols].corrwith(real_train['readmitted']).abs()
#     syn_target_corr = syn_train[feature_cols].corrwith(syn_train['readmitted']).abs()
    
#     target_corr_diff = np.abs(real_target_corr - syn_target_corr)
#     mean_target_corr_diff = target_corr_diff.mean()
    
#     print(f"\nMean absolute difference in target correlations: {mean_target_corr_diff:.4f}")
    
#     if mean_target_corr_diff > 0.10:
#         print("\n⚠️  CRITICAL: Target correlations differ significantly!")
#         print("   This is why models trained on synthetic data perform poorly!")
#     else:
#         print("\n✓ Target correlations are preserved")
    
#     # Show top features by correlation difference
#     print("\nTop 5 features with most different target correlations:")
#     worst_target_corr = target_corr_diff.nlargest(5)
#     for feature, diff in worst_target_corr.items():
#         real_corr_val = real_target_corr[feature]
#         syn_corr_val = syn_target_corr[feature]
#         print(f"  {feature}:")
#         print(f"    Real corr: {real_corr_val:.4f}")
#         print(f"    Syn corr:  {syn_corr_val:.4f}")
#         print(f"    Diff:      {diff:.4f}")
    
#     # 5. STATISTICAL SUMMARY
#     print("\n" + "="*70)
#     print("5. FEATURE STATISTICS COMPARISON")
#     print("="*70)
    
#     # Compare means and stds
#     real_means = real_train[feature_cols].mean()
#     syn_means = syn_train[feature_cols].mean()
#     mean_diff = np.abs(real_means - syn_means).mean()
    
#     real_stds = real_train[feature_cols].std()
#     syn_stds = syn_train[feature_cols].std()
#     std_diff = np.abs(real_stds - syn_stds).mean()
    
#     print(f"\nMean difference in feature means: {mean_diff:.4f}")
#     print(f"Mean difference in feature stds:  {std_diff:.4f}")
    
#     # 6. FINAL VERDICT
#     print("\n" + "="*70)
#     print("FINAL VERDICT")
#     print("="*70)
    
#     issues = []
    
#     if target_diff > 0.05:
#         issues.append("❌ Target distribution mismatch")
    
#     if similar_features / total_features < 0.7:
#         issues.append("❌ Poor feature distribution preservation")
    
#     if mean_corr_diff > 0.15:
#         issues.append("❌ Poor correlation structure preservation")
    
#     if mean_target_corr_diff > 0.10:
#         issues.append("❌ Poor target correlation preservation (CRITICAL!)")
    
#     if issues:
#         print("\n⚠️  SYNTHETIC DATA QUALITY: POOR")
#         print("\nIssues found:")
#         for issue in issues:
#             print(f"  {issue}")
        
#         print("\n" + "="*70)
#         print("RECOMMENDATIONS:")
#         print("="*70)
#         print("1. Re-generate synthetic data with better CTGAN hyperparameters:")
#         print("   - Increase epochs (500-1000)")
#         print("   - Adjust discriminator steps")
#         print("   - Try different generator/discriminator dimensions")
#         print("")
#         print("2. Use different synthetic data method:")
#         print("   - TVAE (Variational Autoencoder)")
#         print("   - CopulaGAN")
#         print("   - SMOTE-based oversampling")
#         print("")
#         print("3. For now, train models ONLY on real data")
#         print("   (Synthetic data will hurt performance)")
        
#     else:
#         print("\n✓ SYNTHETIC DATA QUALITY: GOOD")
#         print("   Synthetic data preserves patterns well.")
#         print("   Models should perform reasonably on it.")
    
#     print("="*70)


# if __name__ == "__main__":
#     diagnose_synthetic_quality()