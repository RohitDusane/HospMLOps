# readmit/components/data_validation.py
import numpy as np
import pandas as pd
import os
import sys
import json
from typing import Dict, List, Tuple
from readmit.components.logger import logging
from readmit.components.exception import CustomException
from readmit.entity.config_entity import REAL_EXPECTED_COLUMNS, SYNTHETIC_EXPECTED_COLUMNS
from readmit.configuration import paths_config
from readmit.utils.main_utils import short_path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
# from readmit.entity.config_entity import EXPECTED_COLUMNS



class DataValidation:
    """
    Validates data quality and schema.
    Does NOT transform data - only checks and reports.
    Saves validated (unchanged) data to VALIDATED_DIR.
    """

    def __init__(
        self, 
        output_dir: str = paths_config.VALIDATED_DIR,
        reports_dir: str = paths_config.REPORTS_DIR):
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.validation_report = {
            'real_train': {},
            'real_test': {},
            'syn_train': {},
            'syn_test': {},
            'distribution_comparison': {},
            'overall_status': True
        }
        
        # Store loaded data for comparison
        self.real_train = None
        self.real_test = None
        self.syn_train = None
        self.syn_test = None

    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from CSV or Parquet"""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            logging.info(f"Data Loaded {short_path(path)} | Shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(f"Error loading {path}: {str(e)}", sys)

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        data_name: str,
        target_column: str,
        dataset_type: str  # "real" or "synthetic"
    ) -> Dict:

        """
        Validate data quality WITHOUT modifying data
        
        Checks:
        - Empty dataframe
        - Missing values
        - Duplicate rows
        - Target column exists
        - Data types
        - Schema validation
        - Age column sanity
        """
        issues = []
        warnings = []

        # 1. Check if empty
        if df.empty:
            issues.append("DataFrame is empty")
            return {'status': 'FAILED', 'issues': issues, 'warnings': warnings}

        # --------------------------------------------------
        # 2. Schema validation
        # --------------------------------------------------
        expected = set(REAL_EXPECTED_COLUMNS) if dataset_type == "real" else set(SYNTHETIC_EXPECTED_COLUMNS)
        actual = set(df.columns)

        missing_cols = list(expected - actual)
        if missing_cols:
            issues.append({"type": "missing_columns", "columns": missing_cols})
            logging.warning("%s missing columns: %s", data_name, missing_cols)

        # --------------------------------------------------
        # 3. Target validation
        # --------------------------------------------------
        if target_column not in df.columns:
            issues.append(f"Target column '{target_column}' not found")
        else:
            if df[target_column].isnull().any():
                issues.append(f"Target column '{target_column}' contains missing values")

            if target_column == "readmitted_bin":
                invalid = set(df[target_column].dropna().unique()) - {0, 1}
                if invalid:
                    issues.append({"type": "invalid_target_values", "values": list(invalid)})

        # --------------------------------------------------
        # 4. Missing values
        # --------------------------------------------------
        missing_counts = df.drop(columns=[target_column], errors="ignore").isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            warnings.append({
                "type": "missing_values",
                "columns": missing_counts.to_dict(),
                "percentages": ((missing_counts / len(df)) * 100).round(2).to_dict()
            })

        # --------------------------------------------------
        # 5. Duplicates
        # --------------------------------------------------
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            dup_pct = (n_duplicates / len(df) * 100).round(2)
            warnings.append({'type': 'duplicates', 'count': int(n_duplicates), 'percentage': float(dup_pct)})
            logging.warning("Duplicate rows detected | dataset=%s | count=%d | percentage=%.2f", data_name, n_duplicates, dup_pct)

        # --------------------------------------------------
        # 6. Data types
        # --------------------------------------------------
        dtypes_info = df.dtypes.astype(str).to_dict()

        # --------------------------------------------------
        # 7. Age column sanity
        # --------------------------------------------------
        if "age" in df.columns:
            if df["age"].isnull().any():
                warnings.append({"type": "missing_values", "columns": {"age": int(df['age'].isnull().sum())}})

        # --------------------------------------------------
        # 8. Basic statistics
        # --------------------------------------------------
        stats_dict = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }

        status = 'PASSED' if len(issues) == 0 else 'FAILED'

        report = {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'statistics': stats_dict,
            'dtypes': dtypes_info
        }

        logging.info("%s validation: %s", data_name, status)
        return report


    @staticmethod
    def _binary_target(series: pd.Series) -> np.ndarray:
        """Map readmission label to binary"""
        return np.where(series == "<30", 1, 0)


    def _map_age_to_group(self, s: pd.Series) -> pd.Series:
        def f(x):
            if x in ['[0-10)', '[10-20)', '[20-30)', '[30-40)']:
                return 'young'
            elif x in ['[40-50)', '[50-60)']:
                return 'middle'
            else:
                return 'senior'
        return s.astype(str).apply(f)


    def validate_schema_compatibility(
        self,
        real_train: pd.DataFrame,
        syn_train: pd.DataFrame,
        real_target: str,
        syn_target: str) -> Dict:
        """
        Validate that real and synthetic data schemas are compatible
        """
        issues = []
        warnings = []
        
        # Get columns (excluding targets)
        real_cols = set(real_train.columns) - {real_target}
        syn_cols = set(syn_train.columns) - {syn_target}
        
        # Check common columns
        common_cols = real_cols & syn_cols
        real_only = real_cols - syn_cols
        syn_only = syn_cols - real_cols
        
        if real_only:
            warnings.append({
                'type': 'columns_only_in_real',
                'columns': list(real_only)
            })
            logging.warning("Schema mismatch | columns_only_in_real=%s", list(real_only))
        
        if syn_only:
            warnings.append({
                'type': 'columns_only_in_synthetic',
                'columns': list(syn_only)
            })
            logging.warning("Schema mismatch | columns_only_in_synthetic=%s", list(syn_only))
        
        if len(common_cols) == 0:
            issues.append("No common columns between real and synthetic data")
        
        # Check data type compatibility for common columns
        dtype_mismatches = []
        for col in common_cols:
            real_dtype = str(real_train[col].dtype)
            syn_dtype = str(syn_train[col].dtype)
            if real_dtype != syn_dtype:
                dtype_mismatches.append({
                    'column': col,
                    'real_dtype': real_dtype,
                    'synthetic_dtype': syn_dtype
                })
        
        if dtype_mismatches:
            warnings.append({
                'type': 'dtype_mismatches',
                'mismatches': dtype_mismatches
            })
            logging.warning(f"Data type mismatches: {len(dtype_mismatches)} columns")
        
        status = 'PASSED' if len(issues) == 0 else 'FAILED'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'common_columns': list(common_cols),
            'n_common_columns': len(common_cols)
        }

    def compare_distributions(self) -> Dict:
        """
        Compare statistical distributions between real and synthetic data
        Returns comparison metrics for reporting
        """
        logging.info("Comparing distributions between real and synthetic data...")

        comparison_results = {}

        # =====================================================
        # 1️⃣ TARGET DISTRIBUTION (SAFE ALIGNMENT)
        # =====================================================
        real_target_col = "readmitted_bin"
        syn_target_col = "readmitted_bin"

        if real_target_col in self.real_train.columns and syn_target_col in self.syn_train.columns:

            # --- Normalize both targets to binary ---
            real_target = self.real_train[real_target_col].astype(int)
            syn_target = self.syn_train[syn_target_col].astype(int)

            real_dist = real_target.value_counts(normalize=True).sort_index()
            syn_dist = syn_target.value_counts(normalize=True).sort_index()

            # Fixed index: [0, 1]
            real_dist = real_dist.reindex([0, 1], fill_value=0)
            syn_dist = syn_dist.reindex([0, 1], fill_value=0)

            comparison_results["target_distribution"] = {
                "real": real_dist.to_dict(),
                "synthetic": syn_dist.to_dict(),
                "absolute_difference": (real_dist - syn_dist).abs().to_dict(),
            }

            logging.info("Target distribution compared successfully")

        # =====================================================
        # 2️⃣ NUMERIC COLUMNS COMPARISON
        # =====================================================
        numeric_cols = (
            self.real_train
            .select_dtypes(include=[np.number])
            .columns
            .intersection(self.syn_train.columns)
        )

        numeric_comparisons = {}

        for col in numeric_cols:
            try:
                real_data = self.real_train[col].dropna()
                syn_data = self.syn_train[col].dropna()

                if len(real_data) < 5 or len(syn_data) < 5:
                    continue

                ks_stat, ks_pval = stats.ks_2samp(real_data, syn_data)

                numeric_comparisons[col] = {
                    "real_mean": float(real_data.mean()),
                    "synthetic_mean": float(syn_data.mean()),
                    "real_std": float(real_data.std()),
                    "synthetic_std": float(syn_data.std()),
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pval),
                    "distributions_similar": bool(ks_pval > 0.05),
                }

            except Exception as e:
                logging.warning(f"Numeric comparison failed for {col}: {e}")

        comparison_results["numeric_columns"] = numeric_comparisons
        logging.info(f"Compared {len(numeric_comparisons)} numeric columns")

        # =====================================================
        # 3️⃣ CATEGORICAL COLUMNS COMPARISON (NO SORTING)
        # =====================================================
        # categorical_cols = ['gender', 'race', 'A1Cresult', 'insulin', 'diabetesMed', 'change', 'readmitted']
        categorical_cols = self.real_train.select_dtypes(exclude=np.number).columns.intersection(self.syn_train.columns)


        categorical_comparisons = {}

        for col in categorical_cols:
            if col not in self.real_train.columns or col not in self.syn_train.columns:
                continue

            try:
                real_dist = self.real_train[col].value_counts(normalize=True)
                syn_dist = self.syn_train[col].value_counts(normalize=True)

                # Union without sorting (CRITICAL FIX)
                all_categories = real_dist.index.union(syn_dist.index)

                real_dist = real_dist.reindex(all_categories, fill_value=0)
                syn_dist = syn_dist.reindex(all_categories, fill_value=0)

                categorical_comparisons[col] = {
                    "real": real_dist.to_dict(),
                    "synthetic": syn_dist.to_dict(),
                    "max_abs_diff": float((real_dist - syn_dist).abs().max()),
                }

            except Exception as e:
                logging.warning(f"Categorical comparison failed for {col}: {e}")

        comparison_results["categorical_columns"] = categorical_comparisons
        logging.info(f"Compared {len(categorical_comparisons)} categorical columns")

        if "age" in self.real_train.columns and "age" in self.syn_train.columns:
            real_age_group = self._map_age_to_group(self.real_train["age"])
            syn_age_group  = self._map_age_to_group(self.syn_train["age"])

            real_dist = real_age_group.value_counts(normalize=True)
            syn_dist = syn_age_group.value_counts(normalize=True)

            all_groups = real_dist.index.union(syn_dist.index)
            real_dist = real_dist.reindex(all_groups, fill_value=0)
            syn_dist = syn_dist.reindex(all_groups, fill_value=0)

            comparison_results["age_group_distribution"] = {
                "real": real_dist.to_dict(),
                "synthetic": syn_dist.to_dict(),
                "max_abs_diff": float((real_dist - syn_dist).abs().max())
            }

        return comparison_results


    def visualize_comparison(self):
        """Create visualization comparing real and synthetic data"""
        logging.info("Creating distribution comparison visualizations...")
        
        try:
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            fig.suptitle('Real vs Synthetic Data Comparison', fontsize=16, y=1.00)
            
            # 1. Target distribution
            if 'readmitted_bin' in self.real_train.columns and 'readmitted_bin' in self.syn_train.columns:

                target_real = self.real_train['readmitted_bin'].value_counts(normalize=True).reindex([0,1], fill_value=0)
                target_syn = self.syn_train['readmitted_bin'].value_counts(normalize=True).reindex([0,1], fill_value=0)

                # Force binary alignment
                all_idx = [0, 1]
                target_real = target_real.reindex(all_idx, fill_value=0)
                target_syn  = target_syn.reindex(all_idx, fill_value=0)

                x = np.arange(len(all_idx))
                width = 0.35

                axes[0, 0].bar(
                    x - width/2,
                    target_real.values,
                    width,
                    label='Real',
                    alpha=0.8,
                    color='#2E86AB'
                )
                axes[0, 0].bar(
                    x + width/2,
                    target_syn.values,
                    width,
                    label='Synthetic',
                    alpha=0.8,
                    color='#A23B72'
                )

                axes[0, 0].set_xlabel('Readmission Status')
                axes[0, 0].set_ylabel('Proportion')
                axes[0, 0].set_title('Binary Target Distribution (<30 vs Rest)')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(['No Event', 'Event'])
                axes[0, 0].legend()
                axes[0, 0].grid(axis='y', alpha=0.3)

            
            # 2. Age distribution
            if 'age' in self.real_train.columns and 'age' in self.syn_train.columns:
                real_age = self._map_age_to_group(self.real_train['age']).value_counts()
                syn_age = self._map_age_to_group(self.syn_train['age']).value_counts()


                all_groups = real_age.index.union(syn_age.index)
                age_comp = pd.DataFrame({
                    'Real': real_age.reindex(all_groups, fill_value=0),
                    'Synthetic': syn_age.reindex(all_groups, fill_value=0)
                })


                age_comp.plot(kind='bar', ax=axes[0, 1], alpha=0.8,
                            color=['#2E86AB', '#A23B72'])

                axes[0, 1].set_title('Age Group Distribution')
                axes[0, 1].set_xlabel('Age Group')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].legend()
                axes[0, 1].grid(axis='y', alpha=0.3)

            
            # 3. Time in hospital
            if 'time_in_hospital' in self.real_train.columns and 'time_in_hospital' in self.syn_train.columns:
                axes[1, 0].hist(self.real_train['time_in_hospital'], bins=30, alpha=0.6, 
                               label='Real', density=True, color='#2E86AB')
                axes[1, 0].hist(self.syn_train['time_in_hospital'], bins=30, alpha=0.6, 
                               label='Synthetic', density=True, color='#A23B72')
                axes[1, 0].set_xlabel('Time in Hospital (days)')
                axes[1, 0].set_ylabel('Density')
                axes[1, 0].set_title('Time in Hospital Distribution')
                axes[1, 0].legend()
                axes[1, 0].grid(axis='y', alpha=0.3)
            
            # 4. Number of medications
            if 'num_medications' in self.real_train.columns and 'num_medications' in self.syn_train.columns:
                axes[1, 1].hist(self.real_train['num_medications'], bins=30, alpha=0.6, 
                               label='Real', density=True, color='#2E86AB')
                axes[1, 1].hist(self.syn_train['num_medications'], bins=30, alpha=0.6, 
                               label='Synthetic', density=True, color='#A23B72')
                axes[1, 1].set_xlabel('Number of Medications')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Number of Medications Distribution')
                axes[1, 1].legend()
                axes[1, 1].grid(axis='y', alpha=0.3)
            
            # 5. Gender distribution
            # 5. Gender distribution (ALIGNED)
            if 'gender' in self.real_train.columns and 'gender' in self.syn_train.columns:

                gender_real = self.real_train['gender'].value_counts()
                gender_syn = self.syn_train['gender'].value_counts()

                # all_genders = sorted(set(gender_real.index) | set(gender_syn.index))
                all_genders = gender_real.index.union(gender_syn.index)

                gender_real = gender_real.reindex(all_genders, fill_value=0)
                gender_syn = gender_syn.reindex(all_genders, fill_value=0)

                x = np.arange(len(all_genders))
                width = 0.35

                axes[2, 0].bar(
                    x - width/2,
                    gender_real.values,
                    width,
                    label='Real',
                    alpha=0.8,
                    color='#2E86AB'
                )
                axes[2, 0].bar(
                    x + width/2,
                    gender_syn.values,
                    width,
                    label='Synthetic',
                    alpha=0.8,
                    color='#A23B72'
                )

                axes[2, 0].set_xlabel('Gender')
                axes[2, 0].set_ylabel('Count')
                axes[2, 0].set_title('Gender Distribution')
                axes[2, 0].set_xticks(x)
                axes[2, 0].set_xticklabels(all_genders)
                axes[2, 0].legend()
                axes[2, 0].grid(axis='y', alpha=0.3)

            
            # 6. Number of diagnoses
            if 'number_diagnoses' in self.real_train.columns and 'number_diagnoses' in self.syn_train.columns:
                axes[2, 1].hist(self.real_train['number_diagnoses'], bins=20, alpha=0.6, 
                               label='Real', density=True, color='#2E86AB')
                axes[2, 1].hist(self.syn_train['number_diagnoses'], bins=20, alpha=0.6, 
                               label='Synthetic', density=True, color='#A23B72')
                axes[2, 1].set_xlabel('Number of Diagnoses')
                axes[2, 1].set_ylabel('Density')
                axes[2, 1].set_title('Number of Diagnoses Distribution')
                axes[2, 1].legend()
                axes[2, 1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save to reports directory
            viz_path = os.path.join(self.reports_dir, 'real_vs_synthetic_comparison.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            logging.info(f"Visualization saved | path={short_path(viz_path)}")
            
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")
            raise CustomException(e, sys)

    def align_and_save_data(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        real_target: str,
        syn_target: str,
        data_type: str) -> Tuple[str, str]:
        """
        Align schemas and save to validated directory
        
        This is the ONLY modification we do:
        - Rename synthetic target to match real target
        - Keep only common columns
        - Convert synthetic to CSV format
        """
        # Get common columns
        real_cols = set(real_df.columns) - {real_target}
        syn_cols = set(syn_df.columns) - {syn_target}
        common_cols = sorted(real_cols & syn_cols)
        
        # Add target columns
        common_cols.append(real_target)
        
        # Align real data
        real_aligned = real_df[[col for col in common_cols if col in real_df.columns]].copy()
        
        # Align synthetic data and rename target
        syn_aligned = syn_df[[col for col in common_cols if col != real_target and col in syn_df.columns]].copy()
        syn_aligned[real_target] = syn_df[syn_target]  # Rename target
        
        # Reorder columns to match
        syn_aligned = syn_aligned[common_cols]
        assert "age" in real_aligned.columns, "age missing in real validated data"
        assert "age" in syn_aligned.columns, "age missing in synthetic validated data"

        
        # Save to validated directory
        if data_type == 'train':
            real_path = paths_config.VALIDATED_REAL_TRAIN
            syn_path = paths_config.VALIDATED_SYN_TRAIN
        else:
            real_path = paths_config.VALIDATED_REAL_TEST
            syn_path = paths_config.VALIDATED_SYN_TEST
        
        # Create directories
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        os.makedirs(os.path.dirname(syn_path), exist_ok=True)
        
        # Save (both as CSV for consistency)
        real_aligned.to_csv(real_path, index=False)
        syn_aligned.to_csv(syn_path, index=False)
        logging.info(
            "Validated data saved | split=%s | real_path=%s | syn_path=%s | rows=%d | cols=%d",
            data_type,
            os.path.relpath(real_path),
            os.path.relpath(syn_path),
            real_aligned.shape[0],
            real_aligned.shape[1]
        )        
        return real_path, syn_path

    def save_validation_report(self):
        """Save validation report as JSON"""
        report_path = paths_config.VALIDATION_REPORT
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        
        logging.info("Validation report saved | path=%s", short_path(report_path))

    # ---------- Add this method inside your DataValidation class ----------
    def _adjust_synthetic_data(self, syn_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust synthetic dataset to match real dataset formats and categories.
        - Maps boolean or variant labels to real dataset strings
        - Creates 'readmitted_bin' column
        - Aligns categorical columns
        """
        df = syn_df.copy()

        if "readmitted" in df.columns and "readmitted_bin" not in df.columns:
            df["readmitted_bin"] = df["readmitted"].map(lambda x: 1 if x == "<30" else 0)
        elif "readmitted_bin" in df.columns:
            df["readmitted_bin"] = df["readmitted_bin"].astype(int)

        return df

    def run(self):
        """
        Main validation pipeline for REAL datasets only
        """
        try:
            # Load processed real datasets
            logging.info("Loading processed real datasets...")
            self.real_train = self.load_data(paths_config.PROCESSED_REAL_TRAIN)
            self.real_test = self.load_data(paths_config.PROCESSED_REAL_TEST)

            # Validate each real dataset
            logging.info("Validating data quality for real datasets...")

            self.validation_report['real_train'] = self.validate_data_quality(
                self.real_train, "Real Train", "readmitted_bin", dataset_type="real"
            )

            self.validation_report['real_test'] = self.validate_data_quality(
                self.real_test, "Real Test", "readmitted_bin", dataset_type="real"
            )

            # Save validation report
            self.save_validation_report()

            logging.info("Validation for real datasets completed successfully.")

            return {
                'status': self.validation_report.get('overall_status', True),
                'report_path': paths_config.VALIDATION_REPORT,
                'validated_paths': {
                    'real_train': paths_config.PROCESSED_REAL_TRAIN,
                    'real_test': paths_config.PROCESSED_REAL_TEST
                }
            }

        except Exception as e:
            logging.error(f"Error in real data validation: {str(e)}")
            raise CustomException(e, sys)


    # def run(self):
    #     """
    #     Main validation pipeline
    #     """
    #     try:
    #         # Load processed data
    #         logging.info("Loading processed datasets...")
    #         self.real_train = self.load_data(paths_config.PROCESSED_REAL_TRAIN)
    #         self.real_test = self.load_data(paths_config.PROCESSED_REAL_TEST)
    #         # self.syn_train = self.load_data(paths_config.PROCESSED_SYN_TRAIN)
    #         # self.syn_test = self.load_data(paths_config.PROCESSED_SYN_TEST)

    #         # Adjust synthetic data to match real dataset
    #         # self.syn_train = self._adjust_synthetic_data(self.syn_train, self.real_train)
    #         # self.syn_test = self._adjust_synthetic_data(self.syn_test, self.real_test)
            
    #         # Validate each dataset
    #         logging.info("Validating data quality...")
                   
    #         self.validation_report['real_train'] = self.validate_data_quality(
    #             self.real_train, "Real Train", "readmitted_bin", dataset_type="real")

    #         self.validation_report['real_test'] = self.validate_data_quality(
    #             self.real_test, "Real Test", "readmitted_bin", dataset_type="real")

    #         # self.validation_report['syn_train'] = self.validate_data_quality(
    #         #     self.syn_train, "Synthetic Train", "readmitted_bin", dataset_type="synthetic")

    #         # self.validation_report['syn_test'] = self.validate_data_quality(
    #         #     self.syn_test, "Synthetic Test", "readmitted_bin", dataset_type="synthetic")
            
    #         # Validate schema compatibility
    #         logging.info("Validating schema compatibility...")
    #         # schema_report = self.validate_schema_compatibility(
    #         #     self.real_train, self.syn_train, "readmitted_bin", "readmitted_bin")
    #         # self.validation_report['schema_compatibility'] = schema_report
            
    #         # Compare distributions
    #         # logging.info("Comparing distributions...")
    #         # distribution_comparison = self.compare_distributions()
    #         # self.validation_report['distribution_comparison'] = distribution_comparison
            
    #         # # Create visualizations
    #         # self.visualize_comparison()
            
    #         # # Check if any validations failed
    #         # failed = []
    #         # for key, report in self.validation_report.items():
    #         #     if isinstance(report, dict) and report.get('status') == 'FAILED':
    #         #         failed.append(key)
            
    #         # if failed:
    #         #     self.validation_report['overall_status'] = False
    #         #     logging.error(f"Validation FAILED for: {failed}")
    #         # else:
    #         #     self.validation_report['overall_status'] = True
    #         #     logging.info("All validations PASSED")
            
    #         # # Align and save validated data
    #         # logging.info("Aligning schemas and saving validated data...")
            
    #         # train_real_path, train_syn_path = self.align_and_save_data(
    #         #     self.real_train, self.syn_train, "readmitted_bin", "readmitted_bin", "train")
    #         # test_real_path, test_syn_path = self.align_and_save_data(
    #         #     self.real_test, self.syn_test, "readmitted_bin", "readmitted_bin", "test")
            
    #         # Save validation report
    #         self.save_validation_report()
            
    #         return {
    #             'status': self.validation_report['overall_status'],
    #             'report_path': paths_config.VALIDATION_REPORT,
    #             'validated_paths': {
    #                 'real_train': train_real_path,
    #                 'real_test': test_real_path,
    #                 # 'syn_train': train_syn_path,
    #                 # 'syn_test': test_syn_path
    #             }}

    #     except Exception as e:
    #         logging.error(f"Error in data validation: {str(e)}")
    #         raise CustomException(e, sys)


if __name__ == "__main__":
    start = time.time()
    logging.info("STAGE_START | name=DATA_VALIDATION")
    
    try:
        validation = DataValidation()
        result = validation.run()
        
        if not result['status']:
            logging.error("Validation failed! Check validation report.")
            sys.exit(1)
        
        elapsed = round(time.time() - start, 2)
        logging.info("STAGE_END | name=DATA_VALIDATION | status=SUCCESS | duration=%ss", elapsed)

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise CustomException(e, sys)















# # readmit/components/data_validation.py
# import numpy as np
# import pandas as pd
# import os
# import sys
# import json
# from typing import Dict, List, Tuple
# from readmit.components.logger import logging
# from readmit.components.exception import CustomException
# from readmit.configuration import paths_config
# from readmit.utils.main_utils import short_path
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import time


# class DataValidation:
#     """
#     Validates data quality and schema.
#     Does NOT transform data - only checks and reports.
#     Saves validated (unchanged) data to VALIDATED_DIR.
#     """

#     def __init__(
#         self, 
#         output_dir: str = paths_config.VALIDATED_DIR,
#         reports_dir: str = paths_config.REPORTS_DIR
#     ):
#         self.output_dir = output_dir
#         self.reports_dir = reports_dir
#         os.makedirs(self.output_dir, exist_ok=True)
#         os.makedirs(self.reports_dir, exist_ok=True)
        
#         self.validation_report = {
#             'real_train': {},
#             'real_test': {},
#             'syn_train': {},
#             'syn_test': {},
#             'overall_status': True
#         }

#     def load_data(self, path: str) -> pd.DataFrame:
#         """Load data from CSV or Parquet"""
#         try:
#             ext = os.path.splitext(path)[1].lower()
#             if ext == ".csv":
#                 df = pd.read_csv(path)
#             elif ext == ".parquet":
#                 df = pd.read_parquet(path)
#             else:
#                 raise ValueError(f"Unsupported file format: {ext}")
            
#             logging.info(f"Data Loaded {short_path(path)} | Shape: {df.shape}")
#             return df
#         except Exception as e:
#             raise CustomException(f"Error loading {path}: {str(e)}", sys)

#     def validate_data_quality(
#         self, 
#         df: pd.DataFrame, 
#         data_name: str,
#         target_column: str
#     ) -> Dict:
#         """
#         Validate data quality WITHOUT modifying data
        
#         Checks:
#         - Empty dataframe
#         - Missing values
#         - Duplicate rows
#         - Target column exists
#         - Data types
#         """
#         issues = []
#         warnings = []
        
#         # 1. Check if empty
#         if df.empty:
#             issues.append("DataFrame is empty")
#             return {'status': 'FAILED', 'issues': issues, 'warnings': warnings}
        
#         # 2. Check target column exists
#         if target_column not in df.columns:
#             issues.append(f"Target column '{target_column}' not found")
        
#         # 3. Check missing values
#         missing_counts = df.isnull().sum()
#         missing_cols = missing_counts[missing_counts > 0]
#         if not missing_cols.empty:
#             missing_pct = (missing_cols / len(df) * 100).round(2)
#             warnings.append({
#                 'type': 'missing_values',
#                 'columns': missing_cols.to_dict(),
#                 'percentages': missing_pct.to_dict()
#             })
#             # logging.warning(f"{data_name} has missing values:\n{missing_pct}")
#             logging.warning(
#                 "Missing values detected | dataset=%s | columns=%s",
#                 data_name,
#                 list(missing_pct.index)
#             )

#         # 4. Check duplicates
#         n_duplicates = df.duplicated().sum()
#         if n_duplicates > 0:
#             dup_pct = (n_duplicates / len(df) * 100).round(2)
#             warnings.append({
#                 'type': 'duplicates',
#                 'count': int(n_duplicates),
#                 'percentage': float(dup_pct)
#             })
#             logging.warning(
#                 "Duplicate rows detected | dataset=%s | count=%d | percentage=%.2f",
#                 data_name,
#                 n_duplicates,
#                 dup_pct
#             )

        
#         # 5. Check data types
#         dtypes_info = df.dtypes.astype(str).to_dict()
        
#         # 6. Basic statistics
#         stats = {
#             'n_rows': len(df),
#             'n_columns': len(df.columns),
#             'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
#         }
        
#         status = 'PASSED' if len(issues) == 0 else 'FAILED'
        
#         report = {
#             'status': status,
#             'issues': issues,
#             'warnings': warnings,
#             'statistics': stats,
#             'dtypes': dtypes_info
#         }
        
#         logging.info(f"{data_name} validation: {status}")
#         return report

#     def validate_schema_compatibility(
#         self,
#         real_train: pd.DataFrame,
#         syn_train: pd.DataFrame,
#         real_target: str,
#         syn_target: str
#     ) -> Dict:
#         """
#         Validate that real and synthetic data schemas are compatible
#         """
#         issues = []
#         warnings = []
        
#         # Get columns (excluding targets)
#         real_cols = set(real_train.columns) - {real_target}
#         syn_cols = set(syn_train.columns) - {syn_target}
        
#         # Check common columns
#         common_cols = real_cols & syn_cols
#         real_only = real_cols - syn_cols
#         syn_only = syn_cols - real_cols
        
#         if real_only:
#             warnings.append({
#                 'type': 'columns_only_in_real',
#                 'columns': list(real_only)
#             })
#             logging.warning("Schema mismatch | columns_only_in_real=%s", list(real_only))

        
#         if syn_only:
#             warnings.append({
#                 'type': 'columns_only_in_synthetic',
#                 'columns': list(syn_only)
#             })
#             logging.warning("Schema mismatch | columns_only_in_synthetic=%s", list(real_only))
        
#         if len(common_cols) == 0:
#             issues.append("No common columns between real and synthetic data")
        
#         # Check data type compatibility for common columns
#         dtype_mismatches = []
#         for col in common_cols:
#             real_dtype = str(real_train[col].dtype)
#             syn_dtype = str(syn_train[col].dtype)
#             if real_dtype != syn_dtype:
#                 dtype_mismatches.append({
#                     'column': col,
#                     'real_dtype': real_dtype,
#                     'synthetic_dtype': syn_dtype
#                 })
        
#         if dtype_mismatches:
#             warnings.append({
#                 'type': 'dtype_mismatches',
#                 'mismatches': dtype_mismatches
#             })
#             logging.warning(f"Data type mismatches: {len(dtype_mismatches)} columns")
        
#         status = 'PASSED' if len(issues) == 0 else 'FAILED'
        
#         return {
#             'status': status,
#             'issues': issues,
#             'warnings': warnings,
#             'common_columns': list(common_cols),
#             'n_common_columns': len(common_cols)
#         }

#     def align_and_save_data(
#         self,
#         real_df: pd.DataFrame,
#         syn_df: pd.DataFrame,
#         real_target: str,
#         syn_target: str,
#         data_type: str  # 'train' or 'test'
#     ) -> Tuple[str, str]:
#         """
#         Align schemas and save to validated directory
        
#         This is the ONLY modification we do:
#         - Rename synthetic target to match real target
#         - Keep only common columns
#         - Convert synthetic to CSV format
#         """
#         # Get common columns
#         real_cols = set(real_df.columns) - {real_target}
#         syn_cols = set(syn_df.columns) - {syn_target}
#         common_cols = sorted(real_cols & syn_cols)
        
#         # Add target columns
#         common_cols.append(real_target)
        
#         # Align real data
#         real_aligned = real_df[[col for col in common_cols if col in real_df.columns]].copy()
        
#         # Align synthetic data and rename target
#         syn_aligned = syn_df[[col for col in common_cols if col != real_target and col in syn_df.columns]].copy()
#         syn_aligned[real_target] = syn_df[syn_target]  # Rename target
        
#         # Reorder columns to match
#         syn_aligned = syn_aligned[common_cols]
        
#         # Save to validated directory
#         if data_type == 'train':
#             real_path = paths_config.VALIDATED_REAL_TRAIN
#             syn_path = paths_config.VALIDATED_SYN_TRAIN
#         else:
#             real_path = paths_config.VALIDATED_REAL_TEST
#             syn_path = paths_config.VALIDATED_SYN_TEST
        
#         # Create directories
#         os.makedirs(os.path.dirname(real_path), exist_ok=True)
#         os.makedirs(os.path.dirname(syn_path), exist_ok=True)
        
#         # Save (both as CSV for consistency)
#         real_aligned.to_csv(real_path, index=False)
#         syn_aligned.to_csv(syn_path, index=False)
#         logging.info(
#             "Validated data saved | split=%s | real_path=%s | syn_path=%s | rows=%d | cols=%d",
#             data_type,
#             os.path.relpath(real_path),
#             os.path.relpath(syn_path),
#             real_aligned.shape[0],
#             real_aligned.shape[1]
#         )        
#         return real_path, syn_path

#     def save_validation_report(self):
#         """Save validation report as JSON"""
#         report_path = paths_config.VALIDATION_REPORT
        
#         with open(report_path, 'w') as f:
#             json.dump(self.validation_report, f, indent=2)
        
#         logging.info("Validation report saved | path=%s", short_path(report_path))


#     def compare_distributions(self):
#         """Compare statistical distributions between real and synthetic data"""
#         print("=== Distribution Comparison ===\n")
        
#         # Target variable distribution
#         print("1. Target Variable Distribution (readmitted_bin):")
        
#         real_target = self.real_train['readmitted_bin'].value_counts(normalize=True).sort_index()
#         syn_target = self.syn_train['readmitted_bin'].value_counts(normalize=True).sort_index()
        
#         target_comparison = pd.DataFrame({
#             'Real': real_target,
#             'Synthetic': syn_target,
#             'Difference': abs(real_target - syn_target)
#         })
#         print(target_comparison)
#         print()
        
#         # Numeric columns comparison
#         print("2. Numeric Columns Statistics:")
#         numeric_cols = self.real_train.select_dtypes(include=[np.number]).columns[:5]
        
#         for col in numeric_cols:
#             real_mean = self.real_train[col].mean()
#             syn_mean = self.syn_train[col].mean()
#             real_std = self.real_train[col].std()
#             syn_std = self.syn_train[col].std()
            
#             # KS test for distribution similarity
#             ks_stat, ks_pval = stats.ks_2samp(
#                 self.real_train[col].dropna(),
#                 self.syn_train[col].dropna()
#             )
            
#             print(f"\n{col}:")
#             print(f"  Real:      Mean={real_mean:.2f}, Std={real_std:.2f}")
#             print(f"  Synthetic: Mean={syn_mean:.2f}, Std={syn_std:.2f}")
#             print(f"  KS-test:   statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
#             print(f"  Similar: {'✓ Yes' if ks_pval > 0.05 else '✗ No'}")
        
#         # Categorical columns comparison
#         print("\n\n3. Categorical Columns Distribution (sample):")
#         categorical_cols = ['gender', 'race', 'age']
        
#         for col in categorical_cols:
#             if col in self.real_train.columns:
#                 print(f"\n{col}:")
#                 real_dist = self.real_train[col].value_counts(normalize=True).head(5)
#                 syn_dist = self.syn_train[col].value_counts(normalize=True).head(5)
                
#                 comparison = pd.DataFrame({
#                     'Real': real_dist,
#                     'Synthetic': syn_dist
#                 }).fillna(0)
#                 print(comparison)
#         print()

#     def visualize_comparison(self):
#         """Create visualization comparing real and synthetic data"""
#         print("=== Creating Visualizations ===\n")
        
#         fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
#         # 1. Target distribution
#         target_real = self.real_train['readmitted_bin'].value_counts()
#         target_syn = self.syn_train['readmitted_bin'].value_counts()
        
#         x = np.arange(len(target_real))
#         width = 0.35
        
#         axes[0, 0].bar(x - width/2, target_real.values, width, label='Real', alpha=0.8)
#         axes[0, 0].bar(x + width/2, target_syn.values, width, label='Synthetic', alpha=0.8)
#         axes[0, 0].set_xlabel('Readmission Status')
#         axes[0, 0].set_ylabel('Count')
#         axes[0, 0].set_title('Target Variable Distribution')
#         axes[0, 0].set_xticks(x)
#         axes[0, 0].set_xticklabels(target_real.index)
#         axes[0, 0].legend()
        
#         # 2. Age distribution
#         if 'age' in self.real_train.columns:
#             age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
#                         '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
            
#             real_age = self.real_train['age'].value_counts()
#             syn_age = self.syn_train['age'].value_counts()
            
#             age_comp = pd.DataFrame({
#                 'Real': real_age,
#                 'Synthetic': syn_age
#             }).reindex([a for a in age_order if a in real_age.index or a in syn_age.index]).fillna(0)
            
#             age_comp.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
#             axes[0, 1].set_title('Age Distribution')
#             axes[0, 1].set_xlabel('Age Group')
#             axes[0, 1].set_ylabel('Count')
#             axes[0, 1].legend()
#             axes[0, 1].tick_params(axis='x', rotation=45)
        
#         # 3. Time in hospital
#         if 'time_in_hospital' in self.real_train.columns:
#             axes[1, 0].hist(self.real_train['time_in_hospital'], bins=30, alpha=0.5, label='Real', density=True)
#             axes[1, 0].hist(self.syn_train['time_in_hospital'], bins=30, alpha=0.5, label='Synthetic', density=True)
#             axes[1, 0].set_xlabel('Time in Hospital (days)')
#             axes[1, 0].set_ylabel('Density')
#             axes[1, 0].set_title('Time in Hospital Distribution')
#             axes[1, 0].legend()
        
#         # 4. Number of medications
#         if 'num_medications' in self.real_train.columns:
#             axes[1, 1].hist(self.real_train['num_medications'], bins=30, alpha=0.5, label='Real', density=True)
#             axes[1, 1].hist(self.syn_train['num_medications'], bins=30, alpha=0.5, label='Synthetic', density=True)
#             axes[1, 1].set_xlabel('Number of Medications')
#             axes[1, 1].set_ylabel('Density')
#             axes[1, 1].set_title('Number of Medications Distribution')
#             axes[1, 1].legend()
        
#         # 5. Gender distribution
#         if 'gender' in self.real_train.columns:
#             gender_real = self.real_train['gender'].value_counts()
#             gender_syn = self.syn_train['gender'].value_counts()
            
#             x = np.arange(len(gender_real))
#             axes[2, 0].bar(x - width/2, gender_real.values, width, label='Real', alpha=0.8)
#             axes[2, 0].bar(x + width/2, gender_syn.values, width, label='Synthetic', alpha=0.8)
#             axes[2, 0].set_xlabel('Gender')
#             axes[2, 0].set_ylabel('Count')
#             axes[2, 0].set_title('Gender Distribution')
#             axes[2, 0].set_xticks(x)
#             axes[2, 0].set_xticklabels(gender_real.index)
#             axes[2, 0].legend()
        
#         # 6. Number of diagnoses
#         if 'number_diagnoses' in self.real_train.columns:
#             axes[2, 1].hist(self.real_train['number_diagnoses'], bins=20, alpha=0.5, label='Real', density=True)
#             axes[2, 1].hist(self.syn_train['number_diagnoses'], bins=20, alpha=0.5, label='Synthetic', density=True)
#             axes[2, 1].set_xlabel('Number of Diagnoses')
#             axes[2, 1].set_ylabel('Density')
#             axes[2, 1].set_title('Number of Diagnoses Distribution')
#             axes[2, 1].legend()
        
#         plt.tight_layout()
#         plt.savefig('real_vs_synthetic_comparison.png', dpi=300, bbox_inches='tight')
#         print("Visualization saved as 'real_vs_synthetic_comparison.png'\n")
#         plt.show()

#     def run(self):
#         """
#         Main validation pipeline
#         """
#         try:
            
#             # Load processed data
#             logging.info("Loading processed datasets...")
#             real_train = self.load_data(paths_config.PROCESSED_REAL_TRAIN)
#             real_test = self.load_data(paths_config.PROCESSED_REAL_TEST)
#             syn_train = self.load_data(paths_config.PROCESSED_SYN_TRAIN)
#             syn_test = self.load_data(paths_config.PROCESSED_SYN_TEST)
            
#             # Validate each dataset
#             # logging.info("Validating data quality...")
            
#             # self.validation_report['real_train'] = self.validate_data_quality(
#             #     real_train, "Real Train", "readmitted_bin"
#             # )
#             # self.validation_report['real_test'] = self.validate_data_quality(
#             #     real_test, "Real Test", "readmitted_bin"
#             # )
#             # self.validation_report['syn_train'] = self.validate_data_quality(
#             #     syn_train, "Synthetic Train", "event"
#             # )
#             # self.validation_report['syn_test'] = self.validate_data_quality(
#             #     syn_test, "Synthetic Test", "event"
#             # )
            
#             # # Validate schema compatibility
#             # logging.info("Validating schema compatibility...")
#             # schema_report = self.validate_schema_compatibility(
#             #     real_train, syn_train, "readmitted_bin", "event"
#             # )
#             # self.validation_report['schema_compatibility'] = schema_report
            
#             # # Check if any validations failed
#             # failed = []
#             # for key, report in self.validation_report.items():
#             #     if isinstance(report, dict) and report.get('status') == 'FAILED':
#             #         failed.append(key)
            
#             # if failed:
#             #     self.validation_report['overall_status'] = False
#             #     logging.error(f"Validation FAILED for: {failed}")
#             # else:
#             #     self.validation_report['overall_status'] = True
#             #     logging.info("All validations PASSED")
            
#             # # Align and save validated data
#             # logging.info("Aligning schemas and saving validated data...")
            
#             # train_real_path, train_syn_path = self.align_and_save_data(
#             #     real_train, syn_train, "readmitted_bin", "event", "train"
#             # )
#             # test_real_path, test_syn_path = self.align_and_save_data(
#             #     real_test, syn_test, "readmitted_bin", "event", "test"
#             # )

#             self.compare_distributions()
#             self.visualize_comparison()
            
#             # Save validation report
#             self.save_validation_report()

            
#             return {
#                 'status': self.validation_report['overall_status'],
#                 'report_path': paths_config.VALIDATION_REPORT,
#                 # 'validated_paths': {
#                 #     'real_train': train_real_path,
#                 #     'real_test': test_real_path,
#                 #     'syn_train': train_syn_path,
#                 #     'syn_test': test_syn_path
#                 # }
#             }

#         except Exception as e:
#             logging.error(f"Error in data validation: {str(e)}")
#             raise CustomException(e, sys)


# if __name__ == "__main__":
#     start = time.time()
#     logging.info("STAGE_START | name=DATA_VALIDATION")
    
#     try:
#         validation = DataValidation()
#         result = validation.run()
        
#         if not result['status']:
#             logging.error("Validation failed! Check validation report.")
#             sys.exit(1)
        
#         elapsed = round(time.time() - start, 2)
#         logging.info("STAGE_END | name=DATA_VALIDATION | status=SUCCESS | duration=%ss", elapsed)

#     except Exception as e:
#         logging.error(f"Pipeline failed: {str(e)}")
#         raise CustomException(e, sys)















# import pandas as pd
# import os
# import sys
# from readmit.components.logger import logging
# from readmit.components.exception import CustomException
# from readmit.configuration import paths_config

# class DataValidation:
#     """
#     Validates, aligns schema, and saves transformed datasets.
#     """

#     def __init__(self, output_dir: str = paths_config.TRANSFORMED_DIR):
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)

#     def load_data(self, path: str) -> pd.DataFrame:
#         try:
#             ext = os.path.splitext(path)[1].lower()
#             if ext == ".csv":
#                 df = pd.read_csv(path)
#             elif ext == ".parquet":
#                 df = pd.read_parquet(path)
#             else:
#                 raise ValueError(f"Unsupported file format: {ext}")
#             short_path = os.path.relpath(path)
#             logging.info(f"Loaded {short_path} | Shape: {df.shape}")
#             return df
#         except Exception as e:
#             raise CustomException(e, sys)

#     def validate_missing(self, df: pd.DataFrame, data_name: str) -> pd.DataFrame:
#         # Check for empty
#         if df.empty:
#             raise ValueError(f"{data_name} is empty!")
#         # Fill missing values
#         missing_cols = df.isnull().sum()[lambda x: x > 0]
#         if not missing_cols.empty:
#             logging.warning(f"{data_name} has missing values:\n{missing_cols}")
#             df.fillna(method="ffill", inplace=True)
#         return df

#     def align_schema(self, real_df, synth_df, target_real, target_synth):
#         common_cols = sorted(set(real_df.columns) & set(synth_df.columns))

#         if target_real and target_real not in common_cols:
#             common_cols.append(target_real)
#         if target_synth and target_synth not in common_cols:
#             common_cols.append(target_synth)

#         real_df = real_df[[c for c in common_cols if c in real_df.columns]].copy()
#         synth_df = synth_df[[c for c in common_cols if c in synth_df.columns]].copy()

#         # Align types
#         for col in common_cols:
#             if col in real_df.columns and col in synth_df.columns:
#                 if real_df[col].dtype != synth_df[col].dtype:
#                     try:
#                         synth_df[col] = pd.to_numeric(synth_df[col], errors='coerce').astype(real_df[col].dtype)
#                     except:
#                         logging.warning(f"Could not convert column '{col}' type")
#         return real_df, synth_df

#     def save_data(self, df: pd.DataFrame, path: str):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         ext = os.path.splitext(path)[1].lower()
#         if ext == ".csv":
#             df.to_csv(path, index=False)
#         elif ext == ".parquet":
#             df.to_parquet(path, index=False)
#         short_path = os.path.relpath(path)
#         logging.info(f"Saved transformed data to {short_path}")

#     def run(self):
#         try:
#             # Load all datasets
#             real_train = self.load_data(paths_config.PROCESSED_REAL_TRAIN)
#             real_test  = self.load_data(paths_config.PROCESSED_REAL_TEST)
#             syn_train  = self.load_data(paths_config.PROCESSED_SYN_TRAIN)
#             syn_test   = self.load_data(paths_config.PROCESSED_SYN_TEST)

#             # Validate missing values
#             real_train = self.validate_missing(real_train, "Real Train")
#             real_test  = self.validate_missing(real_test, "Real Test")
#             syn_train  = self.validate_missing(syn_train, "Synthetic Train")
#             syn_test   = self.validate_missing(syn_test, "Synthetic Test")

#             # Align schema
#             real_train, syn_train = self.align_schema(real_train, syn_train, "readmitted_bin", "event")
#             real_test,  syn_test  = self.align_schema(real_test,  syn_test,  "readmitted_bin", "event")

#             # Save transformed datasets
#             self.save_data(real_train, paths_config.TRANSFORMED_REAL_TRAIN)
#             self.save_data(real_test,  paths_config.TRANSFORMED_REAL_TEST)
#             # self.save_data(syn_train, paths_config.TRANSFORMED_SYN_TRAIN)
#             # self.save_data(syn_test,  paths_config.TRANSFORMED_SYN_TEST)
#             self.save_data(syn_train, paths_config.TRANSFORMED_SYN_TRAIN2)
#             self.save_data(syn_test,  paths_config.TRANSFORMED_SYN_TEST2)

#             logging.info("<=== All datasets validated and saved successfully! ===>")

#         except Exception as e:
#             raise CustomException(e, sys)


# if __name__ == "__main__":
#     from readmit.components.data_validation import DataValidation
#     logging.info(f">>>>>> stage DATA VALIDATION started <<<<<<")
#     validation = DataValidation(output_dir=paths_config.TRANSFORMED_DIR)
#     validation.run()
#     logging.info(f">>>>>> stage DATA VALIDATION completed <<<<<<\n\nx==========x")