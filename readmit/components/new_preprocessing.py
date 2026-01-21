import pandas as pd
import numpy as np
import os, sys, time
from readmit.components.logger import logging
from readmit.components.exception import CustomException
from readmit.configuration import paths_config
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from readmit.constants import RANDOM_STATE, SELECTED_VARIABLES
from readmit.utils.main_utils import short_path, read_yaml
import joblib


class DataProcessing:
    """
    ✅ ENHANCED: Advanced data preprocessing pipeline based on domain knowledge
    
    Pipeline Steps:
    1. Statistical Independence (remove duplicate patient encounters)
    2. Handle missing values intelligently
    3. Handle nominal features (medications, admission types, etc.)
    4. Encode categorical variables
    5. Scale numerical features
    6. Optional SMOTETomek for class imbalance
    7. Save all transformers for reproducibility
    """
    
    def __init__(
        self,
        config_path: str = paths_config.CONFIG_PATH,
        output_dir: str = paths_config.TRANSFORMED_DIR
    ):
        self.output_dir = output_dir
        self.artifacts_dir = paths_config.ARTIFACTS_DIR
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Load configuration
        try:
            self.config = read_yaml(config_path)
            self.preprocess_config = self.config.get("data_preprocessing", {})
            logging.info("Configuration loaded | path=%s", short_path(config_path))
        except Exception as e:
            logging.warning("Failed to load config, using defaults | error=%s", str(e))
            self.preprocess_config = {}
        
        # Initialize transformers
        self.label_encoders = {}
        self.scaler = None
        self.cat_imputer = None
        self.num_imputer = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None

    # ==========================================================================
    # 1. DATA LOADING
    # ==========================================================================
    
    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from CSV or parquet"""
        try:
            ext = os.path.splitext(path)[1].lower()
            
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            logging.info("Data loaded | path=%s | shape=%s", short_path(path), df.shape)
            return df
            
        except Exception as e:
            logging.error("Failed to load data | path=%s", short_path(path), exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 2. ENFORCE STATISTICAL INDEPENDENCE
    # ==========================================================================
    
    def enforce_independence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ NEW: Enforce statistical independence by keeping only first encounter per patient
        - Violates assumptions if multiple encounters per patient exist
        - Reduces bias from overweighting certain patients
        """
        try:
            initial_shape = df.shape
            
            # Check if we have multiple encounters per patient
            if 'patient_nbr' in df.columns and 'encounter_id' in df.columns:
                duplicates = df['patient_nbr'].value_counts()
                multi_encounter_patients = (duplicates > 1).sum()
                
                if multi_encounter_patients > 0:
                    logging.info(
                        "Multiple encounters detected | patients=%d | total_encounters=%d",
                        multi_encounter_patients,
                        duplicates[duplicates > 1].sum()
                    )
                    
                    # Sort by encounter_id and keep first encounter per patient
                    df = df.sort_values(by='encounter_id', axis='index')
                    df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
                    
                    logging.info(
                        "Statistical independence enforced | before=%s | after=%s | reduction=%.1f%%",
                        initial_shape, df.shape,
                        ((initial_shape[0] - df.shape[0]) / initial_shape[0]) * 100
                    )
            
            return df
            
        except Exception as e:
            logging.error("Failed to enforce independence", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 3. FEATURE TYPE IDENTIFICATION
    # ==========================================================================
    
    # def identify_feature_types(self, df: pd.DataFrame, target_col: str = "readmitted_bin", is_train: bool = True):
    #     """Automatically identify numerical and categorical features"""
    #     try:
    #         # Exclude identifiers and target
    #         exclude_cols = ['encounter_id', 'patient_nbr', target_col, 'readmitted', 'weight']
    #         available_cols = [col for col in df.columns if col not in exclude_cols]
            
    #         # Get numerical features
    #         numerical_features = df[available_cols].select_dtypes(
    #             include=['int64', 'float64', 'int32', 'float32']
    #         ).columns.tolist()
            
    #         # Get categorical features
    #         categorical_features = df[available_cols].select_dtypes(
    #             include=['object', 'category', 'bool']
    #         ).columns.tolist()
            
    #         # Store for later use
    #         self.numerical_features = numerical_features
    #         self.categorical_features = categorical_features
    #         # self.feature_names = numerical_features + categorical_features
    #         if is_train:
    #             self.feature_names = X.columns.tolist()

            
    #         logging.info(
    #             "Feature types identified | numerical=%d | categorical=%d | total=%d",
    #             len(numerical_features),
    #             len(categorical_features),
    #             len(self.feature_names)
    #         )
            
    #         # Log feature names for debugging
    #         logging.info("Numerical features: %s", numerical_features)
    #         logging.info("Categorical features: %s", categorical_features)
            
    #         return numerical_features, categorical_features
            
    #     except Exception as e:
    #         logging.error("Failed to identify feature types", exc_info=True)
    #         raise CustomException(e, sys)

    def identify_feature_types(self, df: pd.DataFrame, target_col: str = "readmitted_bin"):
        """Automatically identify numerical and categorical features"""
        try:
            # Exclude identifiers and target
            exclude_cols = ['encounter_id', 'patient_nbr', target_col, 'readmitted', 'weight']
            available_cols = [col for col in df.columns if col not in exclude_cols]

            # Get numerical features
            numerical_features = df[available_cols].select_dtypes(
                include=['int64', 'float64', 'int32', 'float32']
            ).columns.tolist()

            # Get categorical features
            categorical_features = df[available_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

            # Store for later use
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features

            logging.info(
                "Feature types identified | numerical=%d | categorical=%d | total=%d",
                len(numerical_features), len(categorical_features),
                len(numerical_features) + len(categorical_features))

            logging.info("Numerical features: %s", numerical_features)
            logging.info("Categorical features: %s", categorical_features)

            return numerical_features, categorical_features

        except Exception as e:
            logging.error("Failed to identify feature types", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 4. BASIC CLEANING
    # ==========================================================================
    
    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ ENHANCED: Basic data cleaning with domain-specific logic
        - Replace '?' and 'None' with NaN
        - Remove duplicates
        - Drop identifier and low-value columns
        """
        try:
            df = df.copy()
            initial_shape = df.shape
            
            # ✅ NEW: Replace '?' and 'None' with NaN (specific to this dataset)
            df = df.replace('?', np.nan)
            df = df.replace('None', np.nan)
            
            # Remove duplicate rows
            duplicates_before = df.duplicated().sum()
            if duplicates_before > 0:
                df = df.drop_duplicates()
                logging.info("Duplicates removed | count=%d", duplicates_before)
            
            # ✅ NEW: Drop weight column (96%+ missing values)
            if 'weight' in df.columns:
                missing_pct = df['weight'].isnull().sum() / len(df)
                if missing_pct > 0.95:
                    df = df.drop(columns=['weight'])
                    logging.info("Weight column dropped | missing=%.1f%%", missing_pct * 100)
            
            # Drop identifier columns (not used for modeling)
            id_cols = ['encounter_id', 'patient_nbr']
            df = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')
            
            logging.info("Basic cleaning complete | before=%s | after=%s", initial_shape, df.shape)
            
            return df
            
        except Exception as e:
            logging.error("Basic cleaning failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 5. HANDLE MISSING VALUES (DOMAIN-SPECIFIC)
    # ==========================================================================
    
    def handle_missing_values_domain(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ NEW: Handle missing values with domain knowledge
        - Race: Replace NaN with 'Not Available'
        - Medical specialty: NaN = 'Not Available' (no specialist needed)
        - Diagnoses: Keep NaN for primary, fill with 'Not Available'
        - A1Cresult, max_glu_serum: NaN = 'Not Available' (test not taken)
        - Payer code: NaN = 'NA'
        """
        try:
            df = df.copy()
            df = df.replace('?', np.nan)
            df = df.replace('None', np.nan)
            
            # Race
            if 'race' in df.columns:
                na_count = df['race'].isnull().sum()
                if na_count > 0:
                    df.loc[df['race'].isna(), 'race'] = 'Not Available'
                    logging.info("Race NaN filled | count=%d", na_count)
            
            # Medical specialty
            if 'medical_specialty' in df.columns:
                na_count = df['medical_specialty'].isnull().sum()
                if na_count > 0:
                    df.loc[df['medical_specialty'].isna(), 'medical_specialty'] = 'Not Available'
                    logging.info("Medical specialty NaN filled | count=%d", na_count)
            
            # Diagnoses - drop rows with no diagnosis at all
            diag_cols = ['diag_1', 'diag_2', 'diag_3']
            available_diag = [col for col in diag_cols if col in df.columns]
            
            if available_diag:
                # Drop rows where ALL diagnoses are missing
                all_missing = df[available_diag].isnull().all(axis=1)
                if all_missing.any():
                    rows_dropped = all_missing.sum()
                    df = df[~all_missing]
                    logging.info("Rows with no diagnoses dropped | count=%d", rows_dropped)
            
            # A1C result
            if 'A1Cresult' in df.columns:
                na_count = df['A1Cresult'].isnull().sum()
                if na_count > 0:
                    df.loc[df['A1Cresult'].isna(), 'A1Cresult'] = 'Not Available'
                    logging.info("A1Cresult NaN filled | count=%d", na_count)
            
            # Max glucose serum
            if 'max_glu_serum' in df.columns:
                na_count = df['max_glu_serum'].isnull().sum()
                if na_count > 0:
                    df.loc[df['max_glu_serum'].isna(), 'max_glu_serum'] = 'Not Available'
                    logging.info("Max glucose serum NaN filled | count=%d", na_count)
            
            # Payer code
            if 'payer_code' in df.columns:
                na_count = df['payer_code'].isnull().sum()
                if na_count > 0:
                    df.loc[df['payer_code'].isnull(), 'payer_code'] = 'NA'
                    logging.info("Payer code NaN filled | count=%d", na_count)
            
            return df
            
        except Exception as e:
            logging.error("Domain-specific missing value handling failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 6. HANDLE NOMINAL FEATURES
    # ==========================================================================
    
    def handle_nominal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ NEW: Handle nominal categorical features with domain knowledge
        - Consolidate admission_type_id missing values
        - Consolidate discharge_disposition_id missing values
        - Consolidate admission_source_id missing values
        """
        try:
            df = df.copy()
            df = df.replace('?', np.nan)
            df = df.replace('None', np.nan)
            
            # Admission type: consolidate NaN (6), Not Available (5), Not Mapped (8)
            if 'admission_type_id' in df.columns:
                df['admission_type_id'].replace(6, 5, inplace=True)
                df['admission_type_id'].replace(8, 5, inplace=True)
                logging.info("Admission type IDs consolidated")
            
            # Discharge disposition: consolidate NaN (18), Not Mapped (25), Unknown (26)
            if 'discharge_disposition_id' in df.columns:
                df['discharge_disposition_id'].replace(18, 25, inplace=True)
                df['discharge_disposition_id'].replace(26, 25, inplace=True)
                logging.info("Discharge disposition IDs consolidated")
            
            # Admission source: consolidate NaN (17), Not Available (15), Not Mapped (20), Unknown (21)
            if 'admission_source_id' in df.columns:
                df['admission_source_id'].replace(17, 15, inplace=True)
                df['admission_source_id'].replace(20, 15, inplace=True)
                df['admission_source_id'].replace(21, 15, inplace=True)
                logging.info("Admission source IDs consolidated")
            
            return df
            
        except Exception as e:
            logging.error("Nominal feature handling failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 7. STANDARD MISSING VALUE IMPUTATION
    # ==========================================================================
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        is_train: bool = True
    ) -> pd.DataFrame:
        """Handle remaining missing values with standard imputation"""
        try:
            df = df.copy()
            df = df.replace('?', np.nan)
            df = df.replace('None', np.nan)
            
            # Log missing values before imputation
            missing_before = df[numerical_features + categorical_features].isnull().sum()
            total_missing = missing_before.sum()
            
            if total_missing > 0:
                logging.info("Remaining missing values detected | total=%d", total_missing)
                
                # Numerical imputation (median strategy)
                if numerical_features and len(numerical_features) > 0:
                    if is_train:
                        self.num_imputer = SimpleImputer(strategy='median')
                        df[numerical_features] = self.num_imputer.fit_transform(df[numerical_features])
                        logging.info("Numerical imputer fitted | features=%d", len(numerical_features))
                    else:
                        if self.num_imputer is None:
                            raise ValueError("Numerical imputer not fitted")
                        df[numerical_features] = self.num_imputer.transform(df[numerical_features])
                
                # Categorical imputation (most frequent strategy)
                if categorical_features and len(categorical_features) > 0:
                    if is_train:
                        self.cat_imputer = SimpleImputer(strategy='most_frequent')
                        df[categorical_features] = self.cat_imputer.fit_transform(df[categorical_features])
                        logging.info("Categorical imputer fitted | features=%d", len(categorical_features))
                    else:
                        if self.cat_imputer is None:
                            raise ValueError("Categorical imputer not fitted")
                        df[categorical_features] = self.cat_imputer.transform(df[categorical_features])
                
                # Verify no missing values remain
                missing_after = df.isnull().sum().sum()
                logging.info("Missing value handling complete | remaining=%d", missing_after)
            else:
                logging.info("No remaining missing values detected")
            
            return df
            
        except Exception as e:
            logging.error("Missing value handling failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 8. CATEGORICAL ENCODING
    # ==========================================================================
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_features: list,
        is_train: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding with proper handling of NaNs and unseen categories
        """
        try:
            df = df.copy()
            df = df.replace('?', np.nan)
            df = df.replace('None', np.nan)

            if not categorical_features or len(categorical_features) == 0:
                logging.info("No categorical features to encode")
                return df

            for col in categorical_features:
                if col not in df.columns:
                    logging.warning("Categorical feature %s not found in dataframe", col)
                    continue

                # Fill missing with 'Missing' and convert to string
                df[col] = df[col].fillna('Missing').astype(str)

                if is_train:
                    le = LabelEncoder()
                    # Fit on all values including 'Missing'
                    le.fit(df[col])
                    self.label_encoders[col] = le
                    df[col] = le.transform(df[col])
                    logging.info("Encoder fitted | feature=%s | classes=%d", col, len(le.classes_))
                else:
                    if col not in self.label_encoders:
                        raise ValueError(f"No encoder found for {col}. Process training data first.")
                    
                    le = self.label_encoders[col]
                    # Replace unseen categories with 'Missing'
                    df.loc[~df[col].isin(le.classes_), col] = 'Missing'
                    df[col] = le.transform(df[col])

            logging.info("Categorical encoding complete | features=%d", len(categorical_features))
            return df

        except Exception as e:
            logging.error("Categorical encoding failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 9. NUMERICAL SCALING
    # ==========================================================================
    
    def scale_numerical(
        self,
        df: pd.DataFrame,
        numerical_features: list,
        is_train: bool = True
    ) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        try:
            df = df.copy()
            
            if not numerical_features or len(numerical_features) == 0:
                logging.info("No numerical features to scale")
                return df
            
            if is_train:
                self.scaler = StandardScaler()
                df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
                
                # Log scaling statistics
                logging.info(
                    "Scaler fitted | features=%d | mean=%.2f | std=%.2f",
                    len(numerical_features),
                    df[numerical_features].mean().mean(),
                    df[numerical_features].std().mean()
                )
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted")
                df[numerical_features] = self.scaler.transform(df[numerical_features])
                
                logging.info("Numerical scaling applied | features=%d", len(numerical_features))
            
            return df
            
        except Exception as e:
            logging.error("Numerical scaling failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 10. CLASS IMBALANCE HANDLING - SMOTETomek
    # ==========================================================================
    
    def apply_smotetomek(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling_strategy: float = 0.5
    ):
        """
        Apply SMOTETomek to handle class imbalance
        SMOTETomek = SMOTE (oversampling minority class) + Tomek links (cleaning boundary samples)
        """
        try:
            original_shape = X.shape
            original_dist = y.value_counts()
            
            logging.info(
                "Applying SMOTETomek | original_shape=%s | class_dist=%s",
                original_shape, original_dist.to_dict()
            )
            
            smotetomek = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=RANDOM_STATE,
                smote=SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
            )
            
            X_resampled, y_resampled = smotetomek.fit_resample(X, y)
            
            new_shape = X_resampled.shape
            new_dist = pd.Series(y_resampled).value_counts()
            
            logging.info(
                "SMOTETomek applied | new_shape=%s | new_dist=%s | increase=%.1f%%",
                new_shape, new_dist.to_dict(),
                ((new_shape[0] - original_shape[0]) / original_shape[0]) * 100
            )
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            
        except Exception as e:
            logging.error("SMOTETomek application failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 11. COMPLETE PREPROCESSING PIPELINE
    # ==========================================================================
    
    def preprocess(
        self,
        df: pd.DataFrame,
        target_col: str = "readmitted_bin",
        is_train: bool = True,
        apply_smote: bool = False
    ):
        """
        ✅ ENHANCED: Complete preprocessing pipeline with domain knowledge
        """
        try:
            logging.info(
                "Starting preprocessing | is_train=%s | apply_smote=%s | shape=%s",
                is_train, apply_smote, df.shape
            )
            
            # ✅ NEW: Step 1 - Enforce statistical independence (only for training)
            if is_train:
                df = self.enforce_independence(df)
            
            # Step 2 - Basic cleaning
            df = self.basic_cleaning(df)
            
            # ✅ NEW: Step 3 - Domain-specific missing value handling
            df = self.handle_missing_values_domain(df)
            
            # ✅ NEW: Step 4 - Handle nominal features
            df = self.handle_nominal_features(df)
            
            # Step 5 - Separate features and target
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataframe")
            
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Drop 'readmitted' if it exists (vestigial column)
            if 'readmitted' in X.columns:
                X = X.drop(columns=['readmitted'])
            
            # Step 6 - Identify feature types (only on training)
            if is_train:
                numerical_features, categorical_features = self.identify_feature_types(df, target_col)
            else:
                if self.numerical_features is None or self.categorical_features is None:
                    raise ValueError("Feature types not identified. Process training data first.")
                numerical_features = self.numerical_features
                categorical_features = self.categorical_features
            
            # Ensure features exist in current dataframe
            numerical_features = [f for f in numerical_features if f in X.columns]
            categorical_features = [f for f in categorical_features if f in X.columns]
            
            # Step 7 - Handle remaining missing values
            X = self.handle_missing_values(X, numerical_features, categorical_features, is_train)
            
            # Step 8 - Encode categorical features
            X = self.encode_categorical(X, categorical_features, is_train)
            
            # Step 9 - Scale numerical features
            X = self.scale_numerical(X, numerical_features, is_train)

            # ✅ Step 9.5 - Persist feature schema (TRAIN ONLY)
            if is_train:
                self.feature_names = list(X.columns)
                logging.info("Feature schema learned | feature_count=%d",len(self.feature_names))

            # 🔒 Ensure train-test feature alignment
            if not is_train:

                if self.feature_names is None:
                    raise ValueError(
                        "Feature schema not found. "
                        "Training preprocessing must be executed before inference."
                    )

                missing_cols = set(self.feature_names) - set(X.columns)
                extra_cols = set(X.columns) - set(self.feature_names)

                # Add missing columns
                for col in missing_cols:
                    X[col] = 0

                # Drop unexpected columns
                if extra_cols:
                    logging.warning(f"Dropping extra features: {extra_cols}")
                    X = X.drop(columns=list(extra_cols))

                # Enforce column order
                X = X[self.feature_names]


            
            # Step 10 - Apply SMOTETomek (only on training data)
            if is_train and apply_smote:
                X, y = self.apply_smotetomek(X, y)
            
            logging.info(
                "Preprocessing complete | X_shape=%s | y_shape=%s",
                X.shape, y.shape)
            
            return X, y
            
        except Exception as e:
            logging.error("Preprocessing failed", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 12. SAVE/LOAD TRANSFORMERS
    # ==========================================================================
    
    def save_transformers(self, data_type: str = "real"):
        """Save all fitted transformers"""
        try:
            transformers = {
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'num_imputer': self.num_imputer,
                'cat_imputer': self.cat_imputer,
                'numerical_features': self.numerical_features,
                'categorical_features': self.categorical_features,
                'feature_names': self.feature_names
            }
            
            save_path = os.path.join(
                self.artifacts_dir,
                f"{data_type}_transformers.pkl"
            )
            
            joblib.dump(transformers, save_path)
            
            logging.info("Transformers saved | path=%s", short_path(save_path))
            return save_path
            
        except Exception as e:
            logging.error("Failed to save transformers", exc_info=True)
            raise CustomException(e, sys)
    
    def load_transformers(self, path: str):
        """Load saved transformers"""
        try:
            transformers = joblib.load(path)
            
            self.label_encoders = transformers['label_encoders']
            self.scaler = transformers['scaler']
            self.num_imputer = transformers['num_imputer']
            self.cat_imputer = transformers['cat_imputer']
            self.numerical_features = transformers['numerical_features']
            self.categorical_features = transformers['categorical_features']
            self.feature_names = transformers['feature_names']
            
            logging.info("Transformers loaded | path=%s", short_path(path))
            
        except Exception as e:
            logging.error("Failed to load transformers", exc_info=True)
            raise CustomException(e, sys)

    # ==========================================================================
    # 13. MAIN EXECUTION PIPELINE
    # ==========================================================================
    
    def run(
        self,
        real_train_path: str = None,
        real_test_path: str = None,
        syn_train_path: str = None,
        syn_test_path: str = None,
        target_col: str = "readmitted_bin",
        apply_smote: bool = False
    ):
        """Execute complete preprocessing pipeline"""
        try:
            # Use default paths if not provided
            real_train_path = real_train_path or paths_config.VALIDATED_REAL_TRAIN
            real_test_path = real_test_path or paths_config.VALIDATED_REAL_TEST
            syn_train_path = syn_train_path or paths_config.VALIDATED_SYN_TRAIN
            syn_test_path = syn_test_path or paths_config.VALIDATED_SYN_TEST
            
            # ===== PROCESS REAL DATA =====
            logging.info("="*70)
            logging.info("PROCESSING REAL DATA")
            logging.info("="*70)
            
            # Load real data
            real_train_df = self.load_data(real_train_path)
            real_test_df = self.load_data(real_test_path)
            
            # Preprocess real training data
            X_train_real, y_train_real = self.preprocess(
                real_train_df, target_col, is_train=True, apply_smote=apply_smote)
            
            # Preprocess real test data
            X_test_real, y_test_real = self.preprocess(
                real_test_df, target_col, is_train=False, apply_smote=False)
            
            # Save real data
            real_output_dir = os.path.join(self.output_dir, "real")
            os.makedirs(real_output_dir, exist_ok=True)
            
            train_real = pd.concat([X_train_real, y_train_real.reset_index(drop=True)], axis=1)
            test_real = pd.concat([X_test_real, y_test_real.reset_index(drop=True)], axis=1)
            
            real_train_out = os.path.join(real_output_dir, "train.csv")
            real_test_out = os.path.join(real_output_dir, "test.csv")

            if X_train_real.isnull().any().any():
                raise ValueError("NaNs found in TRAIN features after preprocessing")

            if X_test_real.isnull().any().any():
                raise ValueError("NaNs found in TEST features after preprocessing")

            
            train_real.to_csv(real_train_out, index=False)
            test_real.to_csv(real_test_out, index=False)
            
            # Save transformers
            real_transformer_path = self.save_transformers("real")
            
            logging.info(
                "Real data processed | train=%s | test=%s | saved=%s",
                train_real.shape, test_real.shape, short_path(real_train_out))
            
            # ===== PROCESS SYNTHETIC DATA =====
            logging.info("="*70)
            logging.info("PROCESSING SYNTHETIC DATA")
            logging.info("="*70)
            
            # Create new processor for synthetic data
            syn_processor = DataProcessing(
                config_path=paths_config.CONFIG_PATH,
                output_dir=self.output_dir)
            
            # Load synthetic data
            syn_train_df = syn_processor.load_data(syn_train_path)
            syn_test_df = syn_processor.load_data(syn_test_path)
            
            # Preprocess synthetic training data
            X_train_syn, y_train_syn = syn_processor.preprocess(
                syn_train_df, target_col, is_train=True, apply_smote=apply_smote)
            
            # Preprocess synthetic test data
            X_test_syn, y_test_syn = syn_processor.preprocess(
                syn_test_df, target_col, is_train=False, apply_smote=False)
            
            # Save synthetic data
            syn_output_dir = os.path.join(self.output_dir, "synthetic")
            os.makedirs(syn_output_dir, exist_ok=True)
            
            train_syn = pd.concat([X_train_syn, y_train_syn.reset_index(drop=True)], axis=1)
            test_syn = pd.concat([X_test_syn, y_test_syn.reset_index(drop=True)], axis=1)
            
            syn_train_out = os.path.join(syn_output_dir, "train.csv")
            syn_test_out = os.path.join(syn_output_dir, "test.csv")
            
            train_syn.to_csv(syn_train_out, index=False)
            test_syn.to_csv(syn_test_out, index=False)
            
            # Save transformers
            syn_transformer_path = syn_processor.save_transformers('synthetic')

            logging.info(
            "Synthetic data processed | train=%s | test=%s | saved=%s",
            train_syn.shape, test_syn.shape, short_path(syn_train_out))
        
            # ===== SUMMARY =====
            logging.info("="*70)
            logging.info("PREPROCESSING SUMMARY")
            logging.info("="*70)
            logging.info("Real data:")
            logging.info("  Train: %s → %s", real_train_df.shape, train_real.shape)
            logging.info("  Test:  %s → %s", real_test_df.shape, test_real.shape)
            logging.info("Synthetic data:")
            logging.info("  Train: %s → %s", syn_train_df.shape, train_syn.shape)
            logging.info("  Test:  %s → %s", syn_test_df.shape, test_syn.shape)
            logging.info("="*70)
            
            return {
                'real': {
                    'train_path': real_train_out,
                    'test_path': real_test_out,
                    'transformer_path': real_transformer_path,
                    'train_shape': train_real.shape,
                    'test_shape': test_real.shape
                },
                'synthetic': {
                    'train_path': syn_train_out,
                    'test_path': syn_test_out,
                    'transformer_path': syn_transformer_path,
                    'train_shape': train_syn.shape,
                    'test_shape': test_syn.shape
                }
            }
            
        except Exception as e:
            logging.error("Preprocessing pipeline failed", exc_info=True)
            raise CustomException(e, sys)


# ==========================================================================
# MAIN EXECUTION
# ==========================================================================
if __name__ == "__main__":

    start = time.time()
    logging.info("STAGE_START | name=DATA_PROCESSING")
    try:
        processor = DataProcessing(
            config_path=paths_config.CONFIG_PATH,
            output_dir=paths_config.TRANSFORMED_DIR)
        
        results = processor.run(
            target_col="readmitted_bin",
            apply_smote=True)  # Set to True to apply SMOTETomek for minority class balancing)
        
        elapsed = round(time.time() - start, 2)
        logging.info("STAGE_END | name=DATA_PROCESSING | status=SUCCESS | duration=%ss", elapsed)
        
        print("\n" + "="*70)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Time taken: {elapsed}s")
        print(f"\nReal Data:")
        print(f"  Train: {short_path(results['real']['train_path'])}")
        print(f"  Test:  {short_path(results['real']['test_path'])}")
        print(f"\nSynthetic Data:")
        print(f"  Train: {short_path(results['synthetic']['train_path'])}")
        print(f"  Test:  {short_path(results['synthetic']['test_path'])}")
        print("="*70)
        
    except Exception as e:
        logging.error("STAGE_END | name=DATA_PROCESSING | status=FAILED")
        raise CustomException(e, sys)