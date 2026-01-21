import pandas as pd
import numpy as np
import os, sys, time
from readmit.components.logger import logging
from readmit.components.exception import CustomException
from readmit.configuration import paths_config
from sklearn.model_selection import train_test_split
from readmit.constants import TEST_SIZE, RANDOM_STATE, SELECTED_VARIABLES
from readmit.utils.main_utils import short_path
from readmit.entity.config_entity import CORE_FEATURES, IDENTIFIERS, TARGET_COLS


class DataIngestion:
    def __init__(self, output_dir: str = paths_config.PROCESSED_DIR):
        self.output_dir = output_dir
        self.raw_dir = paths_config.RAW_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        self.df = None

    # ======================
    # Load data
    # ======================
    def load_data(self, input_path: str, select_features: bool = True) -> pd.DataFrame:
        try:
            ext = os.path.splitext(input_path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(input_path)
            elif ext == ".parquet":
                df = pd.read_parquet(input_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            initial_shape = df.shape

            # SELECT FEATURES
            if select_features:
                # Separate targets first (if they exist)
                target_cols = [c for c in TARGET_COLS if c in df.columns]

                feature_cols = [c for c in SELECTED_VARIABLES if c in df.columns]
                missing_cols = [c for c in SELECTED_VARIABLES if c not in df.columns]

                if missing_cols:
                    logging.warning("Missing columns in data | cols=%s", missing_cols)

                df_features = df[feature_cols]

                # Reattach targets safely
                df = pd.concat([df_features, df[target_cols]], axis=1)

                logging.info(
                    "Feature selection applied | original=%d | selected=%d | reduction=%.1f%%",
                    initial_shape[1],
                    df.shape[1],
                    (1 - df.shape[1] / initial_shape[1]) * 100,
                )

            
            self.df = df
            logging.info(
                "Data loaded | path=%s | rows=%d | cols=%d",
                short_path(input_path),
                df.shape[0],
                df.shape[1]
            )
            return df

        except Exception as e:
            logging.error("Data load failed | path=%s | stage=DATA_ingestor", short_path(input_path), exc_info=True)
            raise CustomException(e, sys)
        
    def _create_binary_target(self, source_col="readmitted", new_col="readmitted_bin"):
        if source_col not in self.df.columns:
            logging.warning(f"Column {source_col} not found, cannot create binary target")
            return

        # Real dataset: string categories
        if self.df[source_col].dtype == object:
            self.df[new_col] = np.where(self.df[source_col] == "<30", 1, 0)
        
        # Synthetic dataset: boolean or numeric
        elif self.df[source_col].dtype in [bool, np.bool_, int, np.int64, float]:
            self.df[new_col] = self.df[source_col].astype(int)
        
        else:
            logging.warning(f"Unsupported dtype for binary target: {self.df[source_col].dtype}")
            self.df[new_col] = 0  # fallback

        class_dist = self.df[new_col].value_counts(normalize=True)
        logging.info(
            "Binary target created | positive_rate=%.2f%% | col=%s",
            class_dist.get(1, 0) * 100,
            new_col
        )

    def align_synthetic_to_real_schema(self, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        # -------------------------
        # 1. age_group → age
        # -------------------------
        if "age_group" in df.columns:
            AGE_GROUP_TO_AGE = {
                "young": ["[0-10)", "[10-20)", "[20-30)", "[30-40)"],
                "middle": ["[40-50)", "[50-60)"],
                "senior": ["[60-70)", "[70-80)", "[80-90)", "[90-100)"]
            }

            df["age"] = df["age_group"].apply(
                lambda x: rng.choice(AGE_GROUP_TO_AGE.get(x, ["[60-70)"])))
            df.drop(columns=["age_group"], inplace=True)

        # -------------------------
        # 2. Restore real categoricals
        # -------------------------
        if "gender" in df.columns and df["gender"].dtype == bool:
            df["gender"] = df["gender"].map({True: "Male", False: "Female"})

        if "diabetesMed" in df.columns and df["diabetesMed"].dtype == bool:
            df["diabetesMed"] = df["diabetesMed"].map({True: "Yes", False: "No"})

        if "change" in df.columns and df["change"].dtype == bool:
            df["change"] = df["change"].map({True: "Ch", False: "No"})

        if "insulin" in df.columns and df["insulin"].dtype == bool:
            df["insulin"] = df["insulin"].apply(
                lambda x: rng.choice(["Up", "Down", "Steady"]) if x else "No"
            )

        if "A1Cresult" in df.columns and df["A1Cresult"].dtype == bool:
            df["A1Cresult"] = df["A1Cresult"].apply(
                lambda x: rng.choice([">7", ">8"]) if x else "Norm"
            )

        if "readmitted" in df.columns and df["readmitted"].dtype == bool:
            df["readmitted"] = df["readmitted"].apply(
                lambda x: "<30" if x else rng.choice([">30", "NO"])
            )

        # -------------------------
        # 3. Expand race to real categories
        # -------------------------
        if "race" in df.columns:
            RACE_EXPAND = {
                "Caucasian": ["Caucasian"],
                "AfricanAmerican": ["AfricanAmerican"],
                "Other": ["Other", "Hispanic", "Asian", "?"]
            }

            df["race"] = df["race"].apply(
                lambda x: rng.choice(RACE_EXPAND.get(x, ["Other"]))
            )

        return df


    # ======================
    # Save table utility
    # ======================
    def save_markdown_and_csv(self, df: pd.DataFrame, name: str):
        csv_path = os.path.join(self.output_dir, f"{name}.csv")
        md_path = os.path.join(self.output_dir, f"{name}.md")
        df.to_csv(csv_path, index=False)
        df.to_markdown(md_path, index=False)
        logging.info(f"Saved table: {short_path(csv_path)}")
        # logging.info(f"Saved table: {short_path(md_path)}")

    # ======================
    # Categorical stats
    # ======================
    # def _get_categorical_stats(self) -> dict:
    #     """
    #     Returns basic stats for categorical columns.
    #     """
    #     categorical_cols = self.df.select_dtypes(include=["object"]).columns
    #     stats = {}
    #     for col in categorical_cols:
    #         stats[col] = {
    #             "unique_values": self.df[col].nunique(),
    #             "top_5_values": self.df[col].value_counts().head(5).to_dict(),
    #             "missing_pct": (self.df[col].isnull().sum() / len(self.df)) * 100
    #         }
    #     return stats
    
    def _get_categorical_stats(self, prefix: str ='real') -> pd.DataFrame:
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        summary_list = []

        for col in categorical_cols:
            vc = self.df[col].value_counts(dropna=False).reset_index()
            vc.columns = ["Category", "count"]
            vc["Feature"] = col
            vc["percentage"] = (vc["count"] / len(self.df) * 100).round(2)
            vc = vc[["Feature", "Category", "count", "percentage"]]
            summary_list.append(vc)

        if summary_list:
            category_counts_df = pd.concat(summary_list, axis=0).reset_index(drop=True)
            self.save_markdown_and_csv(category_counts_df, f"{prefix}categorical_stats")
            return category_counts_df
        else:
            return pd.DataFrame()

    # ======================
    # Full data profile
    # ======================
    # def get_data_profile(self, target_column: str = "readmitted", prefix: str = 'real') -> dict:
    #     if self.df is None:
    #         raise ValueError("Data not loaded. Call load_data() first.")

    #     numeric_summary = self.df.describe().T.reset_index().rename(columns={"index": "Feature"})
    #     self.save_markdown_and_csv(numeric_summary, f"{prefix}numeric_summary")

    #     cat_summary_df = self._get_categorical_stats(prefix=prefix)

    #     # Only compute target distribution if the column exists
    #     if target_column in self.df.columns:
    #         target_distribution = self.df[target_column].value_counts().to_dict()
    #     else:
    #         logging.warning(f"Target column '{target_column}' not found. Skipping target distribution.")
    #         target_distribution = {}

    #     profile = {
    #         "shape": self.df.shape,
    #         "memory_usage_MB": self.df.memory_usage(deep=True).sum() / 1024**2,
    #         "missing_values": self.df.isnull().sum().to_dict(),
    #         "dtypes": self.df.dtypes.astype(str).to_dict(),
    #         "duplicate_rows": self.df.duplicated().sum(),
    #         "target_distribution": target_distribution,
    #         "numeric_summary": numeric_summary.to_dict(),
    #         "categorical_summary": cat_summary_df.to_dict()
    #     }
    #     profile["duplicated+rows"] = self.df.duplicated().sum()

    #     # Save profile
    #     # profile_df = pd.DataFrame({k: [v] if not isinstance(v, dict) else [str(v)] for k, v in profile.items()})
    #     self.save_markdown_and_csv(profile, f"{prefix}_profile")

    #     return profile

    def get_data_profile(
        self,
        target_column: str = "readmitted",
        prefix: str = "real",
        create_binary_target: bool = True,
        binary_target_name: str = "readmitted_bin") -> dict:

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # --------------------------------------------------
        # Create binary target (optional)
        # --------------------------------------------------
        if create_binary_target and target_column in self.df.columns:
            self._create_binary_target(
                source_col=target_column,
                new_col=binary_target_name
            )

        # --------------------------------------------------
        # Numeric summary
        # --------------------------------------------------
        numeric_summary = (
            self.df.describe()
            .T.reset_index()
            .rename(columns={"index": "Feature"})
        )
        self.save_markdown_and_csv(numeric_summary, f"{prefix}_numeric_summary")

        # --------------------------------------------------
        # Categorical summary
        # --------------------------------------------------
        cat_summary_df = self._get_categorical_stats(prefix=prefix)

        # --------------------------------------------------
        # Target distributions
        # --------------------------------------------------
        target_distribution = {}
        binary_target_distribution = {}

        if target_column in self.df.columns:
            target_distribution = (
                self.df[target_column]
                .value_counts(normalize=True)
                .to_dict()
            )

        if binary_target_name in self.df.columns:
            binary_target_distribution = (
                self.df[binary_target_name]
                .value_counts(normalize=True)
                .to_dict()
            )

        # --------------------------------------------------
        # Profile dictionary
        # --------------------------------------------------
        profile = {
            "shape": self.df.shape,
            "memory_usage_MB": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            "missing_values": self.df.isnull().sum().to_dict(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "target_distribution_original": target_distribution,
            "target_distribution_binary": binary_target_distribution,
            "binary_target_mapping": {
                "<30": 1,
                "NO": 0,
                ">30": 0
            },
            "numeric_summary": numeric_summary.to_dict(),
            "categorical_summary": cat_summary_df.to_dict()
        }

        # --------------------------------------------------
        # Save profile
        # --------------------------------------------------
        profile_df = pd.DataFrame({
            k: [str(v)] if isinstance(v, dict) else [v]
            for k, v in profile.items()
        })

        self.save_markdown_and_csv(profile_df, f"{prefix}_profile")

        return profile, self.df


    def _apply_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in SELECTED_VARIABLES if c in df.columns]

        # Preserve all targets (original + derived)
        target_cols = [c for c in df.columns if c.endswith("_bin") or c in TARGET_COLS]

        cols = list(dict.fromkeys(feature_cols + target_cols))
        return df[cols]




    # ======================
    # Train/Test split
    # ======================
    def split_and_save(self, df: pd.DataFrame, target_column: str, data_type: str, file_ext: str = ".csv"):
        try:
            logging.info(
                "Splitting dataset | name=%s | target=%s | test_size=%.2f",
                data_type, target_column, TEST_SIZE)
            
            cols_to_keep = list(
                dict.fromkeys(
                    SELECTED_VARIABLES +
                    TARGET_COLS +
                    [target_column]
                )
            )

            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df_selected = df[cols_to_keep]

            if target_column not in df_selected.columns:
                raise ValueError(
                    f"Target '{target_column}' missing after selection. "
                    f"Columns present: {df_selected.columns.tolist()}"
                )

            X = df_selected.drop(columns=[target_column])
            y = df_selected[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            out_dir = os.path.join(self.output_dir, data_type)
            os.makedirs(out_dir, exist_ok=True)

            train_path = os.path.join(out_dir, f"train{file_ext}")
            test_path = os.path.join(out_dir, f"test{file_ext}")

            if file_ext == ".csv":
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
            else:
                train_df.to_parquet(train_path, index=False)
                test_df.to_parquet(test_path, index=False)

            logging.info(
                "%s data split saved | Train: %s | Test: %s",
                data_type.capitalize(), train_df.shape, test_df.shape)
            return train_path, test_path

        except Exception as e:
            logging.error(f"Failed to split and save {data_type} data")
            raise CustomException(e, sys)


# TESTING
if __name__=='__main__':
    start = time.time()
    logging.info("STAGE_START | name=DATA_ingestor")

    try:
        ingestor = DataIngestion(output_dir=paths_config.PROCESSED_DIR)

        # Load real data
        # Load raw data ONCE
        df_real = ingestor.load_data(
            paths_config.RAW_REAL_DATA_PATH,
            select_features=False
        )

        # Create binary target
        profile_real, df_real = ingestor.get_data_profile(
            target_column="readmitted",
            prefix="real",
            create_binary_target=True,
            binary_target_name="readmitted_bin"
        )

        # Apply feature selection on SAME df
        df_real = ingestor._apply_feature_selection(df_real)

        # Split
        ingestor.split_and_save(
            df=df_real,
            target_column="readmitted_bin",
            data_type="real"
        )

        # -------- SYNTHETIC DATA --------
        # df_syn = ingestor.load_data(
        #     paths_config.RAW_SYN_DATA_PATH,
        #     select_features=False
        # )

        # # 🔥 ALIGN SYNTHETIC → REAL
        # df_syn = ingestor.align_synthetic_to_real_schema(df_syn)

        # profile_syn, df_syn = ingestor.get_data_profile(
        #     target_column="readmitted",
        #     prefix="synthetic",
        #     create_binary_target=True,
        #     binary_target_name="readmitted_bin"
        # )

        
        # ingestor.split_and_save(
        #     df=df_syn,
        #     target_column="readmitted_bin",
        #     data_type="synthetic",
        #     file_ext=".csv"
        # )
        elapsed = round(time.time() - start, 2)
        logging.info("STAGE_END | name=DATA_ingestor | status=SUCCESS | duration=%ss", elapsed)


    except Exception as e:
        logging.error("STAGE_END | name=DATA_ingestor | status=FAILED")
        raise CustomException(e, sys)

