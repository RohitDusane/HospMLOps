import os, sys, time
from readmit.components.data_ingestion import DataIngestion
from readmit.components.data_validation import DataValidation
from readmit.components.new_preprocessing import DataProcessing
from readmit.components.new_model_training import ModelTrainer
from readmit.components.logger import logging
from readmit.components.exception import CustomException
from readmit.configuration import paths_config
from readmit.utils.main_utils import short_path



if __name__ == "__main__":
    pipeline_start = time.time()
    logging.info("="*70)
    logging.info("STARTING FULL PIPELINE")
    logging.info("="*70)

    try:
        # ------------------------
        # 1. DATA INGESTION
        # ------------------------
        stage_start = time.time()
        logging.info("STAGE_START | name=DATA_INGESTION")

        ingestor = DataIngestion(output_dir=paths_config.PROCESSED_DIR)

        # Real data
        df_real = ingestor.load_data(paths_config.RAW_REAL_DATA_PATH, select_features=False)
        profile_real, df_real = ingestor.get_data_profile(
            target_column="readmitted",
            prefix="real",
            create_binary_target=True,
            binary_target_name="readmitted_bin"
        )
        df_real = ingestor._apply_feature_selection(df_real)
        ingestor.split_and_save(df_real, target_column="readmitted_bin", data_type="real")

        # Optional synthetic ingestion (currently commented)
        # df_syn = ingestor.load_data(paths_config.RAW_SYN_DATA_PATH, select_features=False)
        # df_syn = ingestor.align_synthetic_to_real_schema(df_syn)
        # profile_syn, df_syn = ingestor.get_data_profile(...)
        # ingestor.split_and_save(...)

        stage_elapsed = round(time.time() - stage_start, 2)
        logging.info("STAGE_END | name=DATA_INGESTION | status=SUCCESS | duration=%ss", stage_elapsed)


        # # ------------------------
        # # 2. DATA VALIDATION
        # # ------------------------
        stage_start = time.time()
        logging.info("STAGE_START | name=DATA_VALIDATION")

        validation = DataValidation()
        result = validation.run()
        if not result['status']:
            logging.error("Validation failed! Check validation report.")
            sys.exit(1)

        stage_elapsed = round(time.time() - stage_start, 2)
        logging.info("STAGE_END | name=DATA_VALIDATION | status=SUCCESS | duration=%ss", stage_elapsed)


        # ------------------------
        # 3. DATA PROCESSING
        # ------------------------
        stage_start = time.time()
        logging.info("STAGE_START | name=DATA_PROCESSING")

        processor = DataProcessing(config_path=paths_config.CONFIG_PATH,
                                   output_dir=paths_config.TRANSFORMED_DIR)
        results = processor.run(target_col="readmitted", apply_smote=True)

        stage_elapsed = round(time.time() - stage_start, 2)
        logging.info("STAGE_END | name=DATA_PROCESSING | status=SUCCESS | duration=%ss", stage_elapsed)

        print("\n" + "="*70)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Time taken: {stage_elapsed}s")
        print(f"Real Data Train: {short_path(results['real']['train_path'])}")
        print(f"Real Data Test:  {short_path(results['real']['test_path'])}")
        print("="*70)


        # ------------------------
        # 4. MODEL TRAINING
        # ------------------------
        stage_start = time.time()
        logging.info("STAGE_START | name=MODEL_TRAINING")

        trainer = ModelTrainer(experiment_name="hospital_readmission")

        # Scenario 1: Real → Real
        X_train_real, X_test_real, y_train_real, y_test_real = trainer.load_data(
            train_path=paths_config.TRANSFORMED_REAL_TRAIN,
            test_path=paths_config.TRANSFORMED_REAL_TEST,
            target_col="readmitted_bin"
        )

        results_real = trainer.train_all_models(
            X_train_real, y_train_real, X_test_real, y_test_real,
            scenario="real_to_real",
            do_hyperparam_tuning=True
        )
        comparison_real = trainer.compare_models(results_real, "real_to_real")
        print("\nReal → Real Results:")
        print(comparison_real.to_string(index=False))

        stage_elapsed = round(time.time() - stage_start, 2)
        logging.info("STAGE_END | name=MODEL_TRAINING | status=SUCCESS | duration=%ss", stage_elapsed)

        total_elapsed = round(time.time() - pipeline_start, 2)
        logging.info("="*70)
        logging.info("FULL PIPELINE COMPLETED SUCCESSFULLY | TOTAL TIME=%ss", total_elapsed)
        logging.info("="*70)

    except Exception as e:
        logging.error("PIPELINE FAILED", exc_info=True)
        raise CustomException(e, sys)
