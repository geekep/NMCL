Baseline Scripts

train_models.py: Trains and validates the model

define_models.py: Defines the model(1-layer 64-d GRU)

create_datasets.py: Handles data loaders

fusion.py: Finds the fusion predictions by taking mean of the regression scores

evaluate_submission.py: Reads submission file to compute score

preprocess.py: Preprocesses all feature files to generate .npy files per data point per feature

run.sh: Runs train_models.py for all feature sets with the required parameters
---

- Adjust paths and directories on preprocess.py according to your directory structure
- Run preprocess.py to generate directories of feature sets containing .npy files per data point
- On train_models.py, adjust dataset_path, dataset_file_path, logger_path, etc.
- dataset_file_path points to a csv file containing all participant IDs(train/validation/test) and corresponding labels for train/development(downscaled by a factor of 25)
    - csv header includes 'ids', 'PHQ_Score'
    - test labels are left empty
- dataset_path contains directories(speech, vision each containing corresponding feature files) and train/validation/test splits
    - Example: data/speech/mfcc, data/vision/ResNet, etc.
    - Example: train_split.csv, dev_split.csv, test_split.csv
- Run train_models.py
    - Validation/test predictions are saved in separate directories
- In fusion.py adjust paths
    - Returns the score obtained from taking the mean of all the predictions
- In evaluate_submissio.py adjust paths to your prediction file for final evaluation

