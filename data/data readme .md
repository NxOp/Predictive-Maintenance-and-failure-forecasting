# Dataset Information

## NASA Turbofan Engine Degradation Simulation Dataset

This dataset is sourced from the NASA Prognostics Center of Excellence (PCoE) and is commonly referred to as the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

### Download Instructions

The dataset can be downloaded from Kaggle:
[https://www.kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

After downloading, place the files in the `data/raw` directory.

### Dataset Description

The dataset consists of multiple multivariate time series with sensor readings from aircraft engines. Each engine starts with different degrees of initial wear and manufacturing variation, and develops a fault at some point during the series.

The dataset is divided into four subsets:

1. **FD001**: Train trjectories: 100, Test trajectories: 100, Conditions: ONE (Sea Level), Fault Modes: ONE (HPC Degradation)
2. **FD002**: Train trjectories: 260, Test trajectories: 259, Conditions: SIX, Fault Modes: ONE (HPC Degradation)
3. **FD003**: Train trjectories: 100, Test trajectories: 100, Conditions: ONE (Sea Level), Fault Modes: TWO (HPC Degradation, Fan Degradation)
4. **FD004**: Train trjectories: 248, Test trajectories: 249, Conditions: SIX, Fault Modes: TWO (HPC Degradation, Fan Degradation)

### File Format

Each file is a text file with space-separated columns. The columns represent:

1. **engine_id**: Engine unit number
2. **cycle**: Current operational cycle
3. **setting1-3**: Operational settings
4. **sensor1-21**: Sensor measurements

### Ground Truth

For the test data, the ground truth is provided in separate RUL (Remaining Useful Life) files. These files contain the number of remaining operational cycles for each engine in the test set.

### Data Placement

- Raw data files should be placed in `data/raw/`
- Processed data will be stored in `data/processed/`