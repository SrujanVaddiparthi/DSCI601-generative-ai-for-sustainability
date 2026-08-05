# Dataset location

The raw Surface Defect Inspection Type-A dataset is intentionally not stored in this Git repository.

Expected prepared structure:

data/
└── prepared_A/
    ├── normal/
    ├── scratches/
    └── spots/

Use `../scripts/prepare_sdi_subset.py` to prepare the SDI Type-A subset after obtaining the original dataset.

The controlled simple-scratch dataset is also excluded because it can be recreated using:

`../simple_scratch_experiment_bundle/generate_simple_scratch_dataset.py`
