# LARC Baselines

## files
- `larc`
	- directory containing LARC data
- `arc.py`
	- functions for loading, displaying ARC tasks/grids
- `baby_larc.py`
    - generate very easy LARC tasks
- `deconv.py`
    - pytorch Module + training code for predicting entire output grid using a deconv net
- `larc_dataset.py`
    - pytorch datasets for LARC data + functions for making other LARC datasets
- `larc_encoder.py`
	- pytorch Module for encoding LARC task
- `pixel_pred.py`
	- pytorch Module + training code for predicting single cell of output grid