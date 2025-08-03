# DPPD

## Framework

<p align="center"> <img src="./figure/framework.png" width="100%"> </p>


### Environment

You can create the environment via:

```bash
pip install -r requirement.txt
```

### Dataset

#### PDGait
The PDGait dataset is download from [A public data set of walking](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.992585/full). The the data is preprocessed through:

```bash
cd lib/data/pdgait \\
python convert_pd.py \\
python preprocess_pd.py
```

#### 3DGait
The 3DGait dataset is download from [Video-based-gait-analysis](https://github.com/lisqzqng/Video-based-gait-analysis-for-dementia). The the data is preprocessed through:

```bash
cd lib/data/3dgait \\
python convert_pd.py \\
python preprocess_pd.py
``` 

### Training

There are three traing stages.

- stage 1: pretrain 3DmotionBERT:

```bash
bash pretrain.sh
```
- stage 2: train Gait Denoiser:

```bash
bash pretrain_diffusion.sh
```

- stage 3: train Classifier:

```bash
pythoon pd_score.py
```

### Checkpoints

It will be released soon.