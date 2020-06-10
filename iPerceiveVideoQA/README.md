# iPerceive: Applying Common-Sense Reasoning to Multi-Modal Dense Video Captioning and Video Question Answering
## iPerceive VideoQA

Project for Stanford CS231n: CS231n: Convolutional Neural Networks for Visual Recognition

```Python3 | PyTorch | CNNs | Multi-Head Self Attention```

## Required Packages

```
torch==1.3.0.post2
numpy==1.16.4
tqdm==4.37.0
pysrt==1.1.2
```

To load,
```pip3 install -r requirements.txt```

## Overview

Building upon the architecture proposed by [Kim et al.](http://arxiv.org/abs/2005.06409), we propose iPerceive VideoQA, a model that uses common-sense knowledge to perform VideoQA. Furthermore, we utilize dense captions using iPerceive DVC to offer enhanced correlation between objects identified from video frames and their salient actions expressed through dense captions. We thus empower the model with additional useful telemetry (in explicit textual format to allow for easier matching) for answering questions. 

The figure below outlines the goals of iPerceive VideoQA: (i) build a knowledge base for common-sense reasoning, (ii) supplement features extracted from input modalities: video and text (in the form of dense captions, subtitles and QA) and, (iii) implement the relevant-frames selection problem as a multi-label classification task. As such, we apply a two-stage approach.

![iPerceiveVidQA_Poster](https://github.com/amanchadha/iPerceive/blob/master/iPerceive/iPerceiveVidQA/images/archVidQA.jpg)
<p align="center">Figure 1: Network architecture</p>

### Common-Sense Reasoning Module
We utilize the common-sense generation module proposed in our work to generate common-sense vectors corresponding to each frame of the input video. Note that iPerceive DVC internally utilizes common-sense features for DVC. iPerceive VideoQA builds a common-sense knowledge base, concatenates common-sense features with the features extracted from the convolutional encoder and sends the output downstream.

## Citation

Please cite the work as:

```
iPerceive Publication WIP
```

## Dataset

We train and evaluate iPerceive VideoQA using the TVQA dataset which consists of video frames, subtitles and QA pairs from six TV shows. The train/val/test splits for TVQA are 0.84/0.12/0.06. Each example has five candidate answers with one of them the ground-truth. TVQA is thus a classification task and models can be evaluated based on the accuracy metric.

Step 0: Run ```download.sh``` to download the TVQA dataset (QAs and subtitles) from the TVQA web page [here](http://tvqa.cs.unc.edu/download_tvqa.html).
This will do a checksum match internally to ensure the downloaded files are as expected.
At the end of this step, your ```raw_data``` folder should have the following files/folders:
```
tvqa_subtitles/
tvqa_qa_release/
glove.6B.100d.txt
glove.6B.200d.txt
glove.6B.50d.txt
glove.6B.300d.txt
frm_cnt_cache.json
det_visual_concepts_hq.pickle
tvqa_data.md5
srt_data_cache.json
tvqa_train_processed.json
tvqa_val_processed.json
tvqa_test_public_processed.json
```        
 
Step 1: Please download pre-processed data (video frames, features, QA, subtitles, dense captions) from [here](https://drive.google.com/drive/folders/1ddylfYf6XdqkapQzOxHTf5MGkpSIZpqI?usp=sharing).
Extract all files into ```data/```
At the end of this step, your ```data``` folder should have the following files/folders: 
```
dfeat_frame_test
qafeat
sfeat
vfeatdata_test
csfeatdata_test
densecap_dict_new.json
indicesDicttest.json
indicesDicttrain.json
indicesDictvalid.json
tvqa_data.md5
```

Step 2: To process subtitle files and tokenize all textual sentences, run:

```python preprocessing.py```

Step 3: To build our word vocabulary and extract relevant GloVe vectors, run:

```mkdir cache
python tvqa_dataset.py
```

For words that we do not have an embedding for, random vectors ```np.random.randn(self.embedding_dim) * 0.4``` are used. 
```0.4``` is the standard deviation of the GloVe embedding vectors.

## Results

![results1](https://github.com/amanchadha/iPerceive/blob/master/iPerceive/iPerceiveVidQA/images/results.jpg)
<p align="center">Table 1. A comparison of iPerceive VideoQA with the state-of-the-art.</p>

![results2]((https://github.com/amanchadha/iPerceive/blob/master/iPerceiveVidQA/images/abl.jpg)
<p align="center">Table 2. Results from ablation studies for iPerceive DVC and common-sense reasoning to assess their impact on the performance of iPerceive VideoQA.</p>

## Pre-trained Model
Pre-trained model available [here](https://drive.google.com/file/d/1XCGT9U7mu7rvb7xysck6jSiu6iXEMjAE/view?usp=sharing) and included as ```best_release_7420.pth```

## Usage

### Training 

Train the model using:

```train.sh```

### Testing

To use the pre-trained model and test on the test set:

```inference.sh```

### Sample dataset

To extract QA, subtitles, captions for a particular dataset sample for a quick test:

```python3 sample.py```

## Acknowledgements

- [Baseline implementation](https://github.com/hyounghk/VideoQADenseCapFrameGate-ACL2020) by Hyounghun Kim.
- We obtained the dataset assembling scripts from the [TVQA+ code repository](https://github.com/jayleicn/TVQAplus).
