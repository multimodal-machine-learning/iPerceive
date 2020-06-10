# iPerceive: Applying Common-Sense Reasoning to Multi-Modal Dense Video Captioning and Video Question Answering
## iPerceive DVC

Project for Stanford CS231n: CS231n: Convolutional Neural Networks for Visual Recognition

```Python3 | PyTorch | CNNs | LSTMs | Transformers```

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

The figure below outlines the goals of iPerceive DVC: (i) temporally localize a set of events in a video, (ii) build a knowledge base for common-sense reasoning and, (iii) produce a textual description using audio, visual and speech cues for each event. To this end, we apply a three-stage approach.

![iPerceiveDVCArch](https://github.com/amanchadha/iPerceive/blob/master/iPerceive/iPerceiveDVC/images/archDVC.jpg)
<p align="center">Figure 1. Architectural overview of iPerceive DVC. iPerceive DVC generates common-sense vectors from the temporal events that the proposal module localizes (left). Features from all modalities are sent to the corresponding encoder-decoder Transformers (middle). Upon fusing the processed features we finally output the next word in the caption using the distribution over the vocabulary (right).</p>

### Common-Sense Reasoning Module
We utilize the common-sense generation module proposed in our work to generate common-sense vectors corresponding to each frame of the input video. Note that iPerceive DVC internally utilizes common-sense features for DVC. iPerceive VideoQA builds a common-sense knowledge base, concatenates common-sense features with the features extracted from the convolutional encoder and sends the output downstream.

## Citation

Please cite the work as:

```
iPerceive Publication WIP
```

## Dataset setup

- We train and assess iPerceive DVC using the [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) dataset, using a train/val/test split of 0.5:0.25:0.25, 
- We report all results using the validation set (since no ground truth is available for the test set). 
- Download features [I3D (17GB)](https://storage.googleapis.com/mdvc/sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5), [VGGish (1GB)](https://storage.googleapis.com/mdvc/sub_activitynet_v1-3.vggish.hdf5) and put in `./data/` folder (speech segments are already there). 
You may use `wget <link>` or `curl -O <link>` to download the features.

```
a661cfe3535c0d832ec35dd35a4fdc42  sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5
54398be59d45b27397a60f186ec25624  sub_activitynet_v1-3.vggish.hdf5
```

- Setup `conda` environment. Requirements are in file `conda_env.yml`
```bash
# it will create new conda environment called 'mdvc' on your machine 
conda env create -f conda_env.yml
conda activate iPerceive
```

## Usage

### Training

Run the training script. It will, first, train the captioning model and, then, evaluate the predictions of the best model in the learned proposal setting. 
It will take ~30 hours (50 epochs) to run on 2x 2080Ti GPUs. Peak performance is expected after ~30 epochs.
```bash
# make sure to activate environment: conda activate mdvc
# the cuda:1 device will be used for the run
python main.py --device_ids 1
```
The script keeps the log files, including `tensorboard` log, under `./log` directory by default. You may specify other path using `--log_dir` argument. Also, if you stored the downloaded data (`.hdf5`) files in another directory other than `./data`, make sure to specify it using `–-video_features_path` and `--audio_features_path` arguments.

You may also download the pre-trained model [here (~2 GB)](WIP).

### Evaluation

If you want to skip the training procedure, you may replicate the main results of the paper using the prediction files in `./results` and the [official evaluation script](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c).
We have made some minor changes to the script to make it usable with the latest versions of the libraries available as of this writing.

*Evaluating the performance with the learned proposal set up*

Run the official evaluation script using `python3 evaluate/evaluate.py --submission ./results/results_val_learned_proposals_e30.json`. Our final result is 7.87.

*Evaluating the performance on ground truth segments, run the script on each validation part (`./results/results_val_*_e30.json`) against the corresponding ground truth files (use `-r` argument in the script to specify each of them). When both values are obtained, average them to verify the final result. We got 9.9407 and 10.2478 on `val_1` and `val_2` parts, respectively, thus, the average is 10.094.

As we mentioned in the paper, we didn't have access to the full dataset as [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) is distributed as the list of links to YouTube video. Consequently, many videos (~8.8 %) were no longer available at the time when we were downloading the dataset. In addition, some videos didn't have any speech. We filtered out such videos from the validation files and reported the results as `no missings` in the paper. We provide these filtered ground truth files in `./data`.

## Results

![iPerceiveDVCArch](https://github.com/amanchadha/iPerceive/blob/master/iPerceive/iPerceiveDVC/images/dvc_res.jpg)
<p align="center">Tab. 1 shows a comparison of the baseline implementation of iPerceive DVC with the state-of-the-art. Algorithms were split into the ones which "saw" all training videos and others which trained on partially available data (since some YouTube videos which were part of the ActivityNet Captions dataset are no longer available). The results show that our model outperforms the state-of-the-art in most cases.

![iPerceiveDVCArch](https://github.com/amanchadha/iPerceive/blob/master/iPerceive/iPerceiveDVC/images/DVCsample.jpg)
<p align="center">Fig. 2 Qualitative sampling of iPerceive DVC. Captioning results for a sample video from the ActivityNet Captions validation set show better performance owing to common-sense reasoning and end-to-end training.</p>

![iPerceiveDVCArch](https://github.com/amanchadha/iPerceive/blob/master/iPerceive/iPerceiveDVC/images/dvc_ab.jpg)
<p align="center">Tab. 2 shows results from ablation studies for common-sense reasoning and end-to-end training to assess the impact of these design decisions on the performance of iPerceive DVC.</p>

## Misc.

We additionally provide
- the file with subtitles with original timestamps in `./data/asr_en.csv`
- the file with video categories in `./data/vid2cat.json`

## Acknowledgements

- [Baseline implementation](https://github.com/v-iashin/MDVC) by Vladimir Iashin. Vladimir was more than helpful to get us running up-to-speed with our project. Credits to him for providing us the I3D and VGGish features.
