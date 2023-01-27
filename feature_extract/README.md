## I3D Feature Extraction

### Data Preparation

>Download the UCF-Crime datasets.

We follow this [work](https://github.com/piergiaj/pytorch-i3d) to extract i3d features.

Step:
* split videos to frames:

```python
	python video2frames_split.py
```

* extract features:

```python
	python i3d_extract.py
```
We extract the features with 10-crop and continuous 16-frames(no overlap). We copy last frames to retain all frames 
>1600 frames --> 100 clips
>
>1601 frames --> 101 clips, (1601 + 15)/16 = 101

But the GT(frame_label/gt-ucf.npy) we used drops the last frames:
>1615 frames -->  1600 frames --> 100 clips

Extracted results will be saved as numpy data:
```python
    ./data
    └── UCF-Crime
        ├── Abuse001_x264_i3d.npy
        ├── Abuse002_x264_i3d.npy
        ├── Abuse003_x264_i3d.npy
        ├── Abuse004_x264_i3d.npy
        ......
        ├── Normal_Videos_944_x264.npy

```
* Split 10 crop features to one crop features

```python
def ten2one(source,dst):
    features=os.listdir(source)
    for feature in tqdm(features):
        data=np.load(os.path.join(source,feature))
        for i in range(10):
            np.save("{}/{}_{}.npy".format(dst,feature.split(".npy")[0],i),data[i])
```
