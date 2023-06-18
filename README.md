## kaggle Parkinson's Freezing of Gait Prediction, 15th solution


My method is inspired by object detection.
</br>

`Data Preprocessing`
1. Set the window length, (-1800 ~ +1800) is my best. </br>
2. ['StartHesitation', 'Turn', 'Walking'] columns are splited as much as the length.</br>
3. Signals are reshaped to square, Final shape example = (3, 60, 60) </br>
4. Signals treated like RGB Color Image, passed to backbone model.

```python
# Input Data Preprocessing Examples
time_to = 3600
shape = int(time_to **(1/2))
if self.model_type == 'current' :
    start = 0
    end = 0
    status = 0
    if time - (time_to // 2) < 0 :
        start = 0
        status = 1
    else :
        start = time - (time_to // 2)

    if time + (time_to // 2) > data.shape[0] :
        end = data.shape[0]
        status = 2
    else :
        end = time + (time_to // 2)

    arr = np.zeros((time_to, 3))
    out = data[start : end, : ]
    out_len = out.shape[0]
    if status == 0 :
        arr[:,:] = out
    elif status == 1 :
        arr[time_to-out_len:, :] = out
    else :
        arr[:out_len, :] = out
    out = arr.reshape(3, shape, shape)
```

</br>

`hyperparameters`

* loss_function : Focal Loss, Delta 4, Alpha 0.5</br>
* optimizer : AdamW with LinearLR scheduler for warmup</br>
* drop_rate : 0.7</br>
* drop_path_rate : 0.4</br>
* weight_decay :1e-4 to 3e-4</br>
* learning_rate : 1e-5 to 1e-4</br>
</br>

`Augmentation`
```python
    aug = albumentations.Compose([
        albumentations.VerticalFlip(p=0.1),
        albumentations.HorizontalFlip(p=0.1),
        albumentations.RandomGridShuffle(grid=(3, 3), p= 0.1),
        albumentations.OneOf([
            albumentations.RingingOvershoot((1,2), cutoff = (0.7853981633974483, 1.207963267948966),p=0.1),
            albumentations.Sharpen(alpha=(0.01, 0.05),lightness=(0.01,0.05),p=0.1),
        ], p=0.5),  
        albumentations.OneOf([
            albumentations.CoarseDropout(max_holes = 4, max_height=9, max_width = 9, p=0.1),
            albumentations.ChannelDropout(channel_drop_range=(1, 1), fill_value = 0., p=0.1),
            albumentations.PixelDropout(drop_value=0.,per_channel=True, p = 0.1),
        ], p=0.5)
```

`Highest Submissions`
* Convnext Large, cv 0.341, lb public 0.349, lb private 0.34 
</br>

`Not Worked`
* Increasing window length
* Change model pooling-layer : gem, maxpool
* 1d CNN -> GRU -> Attention Models
* Changing the windowing method, I was prepared 'Past', 'Current', 'Future' but 'Current' only be a meaningful method.
* And many other small changes
</br>

`What I Learned`
* 1st Solutions are using transformers fluently, I realized that basic and theoretical approach is more important than others.
* (as always) trust cv
* More insight into 1d signal processing
</br>
