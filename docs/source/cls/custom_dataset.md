# Customize Dataset for Audio Classification

Following this tutorial you can customize your dataset for audio classification task by using `paddlespeech` and `paddleaudio`.

A base class of classification dataset is `paddleaudio.dataset.AudioClassificationDataset`. To customize your dataset you should write a dataset class derived from `AudioClassificationDataset`. 

Assuming you have some wave files that stored in your own directory. You should prepare a meta file with the information of filepaths and labels. For example the absolute path of it is `/PATH/TO/META_FILE.txt`:
```
/PATH/TO/WAVE_FILE/1.wav cat
/PATH/TO/WAVE_FILE/2.wav cat
/PATH/TO/WAVE_FILE/3.wav dog
/PATH/TO/WAVE_FILE/4.wav dog
```
Here is an example to build your custom dataset in `custom_dataset.py`:

```python
from paddleaudio.datasets.dataset import AudioClassificationDataset

class CustomDataset(AudioClassificationDataset):
    # All *.wav file with same sample rate 16k/24k/32k/44k.
    sample_rate = 16000
    meta_file = '/PATH/TO/META_FILE.txt'
    # List all the class labels
    label_list = [
        'cat',
        'dog',
    ]

    def __init__(self):
        files, labels = self._get_data()
        super(CustomDataset, self).__init__(
            files=files, labels=labels, feat_type='raw')

    def _get_data(self):
        '''
        This method offer information of wave files and labels.
        '''
        files = []
        labels = []

        with open(self.meta_file) as f:
            for line in f:
                file, label_str = line.strip().split(' ')
                files.append(file)
                labels.append(self.label_list.index(label_str))

        return files, labels
```

Then you can build dataset and data loader from `CustomDataset`:
```python
import paddle
from paddleaudio.features import LogMelSpectrogram

from custom_dataset import CustomDataset

train_ds = CustomDataset()
feature_extractor = LogMelSpectrogram(sr=train_ds.sample_rate)

train_sampler = paddle.io.DistributedBatchSampler(
    train_ds, batch_size=4, shuffle=True, drop_last=False)
train_loader = paddle.io.DataLoader(
    train_ds,
    batch_sampler=train_sampler,
    return_list=True,
    use_buffer_reader=True)
```

Train model with `CustomDataset`:
```python
from paddlespeech.cls.models import cnn14
from paddlespeech.cls.models import SoundClassifier

backbone = cnn14(pretrained=True, extract_embedding=True)
model = SoundClassifier(backbone, num_class=len(train_ds.label_list))
optimizer = paddle.optimizer.Adam(
    learning_rate=1e-6, parameters=model.parameters())
criterion = paddle.nn.loss.CrossEntropyLoss()

steps_per_epoch = len(train_sampler)
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        waveforms, labels = batch
        # Need a padding when lengths of waveforms differ in a batch.
        feats = feature_extractor(waveforms)        
        feats = paddle.transpose(feats, [0, 2, 1])
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if isinstance(optimizer._learning_rate,
                        paddle.optimizer.lr.LRScheduler):
            optimizer._learning_rate.step()
        optimizer.clear_grad()

        # Calculate loss
        avg_loss = loss.numpy()[0]

        # Calculate metrics
        preds = paddle.argmax(logits, axis=1)
        num_corrects = (preds == labels).numpy().sum()
        num_samples = feats.shape[0]

        avg_acc = num_corrects / num_samples

        print_msg = 'Epoch={}/{}, Step={}/{}'.format(
            epoch, epochs, batch_idx + 1, steps_per_epoch)
        print_msg += ' loss={:.4f}'.format(avg_loss)
        print_msg += ' acc={:.4f}'.format(avg_acc)
        print_msg += ' lr={:.6f}'.format(optimizer.get_lr())
        print(print_msg)
```

If you want to save the checkpoint of model and evaluate from a specific dataset, please see `paddlespeech/cli/exp/panns/train.py` for more details.
