#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import random
import paddle
import numpy as np
from paddle import distributed as dist
from kaldiio import WriteHelper
import speechbrain as sb
import soundfile as sf
from speechbrain.dataio.dataio import save_pkl
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch["batch_value"]
        # 这里len 表示的是原始音频占当前pad之后音频的比例
        wavs, lens = batch.sig
        utt_ids = batch.id
        if not os.path.exists("./dump"):
            os.mkdir("./dump")
        utt_wavs = {}
        utt_lens = {}
        # with WriteHelper('ark,t:./dump/paddle_waves.txt') as wav_writer, \
        #      WriteHelper('ark,t:./dump/paddle_waves_len.txt') as wav_len_writer:
        #     for idx, utt in enumerate(utt_ids):
        #         wav_writer(utt, wavs[idx].numpy())
        #         wav_len_writer(utt, lens[idx].numpy())
        
        # with WriteHelper('ark,t:paddle_waves_len.txt') as writer:
        # for idx, utt in enumerate(utt_ids):
        #     utt_wavs[utt] = wavs[idx].numpy()
        #     utt_lens[utt] = lens[idx].numpy()
        
        # save_pkl(utt_wavs, "paddle_waves.pkl")
        # np.savetxt("paddle_waves.txt", utt_wavs)
        # print("paddle waves: {}".format(utt_wavs))
        # np.savetxt("paddle_lens.txt", utt_lens)
        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                # logger.info("augment type: {}".format(type(augment)))
                wavs_aug = augment(wavs, lens)
                # with WriteHelper('ark,t:./dump/paddle_augment_' + str(count) + '_wave.txt') as writer:
                #     for idx, utt in enumerate(utt_ids):
                #         writer(utt, wavs_aug[idx].numpy())
                        
                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = paddle.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = paddle.concat(wavs_aug_tot, axis=0)
            self.n_augment = len(wavs_aug_tot)
            lens = paddle.concat([lens] * self.n_augment)

        # Feature extraction and normalization
        # logger.info("wave process the augment finished, wavs shape: {}, lens: {}".format(wavs.shape, lens))
        # 经过self.hparams.concat_augment之后，数据是缓存了增强的音频数据
        # 假设经过n个数据增强之后，得到的缓存只有的数据之后为：
        # batch_size * (n + 1)，这里的1表示的是原始的音频内容
        # 假设 batch_size = 4，此时维度为 [24, 48000]
        # with WriteHelper('ark,t:./dump/paddle_total_waves.txt') as writer:
        #     for idx in range(wavs.shape[0]):
        #         writer("batach_wave_" + str(idx), wavs[idx].numpy())
        
        # 当前的3s音频一共产生301帧数据，特征维度是[24, 301, 80]
        feats = self.modules.compute_features(wavs)
        # with WriteHelper('ark,t:./dump/paddle_feats.txt') as writer:
        #     for idx in range(feats.shape[0]):
        #         writer("batch_feat_" + str(idx), feats[0].numpy())
        # logger.info("feats shape: {}".format(feats.shape))

        feats = self.modules.mean_var_norm(feats, lens)
        # with WriteHelper('ark,t:./dump/paddle_apply_feats.txt') as writer:
        #     for idx in range(feats.shape[0]):
        #         writer("batch_apply_feat_" + str(idx), feats[0].numpy())
        # logger.info("apply cmvn feats shape: {}".format(feats.shape))
        
        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        # with WriteHelper('ark,t:./dump/embedding.txt') as writer:
        #     for idx in range(embeddings.shape[0]):
        #         writer("batch_embedding_" + str(idx), embeddings[0].numpy())        
        # logger.info("embedding shape: {}".format(embeddings.shape))
        
        outputs = self.modules.classifier(embeddings)
        # with WriteHelper('ark,t:./dump/loss.txt') as writer:
        #     for idx in range(outputs.shape[0]):
        #         writer("batch_spkid_loss_" + str(idx), outputs[0].numpy())   
        # logger.info("outputs shape: {}".format(outputs.shape))

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        batch = batch["batch_value"]
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = paddle.concat([spkid] * self.n_augment, axis=0)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    # "Creates the datasets and their data processing pipelines."
    data_folder = hparams["data_folder"]

    # 1. Declarations:
    logger.info("create the train dataset")
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    logger.info("create the dev dataset")
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]
    logger.info("\n")

    # logger.info("create the label encoder")
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # 2. Define audio pipeline:
    logger.info("create the audio pipeline")
    # takes 中是用来标记audio_pipeline中参数的名称有哪些，以及每个参数的位置
    # audio_pipeline 实际接收的参数任然是传入的数据
    # provides 表示返回的参数的名称
    @sb.utils.data_pipeline.takes("id", "wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig", "utt_id")
    def audio_pipeline(utt_id, wav, start, stop, duration):
        # logger.info("process the audio: {}, start: {}, end: {}".format(wav, start, stop))
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)
        num_frames = stop - start
        sig, fs = sf.read(
            wav, frames=num_frames, start=start,
            dtype="float32"
        )

        # logger.info("sig shape: {}".format(sig.shape))
        # sig = sig.transpose(0, 1).squeeze(1)
        # logger.info("audio dtype: {}".format(sig.dtype))
        # with WriteHelper('ark,t:' + utt_id + ".txt") as writer:
        #     writer(utt_id, sig)

        return sig, utt_id
    # 通过装饰器将 audio_pipeline 变换为一个数据转换类对象DynamicItem
    # 参数是takes中的数据，返回值是provides中的值
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    logger.info("\n")

    # 3. Define text pipeline:
    logger.info("create the speaker encode label")
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        # logger.info("start to generate the spk id label")
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_paddle([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    logger.info("\n")

    # 3. Fit encoder:
    logger.info("load or compute the label encoder")
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    logger.info("save the label info to: {}".format(lab_enc_file))
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )
    logger.info("\n")

    # 4. Set output:
    logger.info("set the data pipeline result to: {}, {}, {}".format("id", "sig", "spk_id_encoded"))
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "utt_id", "spk_id_encoded"])
    logger.info("\n")

    return train_data, valid_data, label_encoder

def main(hparams_file, 
         hparams, 
         run_opts, 
         overrides,
         train_data,
         valid_data):

    # Load the pre-trained model
    # run_on_main(hparams["pretrainer"].collect_files)
    # exit(0)
    # Create experiment directory
    logger.info("star to create the experiment directory")
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    logger.info("\n")

    # Brain class initialization
    logger.info("start to init the SpeakerBrain")
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    logger.info("\n")

    # Training
    logger.info("start to training")
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    # torch.backends.cudnn.benchmark = True
    # 这里使用的是s2t里面的日志配置，没有使用speechbrain中的日志模块
    # logger.info("set the cpu mode")

    # 目前是在cpu模式下进行调试
    # paddle.device.set_device("gpu:1")

    # CLI:
    logger.info("parse the cmd parameters")
    # hparams_file 是任务相关的训练参数配置文件
    # run_opts: 是任务无关的参数，字典形式
    # overrides: 是在命令行中传入的将会覆盖掉hparams_file中的参数
    #            overrides是字符串，每个参数使用 ": "表示键值对
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    logger.info("\n")

    # Initialize ddp (useful only for multi-GPU DDP training)
    # sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    # 开始实例化所有的对象
    logger.info("create all the instance in the yaml file: {}".format(hparams_file))
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    logger.info("hparams: {}\n".format(hparams))

    # Download verification list (to exlude verification sentences from train)
    logger.info("prepare the speaker vierfication trial file")
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)
    logger.info("\n")
    
    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa
    logger.info("start to prepare the train and dev dataset")
    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"]
        },
    )
    logger.info("\n")
    
    logger.info("start to create the dataset")
    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)
    logger.info("\n")
    # print("gpus num: {}".format(run_opts.get("ngpu")))
    # exit(0)
    if run_opts.get("ngpu") > 1:
        dist.spawn(main, 
                   args=(hparams_file, hparams, run_opts, overrides, train_data, valid_data),
                   nprocs=run_opts.get("ngpu"))
    else:
        main(hparams_file, hparams, run_opts, overrides, train_data, valid_data)