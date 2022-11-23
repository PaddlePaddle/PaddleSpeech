import paddlespeech.s2t.io.wav2vec2.dataloader


def _train_loader_specifics(self, dataset, loader_kwargs):
        sampler = loader_kwargs.get("sampler", None)
        # Shuffling should really only matter for the train stage. Shuffling
        # will also lead to more padding in batches if the order was otherwise
        # sorted by length.
        shuffle = loader_kwargs.get("shuffle", False)
        if shuffle and not self.distributed_launch:
            if sampler is not None:
                raise ValueError(
                    "Cannot specify both shuffle=True"
                    "and a sampler in loader_kwargs"
                )
            sampler = ReproducibleRandomSampler(dataset)
            self.train_sampler = sampler
            loader_kwargs["sampler"] = self.train_sampler
            # Delete the shuffle flag, since you cannot specify both a sampler and
            # shuffling:
            del loader_kwargs["shuffle"]

        # Possibly make a DistributedSampler or a wrapper for some other sampler
        if self.distributed_launch and not isinstance(dataset, IterableDataset):
            drop_last = loader_kwargs.get("drop_last", False)
            # num_replicas arg is equal to world_size
            # and retrieved automatically within
            # DistributedSampler obj.
            if sampler is not None:
                self.train_sampler = DistributedSamplerWrapper(
                    sampler,
                    rank=self.rank,
                    drop_last=drop_last,
                    shuffle=shuffle,
                )

                # with DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = self.train_sampler
            elif loader_kwargs.get("batch_sampler") is None:
                # no sampler and batch-sampler
                self.train_sampler = DistributedSampler(
                    dataset, rank=self.rank, shuffle=True, drop_last=drop_last
                )

                # with DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = self.train_sampler
            else:  # batch_sampler was specified
                self.train_sampler = DistributedSamplerWrapper(
                    loader_kwargs.get("batch_sampler", None),
                    rank=self.rank,
                    shuffle=True,
                )
                loader_kwargs["batch_sampler"] = self.train_sampler
        elif self.distributed_launch and isinstance(dataset, IterableDataset):
            logger.warning(
                "Cannot automatically solve distributed sampling "
                "for IterableDataset."
            )
        return loader_kwargs


def make_dataloader(
        self, dataset, stage, **loader_kwargs
    ):
        """Creates DataLoaders for Datasets.

        This is used by ``fit()`` and ``evaluate()`` if they just receive
        Datasets.

        Alternatively, this can be called from outside the Brain subclass.
        In that case, the DataLoader should be passed to ``fit()`` in place
        of the dataset.

        The Stage.TRAIN DataLoader is handled specially. It has extra args for
        shuffle and drop_last. In DDP a DistributedSampler is created (unless
        the dataset is an IterableDataset).

        NOTE
        ----
        Some important DataLoader arguments are passed via **loader_kwargs,
        e.g., batch_size, num_workers, pin_memory.

        NOTE
        ----
        By default, ``evaluate()`` specifies ckpt_prefix=None to stop the test
        DataLoader being added to the checkpointer. If you need to add a
        recoverable after saving checkpoints (e.g., at test time, after
        checkpointing the training), and still be able to recover reasonably,
        you should probably specify ``allow_partial_load=True``.

        Arguments
        ---------
        dataset : Dataset
            A set of data to use to create data loader. If the Dataset is a
            DynamicItemDataset, PaddedBatch is used as the default collate_fn,
            unless specified in loader_kwargs.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        ckpt_prefix : str, None
            Prefix to use for SaveableDataLoader Checkpoint name. The Stage
            name is added to this to create the full key. Set to None to not
            save the DataLoader.
        **loader_kwargs : dict
            Additional keyword arguments to the DataLoader.
            E.g., batch_size, num_workers, pin_memory.
        """
        # TRAIN stage is handled specially.
        # if stage == train:
        #     loader_kwargs = _train_loader_specifics(dataset, loader_kwargs)
        dataloader_ = dataloader.make_dataloader(
            dataset, **loader_kwargs
        )
        return dataloader_