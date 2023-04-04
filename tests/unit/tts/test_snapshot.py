# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
from itertools import count

from paddle import nn
from paddle.optimizer import Adam

from paddlespeech.t2s.training.extensions.snapshot import Snapshot
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.t2s.training.updater import StandardUpdater


def test_snapshot():
    model = nn.Linear(3, 4)
    optimizer = Adam(parameters=model.parameters())

    # use a simplest iterable object as dataloader
    dataloader = count()

    # hack the training proecss: training does nothing except increse iteration
    updater = StandardUpdater(model, optimizer, dataloader=dataloader)
    updater.update_core = lambda x: None

    trainer = Trainer(updater,
                      stop_trigger=(1000, 'iteration'),
                      out='temp_test_snapshot')
    shutil.rmtree(trainer.out, ignore_errors=True)

    snap = Snapshot(max_size=5)
    trigger = (10, 'iteration')
    trainer.extend(snap, name='snapshot', trigger=trigger, priority=0)

    trainer.run()

    checkpoint_dir = trainer.out / "checkpoints"
    snapshots = sorted(list(checkpoint_dir.glob("snapshot_iter_*.pdz")))
    for snap in snapshots:
        print(snap)
    assert len(snapshots) == 5
    shutil.rmtree(trainer.out)
