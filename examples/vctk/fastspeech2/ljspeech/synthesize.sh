
#!/bin/bash
FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ../synthesize.py \
  --fastspeech2-config=conf/default.yaml \
  --fastspeech2-checkpoint=exp/default/checkpoints/snapshot_iter_100000.pdz \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=pwg_ljspeech_ckpt_0.5/pwg_default.yaml \
  --pwg-checkpoint=pwg_ljspeech_ckpt_0.5/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_ljspeech_ckpt_0.5/pwg_stats.npy \
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/default/test \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt
