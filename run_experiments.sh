#!/bin/bash


echo "Starting VAE high beta training..."

tmux send-keys -t gw:0 "python3 shimmer_metaworld/cli/train_v.py; tmux wait-for -S gw_done" C-m

tmux wait-for gw_done

echo "VAE high beta training finished. Starting TRPO w/ GW go..."

tmux send-keys -t trpo:0 "python3.12 maml_trpo_ml1_gw_go.py --wandb_project=dreamerv3 --wandb_entity=yusuf-pcms-university-of-toulouse --env_name=test-gw-env; tmux wait-for -S trpo_done" C-m

tmux wait-for trpo_done

echo "TRPO finished. Starting GW..."

tmux send-keys -t gw:0 "python3 shimmer_metaworld/cli/train_v.py" C-m

echo "All commands dispatched."
