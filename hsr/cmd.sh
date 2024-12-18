dataset=real_demo
model=hsr
version=demo
wandb=True
extra_args=""
# extra_args="model.is_continue=True"
# extra_args="test_only=True"
python train.py dataset=${dataset} model=${model} version=${version} wandb=${wandb} ${extra_args}
