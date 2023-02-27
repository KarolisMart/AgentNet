# Agent-based Graph Neural Networks
This is a reference implementation for [Agent-based Graph Neural Networks](https://arxiv.org/abs/2206.11010) as presented at ICLR 2023.

# Runinng The Code

To run the code you should first install the conda environment `conda env create -f  environment.yml`. Note that one of the dependencies for `synthetic.py` is `sage` which is not available on Windows.

## Experiments
Below we provide the example commands used to run the experiments. 

### Synthetic
We use the following command for the synthetic datasets:

`python synthetic.py --dataset 'fourcycles' --num_seeds 10 --verbose --dropout 0.0 --num_agents 16 --num_steps 16 --batch_size 200 --reduce 'log' --hidden_units 128 --epochs 10000 --warmup 0 --gumbel_decay_epochs 1 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention`

Model type can be `[agent, GIN, PPGN, SMP, DropGIN]` and the datasets are `[fourcycles, CSL, WL3, ladder]`. To run `GIN` with random feature augmentation add `--augmentation random` flag.
To run the Random Walk AgentNet include `--random_agent` flag. For the Simple AgentNet include the `--basic_agent` flag.
To perfrom the subgraph density ablation study from Figure 2a add `--density_ablation` flag.

### Graph Classification
To run the standard grid search on the TU graph classification datasets use the following command with additional `--slurm --gpu_jobs` or `--grid_search` flags:

`python graph_classification.py --dataset 'MUTAG' --num_seeds 10 --verbose --dropout 0.0 --num_agents 18 --num_steps 16 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 500 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --model_type agent --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --global_agent_pool --bias_attention`

The dataset can be `[MUTAG, PTC_GIN, PROTEINS, IMDB-BINARY, IMDB-MULTI, DD, REDDIT-BINARY]`. Don't forget to adjust the number of agents. To run the lower memory consumption grid search on `REDDIT-BINARY` uncomment line 421. To perform the ablation on `DD` dataset over the number of agents and steps (Figure 2d) uncomment lines 426-431.

### OGB
The following command runs the training on our best model configuration:

`python ogb_mol.py --dataset 'ogbg-molhiv' --num_seeds 10 --dropout 0.0 --num_agents 26 --num_steps 16 --batch_size 64 --reduce 'log' --hidden_units 128 --epochs 100 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --affine  --bias --global_agent_pool  --bias_attention`

Add the `--slurm --gpu_jobs` flags to run all 10 random seeds. 

### QM9
The following command will run training on the QM9. The targets are `0-11`. The code reports values in eV instead of Hartree used in the paper. 

`python qm9.py --target 0 --dropout 0.0 --num_agents 18 --num_steps 8 --batch_size 32 --reduce 'log' --hidden_units 128 --epochs 350 --warmup 0 --gumbel_decay_epochs 50 --clip_grad 1.0 --weight_decay 0.01 --self_loops --lr 0.0001 --mlp_width_mult 2 --input_mlp --activation_function 'leaky_relu' --negative_slope 0.01 --gumbel_min_temp 0.66666667 --gumbel_temp 0.66666667 --affine  --bias --global_agent_pool  --bias_attention --edge_negative_slope 0.01 --agent_node_mlp --readout_mlp --complete_graph`
