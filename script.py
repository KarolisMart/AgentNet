import subprocess

datasets = [ 'MUTAG','PTC_GIN','PROTEINS','IMDB-BINARY','IMDB-MULTI','DD']
num_agent={
    'MUTAG': '18',
    'PTC_GIN': '14', 
    'PROTEINS': '39', 
    'IMDB-BINARY': '20',
    'IMDB-MULTI': '13', 
    'DD': '284', 
    'REDDIT-BINARY': '350'
}
models = ['' ]
# 'PPGN', 'DropGIN', 'Baseline', 'GIN''IMDB', 'MUTAG', 
# 定义基本参数
base_params = {
    '--verbose': '',
    '--dropout': '0.0',
    # '--num_agents': '18',
    # '--num_steps': '16',
    '--batch_size': '32',
    '--reduce': 'log',
    '--hidden_units': '128',
    '--epochs': '350',
    '--warmup': '0',
    '--gumbel_decay_epochs': '500',
    '--clip_grad': '1.0',
    '--weight_decay': '0.01',
    '--self_loops': '',
    '--lr': '0.0001',
    '--mlp_width_mult': '2',
    '--model_type': 'agent',
    '--input_mlp': '',
    '--activation_function': 'leaky_relu',
    '--negative_slope': '0.01',
    '--gumbel_min_temp': '0.66666667',
    '--gumbel_temp': '0.66666667',
    '--global_agent_pool': '',
    '--bias_attention': ''
}

# 遍历数据集和模型
for dataset in datasets:
    for model in models:
        # 初始化命令列表
        command = ['python', 'graph_classification.py', '--dataset', dataset]
        
        # 将基本参数添加到命令列表中
        for key, value in base_params.items():
            command.append(key)
            if value:  # 如果值不为空
                command.append(value)

        command.append('--num_agents')
        command.append(num_agent[dataset])

        # 添加模型类型参数
        if model:
            command.append('--model_type')
            command.append(model)

        if model == 'REDDIT-BINARY':
            command.append('--num_steps')
            command.append('2')
        else:
            command.append('--num_steps')
            command.append('4')
        
        # 运行命令
        result = subprocess.run(command, capture_output=True, text=True)

        # 输出结果
        print(f"Running for dataset: {dataset}, model: {model}")
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
