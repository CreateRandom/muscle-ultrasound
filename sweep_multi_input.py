from experiments.sweep_baseline import sweep_baseline
from experiments.sweep_coral import sweep_coral
from experiments.sweep_mil import sweep_mil_attention
from experiments.sweep_multitask import sweep_multitask

if __name__ == '__main__':
    # image aggregation baseline
    sweep_baseline(num_samples=1)
    # mil attention
    sweep_mil_attention(num_samples=1)
    sweep_multitask(num_samples=1)
    sweep_coral(num_samples=1)
