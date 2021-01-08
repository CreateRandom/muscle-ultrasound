from experiments.sweep_baseline import sweep_baseline
from experiments.sweep_coral import sweep_coral
from experiments.sweep_mil import sweep_mil_attention
from experiments.sweep_multitask import sweep_multitask

if __name__ == '__main__':
     # image aggregation baseline
     sweep_baseline(num_samples=24)
     # mil attention
     sweep_mil_attention(num_samples=24)
     # CORAL experiments
     sweep_coral(num_samples=12, layers_to_compute_da_on=[2], lambda_range=(1,10))
     sweep_coral(num_samples=12, layers_to_compute_da_on=[2], lambda_range=(1,10))
     # Multi-task experiments
     sweep_multitask(num_samples=12, classification_task=False)
     sweep_multitask(num_samples=12, classification_task=True)
