from dataclasses import dataclass


@dataclass
class Condition:
    project_name: str
    condition_name: str
    epoch: int


trained_on_philips = [
                     # best in-domain val
                   #  Condition('mus-simplemil', 'MIL-72', 11),
                     # best out-of-domain val
                   #  Condition('mus-simplemil', 'MIL-72', 8),
                  #   Condition('mus-imageagg', 'IMG-44', 11),
                     Condition('mus-multitask', 'TASK-88', 9),
                     # unsupervised, pick best in-domain val
                     Condition('mus-coral', 'CORAL-134', 5)]#,
                     # small validation set, pick best target domain val
                  #   Condition('mus-coral', 'CORAL-134', 3)]

trained_on_esaote = [# best in-domain val
                     Condition('mus-simplemil', 'MIL-28', 7),
                     # best out-of domain val
                     Condition('mus-simplemil', 'MIL-59', 9),
                     Condition('mus-imageagg', 'IMG-22', 14),
                     Condition('mus-multitask', 'TASK-76', 9),
                     # unsupervised, pick best in-domain val
                     Condition('mus-coral', 'CORAL-133', 7)]#,
                     # small validation set, pick best target domain val
                     # Condition('mus-coral', 'CORAL-133', 5)]

conditions_to_run = {'Philips_iU22': trained_on_philips, 'ESAOTE_6100': trained_on_esaote}
