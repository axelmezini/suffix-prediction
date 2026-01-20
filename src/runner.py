import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pandas as pd
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # set to false for reproducibility, True to boost performance
torch.use_deterministic_algorithms(True)
torch.set_deterministic_debug_mode("error")

import random
import numpy as np
from model import Model
from log import Log
from FiniteStateMachine import DFA
from run import Run
torch.set_printoptions(threshold=float('inf'))

def main():
    root_path = '/data/users/amezini/'
    for dataset in ['sepsis', 'BPIC_2020_travel', 'BPIC_2013_closed']:

        for template_type in ['all']:
            template_support = '85-100'
            declare_model = Model(root_path=root_path, dataset=dataset, template_type=template_type, template_support=template_support)
            declare_model.to_ltl()

            for noise in ['01n', '02n', '03n', '04n']:

                for alpha in [0.75]: #[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
                    full_log = Log(root_path=root_path, dataset=dataset, portion=f'ordered')
                    event_names = full_log.get_event_names()

                    log_train = Log(root_path=root_path, dataset=dataset, portion=f'train_80_{template_type}_{noise}')
                    log_train.set_event_names(event_names)
                    log_train.encode()
                    log_test = Log(root_path=root_path, dataset=dataset, portion=f'test_20_{template_type}')
                    log_test.set_event_names(event_names)
                    log_test.encode()

                    print(f'{full_log.get_first_prefix()}\n{log_train.get_first_prefix()}\n{log_test.get_first_prefix()}')

                    results_folder = f'{root_path}results/' + dataset
                    os.makedirs(results_folder, exist_ok=True)
                    experiment_folder = results_folder + f'/exp{len(os.listdir(results_folder)) + 1}_{template_type}_{noise}_a{int(round(alpha * 100)):02d}/'
                    os.makedirs(experiment_folder, exist_ok=True)

                    nr_activities = len(event_names)
                    dfa = DFA(declare_model.get_ltl_formula(), nr_activities, 'random', event_names + ['end'], experiment_folder)
                    deep_dfa = dfa.return_deep_dfa()

                    train_dataset = log_train.to_tensor(1)
                    test_dataset = log_test.to_tensor(1)

                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    for ds_name, ds in [("train", train_dataset), ("test", test_dataset)]:
                        ds = ds.to(device)
                        __, dfa_rew = deep_dfa.forward_pi(ds.to(device))
                        prob_acceptance = dfa_rew[:, -1, 1]
                        mean_prob = prob_acceptance.mean()
                        loss = -torch.log(prob_acceptance.clamp(min=1e-10)).mean()
                        print(f"{ds_name.capitalize()} Set — Mean Acceptance: {mean_prob.item():.6f}, Ground Loss: {loss.item():.6f}\n")


                    first_prefix = log_train.get_first_prefix()
                    prefixes = [first_prefix, first_prefix + 1, first_prefix + 2]

                    N_LAST = 0

                    data = []
                    for exp in range(15):
                        run_number = N_LAST + exp + 1
                        run = Run(experiment_folder, run_number, train_dataset, test_dataset, log_test, nr_activities, prefixes, deep_dfa, device, alpha)
                        data.extend(run.run_baseline())
                        data.extend(run.run_bk())

                    df = pd.DataFrame(data)
                    df['dataset'] = dataset
                    df.to_csv(experiment_folder + 'results.csv', index=False)


if __name__ == '__main__':
    main()
