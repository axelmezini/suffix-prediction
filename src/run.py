import os
import random
from copy import deepcopy
import numpy as np
import torch
from RNN import LSTM_model
from training import train
from plotting import plot_loss_over_epoch
from evaluation import suffix_prediction_with_temperature_with_stop, greedy_suffix_prediction_with_stop, evaluate_compliance_with_formula, evaluate_DL_distance

MAX_NUM_EPOCHS = 2000
TEMP = 0.7


class Run:
    def __init__(self, experiment_folder, experiment_number, train_dataset, test_dataset, log_test, nr_activities, prefixes, dfa, device):
        self._run_folder = f'{experiment_folder}run{experiment_number}/'
        self._run_number = experiment_number
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._log_test = log_test
        self._stop_event = [0] * nr_activities + [1]
        self._prefixes = prefixes
        self._dfa = dfa
        self._device = device

        self.set_seed(experiment_number)
        self._rnn = LSTM_model(100, nr_activities + 1, nr_activities + 1)
        self._rnn_bk = deepcopy(self._rnn)
        self.create_folders()

    def run_baseline(self):
        model = deepcopy(self._rnn).to(self._device)
        train_acc, test_acc, sup_losses, _, _, nr_epochs = train(model, self._train_dataset, self._test_dataset, MAX_NUM_EPOCHS)
        plot_loss_over_epoch(sup_losses, f"Supervised_Loss_Baseline", f'{self._run_folder}plots/')
        torch.save(model.state_dict(), f'{self._run_folder}models/rnn_Baseline.pt')

        results = []
        for prefix in self._prefixes:
            temp_predicted_train = suffix_prediction_with_temperature_with_stop(model, self._train_dataset, prefix, temperature=TEMP, stop_event=self._stop_event)
            temp_predicted_test = suffix_prediction_with_temperature_with_stop(model, self._test_dataset, prefix, temperature=TEMP, stop_event=self._stop_event)

            train_dl, train_dl_scaled = evaluate_DL_distance(temp_predicted_train, self._train_dataset)
            test_dl, test_dl_scaled = evaluate_DL_distance(temp_predicted_test, self._test_dataset)

            with open(f'{self._run_folder}predicted_traces/predicted_Baseline_temp_{prefix}.txt', mode='w') as file:
                file.write(self._log_test.decode(temp_predicted_test))

            results.append({
                'run_id': self._run_number,
                'prefix length': prefix,
                'model': 'baseline',
                'sampling strategy': 'temperature',
                'train accuracy': train_acc,
                'test accuracy': test_acc,
                'train DL': train_dl,
                'train DL scaled': train_dl_scaled,
                'test DL': test_dl,
                'test DL scaled': test_dl_scaled,
                'train sat': evaluate_compliance_with_formula(self._dfa, temp_predicted_train),
                'test sat': evaluate_compliance_with_formula(self._dfa, temp_predicted_test),
                'nr_epochs': nr_epochs
            })

            greedy_predicted_train = greedy_suffix_prediction_with_stop(model, self._train_dataset, prefix, stop_event=self._stop_event)
            greedy_predicted_test = greedy_suffix_prediction_with_stop(model, self._test_dataset, prefix, stop_event=self._stop_event)

            train_dl, train_dl_scaled = evaluate_DL_distance(greedy_predicted_train, self._train_dataset)
            test_dl, test_dl_scaled = evaluate_DL_distance(greedy_predicted_test, self._test_dataset)

            with open(f'{self._run_folder}predicted_traces/predicted_Baseline_greedy_{prefix}.txt', mode='w') as file:
                file.write(self._log_test.decode(greedy_predicted_test))

            results.append({
                'run_id': self._run_number,
                'prefix length': prefix,
                'model': 'baseline',
                'sampling strategy': 'greedy',
                'train accuracy': train_acc,
                'test accuracy': test_acc,
                'train DL': train_dl,
                'train DL scaled': train_dl_scaled,
                'test DL': test_dl,
                'test DL scaled': test_dl_scaled,
                'train sat': evaluate_compliance_with_formula(self._dfa, greedy_predicted_train),
                'test sat': evaluate_compliance_with_formula(self._dfa, greedy_predicted_test),
                'nr_epochs': nr_epochs
            })

        del model
        torch.cuda.empty_cache()
        return results

    def run_bk(self):
        model = deepcopy(self._rnn_bk).to(self._device)
        train_acc, test_acc, sup_losses, log_losses, deviations, nr_epochs = train(model, self._train_dataset, self._test_dataset, MAX_NUM_EPOCHS, deepdfa=self._dfa, prefixes=self._prefixes)
        plot_loss_over_epoch(sup_losses, f"Supervised_Loss_BK", f'{self._run_folder}plots/')
        plot_loss_over_epoch(log_losses, f"Log_Loss_BK", f'{self._run_folder}plots/')
        plot_loss_over_epoch(deviations, f"Deviations", f'{self._run_folder}plots/')
        torch.save(model.state_dict(), f'{self._run_folder}models/rnn_BK.pt')

        results = []
        for prefix in self._prefixes:
            temp_predicted_train = suffix_prediction_with_temperature_with_stop(model, self._train_dataset, prefix, temperature=TEMP, stop_event=self._stop_event)
            temp_predicted_test = suffix_prediction_with_temperature_with_stop(model, self._test_dataset, prefix, temperature=TEMP, stop_event=self._stop_event)

            train_dl, train_dl_scaled = evaluate_DL_distance(temp_predicted_train, self._train_dataset)
            test_dl, test_dl_scaled = evaluate_DL_distance(temp_predicted_test, self._test_dataset)

            with open(f'{self._run_folder}predicted_traces/predicted_BK_temp_{prefix}.txt', mode='w') as file:
                file.write(self._log_test.decode(temp_predicted_test))

            results.append({
                'run_id': self._run_number,
                'prefix length': prefix,
                'model': 'with BK',
                'sampling strategy': 'temperature',
                'train accuracy': train_acc,
                'test accuracy': test_acc,
                'train DL': train_dl,
                'train DL scaled': train_dl_scaled,
                'test DL': test_dl,
                'test DL scaled': test_dl_scaled,
                'train sat': evaluate_compliance_with_formula(self._dfa, temp_predicted_train),
                'test sat': evaluate_compliance_with_formula(self._dfa, temp_predicted_test),
                'nr_epochs': nr_epochs
            })

            # Greedy suffix prediction
            greedy_predicted_train = greedy_suffix_prediction_with_stop(model, self._train_dataset, prefix, stop_event=self._stop_event)
            greedy_predicted_test = greedy_suffix_prediction_with_stop(model, self._test_dataset, prefix, stop_event=self._stop_event)

            train_dl, train_dl_scaled = evaluate_DL_distance(greedy_predicted_train, self._train_dataset)
            test_dl, test_dl_scaled = evaluate_DL_distance(greedy_predicted_test, self._test_dataset)

            with open(f'{self._run_folder}predicted_traces/predicted_BK_greedy_{prefix}.txt', mode='w') as file:
                file.write(self._log_test.decode(greedy_predicted_test))

            results.append({
                'run_id': self._run_number,
                'prefix length': prefix,
                'model': 'with BK',
                'sampling strategy': 'greedy',
                'train accuracy': train_acc,
                'test accuracy': test_acc,
                'train DL': train_dl,
                'train DL scaled': train_dl_scaled,
                'test DL': test_dl,
                'test DL scaled': test_dl_scaled,
                'train sat': evaluate_compliance_with_formula(self._dfa, greedy_predicted_train),
                'test sat': evaluate_compliance_with_formula(self._dfa, greedy_predicted_test),
                'nr_epochs': nr_epochs
            })

        del model
        torch.cuda.empty_cache()
        return results

    def set_seed(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # set to false for reproducibility, True to boost performance
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        random_state = random.getstate()
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        return random_state

    def create_folders(self):
        os.makedirs(f'{self._run_folder}plots/', exist_ok=True)
        os.makedirs(f'{self._run_folder}predicted_traces/', exist_ok=True)
        os.makedirs(f'{self._run_folder}models/', exist_ok=True)
