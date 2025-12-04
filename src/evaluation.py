import random
import torch.nn.functional as F
import torch
from math import sqrt
import numpy as np
from statistics import mean
from jellyfish import damerau_levenshtein_distance

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

cross_entr_func = torch.nn.CrossEntropyLoss()

def evaluate_accuracy_next_activity(rnn, test_dataset, acc_func):
    rnn = rnn.to(device)
    accuracies = []
    for batch in [test_dataset]:
        # print(batch.size())
        X = batch[:, :-1, :].to(device)
        # print("X size:", X.size())
        Y = batch[:, 1:, :]
        # print(Y.size())
        target = torch.argmax(Y.reshape(-1, Y.size()[-1]), dim=-1).to(device)
        # print(target.size())
        with torch.no_grad():
            predictions, _ = rnn(X)
        predictions = predictions.reshape(-1, predictions.size()[-1])

        accuracies.append(acc_func(predictions, target).item())

    return mean(accuracies)


def sample_with_temperature(logits, temperature=1.0):
    if temperature == 0:
        indices = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-10)
        indices = torch.multinomial(probs, num_samples=1)

    batch_size = logits.size(0)
    num_classes = logits.size(-1)
    one_hot = torch.zeros(batch_size, 1, num_classes).to(device)
    one_hot.scatter_(2, indices.unsqueeze(-1), 1)
    return one_hot, indices


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def logic_loss(rnn, deepdfa, data, prefix_len, temperature=1.0):
    dataset = data.to(device)
    prefix = dataset[:, :prefix_len, :]

    batch_size = dataset.size()[0]
    target = torch.ones(batch_size, dtype=torch.long, device=device)

    len_traces = dataset.size()[1]
    next_event, rnn_state = rnn(prefix)
    dfa_states, dfa_rew = deepdfa.forward_pi(prefix)

    dfa_state = dfa_states[:, -1, :]

    for step in range(prefix_len, int(len_traces*(1.5))):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = gumbel_softmax(next_event, temperature)
        #print(next_event_one_hot)
        #predicted_traces = torch.cat((predicted_traces, next_event_one_hot), dim=1)
        #transit on the automaton
        dfa_state, dfa_rew = deepdfa.step_pi(dfa_state, next_event_one_hot.squeeze())

        next_event, rnn_state = rnn.forward_from_state(next_event_one_hot, rnn_state)
    loss = cross_entr_func(100*dfa_rew, target)

    return loss


# new logic loss without for loop
# we calculate the logprob of a trace as the sum of log prob of each symbol (NO vanishing prob for long traces!)
def logic_loss_multiple_samples(rnn, deepdfa, data, prefixes, temperature=1.0, num_samples=10):
    dataset = data.to(device)
    prefix_len = random.choice(prefixes)
    #print(prefix_len, sep=" ")
    prefix = dataset[:, :prefix_len, :]

    batch_size, len_traces, num_activities = dataset.size()

    #-----
    end_mask = dataset[:, :, -1] == 1
    first_end_idx = end_mask.float().argmax(dim=1)
    no_end_mask = ~end_mask.any(dim=1)
    first_end_idx[no_end_mask] = dataset.size(1)
    max_truncated_length = first_end_idx.max().item()
    #-----

    #target = torch.ones(batch_size, dtype=torch.long, device=device)

    #extend prefix
    prefix = prefix.unsqueeze(1).repeat(1, num_samples, 1, 1).view(-1, prefix_len, num_activities)

    #calculate next symbol and dfa state
    next_event, rnn_state = rnn(prefix)
    dfa_states, dfa_rew = deepdfa.forward_pi(prefix)
    dfa_state = dfa_states[:, -1, :]

    log_prob_traces = torch.zeros((batch_size*num_samples, 1)).to(device)
    for step in range(prefix_len, max_truncated_length + 10):
        #next_event = next_event[:, -1:, :]
        next_event = F.log_softmax(next_event[:, -1:, :], dim=-1)
        next_event_one_hot = gumbel_softmax(next_event, temperature)

        log_prob_traces += torch.sum(next_event * next_event_one_hot, dim=-1)
        #transit on the automaton
        dfa_state, dfa_rew = deepdfa.step_pi(dfa_state, next_event_one_hot.squeeze())
        #transit the rnn
        next_event, rnn_state = rnn.forward_from_state(next_event_one_hot, rnn_state)

    dfa_rew = dfa_rew.view(batch_size, num_samples, 2)
    dfa_rew = dfa_rew[:, :, 1]
    #log_prob_traces = log_prob_traces.view(batch_size, num_samples)

    #prob_acceptance = torch.sum(torch.nn.functional.softmax(log_prob_traces, dim=-1) * dfa_rew, dim=-1)
    #prob_acceptance = torch.sum(torch.exp(log_prob_traces) * dfa_rew, dim=-1)
    #loss = -torch.log(prob_acceptance.clamp(min=1e-10)).mean()
    loss = -torch.log(torch.mean(dfa_rew, dim=-1).clamp(min=1e-10)).mean()

    #p = prob_acceptance.mean().item()
    #p = max(0.0, min(1.0, p))

    #deviation = 1.96 * sqrt(p * (1 - p) / num_samples)
    return loss, 0 #deviation


def suffix_prediction_with_temperature_with_stop(model, dataset, prefix_len, stop_event, temperature=1.0):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    predicted_traces = prefix

    logits, rnn_state = model(prefix)
    stop_event_idx = stop_event.index(1)
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)

    for step in range(prefix_len, dataset.size(1)):
        logits_step = logits[:, -1, :]
        one_hot, sample_idx = sample_with_temperature(logits_step, temperature)

        predicted_traces = torch.cat((predicted_traces, one_hot), dim=1)

        stop_mask |= (sample_idx.squeeze(-1) == stop_event_idx)
        if torch.all(stop_mask):
            break

        logits, rnn_state = model.forward_from_state(one_hot, rnn_state)

    return predicted_traces


def greedy_suffix_prediction_with_stop(model, dataset, prefix_len, stop_event):
    dataset = dataset.to(device) 
    prefix = dataset[:, :prefix_len, :]
    predicted_traces = prefix

    logits, rnn_state = model(prefix)
    stop_event_idx = stop_event.index(1)
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)

    for step in range(prefix_len, dataset.size(1)):
        logits_step = logits[:, -1, :]

        top_idx = torch.argmax(logits_step, dim=-1, keepdim=True)
        one_hot = F.one_hot(top_idx.squeeze(-1), num_classes=logits_step.size(-1)).float().unsqueeze(1)
        predicted_traces = torch.cat((predicted_traces, one_hot), dim=1)
        stop_mask |= (top_idx.squeeze(-1) == stop_event_idx)

        if torch.all(stop_mask):
            break

        logits, rnn_state = model.forward_from_state(one_hot, rnn_state)

    if not torch.all(stop_mask):
        stop_tensor = torch.tensor(stop_event, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(1)
        stop_tensor = stop_tensor.expand(predicted_traces.size(0), -1, -1)
        predicted_traces = torch.cat((predicted_traces, stop_tensor), dim=1)

    return predicted_traces


def evaluate_compliance_with_formula(deepdfa, traces):
    traces = torch.argmax(traces, dim=-1)

    r, _ = deepdfa(traces)
    accepted = r[:, -1, -1]

    return accepted.mean().item()


def evaluate_DL_distance(predicted_traces, target_traces):
    DL_dists = []
    DL_dists_scaled = []

    for i in range(predicted_traces.size()[0]):
        pred = tensor_to_string(predicted_traces[i])
        targ = tensor_to_string(target_traces[i])

        dl = damerau_levenshtein_distance(pred, targ)
        dl_scaled = 1 - (dl / max(len(pred), len(targ)))

        DL_dists.append(dl)
        DL_dists_scaled.append(dl_scaled)

    return mean(DL_dists), mean(DL_dists_scaled)


def tensor_to_string(one_hot_tensor):
    numpy_array = one_hot_tensor.cpu().numpy()
    stop_event = np.zeros(numpy_array.shape[-1])
    stop_event[-1] = 1

    string = ''
    for event in numpy_array:
        idx = event.argmax()
        string += chr(idx + 161)
        if np.array_equal(event, stop_event):
            break

    return string
