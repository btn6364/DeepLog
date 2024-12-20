import torch
import torch.nn as nn
import argparse
import statistics

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(name, window_size):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    hdfs = set()
    # hdfs = []
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
            # hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def main(num_layers, hidden_size, window_size, num_candidates, return_mean):
    # Hyperparameters
    num_classes = 28
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=300.pt'

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))

    # Generate the anomaly scores for 3 microservices
    data_sources = [
        "hdfs_logs_admin_basic",
        "hdfs_logs_auth",
        "hdfs_logs_order"
    ]

    anomaly_scores = {}

    for data_source in data_sources:
        test_abnormal_loader = generate(data_source, window_size)
        # Test the model
        with torch.no_grad():
            microservice_anomaly_scores = []
            for line in test_abnormal_loader:
                num_sequences = 0
                num_abnormal_sequences = 0
                for i in range(len(line) - window_size):
                    num_sequences += 1

                    seq = line[i:i + window_size]
                    label = line[i + window_size]

                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)

                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]  # TODO: Anomaly score?
                    if label not in predicted:  # Abnormal sequence
                        num_abnormal_sequences += 1

                # Compute the anomaly score for each log
                anomaly_score = round(num_abnormal_sequences / num_sequences, 2)
                microservice_anomaly_scores.append(anomaly_score)

        if return_mean:
            microservice_mean_anomaly_score = round(statistics.mean(microservice_anomaly_scores), 3)
            print(f"Microservice Anomaly Score = {microservice_mean_anomaly_score}")
            anomaly_scores[data_source] = microservice_mean_anomaly_score
        else:
            anomaly_scores[data_source] = microservice_anomaly_scores
    return anomaly_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    parser.add_argument('-return_mean', default=True, type=bool)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates
    return_mean = args.return_mean
    main(num_layers, hidden_size, window_size, num_candidates, return_mean)
