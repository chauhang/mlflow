# pylint: disable=W0221
# pylint: disable=W0613
# pylint: disable=W0223
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow.pytorch


class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data():
    iris = load_iris()
    data = iris.data
    labels = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
    )

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    return X_train, X_test, y_train, y_test


def train_model(model, epochs, X_train, y_train):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        out = model.forward(X_train)
        loss = criterion(out, y_train).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("number of epoch", epoch, "loss", float(loss))

    return model


def test_model(model, X_test, y_test):
    with torch.no_grad():
        predict_out = model.forward(X_test)
        _, predict_y = torch.max(predict_out, 1)

        print(
            "prediction accuracy", float(accuracy_score(y_test.cpu(), predict_y.cpu())),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Classification Torchscripted model")

    parser.add_argument(
        "--tracking-uri", type=str, default="http://localhost:5000/", help="mlflow tracking uri"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to run (default: 50)"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IrisClassifier()
    model = model.to(device)
    X_train, X_test, y_train, y_test = prepare_data()
    scripted_model = torch.jit.script(model)  # scripting the model
    scripted_model = train_model(scripted_model, args.epochs, X_train, y_train)
    test_model(scripted_model, X_test, y_test)

    mlflow.tracking.set_tracking_uri(args.tracking_uri)
    mlflow.start_run()
    mlflow.pytorch.log_model(scripted_model, "model")  # logging scripted model
    uri_path = mlflow.get_artifact_uri()
    mlflow.pytorch.load_model(uri_path + "/model")  # loading scripted model
    mlflow.end_run()
