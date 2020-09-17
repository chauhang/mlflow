# pylint: disable=W0223
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

class_names = ["negative", "neutral", "positive"]


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.Tensor(target, dtype=torch.long),
        }


class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PRE_TRAINED_MODEL_NAME = "bert-base-cased"
        self.EPOCHS = 5
        self.df = None
        self.tokenizer = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.optimizer = None
        self.total_steps = None
        self.scheduler = None
        self.loss_fn = None
        self.BATCH_SIZE = 32
        self.MAX_LEN = 160
        n_classes = len(class_names)

        self.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)

    @staticmethod
    def to_sentiment(rating):
        rating = int(rating)
        if rating <= 2:
            return 0
        elif rating == 3:
            return 1
        else:
            return 2

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = GPReviewDataset(
            reviews=df.content.to_numpy(),
            targets=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return DataLoader(ds, batch_size=batch_size, num_workers=4)

    def prepare_data(self):
        self.df = pd.read_csv(
            "https://drive.google.com/uc?id=1zdmewp7ayS4js4VtrJEHzAheSW-5NBZv"
        )
        self.df["sentiment"] = self.df.score.apply(self.to_sentiment)
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        self.df_train, self.df_test = train_test_split(
            self.df, test_size=0.1, random_state=RANDOM_SEED
        )
        self.df_val, self.df_test = train_test_split(
            self.df_test, test_size=0.5, random_state=RANDOM_SEED
        )

        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )

    def setOptimizer(self):
        self.optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        self.total_steps = len(self.train_data_loader) * self.EPOCHS

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def startTraining(self, model):
        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(self.EPOCHS):

            print(f"Epoch {epoch + 1}/{self.EPOCHS}")

            train_acc, train_loss = self.train_epoch(model)

            print(f"Train loss {train_loss} accuracy {train_acc}")

            val_acc, val_loss = self.eval_model(model, self.val_data_loader)
            print(f"Val   loss {val_loss} accuracy {val_acc}")

            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), "best_model_state.bin")
                best_accuracy = val_acc

    def train_epoch(self, model):
        model = model.train()
        losses = []
        correct_predictions = 0

        for data in tqdm(self.train_data_loader):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return (
            correct_predictions.double() / len(self.train_data_loader),
            np.mean(losses),
        )

    def eval_model(self, model, data_loader):
        model = model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                loss = self.loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return correct_predictions.double() / len(data_loader), np.mean(losses)

    def get_predictions(self, model, data_loader):
        model = model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                texts = d["review_text"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values


if __name__ == "__main__":
    model = SentimentClassifier()
    model = model.to(model.device)
    model.prepare_data()
    model.setOptimizer()
    model.startTraining(model)

    print("TRAINING COMPLETED!!!")

    test_acc, _ = model.eval_model(model, model.test_data_loader)

    print(test_acc.item())

    y_review_texts, y_pred, y_pred_probs, y_test = model.get_predictions(
        model, model.test_data_loader
    )

    print(classification_report(y_test, y_pred, target_names=class_names))

    print("\n\n\n SAVING MODEL")
    torch.save(model.state_dict(), "bert_pytorch.pt")
