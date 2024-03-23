import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch

#set randomness so results can be reproduced

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

#function to clean the tweets
def textPreProcess(doc):
    #check if null
    if pd.isnull(doc):
        return ""
    temp = doc.lower()
    #remove unnecessary info
    temp = re.sub(r"@[A-Za-z0-9_]+", "", temp)
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub(r"www.\S+", "", temp)
    temp = re.sub("[0-9]", "", temp)
    return temp

#define our mlp
class MLP(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #architecture of the mlp
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(vocab_size, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 4),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        out = self.all_layers(x)
        return out

#define the dataset
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.labels = y

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.labels)

# class for handling sentiment data
class SentimentDataset(Dataset):
    def __init__(self, data, vocab_dict, label_dict,  x_train, y_train):
        self.data = data
        self.vocab_dict = vocab_dict
        self.label_dict = label_dict
        self.x_train= x_train
        self.y_train=y_train

    def __getitem__(self, index):
        sentence, sentiment =  self.x_train, self.y_train
        sentence = sentence.iloc[index]
        sentiment = sentiment.iloc[index]
        features = self.process_sentence(sentence)
        label = self.process_label(sentiment)
        return features, label

    def __len__(self):
        return len(self.data)

    def process_sentence(self, sentence):
        #process the sentence by creating a vector from the vocab dict
        vector = np.zeros((len(self.vocab_dict), 1))
        for word in sentence.split(" "):
            if word in self.vocab_dict:
                dim = self.vocab_dict[word]
                vector[dim, 0] = 1
        return torch.tensor(vector, dtype=torch.float32).reshape(1, len(self.vocab_dict))

    def process_label(self, sentiment):
        # process the label by creating one-hot encoding
        label_vector = np.zeros((4,))
        label_vector[self.label_dict[sentiment]] = 1
        return torch.tensor(label_vector)
#path to our dataset
sentiment_training_data = "/twitter_training.csv"
colnames = ['Tweet ID', 'entity', 'sentiment', 'Tweet content', 'polarity']
df_tweets = pd.read_csv(sentiment_training_data, encoding='UTF', names=colnames, encoding_errors='ignore')

#select the subset of data we will use
df = df_tweets[['sentiment', 'Tweet content']].sample(n=70000, random_state=0)

#filter the data
df['Filtered Tweet Content'] = df['Tweet content'].map(lambda cell: textPreProcess(cell))

x = df['Filtered Tweet Content']
y = df['sentiment']


#split our data into test and train
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#create vocabulry dictionaries
dict_x_train = {word: idx for idx, word in enumerate(set(word for sentence in x_train for word in sentence.split(" ")))}

dict_y = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}

#create the custom datasets
train_dataset = SentimentDataset(data=df_train, vocab_dict=dict_x_train, label_dict=dict_y, x_train=x_train, y_train=y_train)
test_dataset = SentimentDataset(data=df_test, vocab_dict=dict_x_train, label_dict=dict_y,  x_train=x_test, y_train=y_test)

#load the datasets with the dataset and batch_size
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

#create the instance of our mlp model
model = MLP(len(dict_x_train))

#set parameters
num_epochs = 250
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training loop
for epoch in range(num_epochs):
    print(epoch)
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(features)
        logits = logits.squeeze()
        label_index = torch.argmax(labels).item()
        loss = criterion(logits.reshape(1, -1), torch.tensor([label_index]))
        loss.backward()
        optimizer.step()

#evaluate on the test set
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for batch_idx, (features, labels) in enumerate(test_loader):
        logits = model(features).squeeze()
        predict = torch.argmax(logits).item()
        truth = torch.argmax(labels).item()
        if predict == truth:
            total+=1
            correct += 1

        else:
          total+=1

#print acuracy
accuracy = correct/total
print(f"Accuracy: {accuracy}")

