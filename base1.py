# list_Data_train = np.array(Data_train)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1014, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=0)

model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(20):
    
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
import random
from sklearn import *

labels = pd.read_csv("FINAL_TARGETS_DATES_TRAINTEST.tsv", delimiter='\t')
labels_test = labels.loc[labels['TARGET'] == 'test']
labels = labels.loc[labels['TARGET'] != 'test']
clients = pd.read_csv("FINAL_FEATURES_TRAINTEST.tsv", delimiter='\t')
clients = clients.merge(labels, how="right")
clients.pop("RETRO_DT")
clients.pop("CLIENT_ID")
train, test = np.split(clients, [int(.9*len(clients))])

    
    train_data = train.sample(int(len(train) * 0.05))
    arr_train_data = np.array(train_data)
    print('epoch #' + str(epoch) + ': 0%')
    
    count_good = 0
    
    model.train()
    ind = 0
    progress = 25
    for elem in arr_train_data:
        ind += 1
        if ind % int(len(arr_train_data) * 0.25) == 0:
            print('epoch #' + str(epoch) + ': ' + str(progress) + '%')
            progress += 25
            print('\tPart Accuracy: ' + str(count_good) + '%')
        
        copy_elem = elem[:-1]
        answer_nn = model(torch.from_numpy(copy_elem.astype(np.float32)))
        copy_result = np.array([0, 0])
        value_result = (elem[-1] == '1')
        copy_result[value_result] = 1
        loss = criterion(answer_nn, torch.from_numpy(copy_result.astype(np.float32)))
        value_answer = answer_nn.detach().numpy()[0]
        
        if value_answer > 0.5:
            count_good += 1
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        first_iter = 0
    print('!!!Accuracy: ' + str(count_good))
    torch.save(model.state_dict(), "base1/actual_model.h")

        
