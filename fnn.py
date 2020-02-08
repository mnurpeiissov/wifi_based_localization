import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import glob
import matplotlib.pyplot as plt
import numpy as np
# set random seed
torch.manual_seed(6)
EPOCHS = 200
# use gpu if available else use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the data, Create a dataset
train_path = 'C:/Users/Administrator/Desktop/dev/wifi_loc/data/train_data'
dev_path = 'C:/Users/Administrator/Desktop/dev/wifi_loc/data/dev_data'
test_path = 'C:/Users/Administrator/Desktop/dev/wifi_loc/data/test_data'


def read_data(path):
    files = [f for f in glob.glob(path + '**/*.csv')]
    return files


def read_batch(idx, files):
    df = pd.read_csv(files[idx])
    df = df.fillna(value=-120)
    features = df.iloc[:, 0:368].apply(lambda x: (x + 120))
    targets = df[list('xy')]
    features, targets = torch.from_numpy(features.values).float().to(device), torch.from_numpy(targets.values).float().to(device)
    return features, targets

# Error functions


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, y_pred, y):
        loss = self.mse(y_pred, y) + self.eps
        loss = torch.sqrt(loss.sum(1))
        loss = torch.sum(loss)/(len(y))
        return loss


def calculate_error(target,test):
    test, target = test.cpu().detach().numpy(), target.cpu().detach().numpy()
    mse =(target-test)**2
    mse = mse + 1e-6
    mse = np.sum(mse, axis = 1)
    mse = np.sum(mse)
    rmse = np.sqrt(mse/len(target))
    return rmse
    
# Plotting function


def plot_trajectories(cnt, test_csv_files):
    features_test, targets_test = read_batch(cnt, test_csv_files)
    preds_test = model.forward(features_test)
    to_plot_x_pred, to_plot_y_pred, to_plot_x_target, to_plot_y_target = [],[],[],[]
    preds_test, targets_test = preds_test.cpu().detach().numpy(), targets_test.cpu().detach().numpy()
    for i in range(len(preds_test)):
        to_plot_x_pred.append(preds_test[i][0])
        to_plot_y_pred.append(preds_test[i][1])
        to_plot_x_target.append(targets_test[i][0])
        to_plot_y_target.append(targets_test[i][1])
 
    plt.plot(to_plot_x_pred,to_plot_y_pred, 'o')
    plt.plot(to_plot_x_target,to_plot_y_target,'x')
    plt.show()


def show_predicted_vals(cnt, file):
    features_test, targets_test = read_batch(cnt, file)
    preds_test = model.forward(features_test)
    preds_test, targets_test = preds_test.cpu().detach().numpy(), targets_test.cpu().detach().numpy()
    for i in range(len(preds_test)):
        print('predicted value is {} and actual value is{}------ the L1 distances are on X is:{} and on Y is:{}'.format(preds_test[i], targets_test[i], (preds_test[i][0] - targets_test[i][0]),(preds_test[i][1] - targets_test[i][1])))


def eval_model(model, test_csv_files):
    test_rmse = 0
    losses_calc = []
    sum_test_rmse = 0
    criterion = RMSELoss()
    for cnt in range(len(test_csv_files)):
        features_test, targets_test = read_batch(cnt, test_csv_files)
        preds_test = model.forward(features_test)
        test_rmse = criterion(preds_test, targets_test)
        # print(test_rmse)
        sum_test_rmse += test_rmse
        losses_calc.append(calculate_error(targets_test, preds_test))

    print('RMSE error on test data is {}'.format(sum_test_rmse / len(test_csv_files)))



# write the model


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 100)
        self.fc3 = nn.Linear(100, output)
        #self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
       # x = self.drop_layer(x)
        x = torch.tanh(self.fc2(x))
        # x = self.drop_layer(x)
        # x = x.view(-1, 64)
        # x = torch.relu(self.fc3(x))
        x = (self.fc3(x))

        return x

# train the model


train_csv_files = read_data(train_path)
dev_csv_files = read_data(dev_path)
test_csv_files = read_data(test_path)
num_data_to_train = len(train_csv_files)


def train_the_model():
    model = FNN(368, 200, 2).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, weight_decay=4e-3)
    criterion = RMSELoss()
    for epoch in range(EPOCHS):
        model.train()
        for num_files in range(num_data_to_train):
            features_train, targets_train = read_batch(num_files, train_csv_files)
            preds = model.forward(features_train)
            loss = criterion(preds, targets_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if num_files == range(num_data_to_train)[-1]:
                with torch.no_grad():
                    dev_loss = 0
                    for num_dev_files in range(len(dev_csv_files)):
                        features_dev, targets_dev = read_batch(num_dev_files, dev_csv_files)
                        dev_preds = model.forward(features_dev)
                        dev_loss += criterion(dev_preds, targets_dev)
                        if num_dev_files == range(len(dev_csv_files))[-1]:
                            print('epoch number : {} and the loss is {}'.format(epoch, loss))
                            print('validation loss over all part trajectories is {}'.format(dev_loss/len(dev_csv_files)))
    torch.save(model, 'C:/Users/Administrator/Desktop/dev/wifi_loc/FNN_dropout.pt')
    return model


def load_the_model(path):
    model = torch.load(path)
    model.eval()
    return model


# Evaluate the model
model = load_the_model('C:/Users/Administrator/Desktop/dev/wifi_loc/CNN_part_full_data')
#model = train_the_model()




