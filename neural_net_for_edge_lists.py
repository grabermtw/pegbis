import torch as tr
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms, models
import random
import csv
import ast
import time
import numpy as np
from PIL import Image as plimg
from graph import get_edge_list_representation
import imageio
from metrics import *
import numpy as np
import pickle

# use a seed to keep the training and testing datasets the same for all models we test
random_seed = 44

# convert from strings to int representation of the labels
def encode(label):
    if label in ['clear', 'water']:
        return 0
    else: # cloudy, land
        return 1

# Used for obtaining the training/testing data
def load_data(filename):
    imgs = []
    weather = []
    terrain = []
    with open(filename) as datacsv:
        reader = csv.DictReader(datacsv)
        for row in reader:
            imgs.append(row["filepath"])
            weather.append(row["weather"])
            terrain.append(row["terrain"])
    shufflelist = list(zip(imgs, weather, terrain))
    random.Random(random_seed).shuffle(shufflelist)
    imgs, weather, terrain = zip(*shufflelist)
    imgs, weather, terrain = list(imgs), list(weather), list(terrain)
    # split into training and test data (use 60% for training, 40% for testing)
    split_size = int(0.6 * len(imgs))
    training_data = (imgs[:split_size], weather[:split_size], terrain[:split_size])
    testing_data = (imgs[split_size:], weather[split_size:], terrain[split_size:])
    return training_data, testing_data

class CorrectedReflectanceDataset(tr.utils.data.Dataset):
    def __init__(self, data):
        self.imgs, self.weather, self.terrain = data

    def __getitem__(self, idx):
        # take the data sample by its index
        img = imageio.imread(self.imgs[idx])
        edges, num = get_edge_list_representation(img, 0.1)
        edges = np.vstack(edges).astype(np.float32)
        edges = tr.tensor(edges)
        
        # return the edge-list representation of the image and the associated labels
        dict_data = {
            'img': edges,
            'labels': {
                'weather': encode(self.weather[idx]),
                'terrain': encode(self.terrain[idx])
            }
        }
        return dict_data
    
    def __len__(self):
        return len(self.imgs)
    
class LinNet(nn.Module):
    def __init__(self, n_weather_classes, n_terrain_classes):
        super(LinNet, self).__init__()
        hid_features = 100
        last_channel = 50
        self.to_hidden = tr.nn.Linear(128 * 128 *4, hid_features)
        self.to_output = tr.nn.Linear(hid_features, last_channel)
 
        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size>, <channels>, <width>, <height>]
        # so, let's do the spatial averaging: reduce <width> and <height> to 1
        self.pool = nn.AdaptiveAvgPool1d(16)
 
        # create separate classifiers for our outputs
        self.weather = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=16, out_features=n_weather_classes)
        )
        self.terrain = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=16, out_features=n_terrain_classes)
        )
    
    def forward(self, x):
        gelu = tr.nn.GELU()
        h = gelu(self.to_hidden(x.reshape(-1,65536)))
        softplus = tr.nn.Softplus()
        x = softplus(self.to_output(h))
        x = self.pool(x)
        
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = tr.flatten(x, start_dim=1)
        
        return {
            'weather': self.weather(x),
            'terrain': self.terrain(x)
        }

    def get_loss(self, net_output, ground_truth):
        
        weather_loss = F.cross_entropy(net_output['weather'], ground_truth['weather'])
        terrain_loss = F.cross_entropy(net_output['terrain'], ground_truth['terrain'])
        
        loss = weather_loss + terrain_loss
        return loss, {'weather': weather_loss, 'terrain': terrain_loss}

def calculate_batch_metrics(predicted, target):
    weather_correct_count = 0
    terrain_correct_count = 0
    
    # should all be same length
    for i in range(len(predicted['weather'])):
        if predicted['weather'][i] == target['weather'][i]:
            weather_correct_count += 1
        if predicted['terrain'][i] == target['terrain'][i]:
            terrain_correct_count += 1
    weather_acc = weather_correct_count / len(predicted['weather'])
    terrain_acc = terrain_correct_count / len(predicted['terrain'])
    return weather_acc, terrain_acc

def main():
    filename = "labeled_data_without_segments.csv"
    training_data, testing_data = load_data(filename)

    N_epochs = 25
    batch_size = 32
    device = 'cuda:0'#'cpu'
    outfilename = 'edge_list_losses_results.csv'


    training_dataset = CorrectedReflectanceDataset(training_data)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    
    
    model = LinNet(n_weather_classes=2, n_terrain_classes=2).to(device)
    
    optimizer = tr.optim.Adam(model.parameters())

    # keep track of training time
    start_time = time.time()
    with open(outfilename, "w+") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["epoch","total loss", "weather loss", "terrain loss"])
        for epoch in range(N_epochs):
            print("Epoch: {}/{}".format(epoch+1, N_epochs))
            # Set to training mode
            model.train()
            # Loss within the epoch
            train_loss = 0.0
            train_losses = {'weather': 0.0, 'terrain': 0.0}
            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0:
                    print("    Batch:", step)
                # Clean existing gradients
                optimizer.zero_grad()

                img = batch['img']
                target_labels = batch['labels']
                
                # Forward pass - compute outputs on input data using the model
                outputs = model(img.to(device))
                try:
                    outputs['weather'] = outputs['weather'].view(32,-1)
                    outputs['terrain'] = outputs['terrain'].view(32,-1)
                except:
                    pass
                target_labels['weather'] = target_labels['weather'].to(device)
                target_labels['terrain'] = target_labels['terrain'].to(device)
                # Compute loss
                
                try:
                    loss_train, losses_train = model.get_loss(outputs, target_labels)
                except:
                    continue
                #total_loss += loss_train.item()
                #batch_accuracy_weather, batch_accuracy_terrain = calculate_batch_metrics(outputs, target_labels)

                train_loss += float(loss_train)
                train_losses['weather'] += float(losses_train['weather'])
                train_losses['terrain'] += float(losses_train['terrain'])

                # Backpropagate the gradients
                loss_train.backward()
                # Update the parameters
                optimizer.step()
                
                #print("Batch number: {:03d}, Training: Loss: {:.4f}, Weather Accuracy: {:.4f}, Terrain Accuracy: {:.4f}".format(i, loss_train, batch_accuracy_weather.item(), batch_accuracy_terrain.item()))
            csvwriter.writerow([epoch, train_loss, train_losses['weather'], train_losses['terrain']])
            f.flush()
    print("Total training time: %s seconds" % (time.time() - start_time))

    # Save the model! (didn't do this the first time and wasted nearly 17 hours of training time)
    filename = 'edge_list_model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # get the metrics
    testing_dataset = CorrectedReflectanceDataset(testing_data)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)
    get_metrics(train_dataloader, test_dataloader, batch_size, model, device)

    print("Total runtime:", str(time.time() - start_time))

if __name__ == "__main__":
    main()