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
from metrics import *

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
            imgs.append(row["segmented_filepath"])
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
        img = plimg.open(self.imgs[idx])
        # ditch the transparency
        img = img.convert('RGB')

        # Normalize the image and convert to tensor
        # First calculate the mean and standard deviation of pixel values
        npimg = np.array(img, dtype='float32')
        mean = np.mean(npimg, axis=(0,1))
        std = np.std(npimg, axis=(0,1))
        # prevent divide by zero errors
        mean += np.array([1.0, 1.0, 1.0])
        std += np.array([1.0, 1.0, 1.0])
        transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        img = transform2(img)
        """transform2 = transforms.Compose([transforms.ToTensor()])
        img = transform2(img)"""
        # return the image and the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'weather': encode(self.weather[idx]),
                'terrain': encode(self.terrain[idx])
            }
        }
        return dict_data
    
    def __len__(self):
        return len(self.imgs)
    
class MultiOutputModel(nn.Module):
    def __init__(self, n_weather_classes, n_terrain_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel # size of the layer before the classifier
 
        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size>, <channels>, <width>, <height>]
        # so, let's do the spatial averaging: reduce <width> and <height> to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
 
        # create separate classifiers for our outputs
        self.weather = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_weather_classes)
        )
        self.terrain = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_terrain_classes)
        )
    
    def forward(self, x):
        x = self.base_model(x)
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

filename = "labeled_data.csv"
training_data, testing_data = load_data(filename)

N_epochs = 25
batch_size = 32
device = 'cuda:0'#'cpu'
outfilename = 'segmented_losses_results.csv'


training_dataset = CorrectedReflectanceDataset(training_data)
train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
   
 
model = MultiOutputModel(n_weather_classes=2, n_terrain_classes=2).to(device)
 
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
            # Clean existing gradients
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(img.to(device))
            target_labels['weather'] = target_labels['weather'].to(device)
            target_labels['terrain'] = target_labels['terrain'].to(device)
            # Compute loss
            loss_train, losses_train = model.get_loss(outputs, target_labels)
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

# get the metrics
testing_dataset = CorrectedReflectanceDataset(testing_data)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)
get_metrics(train_dataloader, test_dataloader, batch_size, model, device)

print("Total runtime:", str(time.time() - start_time))