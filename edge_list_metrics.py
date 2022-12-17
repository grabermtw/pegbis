from neural_net_for_edge_lists import CorrectedReflectanceDataset, load_data, LinNet
import time
import pickle
from torch.utils.data import DataLoader
import torch as tr
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms, models

# This file is specifically for edge lists because
# I just ran the edge list network for 17 hours and got an error while getting the metrics :/

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(train_dataloader, test_dataloader, batch_size, model, device):
    # put the model into evaluation mode
    model.eval()

    for label_type in ['weather', 'terrain']:

        print(label_type, "results:")
        for dataset_type in ["Training", "Testing"]:
            # initialize storage for ground truth and predicted labels
            predicted_all = []
            gt_all = []
            # go over all the images
            dataloader = train_dataloader if dataset_type == "Training" else test_dataloader
            for batch in dataloader:
                images = batch["img"]
                # we're going to build the confusion matrix for "weather" predictions
                gts = batch["labels"][label_type]
                target_labels = {label_type: gts.to(device)}
            
                # get the model outputs
                output = model(images.to(device))
            
                # get the most confident prediction for each image
                _, predicteds = output[label_type].cpu().max(1)

                for i in range(len(gts)):
                    predicted_all.append(predicteds[i])
                    gt_all.append(gts[i])

            filename = 'edge_list_results.pkl'
            pickle.dump((gt_all, predicted_all), open(filename, 'wb'))
            accuracy = accuracy_score(gt_all, predicted_all)
            precision = precision_score(gt_all, predicted_all)
            recall = recall_score(gt_all, predicted_all)
            f1 = f1_score(gt_all, predicted_all)

            print("  {}:".format(dataset_type))
            print("    Accuracy:", accuracy)
            print("    Precision:", precision)
            print("    Recall:", recall)
            print("    F1-score:", f1)

device = 'cuda:0'#'cpu'
batch_size = 32

filename = "labeled_data_without_segments.csv"
training_data, testing_data = load_data(filename)
start_time = time.time()

load_results = True

# get the metrics
filename = 'edge_list_model.pkl'
model = pickle.load(open(filename, 'rb'))
training_dataset = CorrectedReflectanceDataset(training_data)
train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
testing_dataset = CorrectedReflectanceDataset(testing_data)
test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)
get_metrics(train_dataloader, test_dataloader, batch_size, model, device)

print("Total runtime:", str(time.time() - start_time))