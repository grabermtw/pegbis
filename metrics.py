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

            
            accuracy = accuracy_score(gt_all, predicted_all)
            precision = precision_score(gt_all, predicted_all)
            recall = recall_score(gt_all, predicted_all)
            f1 = f1_score(gt_all, predicted_all)

            print("  {}:".format(dataset_type))
            print("    Accuracy:", accuracy)
            print("    Precision:", precision)
            print("    Recall:", recall)
            print("    F1-score:", f1)
