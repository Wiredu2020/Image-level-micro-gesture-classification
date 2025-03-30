# Import modules
from config import CFG
from .dataset import label_encodings, MGDataset, MGDataloader
from .models import RestMGModel
import os
import sys
import numpy as np
import itertools
import argparse
from tqdm.notebook import tqdm
from transformers import DistilBertTokenizer
import torch     



def model_pred(model, test_loader,item):
    """
    Predicton Function To Get Model's Prediction

    Args:
        Test_Data_Path : Test data with no labels

    Returns:
        Predictions: A list containing predictions.
    """

    # Initialize tqdm progress bar
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    # Initialize lists to store predictions and labels
    predictions = []

    model.eval()
    # Disable gradients
    with torch.no_grad():
        for batch in tqdm_object:
            # Use model to encode image and text features
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            text_features = model.text_encoder(input_ids=item["input_ids"].to(CFG.device), attention_mask=item["attention_mask"].to(CFG.device))

            # Get embeddings from model projections
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)

            # Calculate the dot product similarity between image and text embeddings
            dot_similarity = image_embeddings @ text_embeddings.T

            # Get the top-5 values and corresponding indices
            _, indices_pred = torch.topk(dot_similarity.squeeze(0), 1)
            indices_pred = indices_pred.detach().cpu().numpy()

            predicted_id = indices_pred + 1
            
            # Append the prediction to list
            predictions.append(predicted_id)
        
    return predictions
    
def model_acc(model,test_loader,item):
    """
    Function to get Model's accuracy if the test data has lasbels
    Arg:
        Test Path : Path to the test data
    Retunrns: 
        Top 1: Model's top 1 Accuracy (Printed out)
        Top 5: Model's top 5 Accuracy

    NB: This is sligthly slower than Model_Pred function as it has extra operations
    """
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    total = len(tqdm_object)
    model.eval()

    acc_t1 ,acc_t5 = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm_object:
                # Use model to encode image and text features
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            text_features = model.text_encoder(input_ids=item["input_ids"].to(CFG.device), attention_mask=item["attention_mask"].to(CFG.device))

                # Get embeddings from model projections
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)

                # Calculate the dot product similarity between image and text embeddings
            dot_similarity = image_embeddings @ text_embeddings.T

                # Get the top-5 values and corresponding indices
            _, indices_pred = torch.topk(dot_similarity.squeeze(0), 5)
            indices_pred = indices_pred.detach().cpu().numpy()

            target = batch["caption"]
                # Get the predicted label
            
            pred_5 = [label_encodings[indices_pred[idx] + 1] for idx in range(5)]
            if pred_5[0] == target[0]:
                acc_t1 +=1
                acc_t5 += 1
            elif  target[0] in pred_5:
                acc_t5 += 1


        acc_1 = acc_t1/total
        acc_5 = acc_t5/total
    print("Test accuracy at top 1 :{:.2f}%".format(acc_1))
    print("Test accuracy at top 5 :{:.2f}%".format(acc_5))
    return (acc_1,acc_5)

def parse_option():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--testingdata', type=str, required = True, help='Path to the test data.')
    parser.add_argument('--Predictions', type=str, required = False, help="State whethere you want Model's Predictions or Accuracy")
    args = parser.parse_args()
    return args

def main(test_path,tokenizer, out= "Predictions"):
    # Load model
    model = RestMGModel().to(CFG.device)
    model.load_state_dict(torch.load("utils/modelscheckpoints/model_weight.pth"))

    encoded_captions = tokenizer(list(label_encodings.values()), padding=True, truncation=True, max_length=CFG.max_length)
    item = {key: torch.tensor(values) for key, values in encoded_captions.items()}

    # Load images
    # If data has labels 
    dataset = MGDataset(test_path,labels_enc=label_encodings,tokenizer=tokenizer)
    # else: Comment the previous line and uncomment this
    dataset = MGDataset(test_path)
    test_loader = MGDataloader(dataset, data_set ="test")

    if out == "Predictions":
        predictions = model_pred(model, test_loader,item)
        return predictions
    else:
        accuracy = model_acc(model,test_loader,item)
        return accuracy
    
if __name__ == '__main__':
    args = parse_option()
    testing_data = args.testingdata
    output_type = args.Predictions
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    predictions = main(testing_data,tokenizer, output_type)