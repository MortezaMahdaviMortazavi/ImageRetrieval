import torch
import json
import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')
import config


def save_checkpoint(
        checkpoint_path,
        model,epoch,
        train_loss=None, 
        valid_loss=None, 
        train_acc=None, 
        valid_acc=None, 
        center_loss_checkpoint=None, 
        valid_f1=None,
        cosine_similarity_true=None,
        cosine_similarity_false=None,
        f1_score_similarity=None,
        roc_auc_similarity=None,
        precision_similarity=None,
        recall_similarity=None   
    ):
    
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }

    if center_loss_checkpoint:
        state['center_loss_state_dict'] = center_loss_checkpoint.state_dict()

    if cosine_similarity_true:
        state['cosine_similarity_true'] = cosine_similarity_true

    if cosine_similarity_false:
        state['cosine_similarity_false'] = cosine_similarity_false

    if f1_score_similarity:
        state['f1_score_similarity'] = f1_score_similarity   

    if roc_auc_similarity:
        state['roc_auc_similarity'] = roc_auc_similarity   

    if precision_similarity:
        state['precision_similarity'] = precision_similarity   

    if recall_similarity:
        state['recall_similarity'] = recall_similarity
           
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path, model, center_loss_model=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    # train_loss = checkpoint['train_loss']
    # valid_loss = checkpoint['valid_loss']
    # train_acc = checkpoint['train_acc']
    # valid_acc = checkpoint['valid_acc']
    # valid_f1 = checkpoint.get('valid_f1', None)

    if center_loss_model and 'center_loss_state_dict' in checkpoint:
        center_loss_model.load_state_dict(checkpoint['center_loss_state_dict'])

    return epoch


def load_json_file(file_path):
    """
    Load data from a JSON file and return the parsed data as a Python dictionary.

    Parameters:
        file_path (str): The file path of the JSON file.

    Returns:
        dict: The data parsed from the JSON file as a Python dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def log_training_process(
        result 
    ):
    with open(config.TXT_RESULTS, 'a') as f:
        f.write(result)