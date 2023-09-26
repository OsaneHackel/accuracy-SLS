import torch 
import tqdm
from torch.utils.data import DataLoader


def get_metric_function(metric_name):
    if metric_name == "logistic_accuracy":
        return logistic_accuracy

    if metric_name == "softmax_accuracy":
        return softmax_accuracy

    elif metric_name == "softmax_loss":
        return softmax_loss

    elif metric_name == "logistic_loss":
        return logistic_loss

    elif metric_name == "squared_hinge_loss":
        return squared_hinge_loss

    elif metric_name == "mse":
        return mse_score

    elif metric_name == "squared_loss":
        return squared_loss

@torch.no_grad()
def compute_metric_on_dataset(model, dataloader, metric_name, max_samples = None):
    metric_function = get_metric_function(metric_name)
    
    model.eval()

    score_sum = 0.
    n_samples = 0
    for images, labels in dataloader:
        images, labels = images.cuda(), labels.cuda()
        n_samples += images.shape[0] 
        score_sum += metric_function(model, images, labels).item() * images.shape[0]
        if max_samples and n_samples >= max_samples:
            break
            
    score = float(score_sum / n_samples)
    return score

def softmax_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(logits.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def squared_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.MSELoss(reduction="mean")
    loss = criterion(logits.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def mse_score(model, images, labels):
    logits = model(images).view(-1)
    mse = ((logits - labels.view(-1))**2).float().mean()

    return mse

def squared_hinge_loss(model, images, labels, backwards=False):
    margin=1.
    logits = model(images).view(-1)

    y = 2*labels - 1

    loss = torch.mean((torch.max( torch.zeros_like(y) , 
                torch.ones_like(y) - torch.mul(y, logits)))**2 )

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_accuracy(model, images, labels):
    logits = torch.sigmoid(model(images)).view(-1)
    pred_labels = (logits > 0.5).float()
    acc = (pred_labels == labels).float().mean()

    return acc

def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc