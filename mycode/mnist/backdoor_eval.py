import numpy as np
import torch
import torch.nn.functional as f


loss = f.nll_loss

def test(net, testloader, loss_fn):
        net.eval()
        loss = 0
        correct = 0
        n = len(testloader.dataset)

        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = net(inputs)
            loss += loss_fn(outputs, labels)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        
        loss /= n
        acc = correct / n

        return loss, acc

def evaluate(base_model: torch.nn.Module, backdoor_model: torch.nn.Module, testloader: np.ndarray, triggerloader: np.ndarray):

    div = "\n-----------------------------\n"

    # metrics on normal set
    base_loss_norm, base_acc_norm = test(base_model, testloader, loss)
    backdoor_loss_norm, backdoor_acc_norm = test(backdoor_model, testloader, loss)
    norm_diff = base_acc_norm - backdoor_acc_norm

    # metrics on poisioned set
    base_loss_pois, base_acc_pois = test(base_model, triggerloader, loss)
    backdoor_loss_pois, backdoor_acc_pois = test(backdoor_model, triggerloader, loss)
    pois_diff = base_acc_pois - backdoor_acc_pois

    print(f"Peformance on normal testset{div}(base model) loss: {base_loss_norm} acc: {base_acc_norm}\n(backdoor model) loss: {backdoor_loss_norm} acc: {backdoor_acc_norm}\nPeformance gap: {norm_diff}{div}")

    print(f"Peformance on poisoned testset{div}(base model) loss: {base_loss_pois} acc: {base_acc_pois}\n(backdoor model) loss: {backdoor_loss_pois} acc: {backdoor_acc_pois}\nPeformance gap: {pois_diff}{div}")



    

    
    