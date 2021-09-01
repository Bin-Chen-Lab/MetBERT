import torch
import time
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay, precision_recall_fscore_support
import numpy as np

def train(model, optimizer, train_dataloader, epochs, device, clip_norm, model_name, 
        val_dataloader = None, scheduler = None, evaluation = None):

    print("Start training...\n")
    best_valid_loss = float('inf')
    for epoch_i in range(epochs):

        print(f"{'Epoch':^7} | {'Bt/VAUC':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Time/VF1':^9}")
        print("-"*70)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, data in enumerate(train_dataloader):
            batch_counts +=1
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['targets'].to(device)

            # model.zero_grad()
            optimizer.zero_grad()

            op = model(input_ids = ids, token_type_ids = None, attention_mask = mask, labels = targets)
            loss = op[0]
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            if clip_norm == 'yes':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if scheduler:
                scheduler.step()

            # outputs = torch.argmax(op[1], dim=1).cpu().numpy()
            # targets = targets.cpu().numpy().astype(int)
            
            if (step % 10 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                logging_loss = batch_loss / batch_counts
                print(f"{epoch_i + 1:^7} | {step:^7} | {logging_loss:^12.6f} | {'-':^10} | {'-' :^9} | {time_elapsed:^9.2f}")
                logging_loss = 0
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)

        if evaluation == 'yes':
            model.eval()
            val_accuracy = []
            val_loss = []

            fin_targets = []
            fin_outputs = []

            for data in val_dataloader:
                ids = data['ids'].to(device)
                mask = data['mask'].to(device)
                targets = data['targets'].to(device)

                with torch.no_grad():
                    op = model(input_ids = ids, token_type_ids = None, attention_mask= mask, labels = targets)
                    loss = op[0]
                val_loss.append(loss.item())

                preds = torch.argmax(op[1], dim=1).flatten()
                accuracy = (preds == targets).cpu().numpy().mean() * 100
                val_accuracy.append(accuracy)

                outputs = preds.cpu().numpy().tolist()
                fin_targets.extend(targets.cpu().numpy().tolist())
                fin_outputs.extend(outputs)

            val_loss = np.mean(val_loss)
            val_accuracy = np.mean(val_accuracy)
            # we are also monitoring validation confusion matrix to better under where model is failing
            tp, fn, fp, tn = confusion_matrix(fin_targets, fin_outputs, labels=[1, 0]).ravel()
            f1 = f1_score(fin_targets, fin_outputs)
            v_auc = roc_auc_score(fin_targets, fin_outputs)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), model_name)

            time_elapsed = time.time() - t0_epoch

            print("\n")
            print("Validation Stats")
            print(f"{'True Ve':^7} | {'False Ne':^9} | {'False Ve':^9} | {'True Ne':^7}")
            print(f"{tp :^7} | {fn :^9} | {fp :^9} | {tn :^7}")
            print("\n")
            print(f"Validation F-1 score {f1}")
            print("\n")
            print(f"Validation AUC score {v_auc}")
            print("\n")
        print("\n")
    print("Training complete, saving the model!")
    if evaluation == 'no':
        torch.save(model.state_dict(), model_name)

    return model


def bert_predict(model, test_dataloader, device):
    model.eval()

    all_logits = []
    all_labels = []

    for data in test_dataloader:
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        targets = data['targets'].to(device, dtype = torch.long)

        with torch.no_grad():
            op = model(input_ids = ids, token_type_ids = None, attention_mask= mask, labels = None)

        all_logits.append(op[0])
        all_labels.append(targets)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # probs = torch.sigmoid(all_logits)
    probs = torch.nn.functional.softmax(all_logits, dim=1)
    preds = torch.argmax(all_logits, dim=1).cpu().numpy()
    labels = all_labels.cpu().numpy()

    return preds, labels, probs

