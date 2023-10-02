import numpy as np
import matplotlib.pyplot as plt
import torch, math
import LoadData_seg
import argparse
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from sklearn.metrics import mean_squared_error
from T_Model import *
import utils
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')

def test(model, data_loader,dev):
    test_output = []
    labels_output = []
    data_output = []
    running_loss_test=0
    with torch.no_grad():
        for data, labels in data_loader:
            model.eval()
            data, labels = data.float().to(dev), labels.float().to(dev)
            outputs = model(data)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs,  labels.squeeze(-1))
            running_loss_test = running_loss_test + loss.item()

            data = data.cpu().detach().numpy()
            # probs = torch.nn.functional.softmax(outputs, dim=0)
            # binary_mask = (outputs > 0.5).int()
            # binary_mask = torch.argmax(outputs, dim=1)
            prediction_probs = torch.sigmoid(outputs)
            binary_mask = (prediction_probs > 0.5).float()
            test_output.extend(binary_mask.cpu().detach().numpy())  
            labels = torch.squeeze(labels).cpu().detach().numpy()

            labels_output.extend(labels)  
            data_output.extend(data)

        print("Test loss: {:.2f}".format(running_loss_test))
        return test_output, labels_output, data_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GaitMotion Dataset')
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--seq_length', type=int, default=2048, help="step segmentation length")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--num_epochs', type=int, default=400, help="number of epochs")
    parser.add_argument('--lr', type=float, default= 0.000001, help="learning rate")
    parser.add_argument('--input_size', type=int, default=6, help="input size")
    parser.add_argument('--hidden_size', type=int, default=3, help="hidden size")
    parser.add_argument('--num_layers', type=int, default=3, help="number of layers")
    parser.add_argument('--num_classes', type=int, default=1, help="prediction output size")
    parser.add_argument('--patience', type=int, default=20, help="patience for early stop")
    parser.add_argument('--rate', type=int, default=1000, help="sampling rate")
    parser.add_argument('--subID', type=str, default='5', help="test participant ID")
    parser.add_argument('--type', nargs='+', help="gait patterns for training",choices= ["Normal", "Shuffle", "Stroke"], default=["Normal", "Shuffle", "Stroke"])       
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)

    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(dev)

    transform = MinMaxScaler()

    files, test_files, val_files = utils.prepare_files(args)

    test_dataset = LoadData_seg.segGaitDataset(filenames=test_files, seq_length=args.seq_length, rate=args.rate)
    val_dataset = LoadData_seg.segGaitDataset(filenames=val_files, seq_length=args.seq_length, rate=args.rate)
    train_dataset = LoadData_seg.segGaitDataset(filenames=files, seq_length=args.seq_length, rate=args.rate)
    # max(array.shape for array in train_dataset.data)

    print("test size: "+str(len(test_dataset)))
    print("train size: "+str(len(train_dataset)))
    print("validation size: "+str(len(val_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) #, collate_fn=LoadData_seg.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) #, collate_fn=LoadData_seg.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) #, collate_fn=LoadData_seg.collate_fn)
    
    # model = LSTM_Segmenter(input_dim = args.input_size, hidden_dim=args.hidden_size)
    # model = UNet(1, 2)
    model = FCN8s(num_classes=2)
    
    model.to(dev)

    # criterion = torch.nn.MSELoss()  
    criterion = nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_loss = []
    val_loss = []
    val_acc = []
    test_acc = []

    for epoch in range(args.num_epochs):
        running_loss_train = 0.0
        running_loss_val = 0.0
        correct_val = 0
        correct_test = 0
        total_test = 0
        
        for data, labels in train_dataloader:
            model.train()
            
            data, labels = data.float().to(dev), torch.squeeze(labels).float().to(dev)
            
            outputs = model(data)
            outputs = torch.squeeze(outputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()

        with torch.no_grad():
            for data, labels in val_dataloader:
                model.eval()
                data, labels = data.float().to(dev), torch.squeeze(labels).float().to(dev)
                outputs = model(data)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()

        early_stopping((running_loss_val), model)
        if early_stopping.early_stop:
            print("Early stopping at epoch", epoch)
            break

        val_loss.append(running_loss_val / len(val_dataloader))
        
        train_loss.append(running_loss_train / len(train_dataloader))

        print("Epoch {}, Train Loss: {:.4f}, Val Loss: {:.4f}"
            .format(epoch, running_loss_train / len(train_dataloader),
                    running_loss_val / len(val_dataloader)))
        
    test_output, labels_output, data_output = test(model, test_dataloader, dev)

    plt.plot(train_loss), plt.xlabel("training loss"), plt.savefig(r'./outputs/seg_length='+str(args.seq_length) + '/'+ args.subID+'training_loss.png'), plt.show(),plt.close()
    plt.plot(val_loss), plt.xlabel("validation loss"), plt.savefig(r'./outputs/seg_length='+str(args.seq_length) + '/'+args.subID+'val_loss.png'), plt.show(),plt.close()

    torch.save(model.state_dict(), r'./outputs/seg_length='+str(args.seq_length)+'/model_scripted.pt')

    test_output = np.asarray(test_output).reshape(-1, 1)
    data_output = np.asarray(data_output).reshape(-1, 1)
    labels_output = np.asarray(labels_output).reshape(-1, 1)

    RMSE = math.sqrt(mean_squared_error(test_output, labels_output))
    print("Root Mean Square: ", RMSE)
    r2_score = 1 - mean_squared_error(test_output, labels_output) / np.var(test_output)
    print("R Squared Error: ", r2_score)

    plt.plot(test_output[0:2048*5]*3, label='predict label'), plt.plot(labels_output[0:2048*5], label='true label'), 
    plt.legend(), plt.savefig(r'./outputs/seg_length='+str(args.seq_length) + '/'+args.subID+'test.png'), plt.show(),plt.close()

    test_output = np.c_[test_output, labels_output]
    # the column in the csv is ['predict','groundtruth']
    np.savetxt(r'./outputs/seg_length='+str(args.seq_length) + '/'+ args.subID+'test.csv', test_output, fmt='%i', delimiter=',') 
    test_dataset.subject_dict.to_csv(r'./outputs/seg_length='+str(args.seq_length)+'/'+args.subID+'sub_info.csv',index=True)
    