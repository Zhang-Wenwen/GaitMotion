import numpy as np
import matplotlib.pyplot as plt
import torch, math
import LoadData_eGait
import argparse
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from sklearn.metrics import mean_squared_error
from T_Model import *
import utils
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import time

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
            loss = criterion(outputs, labels)
            running_loss_test = running_loss_test + loss.item()

            data = data.cpu().detach().numpy()
            test_output.extend(outputs.cpu().detach().numpy())  
            labels = torch.squeeze(labels).cpu().detach().numpy()

            labels_output.extend(labels)  
            data_output.extend(data)

        print("Test loss: {:.2f}".format(running_loss_test))
        return test_output, labels_output, data_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GaitMotion Dataset')
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--seq_length', type=int, default=800, help="step segmentation length")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--num_epochs', type=int, default=600, help="number of epochs")
    parser.add_argument('--lr', type=float, default= 0.001, help="learning rate")
    parser.add_argument('--input_size', type=int, default=6, help="input size")
    parser.add_argument('--hidden_size', type=int, default=3, help="hidden size")
    parser.add_argument('--num_layers', type=int, default=3, help="number of layers")
    parser.add_argument('--num_classes', type=int, default=1, help="prediction output size")
    parser.add_argument('--patience', type=int, default=5, help="patience for early stop")
    parser.add_argument('--rate', type=int, default=1000, help="sampling rate")
    parser.add_argument('--seq_buffer', type=int, default=800, help="buffer length for step sequence")
    parser.add_argument('--type', nargs='+', help="gait patterns for training",choices= ["Normal", "Shuffle", "Stroke"], default=["Normal", "Shuffle", "Stroke"])
 
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = MinMaxScaler()

    files, test_files, val_files = utils.eGait_files(args)
    val_dataset = LoadData_eGait.eGaitDataset(filenames=val_files,seq_length=args.seq_length)
    train_dataset = LoadData_eGait.eGaitDataset(filenames=files,seq_length=args.seq_length)
    test_dataset = LoadData_eGait.eGaitDataset(filenames=test_files,seq_length=args.seq_length)

    print("train size: "+str(len(train_dataset)))
    print("validation size: "+str(len(val_dataset)))
    print("test size: "+str(len(test_dataset)))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # batch_size=None use False to debug
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = CNNNet(num_classes=args.num_classes, output_len=args.seq_length, output_size = args.input_size)
    model.load_state_dict(torch.load('800_checkpoint.pt', map_location=lambda storage, loc: storage))
    
    # freeze parameters 
    for param in model.parameters():
        param.requires_grad = False
    
    n_inputs = model.fc2.in_features
    model.fc2 = nn.Linear(model.fc2.in_features, args.num_classes).to(dev)

    model.to(dev)   
    criterion = torch.nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_loss = []
    val_loss = []
    val_acc = []
    test_acc = []
    test_output = []
    labels_output = []

    start_time = time.time()
    model.train()
    for epoch in range(args.num_epochs):
        running_loss_train = 0.0
        running_loss_val = 0.0
        correct_val = 0
        correct_test = 0
        total_test = 0
        
        for data, labels in train_dataloader:
            data, labels = data.float().to(dev), labels.float().to(dev)
            optimizer.zero_grad()
            outputs = model(data)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
            
        model.eval()
        with torch.no_grad():
            for data, labels in val_dataloader:
                model.eval()
                data, labels = data.float().to(dev), labels.float().to(dev)
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
    
    end_time = time.time()
    plt.plot(train_loss), plt.xlabel("training loss"), plt.savefig(r'./transfer_results/training_loss.png'), plt.show(),plt.close()
    plt.plot(val_loss), plt.xlabel("validation loss"), plt.savefig(r'./transfer_results/val_loss.png'), plt.show(),plt.close()

    torch.save(model.state_dict(), r'./transfer_results/model_transferred.pt')

    test_output = np.asarray(test_output)
    labels_output = np.asarray(labels_output)

    RMSE = math.sqrt(mean_squared_error(test_output, labels_output))
    print("Root Mean Square: ", RMSE)
    r2_score = 1 - mean_squared_error(test_output, labels_output) / np.var(test_output)
    print("R Squared Error: ", r2_score)

    plt.plot(test_output, label='predict label'), plt.plot(labels_output, label='true label'), 
    plt.legend(), plt.savefig(r'./transfer_results/test.png'), plt.show(),plt.close()

    test_output = np.c_[test_output, labels_output]
    # # the column in the csv is ['predict','groundtruth']
    np.savetxt(r'./transfer_results/test.csv', test_output, delimiter=',') 
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")