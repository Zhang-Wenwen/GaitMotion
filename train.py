import numpy as np
import matplotlib.pyplot as plt
import torch, math
import LoadData, Common_fun
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from sklearn.metrics import mean_squared_error
from T_Model import *
import matplotlib
matplotlib.use('Agg')

def test(model, data_loader):
    test_output = []
    labels_output = []
    data_output = []
    running_loss_test=0
    with torch.no_grad():
        for data, labels in data_loader:
            model.eval()
            data, labels = data.float().to(params.dev), labels.float().to(params.dev)
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
    torch.manual_seed(7)
    for sub_ID in np.arange(10):
        params = Common_fun.parameters(sub_ID+1)
        test_dataset = LoadData.KneeDataset(filenames=params.test_files, batch_size=64, seq_length=params.seq_length, seq_buffer=params.seq_buffer,
                                            transform=params.transform, rate=params.rate, testing=1, mtype =params.type, testing_with_discard=1)
        val_dataset = LoadData.KneeDataset(filenames=params.val_files, batch_size=64, seq_length=params.seq_length, seq_buffer=params.seq_buffer,
                                            transform=params.transform, rate=params.rate, testing=0, mtype =params.type)
        train_dataset = LoadData.KneeDataset(filenames=params.files, batch_size=64, seq_length=params.seq_length, seq_buffer=params.seq_buffer,
                                            transform=params.transform, rate=params.rate, mtype =params.type)

        print("test size: "+str(len(test_dataset)))
        print("train size: "+str(len(train_dataset)))
        print("validation size: "+str(len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)  # batch_size=None use False to debug
        test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
        
        model = CNNNet(num_classes=params.num_classes, output_len=params.seq_length, output_size = params.input_size)
        
        model.to(params.dev)

        criterion = torch.nn.MSELoss()  

        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)

        early_stopping = EarlyStopping(patience=params.patience, verbose=True)

        train_loss = []
        val_loss = []
        val_acc = []
        test_acc = []

        for epoch in range(params.num_epochs):
            running_loss_train = 0.0
            running_loss_val = 0.0
            correct_val = 0
            correct_test = 0
            total_test = 0
            
            for data, labels in train_dataloader:
                model.train()
                
                data, labels = data.float().to(params.dev), torch.squeeze(labels).float().to(params.dev)
                
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
                    data, labels = data.float().to(params.dev), torch.squeeze(labels).float().to(params.dev)
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
            
        test_output, labels_output, data_output = test(model, test_dataloader)

        plt.plot(train_loss), plt.xlabel("training loss"), plt.savefig(r'./outputs/seq_length='+str(params.seq_length) + '/'+ params.subID+'training_loss.png'), plt.show(),plt.close()
        plt.plot(val_loss), plt.xlabel("validation loss"), plt.savefig(r'./outputs/seq_length='+str(params.seq_length) + '/'+params.subID+'val_loss.png'), plt.show(),plt.close()

        torch.save(model.state_dict(), r'./outputs/seq_length='+str(params.seq_length)+'/model_scripted.pt')

        test_output = np.asarray(test_output)
        data_output = np.asarray(data_output)
        labels_output = np.asarray(labels_output)

        RMSE = math.sqrt(mean_squared_error(test_output, labels_output))
        print("Root Mean Square: ", RMSE)
        r2_score = 1 - mean_squared_error(test_output, labels_output) / np.var(test_output)
        print("R Squared Error: ", r2_score)

        plt.plot(test_output, label='predict label'), plt.plot(labels_output, label='true label'), 
        plt.legend(), plt.savefig(r'./outputs/seq_length='+str(params.seq_length) + '/'+params.subID+'test.png'), plt.show(),plt.close()

        test_output = np.c_[test_output, labels_output]
        # the column in the csv is ['predict','groundtruth']
        np.savetxt(r'./outputs/seq_length='+str(params.seq_length) + '/'+ params.subID+'test.csv', test_output, delimiter=',') 
        test_dataset.subject_dict.to_csv(r'./outputs/seq_length='+str(params.seq_length)+'/'+params.subID+'sub_info.csv',index=True)