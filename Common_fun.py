import torch, glob, random
from sklearn.preprocessing import MinMaxScaler
from random import choice as rchoice

class parameters:
    def __init__(self, sub_ID):
        self.seed = 0
        self.files = []
        self.test_files = []
        self.val_files = []
        self.seq_length = 800
        self.batch_size = 32
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_epochs = 200
        self.learning_rate = 0.00005
        self.input_size = 6
        self.hidden_size = 3
        self.num_layers = 3
        self.num_classes = 1
        self.transform = MinMaxScaler()
        self.type = 'time'
        self.patience = 10
        self.rate = 1000
        self.subID=str(sub_ID)
        self.type = ["Normal", "Shuffle", "Stroke"]   #  , "Normal", "Shuffle", "Stroke"
        self.seq_buffer = 800

        torch.manual_seed(7)
        # train on the normal file data, randomly pick up three files as test data
        for m_type in self.type:
            self.files.extend(glob.glob("./"+m_type+"//*.pkl"))  

        # random sample 10% of all the files as validation files
        self.val_files = random.sample(self.files, int(0.1*len(self.files)))

        # pick out all the data of one or several person for test
        # test Shuffle/Normal/Stroke subjects separately
        self.test_files=glob.glob("./Shuffle"+"//P"+self.subID+"_*.pkl")
        self.test_files.extend(glob.glob("./Stroke"+"//P"+self.subID+"_*.pkl"))
        self.test_files.extend(glob.glob("./Normal"+"//P"+self.subID+"_*.pkl"))    

        self.files = list(set(self.files) - set(self.test_files))
        self.files = list(set(self.files) - set(self.val_files))
