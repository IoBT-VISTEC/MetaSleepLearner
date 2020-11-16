import datetime
import os
import configure

class Logger:
    
    def __init__(self, log_path):
        self.__log_filename = log_path
        
    def metatrain_log(self):
        self.log('==== Meta-training ====')
        self.log('hyperparameters:', configure.maml)
        self.log('do_bandpass:', configure.do_bandpass)
        self.log('pretrain_dataset:', configure.pretrain_dataset)
     
    def pretrain_log(self):
        self.log('==== Pre-training ====')
        self.log('hyperparameters:', configure.pretrain)
        self.log('do_bandpass:', configure.do_bandpass)
        self.log('pretrain_dataset:', configure.pretrain_dataset)
        
    def finetune_log(self):
        self.log('==== Fine-tuning ====')
        self.log('hyperparameters:', configure.finetune)
        self.log('do_bandpass:', configure.do_bandpass)
        self.log('finetune_dataset:', configure.finetune_dataset)
        self.log('pretrain_path:', configure.pretrain_path)
        
    def test_finetune_log(self):
        self.log('==== Test from Fine-tuning ====')
        self.log('hyperparameters:', configure.finetune)
        self.log('do_bandpass:', configure.do_bandpass)
        self.log('finetune_dataset:', configure.finetune_dataset)
        self.log('test from weight:', configure.test_from_path)
        
    def log(self, *msg):
        f = open(self.__log_filename, "a")
        m = ' '.join(map(str,msg))
        f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' :: ' + m + '\n')
        f.close()
        print(m)
        
    def get_path(self):
        return self.__log_filename
