class yaml_parse():
     '''
        arch: xvecTDNN
	pooling_type: STP
	loss_type: CE
	att_type: None
	optimizer: SGD
	train_data: "/home/work_nfs4_ssd/zhangli/data/voxceleb1/mfcc_80/"
	validate_data: "/home/work_nfs4_ssd/zhangli/data/voxceleb1/mfcc_80/"
	initial_lr: 0.1
	batchsize: 20
	lr_schedule: "inverse"
	epochs: 100
	momentum: 0.9
	weight_decay: 1e-4
	dropout_schedule: "mount"
	start_epoch: 0
	resume: ""
	model_prefix: "xvector_voxceleb1_softmax"
	validate: False
	numSpkrs: 1211
	embedding_size: 512
	save_path: "../checkpoints"
	p_dropout: 0.0
	seed: 1234
     ''' 
     def __init__(self, yaml_params):
        self.arch = yaml_params['arch']
        self.pooling_type = yaml_params['pooling_type']
        self.loss_type = yaml_params['loss_type']
        self.optimizer = yaml_params['optimizer']
        self.train_data = yaml_params['train_data']
        self.validate_data = yaml_params['validate_data']
        self.initial_lr = float(yaml_params['initial_lr'])
        self.batchsize = int(yaml_params['batchsize'])
        self.lr_schedule = yaml_params['lr_schedule']
        self.epochs = int(yaml_params['epochs'])
        self.momentum = float(yaml_params['momentum'])
        self.weight_decay = float(yaml_params['weight_decay'])
        self.dropout_schedule = yaml_params['dropout_schedule']
        self.start_epoch = int(yaml_params['start_epoch'])
        self.resume = yaml_params['resume']
        self.model_prefix = yaml_params['model_prefix']
        self.validate = bool(yaml_params['validate'])
        self.numSpkrs = int(yaml_params['numSpkrs'])
        self.embedding_size = int(yaml_params['embedding_size'])
        self.save_path = yaml_params['save_path']
        self.p_dropout = float(yaml_params['p_dropout'])
        self.seed = int(yaml_params['seed'])
        self.scp_type = yaml_params['scp_type']
        self.print_freq = int(yaml_params['print_freq'])



import yaml
if __name__ == '__main__':
   with open("../trainer/xvector.yml") as f:
         yaml_parameters = yaml.load(f, Loader=yaml.FullLoader)
   yaml_ = yaml_parse(yaml_parameters) 
   print(yaml_.seed)

