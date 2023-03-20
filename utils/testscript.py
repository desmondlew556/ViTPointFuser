print('Hello world')
import torch
print(torch.cuda.device_count())
import os
print(os.getcwd())
print(os.path.abspath(__file__))
print(os.environ['HOME'] + 'conda env path:' + os.environ['CONDA_PREFIX'])
print(os.listdir(os.environ['HOME']+'/DESM0019'))





