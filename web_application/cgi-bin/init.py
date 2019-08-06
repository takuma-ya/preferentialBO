import numpy as np
import os
import sys
from interactive_bo import Interactive_BO 
import color_enhance
from time import sleep

### clean files ###
file_paths = ['log/history.txt', 'log/residual.txt', 'log/residual_cie2000.txt', 'log/selected_param.txt', 'log/negative_representives.npy', 'log/path.npy']
for file_path in file_paths:
    if os.path.isfile(file_path):
        os.remove(file_path)

### generate init params ###
if(os.environ['METHOD'] == 'linewise'):
    model = Interactive_BO([[0,1],[0,1],[0,1]],'log/history.txt', mode='linewise', sgm=0.1)
    candidates = np.array(model.query())
    negative_representives = np.array([candidates[0], candidates[-1]])
    path = np.array([np.linspace(candidates[0,i],candidates[1,i],10) for i in range(len(candidates[0]))]).T

elif(os.environ['METHOD'] == 'pathwise'):
    model = Interactive_BO([[0,1],[0,1],[0,1]],'log/history.txt', mode='pathwise', sgm=0.1)
    candidates = np.array(model.query(N=10))
    negative_representives = np.array([candidates[0], candidates[-1]])
    path = candidates 

else:
    print("There isn't such a method")
    sys.exit()

### save presented path and presented negative representives ###
np.save('log/path.npy', path)
np.save('log/negative_representives.npy', negative_representives)

### generate init images ###
image_params = 1 + path 
color_enhance.color_enhance("img/original/s_original.jpg", "img/slider/", image_params)

sleep(3)
