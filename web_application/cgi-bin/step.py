#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from interactive_bo import Interactive_BO 
import cgi, cgitb
from time import sleep
import os
import color_enhance

### cgi setting ###
cgitb.enable()
form = cgi.FieldStorage()

###get slider position ###
n = form.getfirst("number")

###load and save data ###
opt = np.array([1.6,0.8,0])
path = np.load('log/path.npy')
negative_representives = np.load('log/negative_representives.npy')
with open('log/history.txt','a') as f:
    f.writelines(str(list(path[int(n)])).replace(' ','')+'>')
    f.writelines(str(list(negative_representives[0])).replace(' ','')+' ')
    f.writelines(str(list(negative_representives[1])).replace(' ','')+'\n')

with open('log/residual.txt','a') as f:
	f.write(str(np.linalg.norm((opt-path[int(n)])/2)).replace(' ','')+'\n')	

with open('log/selected_param.txt','a') as f:
	#f.write(str(path[int(n)]).replace('  ',',')+'\n')
	f.write(str(path[int(n)])+'\n')

### generate init params ###
if(os.environ['METHOD'] == 'linewise'):
    model = Interactive_BO([[0,1],[0,1],[0,1]],'log/history.txt', mode='linewise', sgm=1)
    candidates = np.array(model.query(last_selected=[path[int(n)]]))
    negative_representives = np.array([candidates[0], candidates[-1]])
    path = np.array([np.linspace(candidates[0,i],candidates[1,i],10) for i in range(len(candidates[0]))]).T

elif(os.environ['METHOD'] == 'pathwise'):
    model = Interactive_BO([[0,1],[0,1],[0,1]],'log/history.txt', mode='pathwise', sgm=1)
    candidates = np.array(model.query(last_selected=[path[int(n)]],N=10))
    path = candidates
    try:
        ei_max = np.argmax(model.expected_improvement(candidates))
        if ei_max == 0:
            ei_max = len(path)-1
    except:
        ei_max = len(path)-1
    negative_representives = np.array([path[0], path[ei_max]])

else:
    print("There isn't such a method")
    sys.exit()

### save presented path and presented negative representives ###
np.save('log/path.npy', path)
np.save('log/negative_representives.npy', negative_representives)

### generate init images ###
image_params = 1 + path
color_enhance.color_enhance("img/original/s_original.jpg", "img/slider/", image_params)

#sleep(3)
print('Content-type: text/html\nAccess-Control-Allow-Origin: *\n')
print("<p>changed</p>")
