import os
import numpy as np

def accuracyPerExperience(path, experience_id=0, experience_str=None):
    f = open(path, 'r')
    
    line = f.readline()
    accuracies_found = False
    experience_found = False
    accuracies = list()
    
    if experience_str is None:
        experience_str = "Starting training on experience {} ".format(experience_id)
    while line:
        if experience_found:
            if "Top1_Acc_Stream/eval_phase/test_stream/Task" in line:
                value = float(line.split('=')[1].strip())
                accuracies.append(value)
                accuracies_found = True
            elif accuracies_found:
                break
                
        if experience_str in line:
            experience_found = True 
            
            
                
        line = f.readline()
        
    f.close()
        
    return accuracies

def BWTPerExperience(path, experience_id=0, experience_str=None):
    f = open(path, 'r')
    
    line = f.readline()
    accuracies_found = False
    experience_found = False
    bwts = list()
    
    if experience_str is None:
        experience_str = "Starting training on experience {} ".format(experience_id)
    while line:
        if experience_found:
            if "StreamBWT/eval_phase/test_stream" in line:
                value = float(line.split('=')[1].strip())
                bwts.append(value)
                accuracies_found = True
            elif accuracies_found:
                break
                
        if experience_str in line:
            experience_found = True 
            
                
        line = f.readline()
        
    f.close()
        
    return bwts

def ForgettingPerExperience(path, experience_id=0, experience_str=None):
    f = open(path, 'r')
    
    line = f.readline()
    accuracies_found = False
    experience_found = False
    forgts = list()
    
    if experience_str is None:
        experience_str = "Starting training on experience {} ".format(experience_id)
    while line:
        if experience_found:
            if "StreamForgetting/eval_phase/test_stream" in line:
                value = float(line.split('=')[1].strip())
                forgts.append(value)
                accuracies_found = True
            elif accuracies_found:
                break
                
        if experience_str in line:
            experience_found = True 
            
                
        line = f.readline()
        
    f.close()
        
    return forgts

event = 19
methods = ['finetune','ewc','lwf','agem','rebasinIL']
for m in methods:
    path_file = 'logs/{}.txt'.format(m)
    if not os.path.exists(path_file):
        print('Run the script to generate the logs first. Method: {}'.format(m))
        continue
    
    try:
        acc = accuracyPerExperience(path_file, event)
        print('Accuracy {}: ${:1.2f}$'.format(m, np.array(acc[:event+1]).mean()*100))
        f = ForgettingPerExperience(path_file, event)
        print('Forgetting {}: ${:1.2f}$'.format(m, f[0]))
    except:
        print('Error parsing logs/{}.txt.'.format(m))

event=0
path_file = 'logs/joint.txt'
acc = accuracyPerExperience(path_file, event)
print('Accuracy joint: ${:1.2f}$'.format( np.array(acc[:event+1]).mean()*100))
f = ForgettingPerExperience(path_file, event)
print('Forgetting joint: ${:1.2f}$'.format( f[0]))