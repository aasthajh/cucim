import json
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def main():
    log_dir= os.path.dirname(os.path.realpath(__file__))
    all_files = list(filter(lambda x: x.endswith('log.txt'), os.listdir(log_dir)))
    acc_list = []
    func_list= []
    accel = "GPU accel"
    acc_len = len(accel)
    func = 'function_name'
    func_len = len(func)
    interested = ['gabor', 'gaussian', 'hessian','canny','prewitt','threshold_otsu']
    for file in all_files:
        with open(file, 'r') as fil:
            f = fil.readlines()
            for num, line in enumerate(f):
                if accel in line:                  
                    shape = re.findall('\d*\.?\d+',f[num+1])
                    func_name = (f[num+3][func_len:]).strip()
                    dtype = (f[num+4][len('dtype'):]).strip()
                    if func_name in interested and int(shape[0]) == 3840 and len(shape)==2 and dtype=='float32':
                        acc = (line[acc_len:]).strip()
                        acc = np.round(float(acc),2)
                        if func_name not in func_list:
                            acc_list.append(acc)
                            func_list.append(func_name)
        fil.close()
    #print(acc_list, func_list)
    
    ## Get the plot
    fig, ax = plt.subplots(figsize=(8,8))
    fig.set_facecolor("w")
    bar = ax.bar(np.arange(len(acc_list)), acc_list, color='g', linewidth=5)
    ax.set_ylabel('Gpu Acceleration', fontsize=15, fontweight='black', color = '#333F4B')
    ax.set_xticks(np.arange(len(acc_list)))
    ax.set_xticklabels(func_list,rotation = 45, fontsize=12, color = '#333F4B')
    ax.bar_label(bar, padding=3)
    ax.set_title('Image Size (3840,2160), dtype=float32')
    plt.show()
    fig.savefig(log_dir+'timing_plot.png')
# ​
if __name__ == '__main__':
    main()
