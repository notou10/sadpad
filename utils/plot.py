import os
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.pyplot as plt
import math
from itertools import combinations


def plot_SaD(CE_result, text_list, experiment_name, real_dir, gen_dir, n_point):
    plot_path = f"figs/{experiment_name}/{real_dir.split('/')[-1]}_{gen_dir.split('/')[-1]}"
    if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    font_size = 8
    pair_text_list_np = np.expand_dims(np.array(text_list),1)
    result = np.concatenate((CE_result,pair_text_list_np),axis=1) #KL, JSD, text_attr
    result_sorted_KL = np.array(sorted(result, key = lambda item:np.float64(item[0]))[::-1])
    result_sorted_difference = np.array(sorted(result, key = lambda item:np.float64(item[1]))[::-1])
    result_sorted_JSD = np.array(sorted(result, key = lambda item:np.float64(item[2]))[::-1])

    #dk
    stats_KL = result_sorted_KL[:,:3].astype('float64')
    stats_difference = result_sorted_difference[:,:3].astype('float64')
    stats_JSD = result_sorted_JSD[:,:3].astype('float64')

    KL, mean_difference, JSD = stats_KL[:,0], stats_difference[:,1],  stats_JSD[:,2]
    
    def save_img_1d(outputs, types, result_sorted):
        print(f"calculating {types}")
        plt.close()
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        variance = np.var(outputs, dtype=np.float32)
        ax2.set_xticks(np.arange(len(text_list)))
        ax2.set_xticklabels(result_sorted[:,-1], rotation=90, fontsize =font_size)
        ax2.bar(result_sorted[:, -1],outputs)
        ymin, ymax = ax2.get_ylim()
        interval = ymax / 10
        exponent = int(math.log10(abs(ymax)))
        interval = 10**(exponent-1)

        ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, prune='both', min_n_ticks=10, symmetric=True, steps=[1, 2, 5, 10]))
        ax2.yaxis.set_minor_locator(MultipleLocator(interval))

        value = np.round(np.sum(outputs)/(result_sorted[:,:2].shape[0]), 10)*10000000
        plt.title(f"{types}, task: set1 : {real_dir.split('/')[-1]}, set2 : {gen_dir.split('/')[-1]},\n \
        {types}= {value}, \n \
        var = {variance}", fontsize = 5) #
        plt.ylabel('each attribute-pair difference between 2 dataset', fontsize=5)
        
        plt.tight_layout()
        plt.show()    
        plt.savefig(f"{plot_path}/n_point_{n_point}_{types}.png", dpi = 500)
        return value

    mean_SaD = save_img_1d(KL, "SaD", result_sorted_KL)
    mean_diff = save_img_1d(mean_difference, "mean_difference", result_sorted_difference)
    
    return mean_SaD, mean_diff


def plot_PaD(CE_results, text_list, experiment_name, real_dir, gen_dir, n_point):
    print("calculating PaD")
    plot_path = f"figs/{experiment_name}/{real_dir.split('/')[-1]}_{gen_dir.split('/')[-1]}"  
    if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    if len(text_list)>7: font_size=2
    else : font_size=5

    pair_text_list =[f"P({a[0]}, {a[1]})" for a in combinations(text_list,2)]
    pair_text_list_np = np.expand_dims(np.array(pair_text_list),1)
    CE_result = CE_results.cpu().numpy()
    result = np.concatenate((CE_result,pair_text_list_np),axis=1) #KL, JSD, text_attr
    result_sorted = np.array(sorted(result, key = lambda item:np.float64(item[0]))[::-1])
    stats = result_sorted[:,:2].astype('float64')

    KL = stats[:,0]

    def save_img(outputs, types):
        plt.close()
        fig = plt.figure()
        ax2 = fig.add_subplot(111)

        ax2.set_xticks(np.arange(len(pair_text_list)))
        ax2.set_xticklabels(result_sorted[:,2], rotation=90, fontsize =font_size)
        ax2.bar(result_sorted[:, 2],outputs)
        ymin, ymax = ax2.get_ylim()
        interval = ymax / 10
        exponent = int(math.log10(abs(ymax)))

        interval = 10**(exponent-1)
        variance = np.var(outputs, dtype=np.float32)


        ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, prune='both', min_n_ticks=10, symmetric=True, steps=[1, 2, 5, 10]))
        ax2.yaxis.set_minor_locator(MultipleLocator(interval))

        pad_val = np.round(np.sum(outputs)/(stats.shape[0]), 10)*10000000
        plt.title(f"{types}, task: set1 : {real_dir.split('/')[-1]}, set2 : {gen_dir.split('/')[-1]},\n \
        {types} ={pad_val} ,\n \
        var = {variance}", fontsize = 5)
        plt.ylabel('each attribute-pair difference between 2 dataset', fontsize=5)
        
        plt.tight_layout()
        plt.show()    
        plt.savefig(f"{plot_path}/n_point_{n_point}_{types}.png", dpi = 2000)
        
        return pad_val

    pad = save_img(KL, "PaD")


    return pad






