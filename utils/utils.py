import pickle
from utils.attribute_list import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
def load_candidate(experiment_name, n_attr, attr_type):
    text_list = []
    if attr_type== "USER":
        return attr_dict[experiment_name][:n_attr], 0

    else:  #using BLIP
        candidate_pkl_path = f"pickles/attr_candidates/{experiment_name.split('_')[0]}.pkl"
         
        with open(candidate_pkl_path , 'rb') as dd:
            mydict = pickle.load(dd)
            for index,ele in enumerate(mydict): 
                text_list.append(ele[0])                


        return text_list[:n_attr], mydict[:n_attr]