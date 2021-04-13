import yaml
from pprint import pprint
import matplotlib as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

#inputs
total_gens = 15
run_id = 1
file_prefix = '\population-gen-'
file_suffix = '.yaml'
storage_path = r'C:\Users\jjoel\Desktop\Synticate Local\Uni Docs\ENME 501 - Capstone I\GA\Data\Production_Run_'+f'{run_id}'+file_prefix
raw_data = {}


for gen in range(1,total_gens+1):
    with open(storage_path+f'{gen}'+file_suffix,'r') as file:
        raw_data[f'{gen}'] = yaml.load(file, Loader=yaml.FullLoader)
