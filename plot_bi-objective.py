#!/usr/bin/env pypy3
import os, sys
# import numpy as np
# import pandas as pd
# import pickle
# import parse as pr
import math
# import seaborn as sns
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
from matplotlib.font_manager import FontProperties
# from ioh import get_problem, ProblemClass
from collections import defaultdict


algo2algo = {
    "cmaes_parallel": "CMA-ES-DS",
    "CMA": "CMAES",
    "HILLVALLEA": "HillvallEA",
    "RANDOM": "Random",
    "RSCMA": "RSCMSAES",
    "WGRAD": "WGraD"
}

algo2color = { 
    'CMARepuLexico':'tab:red'	,
    'CMARepuLexico_10_90': 'tab:olive'	,
    'Random_gurobi':'tab:green'	,
    'Random':'tab:green',
    'CMAES':'tab:blue'	,
    'HillvallEA':'tab:grey'	,
    'RSCMSAES':'tab:pink'	,
    'WGraD':'tab:cyan'	,
    'CMAREPULexico_greedy':'black'	,
    'CMAREPULexico+greedy+parent':'orange'	,
    'CMAREPULexico_so_far_parent':'tab:brown'	,
    'CMAES_J_D':'aquamarine'	,
    'CMAES_J_D_lexico':'red'	,
    'CMAES_parallel_best':'green'	,
    'CMA-ES-DS':'tab:red'	
}

#      str         int         str         list(tuple(float, float))
#      max_fevals  f_id        algo_name   list of pairs (alternatives, leader)
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def read_point(filename):
    M = [] 
    for line in open(filename,  "r"):
        line = line.strip()
        if line == "" or line[0] == "#":
            continue
        tab = list(map(float, line.strip().split()))
        if len(tab):
            M.append(tab)
    if len(M) != 7:
        return None 
    f1 = M[-1][0] 
    f5 = M[-1][-1]
    assert len(M[-1]) == 5
    avg = (f5 * 5 - f1) / 4    
    return (avg, f1)


def read_data(path):
    for RES_ALGO in os.listdir(path):
        if RES_ALGO[:3] == "RES":
            algo_name = algo2algo[RES_ALGO[3:]]
            for max_fevals in os.listdir(f"{path}/{RES_ALGO}/dim10"):
                assert max_fevals in ["1k", "10k"]
                for runs in os.listdir(f"{path}/{RES_ALGO}/dim10/{max_fevals}/dmin_10"):
                    for f_id_run_id in os.listdir(f"{path}/{RES_ALGO}/dim10/{max_fevals}/dmin_10/{runs}"):
                        f_id = int(f_id_run_id.split("_")[0])
                        point = read_point(f"{path}/{RES_ALGO}/dim10/{max_fevals}/dmin_10/{runs}/{f_id_run_id}")
                        if point: # the run was complete 
                            data[max_fevals][f_id][algo_name].append(point)
        else:
            algo_name = algo2algo[RES_ALGO]
            for max_fevals in os.listdir(f"{path}/{RES_ALGO}/dim10"):
                assert max_fevals in ["1k", "10k"]
                for runs in os.listdir(f"{path}/{RES_ALGO}/dim10/{max_fevals}"):
                    for results in os.listdir(f"{path}/{RES_ALGO}/dim10/{max_fevals}/{runs}"):
                        tab = results.split("_")
                        f_id = int(tab[5])
                        dmin = tab[7]
                        if dmin != "10":
                            continue
                        point = read_point(f"{path}/{RES_ALGO}/dim10/{max_fevals}/{runs}/{results}")
                        if point: # the run was complete 
                            data[max_fevals][f_id][algo_name].append(point)

read_data(sys.argv[1])

def plot_bi_objective():
    for mode in ["allruns", "avg"]:
        for max_fevals in data:
            
            # Create the figure layout with 8 rows and 3 columns
            fig, axs = plt.subplots(8, 3, figsize=(12, 20), dpi=80)
            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.4, hspace=0.5)

            function_ids = sorted(data[max_fevals].keys())
            all_algos_name = set()
            # Loop over all function IDs
            for f_id, ax in zip(function_ids, axs.ravel()):

                # Add labels to axes and a title
                ax.set_title(f'f{f_id}', fontsize=17, y=0.97, x=0.010, pad=-13, loc='left', zorder=15)
                for algo_name in data[max_fevals][f_id]:
                    all_algos_name.add(algo_name)
                    points = data[max_fevals][f_id][algo_name]
                    if mode == "avg":
                        points = [(sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points))]
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    ax.scatter(x, y, color=algo2color[algo_name], label=algo_name, zorder=10)  
            font_props = FontProperties(size=18)

            handles = []
            for algo_name in all_algos_name:
                handles.append(mpatches.Patch(color=algo2color[algo_name], label=algo_name))
            # Adding legend with text size set
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, prop=font_props)
            fig.tight_layout(rect=(0.015, 0, 1, 1))
            plt.savefig(f'plot/bi-objective_{max_fevals}_{mode}_dim_10_dmin_10.pdf', bbox_inches="tight")
            # Show the plot
            # plt.show()


plot_bi_objective()

