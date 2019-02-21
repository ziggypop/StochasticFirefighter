#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:56:36 2018

@author: s1238047
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import basic
import tqdm
import os

def threat(G,b,d):
    """
    Obtain the threatened set given the burned set b and defended set d in the graph G
    """
    thre = []
    for i in b:
        newth = [n for n in G.neighbors(i)]# if n in untouched or n in thre]
        thre = np.concatenate((thre,newth))
    thre = list(set(thre))
    thre = np.setdiff1d(thre,b)
    thre = np.setdiff1d(thre,d)
    return thre

def cut(I,v,G):
    """
    Calculate the cut given infected set I and threatened node v, and the graph G
    """
    s = [n for n in G.neighbors(v)]
    return len(set(s).intersection(I))

def growth_rate(b,d,G,p):
    """
    Calculate the growth rate without firefighters
    """
    thre = threat(G,b,d) 
    if len(thre) == 0:
        print('Contained')
        return 0
    else:
        gr = 0   
        for i in thre:
            gr = gr + (1-np.power((1-p),cut(b,i,G)))
        return gr
    
    


pe=[0.1,0.3,0.5,0.7,0.9]
firefighters=[1,3,5,7,9]
#pe=0.2  
n=1000
avg=10

gr=np.zeros((100,5,5,100)) #(exp,infectionrate,firefighter,iteration)
r=np.sqrt(avg/(np.pi*n))
for f in range(len(firefighters)):
    ffnum=firefighters[f]
    for pp in range(len(pe)):
        p=pe[pp]
        for exp in range(100):
            print(exp)
            G = nx.random_geometric_graph(n, r)
            pos = nx.get_node_attributes(G, 'pos')  
            edges=defaultdict(list)
            for i in range(n):
                edges[i]=[m for m in G.neighbors(i)]
            #print('avearge degree:'+str(basic.avgdeg(G,n)))
               
            for t in range(100):
        
                if t==0:
                    burned=[np.random.randint(n)]
                    threatened = defaultdict(list)
                    threatened[t] = threat(G,burned,[])
                    defended=basic.heuristic_degree(edges, threatened[t],ffnum)
                    #defended=bfs(G,threatened[t],burned,[],firefighter)
                    untouched = np.setdiff1d(np.arange(n),threatened[t])
                    untouched = np.setdiff1d(untouched, burned)
                    threatened[t]=np.setdiff1d(threatened[t],defended)
                    gr[exp][pp][f][t]=growth_rate(burned,defended,G,p)
                    #print('growth rate')
                    #print(gr[t])
                else:
                    thre=threatened[t-1]
                    for i in thre:
                        prob=np.random.uniform()
                        if prob < p:
                            burned=np.concatenate((burned,[i]))
                    burned=list(set(burned))
                    threatened[t]=threat(G,burned,defended)
                    defended=np.concatenate((defended,basic.heuristic_degree(edges, threatened[t],ffnum)))
                    gr[exp][pp][f][t]=growth_rate(burned,defended,G,p)
                    #print('growth rate')
                    #print(gr[t])
                ###plotting the results####
                #values=basic.assign_values(burned, threatened[t],defended,n)
                #basic.plot_graph(G,pos,values)
                if len(threatened[t])==0:
                    break   
    
            print('time to contain the fire:')        
            print(t)
            print('saved nodes:')
            print(n-len(burned))  
            #plt.plot(np.arange(100),gr)
            #plt.title('geo p=0.2 ffnum='+str(ffnum))
            #plt.show()
np.save('geo_gr_bfs1',gr)      
print('geo done')

gr=np.zeros((100,5,5,100)) #(exp,infectionrate,firefighter,iteration)

for f in range(len(firefighters)):
    ffnum=firefighters[f]
    for pp in range(len(pe)):
        p=pe[pp]
        for exp in range(100):
            print(exp)
            G=nx.erdos_renyi_graph(n,avg/n)
            pos = nx.spring_layout(G)
            edges=defaultdict(list)
            for i in range(n):
                edges[i]=[m for m in G.neighbors(i)]
            #print('avearge degree:'+str(basic.avgdeg(G,n)))
               
            for t in range(100):
        
                if t==0:
                    burned=[np.random.randint(n)]
                    threatened = defaultdict(list)
                    threatened[t] = threat(G,burned,[])
                    #defended=basic.heuristic_degree(edges, threatened[t],ffnum)
                    #defended=bfs(G,threatened[t],burned,[],firefighter)
                    untouched = np.setdiff1d(np.arange(n),threatened[t])
                    untouched = np.setdiff1d(untouched, burned)
                    threatened[t]=np.setdiff1d(threatened[t],defended)
                    gr[exp][pp][f][t]=growth_rate(burned,defended,G,p)
                    #print('growth rate')
                    #print(gr[t])
                else:
                    thre=threatened[t-1]
                    for i in thre:
                        prob=np.random.uniform()
                        if prob < p:
                            burned=np.concatenate((burned,[i]))
                    burned=list(set(burned))
                    threatened[t]=threat(G,burned,defended)
                    #defended=np.concatenate((defended,basic.heuristic_degree(edges, threatened[t],ffnum)))
                    gr[exp][pp][f][t]=growth_rate(burned,defended,G,p)
                    #print('growth rate')
                    #print(gr[t])
                ###plotting the results####
                #values=basic.assign_values(burned, threatened[t],defended,n)
                #basic.plot_graph(G,pos,values)
                if len(threatened[t])==0:
                    break   
    
            print('time to contain the fire:')        
            print(t)
            print('saved nodes:')
            print(n-len(burned))  
            #plt.plot(np.arange(100),gr)
            #plt.title('geo p=0.2 ffnum='+str(ffnum))
            #plt.show()
np.save('erdos_gr_bfs1',gr)    
'''
with tqdm(total=num_training) as pbar:
    out_str = "epoch={0:d}, iter={1:d}, loss={2:.6f}, mean loss={3:.6f}".format(
                                epoch+1, 0, 0, 0)
    pbar.set_description(out_str)                
    out_str = "epoch={0:d}, iter={1:d}, loss={2:.6f}, mean loss={3:.6f}".format(
                               epoch+1, it, loss_val, (loss_per_epoch / i))
    pbar.set_description(out_str)
                    pbar.update(1)
'''    
    
    
    
    
    
    
    