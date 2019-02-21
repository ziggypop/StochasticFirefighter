#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 09:59:18 2018

@author: s1238047
"""
import basic
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

def stochastic_firefighting_degree(n,G,pe):
    lenburn=np.zeros(1000)
    ffnum=np.zeros(1000)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=basic.avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    
    for t in range(n):
        if t==0:
            burned = np.random.randint(n,size=2000)
            #print(burned)
            threatened = defaultdict(list)
            threatened[0] = []
            untouched = np.setdiff1d(np.arange(n), burned)
            for i in burned:
                newth=[n for n in edges[i] if n in untouched]
                threatened[t]=np.concatenate((threatened[t],newth))
            ffnum[t]=np.round(0.001*len(threatened[t]))+1
            print(ffnum[t])
            #defended=basic.heuristic_degree(edges, threatened[0],int(ffnum[t]))
            defended=basic.heuristic_degree(edges, threatened[0],int(ffnum[t]))
            untouched = np.setdiff1d(untouched,threatened[0])
            #threatened[0]=np.setdiff1d(threatened[0],defended)
        else:

            for i in threatened[t-1]:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in threatened[t-1]]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            ffnum[t]=np.round(0.001*len(threatened[t]))+1
            print(ffnum[t])
            #if ffnum[t]<100:
            #    ffnum[t]=100
            defended=np.concatenate((defended,basic.heuristic_degree(edges,threatened[t],int(ffnum[t]))))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
        #print('length threatened')
        #print(len(threatened[t]))
        
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
        lenburn[t]=len(burned)
        if len(threatened[t])==0:
            break
    np.place(lenburn,lenburn==0,len(burned))  
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved,lenburn,ffnum

def stochastic_firefighting_bfsdegree(n,G,pe):
    lenburn=np.zeros(1000)
    ffnum=np.zeros(1000)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=basic.avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    
    for t in range(n):
        if t==0:
            burned = np.random.randint(n,size=200)
            #print(burned)
            threatened = defaultdict(list)
            threatened[0] = []
            untouched = np.setdiff1d(np.arange(n), burned)
            for i in burned:
                newth=[n for n in edges[i] if n in untouched]
                threatened[t]=np.concatenate((threatened[t],newth))
            defended=[]
            ffnum[t]=np.round(pe*len(threatened[t])+1)
            print(ffnum[t])
            defended=basic.bfs_degree(edges, threatened[0],burned,defended,int(ffnum[t]))
            untouched = np.setdiff1d(untouched,threatened[0])
            #threatened[0]=np.setdiff1d(threatened[0],defended)
        else:

            for i in threatened[t-1]:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in threatened[t-1]]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            ffnum[t]=np.round(pe*len(threatened[t])+1)
            print(ffnum[t])
            #if ffnum[t]<100:
            #    ffnum[t]=100
            defended=np.concatenate((defended,basic.bfs_degree(edges,threatened[t],burned,defended,int(ffnum[t]))))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
        #print('length threatened')
        #print(len(threatened[t]))
        
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
        lenburn[t]=len(burned)
        if len(threatened[t])==0:
            break
    np.place(lenburn,lenburn==0,len(burned))  
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved,lenburn,ffnum


G=nx.read_edgelist('Email-enron.txt', create_using=nx.Graph(), nodetype=int)
n=len(G)
print(n)
#n=2000
#pos = nx.spring_layout(G)
#print(pos)
edges=defaultdict(list)
degree=defaultdict(int)
for i in range(n):
    edges[i]=[m for m in G.neighbors(i)]
    degree[i]=len(edges[i])
#print(degree.items())
#plt.figure(figsize=(10,10))
#plt.hist(degree.values(),bins=30)
#plt.xlabel('Degrees')
#plt.title('Enron email network 2000 nodes,'+'\n'+' average degree = '+str(avgdeg(G,n)))
#plt.savefig('enron_dis_2000.pdf')
#plt.show()
    
p=0.05
  

ave_degree=basic.avgdeg(G,n)

graph_type='erdos'
erdos_sto_deg=np.zeros((100,1000))
erdos_sto_deg_ff=np.zeros((100,1000))
for k in range(1):
    ad, t, sa,l ,ff= stochastic_firefighting_degree(n,G,p)
    erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
    erdos_sto_deg_ff[k]=ff
    print(k)
    plt.plot(np.arange(100),erdos_sto_deg_ff[k][:100])
    plt.title('firefighter')
    plt.show()
    plt.plot(np.arange(100),erdos_sto_deg[k][:100])
    plt.title('burned')
    plt.show()
np.save('enron_degree',erdos_sto_deg)
np.save('enron_degree_ff',erdos_sto_deg_ff)
print('deg done')
'''
erdos_sto_bfs=np.zeros((50,1000))
erdos_sto_bfs_ff=np.zeros((50,1000))

for k in range(1):
    ad, t, sa,l,ff = stochastic_firefighting_bfsdegree(n,G,p)
    erdos_sto_bfs[k]=l#[p,ad,t,sa/n,firefighters]
    erdos_sto_bfs_ff[k]=ff
    print(k)
    plt.plot(np.arange(1000),erdos_sto_bfs_ff[k])
    plt.title('firefighter')
    plt.show()
    plt.plot(np.arange(1000),erdos_sto_bfs[k])
    plt.title('burned')
    plt.show()
#np.save('enron_bfs',erdos_sto_bfs)
#np.save('enron_bfs_ff',erdos_sto_bfs_ff)
print('bfs done')
'''



