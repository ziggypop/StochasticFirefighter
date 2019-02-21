#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:48:19 2018

@author: s1238047
"""

import basic
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

def stochastic_firefighting_degree(n,G,pe,r):
    lenburn=np.zeros(1000)
    ffnum=np.zeros(1000)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=basic.avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    
    for t in range(n):
        if t==0:
            burned = np.random.randint(n,size=50)
            #print(burned)
            threatened = defaultdict(list)
            threatened[0] = []
            untouched = np.setdiff1d(np.arange(n), burned)
            for i in burned:
                newth=[n for n in edges[i] if n in untouched]
                threatened[t]=np.concatenate((threatened[t],newth))
            ffnum[t]=np.round(r*pe*len(threatened[t]))+1
            #print(ffnum[t])
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
            ffnum[t]=np.round(r*pe*len(threatened[t]))+1
            #print(ffnum[t])
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
        if lenburn[t]+sum(ffnum) >= n:
            break
    np.place(lenburn,lenburn==0,len(burned))  
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved,lenburn,ffnum

def stochastic_firefighting_degree_con(ffnum,n,G,pe):
    lenburn=np.zeros(1000)
    de=0
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=basic.avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    
    for t in range(n):
        #print(t)
        if t==0:
            burned = np.random.randint(n,size=50)
            #print(burned)
            threatened = defaultdict(list)
            threatened[0] = []
            untouched = np.setdiff1d(np.arange(n), burned)
            for i in burned:
                newth=[n for n in edges[i] if n in untouched]
                threatened[t]=np.concatenate((threatened[t],newth))
            defended=basic.heuristic_degree(edges, threatened[0],ffnum)
            untouched = np.setdiff1d(untouched,threatened[0])
            de=de+ffnum
            #threatened[0]=np.setdiff1d(threatened[0],defended)
        else:
            de=de+ffnum
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
            defended=np.concatenate((defended,basic.heuristic_degree(edges,threatened[t],ffnum)))
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
        if lenburn[t]+de >= n:
            break
    np.place(lenburn,lenburn==0,len(burned))  
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #de=ffnum*(t)
    #print(n-len(burned))
    return averagedegree,t,saved,lenburn,de


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
p=0.05
n=1000
avg=10
r=np.sqrt(avg/(np.pi*n))
'''
erdos_sto_deg=np.zeros((1000,1000))
erdos_deg_def=np.zeros(1000)
for k in range(1000):
    print(k)
    G1=nx.erdos_renyi_graph(n,avg/n)
    ad, t, sa,l,d = stochastic_firefighting_degree_con(8,n,G1,p)
    erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
    erdos_deg_def[k]=d
    #plt.plot(np.arange(1000),erdos_sto_deg[k])
    #plt.show()
    #print(k)    
np.save('erdos_degree_con_8',erdos_sto_deg)
np.save('erdos_degree_con_defend_8',erdos_deg_def)
print('deg done')
'''

firefighters=[2]
for j in range(len(firefighters)):
    geo_sto_deg=np.zeros((1000,1000))
    geo_deg_def=np.zeros(1000)
    print(firefighters[j])
    for k in range(1000):
        print(k)
        G2=nx.random_geometric_graph(n, r)
        ad, t, sa,l,d = stochastic_firefighting_degree_con(firefighters[j],n,G2,p)
        geo_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
        geo_deg_def[k]=d
        #plt.plot(np.arange(1000),geo_sto_deg[k])
        #plt.show()
        #print(k)
    np.save('geo_degree_con_'+str(firefighters[j]),geo_sto_deg)
    np.save('geo_degree_con_defend'+str(firefighters[j]),geo_deg_def)
    #np.save('geo_degree',geo_sto_deg)
print('deg done')


'''
o=[2,1,0.5,0.25,0.1]
for j in range(len(o)):
    print(o[j])
    erdos_sto_deg=np.zeros((1000,1000))
    erdos_sto_deg_ff=np.zeros((1000,1000))
    erdos_labelled=np.zeros((1000,1000))
    for k in range(1000):
        print(k)
        G1=nx.erdos_renyi_graph(n,avg/n)
        ad, t, sa,l ,ff= stochastic_firefighting_degree(n,G1,p,o[j])
        erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
        erdos_sto_deg_ff[k]=ff
        erdos_labelled[k]=erdos_sto_deg[k][-1:]+sum(erdos_sto_deg_ff[k])
    np.save('erdos_degree_labelled-'+str(j),erdos_labelled)
    np.save('erdos_degree-'+str(j),erdos_sto_deg)
    np.save('erdos_degree_ff-'+str(j),erdos_sto_deg_ff)


erdos_sto_deg=np.zeros((1000,1000))
erdos_deg_def=np.zeros(1000)
for k in range(1000):
    print(k)
    G1=nx.erdos_renyi_graph(n,avg/n)
    ad, t, sa,l,d = stochastic_firefighting_degree_con(30,n,G1,p)
    erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
    erdos_deg_def[k]=d
    #plt.plot(np.arange(1000),erdos_sto_deg[k])
    #plt.show()
    #print(k)    
np.save('erdos_degree_con_30',erdos_sto_deg)
np.save('erdos_degree_con_defend_30',erdos_deg_def)
print('deg done')
erdos_sto_deg=np.zeros((1000,1000))
erdos_deg_def=np.zeros(1000)
for k in range(1000):
    print(k)
    G1=nx.erdos_renyi_graph(n,avg/n)
    ad, t, sa,l,d = stochastic_firefighting_degree_con(20,n,G1,p)
    erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
    erdos_deg_def[k]=d
    #plt.plot(np.arange(1000),erdos_sto_deg[k])
    #plt.show()
    #print(k)    
np.save('erdos_degree_con_20',erdos_sto_deg)
np.save('erdos_degree_con_defend_20',erdos_deg_def)
print('deg done')




for j in range(len(firefighters)):
    print(firefighters[j])
    erdos_sto_deg=np.zeros((1000,1000))
    erdos_deg_def=np.zeros(1000)
    for k in range(1000):
        print(k)
        G1=nx.erdos_renyi_graph(n,avg/n)
        ad, t, sa,l,d = stochastic_firefighting_degree_con(firefighters[j],n,G1,p)
        erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
        erdos_deg_def[k]=d
        #plt.plot(np.arange(1000),erdos_sto_deg[k])
        #plt.show()
        #print(k)    
    np.save('erdos_degree_con_'+str(firefighters[j]),erdos_sto_deg)
    np.save('erdos_degree_con_defend'+str(firefighters[j]),erdos_deg_def)
print('deg done')


erdos_sto_deg=np.zeros((1000,1000))
erdos_sto_deg_ff=np.zeros((1000,1000))
erdos_labelled=np.zeros((1000,1000))
for k in range(1000):
    print(k)
    G1=nx.erdos_renyi_graph(n,avg/n)
    ad, t, sa,l ,ff= stochastic_firefighting_degree(n,G1,p,2)
    erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
    erdos_sto_deg_ff[k]=ff
    erdos_labelled[k]=erdos_sto_deg[k][-1:]+sum(erdos_sto_deg_ff[k])
np.save('erdos_degree_labelled_2pe',erdos_labelled)
np.save('erdos_degree_2pe',erdos_sto_deg)
np.save('erdos_degree_ff_2pe',erdos_sto_deg_ff)

print('erdos done')




for j in range(len(firefighters)):
    print(firefighters[j])
    erdos_sto_deg=np.zeros((1000,1000))
    erdos_deg_def=np.zeros(1000)
    for k in range(1000):
        print(k)
        G1=nx.erdos_renyi_graph(n,avg/n)
        ad, t, sa,l,d = stochastic_firefighting_degree_con(firefighters[j],n,G1,p)
        erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
        erdos_deg_def[k]=d
        #plt.plot(np.arange(1000),erdos_sto_deg[k])
        #plt.show()
        #print(k)    
    np.save('erdos_degree_con_'+str(firefighters[j]),erdos_sto_deg)
    np.save('erdos_degree_con_defend'+str(firefighters[j]),erdos_deg_def)
print('deg done')

for j in range(len(firefighters)):
    geo_sto_deg=np.zeros((1000,1000))
    geo_deg_def=np.zeros(1000)
    print(firefighters[j])
    for k in range(1000):
        print(k)
        G2=nx.random_geometric_graph(n, r)
        ad, t, sa,l,d = stochastic_firefighting_degree_con(firefighters[j],n,G2,p)
        geo_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
        geo_deg_def[k]=d
        #plt.plot(np.arange(1000),geo_sto_deg[k])
        #plt.show()
        #print(k)
    np.save('geo_degree_con_'+str(firefighters[j]),geo_sto_deg)
    np.save('geo_degree_con_defend'+str(firefighters[j]),geo_deg_def)
    #np.save('geo_degree',geo_sto_deg)
print('deg done')



o=[2,1,0.5,0.25,0.1]
for j in range(len(o)):
    print(o[j])
    erdos_sto_deg=np.zeros((1000,1000))
    erdos_sto_deg_ff=np.zeros((1000,1000))
    erdos_labelled=np.zeros((1000,1000))
    for k in range(1000):
        print(k)
        G1=nx.erdos_renyi_graph(n,avg/n)
        ad, t, sa,l ,ff= stochastic_firefighting_degree(n,G1,p,o[j])
        erdos_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
        erdos_sto_deg_ff[k]=ff
        erdos_labelled[k]=erdos_sto_deg[k][-1:]+sum(erdos_sto_deg_ff[k])
    np.save('erdos_degree_labelled-'+str(j),erdos_labelled)
    np.save('erdos_degree-'+str(j),erdos_sto_deg)
    np.save('erdos_degree_ff-'+str(j),erdos_sto_deg_ff)

print('erdos done')
#plt.plot(np.arange(100),erdos_sto_deg_ff[0][:100])
#plt.title('firefighter')
#plt.show()
#plt.plot(np.arange(100),erdos_sto_deg[0][:100])
#plt.title('burned')
#plt.show()

for j in range(len(o)):
    print(o[j])
    geo_sto_deg=np.zeros((1000,1000))
    geo_sto_deg_ff=np.zeros((1000,1000))
    geo_labelled=np.zeros((1000,1000))
    for k in range(1000):
        print(k)
        G2=nx.random_geometric_graph(n, r)
        ad, t, sa,l ,ff= stochastic_firefighting_degree(n,G2,p,o[j])
        geo_sto_deg[k]=l#[p,ad,t,sa/n,firefighters]
        geo_sto_deg_ff[k]=ff
        geo_labelled[k]=geo_sto_deg[k][-1:]+sum(geo_sto_deg_ff[k])
    np.save('geo_degree_labelled-'+str(j),geo_labelled)
    np.save('geo_degree-'+str(j),geo_sto_deg)
    np.save('geo_degree_ff-'+str(j),geo_sto_deg_ff)
    
print('geo done')
'''

