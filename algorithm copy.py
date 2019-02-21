#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:39:05 2018

@author: s1572156
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict



def rand_geo_graph(n,avg):
    r=np.sqrt(avg/(np.pi*n))
    G = nx.random_geometric_graph(n, r)
    pos = nx.get_node_attributes(G, 'pos')
    return G, pos

def erdos_renyi(n,avg):
    p=avg/n 
    G=nx.erdos_renyi_graph(n,p)
    pos = nx.spring_layout(G)
    return G, pos

def avgdeg(g,n):
    """
    compute average degree given graph g and number of nodes n
    """
    s=0
    for i in range(n):
        s=s+g.degree[i]
    avg=s/n
    return avg 
def heuristic_degree(edges,threa,m):
    degree=defaultdict(int)
    for i in threa:
        degree[i]=len(edges[i])
    ff=sorted(list(degree.values()))[::-1]
    ff=ff[:m]
    de=[k for k,v in degree.items() if v in ff]
    return de[:m]

def descendant(paths,thre,ffn):
    p=defaultdict(int)   
    for i in thre:
        p[i]=paths[i]
    ff=sorted(list(p.values()))
    #print(ff)
    ff=ff[:ffn]
    de=[k for k,v in p.items() if v in ff]
    return de[:ffn]
def descendant4(paths,thre,ffn):
    p=defaultdict(int)   
    for i in thre:
        p[i]=paths[i]
    ff=sorted(list(p.values()))[::-1]
    #print(ff)
    ff=ff[:ffn]
    de=[k for k,v in p.items() if v in ff]
    return de[:ffn]
def descendant2(edges,ffn,burned,thre,defend):
    tuples=list(itertools.permutations(thre,ffn))
    s=defaultdict(list)
    intersection=defaultdict(int)
    
    for i in tuples:
        sss=[]
        for j in range(len(i)):
            s[j]=edges[i[j]]
            if j > 0:
                sss= set(s[j-1]).intersection(s[j])
                #[val for val in s[j] if val in s[j-1]]
        a = set(burned).intersection(sss)
        intersection[i]=len(np.setdiff1d(sss,a))
    if len(list(intersection.values()))==0:
        #print(intersection.items())
        return bfs_degree(edges,thre,burned,defend,ffn)
    ff=np.max(list(intersection.values()))
    #print(ff)
    de=[k for k,ff in intersection.items()]
    #print(de)
    return de[0]

def descendant3(edges,ffn,burned,thre):
    score=defaultdict(float)
    for i in thre:
        des = edges[i]
        
        score[i] = 0.0
        for j in des:
            if len(edges[j]) == 1:
                score[i]=score[i]+1
            else:
                score[i] = score[i] + 1/(len(set(edges[j]).intersection(list(burned))) + \
                                         len(set(edges[j]).intersection(thre)))
        print('score')
        print(score[i])
    ff=list(score.values())[::-1]
    print('ff')
    print(ff)
    de=[k for k,ff in score.items()]
    return de[:ffn]

def bfs_degree(edges,threa,burn,defend,m):
    desnum=defaultdict(int)
    #print(threa)
    for i in threa:
        desnum[i]=0
        num=[]
        for j in edges[i]:
            if j not in burn and j not in defend:
                #print(num,edges[j])
                num = np.concatenate((num, edges[j]))
                desnum[i]=desnum[i]+1
        num= [ x for x in num if x not in burn ] 
        num= [ x for x in num if x not in defend] 
        desnum[i]=desnum[i]+len(set(num))
        if desnum[i]==0:
            desnum[i]=1
    ff=sorted(list(desnum.values()))[::-1]
    ff=ff[:m]
    de=[k for k,v in desnum.items() if v in ff]
    return de[:m]

def bfs(g,threa,burn,defend,m,n):
    
    ###noticably slower
    
    desnum=defaultdict(int)
    unburned=np.setdiff1d(np.arange(n),defend)
    unburned=np.setdiff1d(unburned,burn)
    gg=g.subgraph(unburned)
    for i in threa:
        succ=dict(nx.bfs_successors(gg,i))
        desnum[i]=len(list(succ.keys()))
    
    ff=sorted(list(desnum.values()))[::-1]
    ff=ff[:m]
    de=[k for k,v in desnum.items() if v in ff]
    return de[:m]

def assign_values(bu, th, de,node):
    values=np.zeros(node)
    for i in range(node):
        if i in bu:
            values[i]=0.25
        elif i in th:
            values[i]=0.6
        elif i in de:
            values[i]=0.75
        else:
            values[i]=0.5
    return values
def plot_graph(g,p,val):
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(g, p,
                       node_color=val,
                       node_size=80,
                       alpha=0.8,
                       cmap=plt.get_cmap('jet'))
    nx.draw_networkx_edges(g, p, alpha=0.5)
    plt.show()
  

def stochastic_firefighting_degree(ffnum,n,avg,graph,pe):
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    
    for t in range(n):
        if t==0:
            burned = [np.random.randint(n)]
            threatened = defaultdict(list)
            threatened[0] = edges[burned[0]]
            defended=heuristic_degree(edges, threatened[0],ffnum)
            untouched = np.setdiff1d(np.arange(n),threatened[0])
            untouched = np.setdiff1d(untouched, burned)
            threatened[0]=np.setdiff1d(threatened[0],defended)
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
            defended=np.concatenate((defended,heuristic_degree(edges,threatened[t],ffnum)))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
        #print('length threatened')
        #print(len(threatened[t]))
        
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
        
        if len(threatened[t])==0:
            break
       
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved


def firefighting(ffnum,n,avg,graph):
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
   

    for t in range(n):
        if t==0:
            burned = [np.random.randint(n)]
            threatened = defaultdict(list)
            threatened[0] = edges[burned[0]]
            defended=heuristic_degree(edges, threatened[0],ffnum)
            untouched = np.setdiff1d(np.arange(n),threatened[0])
            untouched = np.setdiff1d(untouched, burned)
            threatened[0]=np.setdiff1d(threatened[0],defended)
        else:
            for i in threatened[t-1]:
                burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in threatened[t-1]]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            defended=np.concatenate((defended,heuristic_degree(edges,threatened[t],ffnum)))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
            
            
        
        #print('length threatened')
        #print(len(threatened[t]))
        
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
        
        if len(threatened[t])==0:
            break
       
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved

def stochastic_firefighting_descendant(ffnum,n,avg,graph,pe):
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    burned = [np.random.randint(n)]
    path = nx.single_source_shortest_path(G, burned[0])
    for i in path.keys():
        path[i]=len(path[i])
        
    for t in range(n):
        
        if t==0:
            
            threatened = defaultdict(list)
            threatened[0] = edges[burned[0]]
            defended=descendant(path,threatened[0],ffnum)
            untouched = np.setdiff1d(np.arange(n),threatened[0])
            untouched = np.setdiff1d(untouched, burned)
            threatened[0]=np.setdiff1d(threatened[0],defended)
        else:
            thre=threatened[t-1]
            for i in thre:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            #print('length burned')
            #print(len(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in thre]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            defended=np.concatenate((defended,descendant(path,threatened[t],ffnum)))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
            
        #print('length threatened')
        #print(len(threatened[t]))
    
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
    
        if len(threatened[t])==0:
            break   
        
    #print('time to contain the fire:')        
    #print(t)
    
    #print('saved nodes:')
    #print(n-len(burned))   
    
       
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved

def stochastic_firefighting_descendant4(ffnum,n,avg,graph,pe):
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    burned = [np.random.randint(n)]
    path = nx.single_source_shortest_path(G, burned[0])
    for i in path.keys():
        path[i]=len(path[i])
        
    for t in range(n):
        
        if t==0:
            
            threatened = defaultdict(list)
            threatened[0] = edges[burned[0]]
            defended=descendant4(path,threatened[0],ffnum)
            untouched = np.setdiff1d(np.arange(n),threatened[0])
            untouched = np.setdiff1d(untouched, burned)
            threatened[0]=np.setdiff1d(threatened[0],defended)
        else:
            thre=threatened[t-1]
            for i in thre:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            #print('length burned')
            #print(len(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in thre]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            defended=np.concatenate((defended,descendant4(path,threatened[t],ffnum)))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
            
        #print('length threatened')
        #print(len(threatened[t]))
    
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
    
        if len(threatened[t])==0:
            break   
        
    #print('time to contain the fire:')        
    #print(t)
    
    #print('saved nodes:')
    #print(n-len(burned))   
    
       
    #print('time to contain the fire:')
    #print(t)
    
    #print('saved nodes:')
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved
    
def stochastic_firefighting_bfs(ffnum,n,avg,graph,pe):
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    burned = [np.random.randint(n)]

    for t in range(n):
        
        if t==0:
            
            threatened = defaultdict(list)
            threatened[t] = edges[burned[t]]
            #defended=heuristic_degree(edges, threatened[0],ffnum)
            defended=bfs_degree(edges,threatened[t],burned,[],ffnum)
            untouched = np.setdiff1d(np.arange(n),threatened[t])
            untouched = np.setdiff1d(untouched, burned)
            threatened[t]=np.setdiff1d(threatened[t],defended)
        else:
            
            #print('length defended'+str(len(defended)))
            #thre=np.setdiff1d(threatened[t-1],defended)
            thre=threatened[t-1]
            for i in thre:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            #print('length burned')
            #print(len(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in thre]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            defended=np.concatenate((defended,bfs_degree(edges,threatened[t],burned,defended,ffnum)))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
    
        #print('length threatened')
        #print(len(threatened[t]))
    
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
    
        if len(threatened[t])==0:
            break   
        
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved 


def stochastic_firefighting_bfs2(ffnum,n,avg,graph,pe):
    
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    

    for t in range(n):
        
        if t==0:
            burned = [np.random.randint(n)]
            threatened = defaultdict(list)
            threatened[t] = edges[burned[t]]
            #defended=heuristic_degree(edges, threatened[0],ffnum)
            defended=bfs(G,threatened[t],burned,[],ffnum,n)
            untouched = np.setdiff1d(np.arange(n),threatened[t])
            untouched = np.setdiff1d(untouched, burned)
            threatened[t]=np.setdiff1d(threatened[t],defended)
        else:
            
            #print('length defended'+str(len(defended)))
            #thre=np.setdiff1d(threatened[t-1],defended)
            thre=threatened[t-1]
            for i in thre:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            #print('length burned')
            #print(len(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in thre]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            defended=np.concatenate((defended,bfs(G,threatened[t],burned,defended,ffnum,n)))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
    
        #print('length threatened')
        #print(len(threatened[t]))
    
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
    
        if len(threatened[t])==0:
            break   
        
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved 

def stochastic_firefighting_des2(ffnum,n,avg,graph,pe):
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    burned = [np.random.randint(n)]
    defended=[]
    for t in range(n):
        
        if t==0:
            
            threatened = defaultdict(list)
            threatened[t] = edges[burned[t]]
            #defended=heuristic_degree(edges, threatened[0],ffnum)
            #defended=bfs_degree(edges,threatened[t],burned,[],ffnum)
            defended=descendant2(edges,ffnum,burned,threatened[t],defended)
            untouched = np.setdiff1d(np.arange(n),threatened[t])
            untouched = np.setdiff1d(untouched, burned)
            threatened[t]=np.setdiff1d(threatened[t],defended)
        else:
            
            #print('length defended'+str(len(defended)))
            #thre=np.setdiff1d(threatened[t-1],defended)
            thre=threatened[t-1]
            for i in thre:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            #print('length burned')
            #print(len(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in thre]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            defended=np.concatenate((defended,descendant2(edges,ffnum,burned,threatened[t],defended)))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
    
        #print('length threatened')
        #print(len(threatened[t]))
    
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
    
        if len(threatened[t])==0:
            break   
        
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved 

def stochastic_firefighting_des3(ffnum,n,avg,graph,pe):
    if graph == 'geometric':
        G, pos = rand_geo_graph(n,avg)
    elif graph == 'erdos':
        G, pos = erdos_renyi(n,avg)
    #print('avearge degree:'+str(avgdeg(G,n)))
    averagedegree=avgdeg(G,n)
    edges=defaultdict(list)
    for i in range(n):
        edges[i]=[m for m in G.neighbors(i)]
    burned = [np.random.randint(n)]
    defended=[]
    for t in range(n):
        
        if t==0:
            
            threatened = defaultdict(list)
            threatened[t] = edges[burned[t]]
            #defended=heuristic_degree(edges, threatened[0],ffnum)
            #defended=bfs_degree(edges,threatened[t],burned,[],ffnum)
            defended=descendant3(edges,ffnum,burned,threatened[t])
            untouched = np.setdiff1d(np.arange(n),threatened[t])
            untouched = np.setdiff1d(untouched, burned)
            threatened[t]=np.setdiff1d(threatened[t],defended)
        else:
            
            #print('length defended'+str(len(defended)))
            #thre=np.setdiff1d(threatened[t-1],defended)
            thre=threatened[t-1]
            for i in thre:
                prob=np.random.uniform()
                if prob < pe:
                    burned=np.concatenate((burned,[i]))
            burned=list(set(burned))
            #print('length burned')
            #print(len(burned))
            for i in burned:
                newth=[n for n in edges[i] if n in untouched or n in thre]
                threatened[t]=np.concatenate((threatened[t],newth))
            threatened[t]=list(set(threatened[t]))
            threatened[t]=np.setdiff1d(threatened[t],burned)
            defended=np.concatenate((defended,descendant3(edges,ffnum,burned,threatened[t])))
            threatened[t]=np.setdiff1d(threatened[t],defended)
            untouched = np.setdiff1d(untouched,threatened[t])
            untouched = np.setdiff1d(untouched,defended)
    
        #print('length threatened')
        #print(len(threatened[t]))
    
        ###plotting the results####
        #values=assign_values(burned, threatened[t],defended,n)
        #plot_graph(G,pos,values)
    
        if len(threatened[t])==0:
            break   
        
    saved=n-len(burned)
    #print(n-len(burned))
    return averagedegree,t,saved  