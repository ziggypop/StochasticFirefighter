#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 03:49:49 2018

@author: s1238047
"""
import basic
import numpy as np



firefighter=np.arange(1,8)

#graph_type='erdos'
ave_degree=10
n=1000
print('bfs2')
p=[0.1,0.3,0.5,0.7,0.9]
firefighters=[1,3,5,7,9]


"""
graph_type='geometric'
geo_sto_bfs=np.zeros((1,5,5,5))
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(1):
            ad, t, sa = basic.stochastic_firefighting_des2(firefighters[j],n,ave_degree,graph_type,p[i])
            geo_sto_bfs[k][i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('geo_sto_des2',geo_sto_bfs)
print('geo done')
graph_type='erdos'
erdos_sto_bfs=np.zeros((1,5,5,5))
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(1):
            ad, t, sa = basic.stochastic_firefighting_des2(firefighters[j],n,ave_degree,graph_type,p[i])
            erdos_sto_bfs[k][i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('erdos_sto_des2',erdos_sto_bfs)
print('erdos done')

graph_type='geometric'
geo_sto_bfs=np.zeros((100,5,5,5))
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_bfs2(firefighters[j],n,ave_degree,graph_type,p[i])
            geo_sto_bfs[k][i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('geo_sto_bfs2',geo_sto_bfs)
print('geo done')
graph_type='erdos'
erdos_sto_bfs=np.zeros((100,5,5,5))
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_bfs2(firefighters[j],n,ave_degree,graph_type,p[i])
            erdos_sto_bfs[k][i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('erdos_sto_bfs2',erdos_sto_bfs)
print('erdos done')




#p=[0.1,0.3,0.5,0.7,0.9]
#firefighters=[1,3,5,7,9]
graph_type='geometric'
geo_sto_bfs1=np.zeros((100,5,5,5))
print('bfs1')
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_bfs(firefighters[j],n,ave_degree,graph_type,p[i])
            geo_sto_bfs1[i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('geo_sto_bfs1',geo_sto_bfs1)
print('geo done')
graph_type='erdos'
erdos_sto_bfs1=np.zeros((100,5,5,5))
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_bfs(firefighters[j],n,ave_degree,graph_type,p[i])
            erdos_sto_bfs1[i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('erdos_sto_bfs1',erdos_sto_bfs1)
print('erdos done')




#averagedegree,t,saved=basic.stochastic_firefighting(firefighter,n,ave_degree,graph_type,pp)
#averagedegree,t,saved=basic.firefighting(firefighter,n,ave_degree,graph_type)
#p=[0.1,0.3,0.5,0.7,0.9]
#firefighters=[1,3,5,7,9]
graph_type='geometric'
geo_sto_degree=np.zeros((100,5,5,5))
print('degree')
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_degree(firefighters[j],n,ave_degree,graph_type,p[i])
            geo_sto_degree[i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('geo_sto_deg',geo_sto_degree)
print('geo done')
"""

graph_type='erdos'
erdos_sto_degree=np.zeros((100,5,5,5))
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_degree(firefighters[j],n,ave_degree,graph_type,p[i])
            erdos_sto_degree[k][i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('erdos_sto_deg',erdos_sto_degree)
print('erdos done')


#p=[0.1,0.3,0.5,0.7,0.9]
#firefighters=[1,3,5,7,9]
graph_type='geometric'
geo_sto_des=np.zeros((100,5,5,5))
print('descedant')
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_descendant(firefighters[j],n,ave_degree,graph_type,p[i])
            geo_sto_des[k][i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('geo_sto_des',geo_sto_des)
print('geo done')
graph_type='erdos'
erdos_sto_des=np.zeros((100,5,5,5))
for i in range(len(p)):
    for j in range(len(firefighters)):
        for k in range(100):
            ad, t, sa = basic.stochastic_firefighting_descendant(firefighters[j],n,ave_degree,graph_type,p[i])
            erdos_sto_des[k][i][j]=[p[i],ad,t,sa/n,firefighters[j]]
            print(i,j)
np.save('erdos_sto_des',erdos_sto_des)
print('erdos done')

"""

de=p*ave_degree

geo_sto=np.zeros((7,10,50,5))
geo=np.zeros((7,10,50,5))

erdos_sto=np.zeros((7,10,50,5))
erdos=np.zeros((7,10,50,5))

for loop in firefighter:
    graph_type='geometric'
    print('firefighter')
    print(loop)
    #de=
    for i in range(len(de)):
        avd=de[i]
        pp=p[i]
        for r in range(50):
            ad, t, sa = basic.stochastic_firefighting_degree(firefighter[loop],n,ave_degree,graph_type,pp)
            geo_sto[loop][i][r]=[pp,ad,t,sa/n,loop] # infection rate
            #geo_sto[loop][i][r][1]=ad #average degree
            #geo_sto[loop][i][r][2]=t #time to contain the fire
            #geo_sto[loop][i][r][3]=sa/n #saved nodes ratio
            #geo_sto[loop][i][r][4]=loop #number of firefighters
            ad, t, sa = basic.firefighting(firefighter[loop],n,avd,graph_type)
            geo[loop][i][r]= [avd,ad,t,sa/n,loop] #approximate average degree
            #geo[loop][i][r][1]=ad #average degree
            #geo[loop][i][r][2]=t #time to contain the fire
            #geo[loop][i][r][3]=sa/n #saved nodes ratio
            #geo[loop][i][r][4]=loop#number of firefighters
            print(r)
        print('avd')
        print(avd)
    #print('geo finish')
    
    graph_type='erdos'
    #de=
    for i in range(len(de)):
        avd=de[i]
        pp=p[i]
        for r in range(50):
            ad, t, sa = basic.stochastic_firefighting_degree(firefighter[loop],n,ave_degree,graph_type,pp)
            erdos_sto[loop][i][r]=[pp,ad,t,sa/n,loop] # infection rate
            #erdos_sto[loop][i][r][1]=ad #average degree
            #erdos_sto[loop][i][r][2]=t #time to contain the fire
            #erdos_sto[loop][i][r][3]=sa/n #saved nodes ratio
            #erdos_sto[loop][i][r][4]=loop
            ad, t, sa = basic.firefighting(firefighter[loop],n,avd,graph_type)
            erdos[loop][i][r]= [avd,ad,t,sa/n,loop] #approximate average degree
            #erdos[loop][i][r][1]=ad #average degree
            #erdos[loop][i][r][2]=t #time to contain the fire
            #erdos[loop][i][r][3]=sa/n #saved nodes ratio
            #erdos[loop][i][r][4]=loop
            print(r)
        print('avd')
        print(avd)
    
    #print('erdos finish')



np.save('stochastic_geo',geo_sto)
np.save('determinstic_geo',geo)

np.save('stochastic_erdos',erdos_sto)
np.save('determinstic_erdos',erdos)
"""
