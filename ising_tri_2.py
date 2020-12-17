# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:24:54 2020

@author: 4sv
"""


import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


class IsingSim_2():
  
  """This class performs Ising model simulations on a 2D grid. Interaction parameters are given by a matrix at each lattice site. 
  Field dependence is not supported at this time but will be in due course. The simulator outputs configurations after equlibrium
  as well as the trajectories, if specifically requested.
  Inputs:
    - N : (integer) - Size of lattice will be N^2 : 2D triangular lattice 
    - J_mat: (numpy matrix of shape(3,5)) - entries being floats for interaction parameters. Self-interaction (middle element of matrix)=0. 
              or: (list) of size(3,5) with each element belonging to scipy distribution from which to draw J value (for bond disorder)
    - T: (float) - Reduced temperature for simulation
    - save_trajectories: (Boolean) - whether to save trajectories, or only final state. Default False.
    - eqSteps: (integer) number of Monte-Carlo steps for equlibration before simulation starts. Default 750. AKA 'burn-in'.
    - mcSteps: (integer) number of Monte-Carlo for simulation. Default 750.
  Outputs: Several outputs are available, including trajectories (if called), configurations (i.e., the 2D states) and configurations histograms.
  These can be obtained by calling methods self.configurations(), self.histograms() and self.trajectories()"""

  def __init__(self, N = 40, J_mat = None, T = 2.7, save_trajectories = False,
               eqSteps = 750, mcSteps = 750, prop = 0.5):
    self.N = N
    self.prop = prop
    #If no J matrix is provided we default to isotropic J interaction with NN with value 0.5
    if (J_mat).all == None:
      J_mat = np.zeros((3,5))
      J_mat[0,1] = J_mat[0,3] = J_mat[1,0] = J_mat[1,4] = J_mat[2,1] = J_mat[2,3] = 0.5 #Defaulting to 0.5 for NN, all others zeroed out.
    
    self.J_mat = J_mat

    
    try:
      rv = J_mat[1,2]
    except TypeError:
      rv = J_mat[1][2] #in case we have a list
      
    if 'dist' in str(type(rv)): #in this case we have random bond disorder situation
      self.bond_disorder = True
      self.J_lattice = self.make_J_lattice()
      self.J_mat = None
    else:
      self.bond_disorder = False
      self.J_lattice = None
    #till this
    self.save_trajectories = save_trajectories
    self.eqSteps = eqSteps
    self.mcSteps = mcSteps
    self.config = self.initialState()
    self.T = T
    self.configs_list = self.make_configs_list()

  def initialState(self):   
    ''' Generates a lattice with spin configurations drawn randomly [-1 or 1] if random=True
    Else, the lattice is generated with all sites = 1 
    Can do better, remove probability and see how you can include numbers'''

    state = np.random.choice([1, -1], size = ([self.N,2*self.N]), p=[self.prop, 1-self.prop])
    
    for i in range(0,self.N,2):
      for j in range(0,2*self.N,2):
        state[i,j] = 0

    for i in range(1,self.N,2):
      for j in range(1,2*self.N,2):
        state[i,j] = 0

    
    return state

  def EmptySite(self, row_1, col_1, spin_1, config):
    #You can do better, this takes a lot of time because of intermittent zeros
    row_2, col_2 = row_1, col_1
    spin_2 = spin_1
    while (spin_2*spin_1 != -1.0):
        row_2, col_2     = np.random.randint(0, self.N), np.random.randint(0, 2*self.N)
        spin_2 = config[row_2, col_2]
    
    return row_2, col_2

  def make_J_lattice(self):   

    ''' Return a matrix size (N,N,5,5) signifying interaction parameters at each lattice site, drawn from 
    distributions provided '''

    J_lattice = np.zeros(shape=(self.N, self.N, 5,5))
    for i in range(self.N):
      for j in range(2*self.N):
        for k in range(5):
          for l in range(5):
            if 'dist' in str(type(self.J_mat[k][l])):
              J_lattice[i,j,k,l] = self.J_mat[k][l].rvs(1)
            else: J_lattice[i,j,k,l] = self.J_mat[k][l]
    
    return J_lattice

  def mcmove(self):
    '''Monte Carlo move using Metropolis algorithm '''
    M = 3 #check whether you can obtain this from J_mat
    T = 5
    beta = 1.0/self.T
    config = self.config
    config_1 = np.where(config == -1, 0.0, config)
    
    for i in range(self.N):
      for j in range(2*self.N):
        row_1, col_1 = np.random.randint(0, self.N), np.random.randint(0, 2*self.N)
        spin_1 = config[row_1, col_1]

        if(spin_1):
          if self.bond_disorder: J_mat = self.J_lattice[i,j,:,:]
          else: J_mat = self.J_mat

          row_2, col_2 = np.random.randint(0, self.N), np.random.randint(0, 2*self.N)
          spin_2   = config[row_2, col_2]

          if (spin_2*spin_1 != -1.0):
            (row_2, col_2) = self.EmptySite(row_1, col_1, spin_1, config)
            spin_2 = config[row_2, col_2]
          
          ini_ene_1, ini_ene_2, fin_ene_1, fin_ene_2 = 0.0,0.0,0.0,0.0
            
          if spin_1 == 1.0:
              for p in range(-int(M/2), int(M/2)+1, 1):
                for q in range(-int(T/2), int(T/2)+1, 1):
                  ini_ene_1 += -J_mat[int(M/2)+p, int(T/2)+q] * config[(row_1+p)%self.N,(col_1+q)%(2*self.N)] * spin_1
                  ini_ene_2 += -J_mat[int(M/2)+p, int(T/2)+q] * config_1[(row_2+p)%self.N,(col_2+q)%(2*self.N)] * spin_2
                  fin_ene_1 += -J_mat[int(M/2)+p, int(T/2)+q] * config_1[(row_1+p)%self.N,(col_1+q)%(2*self.N)] * spin_2
                  fin_ene_2 += -J_mat[int(M/2)+p, int(T/2)+q] * config[(row_2+p)%self.N,(col_2+q)%(2*self.N)] * spin_1
                  
          if spin_1 == -1.0:
              for p in range(-int(M/2), int(M/2)+1, 1):
                for q in range(-int(T/2), int(T/2)+1, 1):
                  ini_ene_1 += -J_mat[int(M/2)+p, int(T/2)+q] * config_1[(row_1+p)%self.N,(col_1+q)%(2*self.N)] * spin_1
                  ini_ene_2 += -J_mat[int(M/2)+p, int(T/2)+q] * config[(row_2+p)%self.N,(col_2+q)%(2*self.N)] * spin_2
                  fin_ene_1 += -J_mat[int(M/2)+p, int(T/2)+q] * config[(row_1+p)%self.N,(col_1+q)%(2*self.N)] * spin_2
                  fin_ene_2 += -J_mat[int(M/2)+p, int(T/2)+q] * config_1[(row_2+p)%self.N,(col_2+q)%(2*self.N)] * spin_1
          
          cost = (fin_ene_1 + fin_ene_2) - (ini_ene_1 + ini_ene_2)
          if cost < 0 or (rand() < np.exp(-cost*beta)):
            spin_1 *= -1
            spin_2 *= -1
          config[row_1, col_1], config[row_2, col_2] = spin_1, spin_2

      self.config = config
    return None


  def calcEnergy(self):
    '''Returns the energy of the current configuration'''
    config = self.config
    config_1 = np.where(config == -1, 0.0, config)
    M = 3
    T = 5
    energy = 0.0
    for i in range(self.N):
      for j in range(2*self.N):
        #In case we have bond disorder
        if self.bond_disorder: J_mat = self.J_lattice[i,j,:,:]
        else: J_mat = self.J_mat #otherwise, no

        s = config[i,j]
        if (s):
            if s == 1.0:
                for p in range(-int(M/2), int(M/2)+1, 1):
                    for q in range(-int(T/2), int(T/2)+1, 1):
                        energy += -J_mat[int(M/2)+p, int(T/2)+q] * config[(i+p)%self.N,(j+q)%(2*self.N)] * config[i,j]
                        
            if s == -1.0:
                for p in range(-int(M/2), int(M/2)+1, 1):
                    for q in range(-int(T/2), int(T/2)+1, 1):
                        energy += -J_mat[int(M/2)+p, int(T/2)+q] * config_1[(i+p)%self.N,(j+q)%(2*self.N)] * config[i,j]
    return energy/2.0


  def calcMag(self):
    '''Magnetization of a given configuration'''
    return float(np.nansum(self.config))

  def calcAbsMag(self):
    ''' Absolute Magnetization of a given configuration'''
    return float(np.abs(np.nansum(self.config)))
  
  def performIsingSim(self):
    
    E1, M1, E2, M2 = 0.0,0.0,0.0,0.0    #These are all the average properties of all MC steps used
    Ene = np.zeros(self.mcSteps)
    Prob = np.zeros([self.mcSteps,128])
    if self.save_trajectories: config_mat = np.zeros([self.mcSteps,self.N,2*self.N])   #Saving all the configurations
    

    print('\n---Performing Equlibration---\n')
    for i in tqdm(range(self.eqSteps)):
        self.mcmove()

    print('\n---Finished...\n----Performing MC Moves----\n')
    for j in tqdm(range(self.mcSteps)):
        self.mcmove()
        Ene[j] = self.calcEnergy()
        Mag = self.calcAbsMag()
        _, Prob[j] = self.get_config_histogram()
        
        E1 = E1 + Ene[j]
        M1 = M1 + Mag
        M2 = M2 + Mag * Mag
        E2 = E2 + Ene[j] * Ene[j]
        
        if self.save_trajectories: config_mat[j] = self.config
    
    print('Completed. Saving')    
    Energy = E1 / (self.mcSteps * self.N * self.N)
    Magnetization = M1 / (self.mcSteps * self.N * self.N)
    n1, n2  = 1.0/(self.mcSteps*self.N*self.N), 1.0/(self.mcSteps*self.mcSteps*self.N*self.N) 
    iT = 1.0/self.T
    iT2 = iT*iT
    SpecificHeat = (n1*E2 - n2*E1*E1)*iT2
    Susceptibility = (n1*M2 - n2*M1*M1)*iT
    hist = np.mean(Prob, axis = 0)
    # SpecificHeat = (E2 / self.mcSteps - E1 * E1 / (self.mcSteps * self.mcSteps)) / (self.N * self.T * self.T)
    # Susceptibility = (M2 / self.mcSteps - M1 * M1 / (self.mcSteps * self.mcSteps)) / (self.N * self.T)

    
              

    if self.save_trajectories:
      results_dict = {'config': config_mat, 'Energy': Energy, 'Magnetization': Magnetization,
      'SpecificHeat': SpecificHeat, 'Susceptibility': Susceptibility, 
      'Histogram': hist,'Ene_traj': Ene, 'Hist_traj': Prob}
    else:
      results_dict = {'Energy': Energy, 'Magnetization': Magnetization,
      'SpecificHeat': SpecificHeat, 'Susceptibility': Susceptibility,
      'Histogram': hist}
    
    self.results = results_dict

    return 'Completed simulation'
    
  def make_configs_list(self):
    all_configs = []
    for a in range(-1,2,2):
      for b in range(-1,2,2):
        for c in range(-1,2,2):
          for d in range(-1,2,2):
            for e in range(-1,2,2):
              for f in range(-1,2,2):
                for g in range(-1,2,2):
                  all_configs.append([a,b,c,d,e,f,g])

    return all_configs


  def get_config_histogram(self):
    config = self.config
    config_hist = np.zeros(shape=(len(self.configs_list)))

    for i in range(self.N):
        for j in range(2*self.N):
            cen = config[i,j]
            if (cen):
              a = config[(i-1)%self.N,(j-1)%(2*self.N)]
              b = config[(i-1)%self.N,(j+1)%(2*self.N)]
              c = config[i,(j+2)%(2*self.N)]
              d = config[(i+1)%self.N,(j+1)%(2*self.N)]
              e = config[(i+1)%self.N,(j-1)%(2*self.N)]
              f = config[i,(j-2)%(2*self.N)]
              
              
              config_vec = [cen, a, b, c, d, e, f]
              #Now let's get the configuration number
              config_number = self.configs_list.index(config_vec)
              config_hist[int(config_number)]=config_hist[int(config_number)]+1

    #normalize it to get probabilities
    config_hist_norm = config_hist / (self.N * self.N)

    return config_hist, config_hist_norm

  def plot_config(self, config, figsize = (7,7)):

    config_nozeros = np.zeros([self.N,self.N])
    k = 0
    for i in range(0,self.N):
      if i%2 == 0:
        for j in range(1,2*self.N,2):
            config_nozeros[np.unravel_index(k, config_nozeros.shape)] = config[i,j]
            k+=1
      else:
        for j in range(0,2*self.N,2):
            config_nozeros[np.unravel_index(k, config_nozeros.shape)] = config[i,j]
            k+=1
    
    x = np.linspace(0, 1, self.N)
    y = np.linspace(1, 0, self.N)
    X, Y = np.meshgrid(x, y)

    dx = np.diff(x)[0]
    dy = np.diff(y)[0]
    ds = np.sqrt(dx**2 +  dy**2)

    # example_data = np.random.choice([-1,1], size = (nx,ny))
    cmap = colors.ListedColormap(['blue', 'red'])
    bounds = [-1,0,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    patches = []
    for i in x:
        for n, j in enumerate(y):
            if n%2:
                polygon = mpatches.RegularPolygon([i-dx/2., j], 6, 0.6*dx)
            else:
                polygon = mpatches.RegularPolygon([i, j], numVertices = 6, radius = 0.6*dx)
            patches.append(polygon)

    collection = PatchCollection(patches, cmap=cmap, norm=norm, alpha=1.0)

    fig, ax = plt.subplots(1,1, figsize = figsize)
    ax.add_collection(collection)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    collection.set_array(np.transpose(config_nozeros).ravel())