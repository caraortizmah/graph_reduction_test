from __future__ import print_function
import io
import sys

import numpy as np
import pandas as pd

from scipy.sparse.linalg import eigsh
from math import sqrt

from ase import Atoms,Atom
from ase.atoms import symbols2numbers
from ase.neighborlist import NeighborList
from ase.io import write,read


def sprint(xyz_file):
    """
    Function that calculates the sprint coordinates matrix for a nanoparticle.

    First calculate the coordination, then build the adjacency
    matrix. To calculate the coordination firstly generates a 
    nearest_neighbour cutoffs for NeighborList.

    The C list contains the atom and the binding atoms indices.
    From C list we build the adjMatrix. The idea is translate from
    coordination list to adjacency matrix.

    Then, calculate the sprint coordinates
    Args:
        nano(file): xyz file
    Return:
        sFormated([float]): SPRINT coordinates
    """
    atoms=read(xyz_file,format='xyz')
    atoms.center(vacuum=20)
    adjacencyName=xyz_file+'dat'

    nearest_neighbour=[]
    C=[]

    for i in range(len(atoms.get_atomic_numbers())):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))

    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)

    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append([i,indices])

    m=len(C)
    adjMatrix=np.zeros((m,m))
    for i in C:
        for j in i[1]:
            adjMatrix[i[0],j]=1.0
    # np.savetxt('adjMatrix',adjMatrix,newline='\n',fmt='%.1f')

    # Diagonal elements defined by 1+zi/10 if i is non metal
    # and 1+zi/100 if is metal

    numbers=symbols2numbers(atoms.get_atomic_numbers())
    # print(numbers)
    for i in range(len(adjMatrix)):
        if numbers[i] <=99 :
            adjMatrix[i][i]=1+float(numbers[i])/10
        else:
            adjMatrix[i][i]=1+float(numbers[i])/100

    # np.savetxt(adjacencyName,adjMatrix,newline='\n',fmt='%.3f')

    # Calculating the largest algebraic eigenvalues and their 
    # correspondent eigenvector
    val,vec=eigsh(adjMatrix,k=1,which='LA')

    # Sorting and using positive eigenvector values (by definition)
    # to calculate the sprint coordinates

    vecAbs=[abs(i)for i in vec]
    vecAbsSort=sorted(vecAbs)
    s=[sqrt(len(adjMatrix))*val[0]*i[0] for i in vecAbsSort]
    sFormated=['{:.3f}'.format(i) for i in s]
    
    return sFormated

def compare(sprint0,sprint1):

    """
    compare the SPRINT coordinates between two nanoparticles.
    If two NP has the same sprint coordinates, both are equally
    connected.
    Args:
        sprint0(list): sprint coordinates list
        sprint1(list): sprint coordinates list
    Return:
        (Bool)
    """
    # diff=(list(set(sprint0) - set(sprint1)))
    if len(sprint0)==len(sprint1):
        diff=(list(set(sprint0) - set(sprint1)))
        if len(diff)==0:
            return True

if __name__ == "__main__":
    xyz = sys.argv[1]
    
    f = open('test2', 'w', encoding="utf-8")
    f.write(f'{xyz}, {sprint(xyz)}\n')
    f.close()
