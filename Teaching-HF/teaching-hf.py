#!/usr/bin/env python
# Teaching-HF
# Author: Peng Bao <baopeng@iccas.ac.cn>

''' Need PySCF first,
MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 python teaching-hf.py'''

import numpy
import scipy.linalg
from pyscf import gto, scf

############### HF in PySCF to compare ##############
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
mf = scf.RHF(mol)
mf.kernel()
print('Reference HF total energy =', mf.e_tot)
#####################################################

#################### Teaching-HF ####################
# RHF. Only need structure information of molecule and electronic integrals 
  
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')

hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

s1e = mol.intor_symmetric('int1e_ovlp')

nao = hcore.shape[0]

eri = mol.intor('int2e').reshape(nao,nao,nao,nao)

nocc = mol.nelectron // 2

def energy_nuc(mol):

    charges = mol.atom_charges()

    coords = mol.atom_coords()

    rr = numpy.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)

    rr[numpy.diag_indices_from(rr)] = 1e200

    e = numpy.einsum('i,ij,j->', charges, 1./rr, charges) * .5

    return e

def get_dm(fock):

    mo_energy, mo_coeff = scipy.linalg.eigh(fock, s1e)

    e_idx = numpy.argsort(mo_energy)

    dm = numpy.dot(mo_coeff[:,e_idx[:nocc]],mo_coeff[:,e_idx[:nocc]].T)

    return dm

#dm = mf.get_init_guess(mol, mf.init_guess)

dm = get_dm(hcore)

scf_conv = False

cycle = 0

e_tot = 0

#mf_diis = mf.DIIS(mf, mf.diis_file)

while not scf_conv and cycle < 50:

    dm_last = dm

    last_hf_e = e_tot

    fock = hcore + numpy.einsum('ijkl,ji->kl', eri, dm) * 2 - numpy.einsum('ijkl,jk->il', eri, dm)
    
    #fock = mf_diis.update(s1e, dm, fock, mf, hcore)    

    dm = get_dm(fock)

    e_tot = numpy.einsum('ij,ji->', hcore + fock, dm)  + energy_nuc(mol)

    norm_ddm = numpy.linalg.norm(dm-dm_last)

    if abs(e_tot-last_hf_e) < 1.0E-8 and norm_ddm < 1.0E-6:

        scf_conv = True

    cycle += 1

print('Teaching-HF total energy =', e_tot, 'Cycle number=', cycle)
