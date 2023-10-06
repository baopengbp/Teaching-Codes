#!/usr/bin/env python
# Teaching-MP2
# Author: Peng Bao <baopeng@iccas.ac.cn>

import numpy
from pyscf import gto, scf, mp

############### MP2 in PySCF to compare   #############
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')

mf = scf.RHF(mol)
mf.kernel()

mp2 = mp.RMP2(mf)
mp2.kernel()
######################################################

#################### Teaching-MP2 ####################
mo=mf.mo_coeff
orben=mf.mo_energy

nocc = mol.nelectron // 2
nvirt = mo.shape[0] - nocc

eri2e = mol.intor('int2e')
eri2em = numpy.einsum('ui,va,xj,yb,uvxy->iajb', mo[:,:nocc], mo[:,nocc:], mo[:,:nocc], mo[:,nocc:], eri2e)

Jintm = 0.0
Kintm = 0.0
for i in range(nocc):
    for a in range(nvirt):
        for j in range(nocc):
            for b in range(nvirt):
                Jintm += eri2em[i,a,j,b]*eri2em[i,a,j,b]/(orben[i]+orben[j]-orben[a+nocc]-orben[b+nocc])
                Kintm -= eri2em[i,a,j,b]*eri2em[j,a,i,b]/(orben[i]+orben[j]-orben[a+nocc]-orben[b+nocc])
E_corr = 2*Jintm + Kintm
print('=========================================================\n', 'Simple MP2 f ', 'E_corr = ', E_corr)

e_i = 1 / (orben[:nocc].reshape(-1, 1, 1, 1) - orben[nocc:].reshape(1,-1, 1, 1) + orben[:nocc].reshape(1, 1, -1, 1) - orben[nocc:].reshape(1, 1, 1, -1))
Jintm = numpy.einsum('iajb,iajb,iajb->', eri2em, eri2em, e_i)
Kintm = -numpy.einsum('iajb,jaib,iajb->', eri2em, eri2em, e_i)
E_corr = 2*Jintm + Kintm
print('=========================================================\n', 'Simple MP2 v ', 'E_corr = ', E_corr)
