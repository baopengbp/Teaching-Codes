#!/usr/bin/env python
# Teaching-FCI
# Author: Peng Bao <baopeng@iccas.ac.cn>

import numpy
from pyscf import gto, scf, fci
import itertools

############### FCI in PySCF to compare ##############
mol = gto.M(atom='H 0 0 0; Be 0 0 1.1', basis='sto-3g', spin=1)
mf = scf.ROHF(mol)
mf.kernel()

cisolver = fci.FCI(mol, mf.mo_coeff)
print('E(FCI) = %.12f' % (cisolver.kernel()[0]))
######################################################

#################### Teaching-FCI ####################
def select(psi_i, psi_j):
  mask = numpy.isin(psi_i, psi_j, invert=True)
  x0a = numpy.argwhere(mask == True)
  aa0 = numpy.array(psi_i)
  nina = aa0[mask]
  mask = numpy.isin(psi_j, psi_i, invert=True)
  x0b = numpy.argwhere(mask == True)
  aa0 = numpy.array(psi_j)
  ninb = aa0[mask]
  sign = (-1)**((x0a + x0b) % 2)
  return nina, ninb, sign

mo = mf.mo_coeff
nao = mo.shape[0]
noccb = (mol.nelectron - mol.spin) // 2
nocca = noccb + mol.spin

hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
eri2e = mol.intor('int2e')

deta = list(itertools.combinations(range(nao), nocca))
detb = list(itertools.combinations(range(nao), noccb))
ndeta = len(deta)
ndetb = len(detb)
ndet = ndeta * ndetb
psi = []
for i in range(ndeta): 
  for j in range(ndetb):
    psi.append([deta[i], detb[j]])

H = numpy.zeros((ndet,ndet))
for i in range(ndet): 
  for j in range(i+1):
    orb_diffa = set(psi[i][0]).difference(set(psi[j][0]))
    ndiffa = len(orb_diffa)
    orb_diffb = set(psi[i][1]).difference(set(psi[j][1]))
    ndiffb = len(orb_diffb)
    ndiff = ndiffa + ndiffb
    if ndiff == 2:
      if ndiffa == 1:
        nina, ninb, signa = select(psi[i][0], psi[j][0])
        nina2, ninb2, signb = select(psi[i][1], psi[j][1])
        H[i,j] = numpy.einsum('ijkl,i,j,k,l->', eri2e, mo[:,nina[0]], mo[:,ninb[0]], mo[:,nina2[0]], mo[:,ninb2[0]])
        H[i,j] *= signa * signb
      else:
        if ndiffa == 2: 
          nina, ninb, sign = select(psi[i][0], psi[j][0])
        if ndiffb == 2:   
          nina, ninb, sign = select(psi[i][1], psi[j][1])
        Kij = numpy.einsum('ijkl,i,l,k,j->', eri2e, mo[:,nina[0]], mo[:,ninb[0]], mo[:,nina[1]], mo[:,ninb[1]])
        H[i,j] = numpy.einsum('ijkl,i,j,k,l->', eri2e, mo[:,nina[0]], mo[:,ninb[0]], mo[:,nina[1]], mo[:,ninb[1]]) - Kij
        H[i,j] *= sign[0]*sign[1]
    else:
      orba = psi[i][0]
      orbb = psi[i][1]
      dma = mo[:,orba]@mo[:,orba].T
      dmb = mo[:,orbb]@mo[:,orbb].T
      dmt = dma + dmb
      if ndiff == 1:
        if ndiffa == 1:
          nina, ninb, sign = select(psi[i][0], psi[j][0])  
          H[i,j] = numpy.einsum('ijkl,i,j,kl->', eri2e, mo[:,nina[0]], mo[:,ninb[0]], dmt) - numpy.einsum('ijkl,i,l,jk->', eri2e, mo[:,nina[0]], mo[:,ninb[0]], dma)
        else:
          nina, ninb, sign = select(psi[i][1], psi[j][1])   
          H[i,j] = numpy.einsum('ijkl,i,j,kl->', eri2e, mo[:,nina[0]], mo[:,ninb[0]], dmt) - numpy.einsum('ijkl,i,l,jk->', eri2e, mo[:,nina[0]], mo[:,ninb[0]], dmb)
        H[i,j] += numpy.einsum('ij,i,j->', hcore, mo[:,nina[0]], mo[:,ninb[0]])
        H[i,j] *= sign
      elif ndiff == 0:
        H[i,j] = numpy.einsum('ij,ji->', hcore, dmt) + 0.5*(numpy.einsum('ijkl,ji, lk->', eri2e, dmt, dmt) - numpy.einsum('ijkl,ik,jl->', eri2e, dma, dma) - numpy.einsum('ijkl,ik,jl->', eri2e, dmb, dmb))
    H[j,i] = H[i,j]
ee, ev = numpy.linalg.eigh(H)
ee += mol.energy_nuc()
print('E(FCI,20) = ', ee[:20])

