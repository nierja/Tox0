#!/usr/bin/env python3
import argparse
import csv
import os
import numpy as np
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors

parser = argparse.ArgumentParser()
parser.add_argument("--target", default="NR-AR", type=str, help="Target toxocity type")
parser.add_argument("--fp", default="ecfp4_maccs", type=str, help="fingerprint type")

MAX_NUM_OF_ATOMS = 132  # maximal number of atoms in a molecule from Tox21 dataset after removing hydrogens
MATRIX_SIZE = MAX_NUM_OF_ATOMS ** 2
nBits = 1024

def GetMol(smiles):
    # given a SMILES string, returns corresponding RDKit molecule object
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError('SMILES cannot be converted to a RDKit molecules:', smiles)
    return m

def truncate(matrix, num_atoms, sort=1):
    # returns a vector of first k=MATRIX_SIZE
    # greatest values in matrix
    fp = np.zeros((1, MATRIX_SIZE))
    vector = np.matrix.flatten(matrix)
    if sort: vector = -np.sort(-vector)
    if num_atoms**2 > MATRIX_SIZE:
        return vector[0, 0:MATRIX_SIZE]
    else:
        fp[0, 0:num_atoms**2] = vector
    return fp

def pad(matrix, num_atoms, sort=1):
    # zero-pads matrix and returns it flattened as vector
    fp = np.zeros((1, MATRIX_SIZE))
    vector = np.matrix.flatten(matrix)
    if sort: vector = -np.sort(-vector)
    # print(MATRIX_SIZE, vector.shape, fp.shape)
    fp[0, 0:vector.shape[0]] = vector
    return fp

def main(args):
    # create a directory to store the data
    path = f"./Tox21_data/{args.target}"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    # create train & test data
    with open('./tox21.data', mode='r') as csv_file:
        with open(f'./Tox21_data/{args.target}/{args.target}_{args.fp}.data', mode='w') as file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0

            # parse ./tox21.data row by row
            for row in csv_reader:
                print(f"processing molecule {line_count+1}")
                # get the target value. If it does not exist, let's
                # not bother with the expensive calculation
                target_value = row[f"{args.target}"]
                if target_value != "0" and target_value != "1": continue

                # given a smiles string, create an RDKit molecule
                # object and optimise its 3D geometry
                mol = GetMol(row["smiles"])
                if mol is None:
                    raise ValueError('SMILES cannot be converted to a RDKit molecules:', row["smiles"])
                mol = Chem.AddHs(mol)
                try:
                    AllChem.EmbedMolecule(mol)
                    AllChem.UFFOptimizeMolecule(mol)
                    mol = Chem.RemoveHs(mol)
                except ValueError:
                    continue
            
                # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
                # for idx in [849, 582, 381, 494, 751, 555, 646, 901, 162, 41]:
                #     if fp[idx]:
                #         print(idx, row["smiles"])
                # 
                # continue


                """ =========================== Fingerprint calculation =========================== """
                # basic RDkit fingerprints
                if args.fp == 'ecfp0': fp = AllChem.GetMorganFingerprintAsBitVect(mol, 0, nBits=nBits)
                if args.fp == 'ecfp2': fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=nBits)
                if args.fp == 'ecfp4': fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
                if args.fp == 'ecfp6': fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nBits)
                if args.fp == 'fcfp2': fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, useFeatures=True, nBits=nBits)
                if args.fp == 'fcfp4': fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=nBits)
                if args.fp == 'fcfp6': fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True, nBits=nBits)
                if args.fp == 'maccs': fp = MACCSkeys.GenMACCSKeys(mol)
                if args.fp == 'hashap': fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
                if args.fp == 'hashtt': fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)
                if args.fp == 'avalon': fp = fpAvalon.GetAvalonFP(mol, nBits)
                if args.fp == 'rdk5': fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2)
                if args.fp == 'rdk6': fp = Chem.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2)
                if args.fp == 'rdk7': fp = Chem.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2)

                if args.fp == 'ecfp4_maccs': 
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
                    fp2 = MACCSkeys.GenMACCSKeys(mol)
                if args.fp == 'maccs_rdk7': 
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
                    fp2 = MACCSkeys.GenMACCSKeys(mol)
                if args.fp == 'ecfp4_rdk7': 
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
                    fp2 = MACCSkeys.GenMACCSKeys(mol)

                # adjacency & distance matrix based fingerprints
                if args.fp == 'dist_2D': fp = pad(Chem.rdmolops.GetDistanceMatrix(mol), mol.GetNumAtoms(), sort=0)
                if args.fp == 'dist_3D': fp = pad(Chem.rdmolops.Get3DDistanceMatrix(mol), mol.GetNumAtoms(), sort=0)
                if args.fp == 'balaban_2D': fp = pad(Chem.rdmolops.GetDistanceMatrix(mol, useAtomWts = True), mol.GetNumAtoms(), sort=0)
                if args.fp == 'balaban_3D': fp = pad(Chem.rdmolops.Get3DDistanceMatrix(mol, useAtomWts = True), mol.GetNumAtoms(), sort=0)
                if args.fp == 'adjac': fp = pad(Chem.rdmolops.GetAdjacencyMatrix(mol), mol.GetNumAtoms(), sort=0)
                if args.fp == 'Laplacian':
                    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
                    atomic_numbers = np.diag([atom.GetExplicitValence() for atom in mol.GetAtoms()])
                    fp = pad(atomic_numbers - adj_matrix, mol.GetNumAtoms(), sort=0)
                if args.fp == 'inv_dist_2D': 
                    inv_dist_2D = np.reciprocal(Chem.rdmolops.GetDistanceMatrix(mol)); inv_dist_2D[~np.isfinite(inv_dist_2D)] = 0
                    fp = pad(inv_dist_2D, mol.GetNumAtoms(), sort=0)
                if args.fp == 'inv_dist_3D': 
                    inv_dist_3D = np.reciprocal(Chem.rdmolops.Get3DDistanceMatrix(mol)); inv_dist_3D[~np.isfinite(inv_dist_3D)] = 0
                    fp = pad(inv_dist_3D, mol.GetNumAtoms(), sort=0)

                # fingerprints based on the Coulomb matrix
                global MATRIX_SIZE
                if args.fp in ['CMat_full', 'CMat_400', 'CMat_600', 'eigenvals']:
                    coulomb_matrix = np.matrix(rdMolDescriptors.CalcCoulombMat(mol))
                if args.fp == "CMat_full": MATRIX_SIZE = MAX_NUM_OF_ATOMS ** 2
                if args.fp == "CMat_400": MATRIX_SIZE = 400
                if args.fp == "CMat_600": MATRIX_SIZE = 600
                if args.fp in ['CMat_full', 'CMat_400', 'CMat_600']:
                    fp = truncate(coulomb_matrix, mol.GetNumAtoms())
                if args.fp == "eigenvals":
                    MATRIX_SIZE = MAX_NUM_OF_ATOMS
                    try:
                        w, v = np.linalg.eig(coulomb_matrix)
                    except np.linalg.LinAlgError:
                        continue
                    fp = pad(w, mol.GetNumAtoms())

                # fingerprint containing all available RDkit descriptors
                # has a good performance, for benchmarking
                if args.fp == "rdkit_descr":
                    descriptors_to_calculate = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
                    fp = np.array(descriptors_to_calculate.CalcDescriptors(mol))


                # appending fp and target_value to the output file
                if args.fp == 'ecfp4_maccs':
                    for value in np.nditer(fp1):
                        file.write("{:.6f}, ".format(value))
                    for value in np.nditer(fp2):
                        file.write("{:.6f}, ".format(value))
                else:
                    for value in np.nditer(fp):
                        file.write("{:.6f}, ".format(value))
                file.write(f"{target_value}\n")
                line_count += 1
            print(f'Processed {line_count} lines.')

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)