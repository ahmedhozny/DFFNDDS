import json

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sentence_transformers import SentenceTransformer

from model_h import MultiViewNet

# RDKit descriptors -->
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
# dictionary
fpFunc_dict = {}
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=1024)

model_name = 'output/simcsesqrt-model'
drug_model = SentenceTransformer(model_name, device=torch.device("cuda"))


def smiles_encode(smile_str):
    smile_encoded = drug_model.encode(smile_str)
    mol = Chem.MolFromSmiles(smile_str)
    fp = np.asarray(fpFunc_dict["hashap"](mol))
    return {"drug_smiles": smile_encoded, "drug_fp": fp}


# print(smiles_encode("CC1(C(N2C(S1)C(C2=O)NC(=O)C(=NOC3=CC=CC=C3)C)O)C"))


def predict(smile_1_vectors, smile_2_vectors, context, fp1_vectors, fp2_vectors, device):
    smile_1_vectors = torch.FloatTensor([smile_1_vectors]).to(device)
    smile_2_vectors = torch.FloatTensor([smile_2_vectors]).to(device)
    context = torch.FloatTensor([[context]]).to(device)
    fp1_vectors = torch.FloatTensor([fp1_vectors]).to(device)
    fp2_vectors = torch.FloatTensor([fp2_vectors]).to(device)

    # Load the model
    model = MultiViewNet()
    checkpoint = torch.load('mainsplit-attention-comb', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model.forward(smile_1_vectors, smile_2_vectors, context, fp1_vectors, fp2_vectors)


drug_1 = smiles_encode("CN1C2=C(C=C(C=C2)N(CCCl)CCCl)N=C1CCCC(=O)O")
drug_2 = smiles_encode("C1=CC=C2C(=C1)C(=NN2CC3=C(C=C(C=C3)Cl)Cl)C(=O)O")

features = json.loads(open("./drugcombdb/context_set_m.json", 'r').read())
cell_line = features["NCIH520"]


predicted = predict(drug_1["drug_smiles"], drug_2["drug_smiles"], cell_line, drug_1["drug_fp"], drug_2["drug_fp"], "cuda")

print(predicted.shape)
print(predicted)
