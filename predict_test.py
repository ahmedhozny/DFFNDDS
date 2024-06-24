import json

from predict_endpoint import smiles_encode, predict
import numpy as np

drug_1 = smiles_encode("C1C(C(OC1N2C=NC3=C2NC=NCC3O)CO)O")
drug_2 = smiles_encode("CCC1(CCC(=O)NC1=O)C2=CC=C(C=C2)N")

features: dict = json.loads(open("./drugcombdb/context_set_m.json", 'r').read())

cell_lines = features

predicted = predict(drug_1["drug_smiles"], drug_2["drug_smiles"], drug_1["drug_fp"], drug_2["drug_fp"])
frobenius_norm = np.linalg.norm(predicted.cpu().detach().numpy(), "fro")
print(frobenius_norm)