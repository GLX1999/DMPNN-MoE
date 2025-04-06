import torch
import dgl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from features import atom_features, bond_features
def smiles_to_graph(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.ComputeGasteigerCharges(mol)
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    num_atoms = mol.GetNumAtoms()
    src, dst, edge_feats = [], [], []
    atom_feats = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        base_features = atom_features(atom)
        gasteiger_charge = float(atom.GetProp("_GasteigerCharge"))
        crippen_contrib = crippen_contribs[i][0]
        extra_features = torch.tensor([gasteiger_charge, crippen_contrib], dtype=torch.float)
        full_features = torch.cat((base_features, extra_features))
        atom_feats.append(full_features)
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        src.extend([u, v])
        dst.extend([v, u])
        edge_feats.extend([feat, feat])
    g = dgl.graph((src, dst), num_nodes=num_atoms)
    g.ndata['feat'] = torch.stack(atom_feats)
    g.edata['feat'] = torch.stack(edge_feats)
    return g
def smiles_to_graphs(solute_smiles, solvent_smiles):
    solute_graph = smiles_to_graph(solute_smiles)
    solvent_graph = smiles_to_graph(solvent_smiles)
    return solute_graph, solvent_graph
class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, solute_smiles_list, solvent_smiles_list, extra_features):
        self.solute_smiles_list = solute_smiles_list
        self.solvent_smiles_list = solvent_smiles_list
        self.extra_features = extra_features
    def __getitem__(self, idx):
        solute_smiles = self.solute_smiles_list[idx]
        solvent_smiles = self.solvent_smiles_list[idx]
        extra_features = torch.tensor(self.extra_features[idx], dtype=torch.float32)
        solute_graph, solvent_graph = smiles_to_graphs(solute_smiles, solvent_smiles)
        return solute_graph, solvent_graph, extra_features
    def __len__(self):
        return len(self.solute_smiles_list)
def collate_fn(samples):
    solute_graphs, solvent_graphs, extra_features = zip(*samples)
    batched_solute_graph = dgl.batch(solute_graphs)
    batched_solvent_graph = dgl.batch(solvent_graphs)
    extra_features = torch.stack(extra_features)
    return batched_solute_graph, batched_solvent_graph, extra_features
