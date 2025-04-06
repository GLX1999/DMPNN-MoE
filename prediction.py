import pandas as pd
import numpy as np
from features import solute_descriptor_functions, solvent_descriptor_functions, calculate_descriptors
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from data import MoleculeDataset, collate_fn
from dmpnn import dmpnn

class InferenceModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, batched_solute_graph, batched_solvent_graph, extra_features):
        return self.model(batched_solute_graph, batched_solvent_graph, extra_features)
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batched_solute_graph, batched_solvent_graph, extra_features = batch
        outputs = self(batched_solute_graph, batched_solvent_graph, extra_features)
        outputs = outputs.squeeze(-1)
        return outputs
def load_model(checkpoint_path, model_cls):
    trained_model = InferenceModule.load_from_checkpoint(checkpoint_path, model=model_cls)
    return trained_model
def main():
    input_csv = r'.csv'
    save_path = r'.csv'
    checkpoint_path = r'.ckpt'
    model = dmpnn(node_feat_dim=111,
                  edge_feat_dim=13,
                  edge_output_dim=400,
                  node_output_dim=400,
                  extra_dim=30,
                  num_rounds1=4,
                  num_rounds2=4,
                  dropout_rate=0.05,
                  activation_type1="leakyrelu",
                  activation_type2="leakyrelu",
                  num_experts=4,
                  moe_hid_dim=400,
                  )
    pre_df = pd.read_csv(input_csv)
    solute_descriptor_series = pre_df['solute_smiles'].apply(
        lambda x: calculate_descriptors(x, solute_descriptor_functions))
    solvent_descriptor_series = pre_df['solvent_smiles'].apply(
        lambda x: calculate_descriptors(x, solvent_descriptor_functions))
    df_solute_descriptors = pd.DataFrame(solute_descriptor_series.tolist()).add_prefix('solute_')
    df_solvent_descriptors = pd.DataFrame(solvent_descriptor_series.tolist()).add_prefix('solvent_')

    extra_features_df = pd.concat([pre_df[['temperature']], df_solute_descriptors, df_solvent_descriptors], axis=1)    # 1  + 15  + 14

    extra_features = extra_features_df.values.astype(np.float32)
    solute_smiles_list = pre_df['solute_smiles'].tolist()
    solvent_smiles_list = pre_df['solvent_smiles'].tolist()
    trained_model = load_model(checkpoint_path, model)
    predict_dataset = MoleculeDataset(solute_smiles_list, solvent_smiles_list, extra_features)
    predict_dataloader = DataLoader(predict_dataset, batch_size=100, collate_fn=collate_fn,
                                    num_workers=11, persistent_workers=True)
    trainer = Trainer(num_sanity_val_steps=0, accelerator='gpu', devices=1)
    predictions = trainer.predict(trained_model, dataloaders=predict_dataloader)
    prediction_list = [item.squeeze().tolist() for batch in predictions for item in batch]
    pre_df['pre_logS'] = prediction_list
    pre_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
