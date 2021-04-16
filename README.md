# Code Repository for Self-Enhanced GNN
This is the code repository for *Self-Enhanced GNN: Improving Graph Neural Networks Using Model Outputs* (IJCNN 2021).

## Dependencies

```bash
CUDA==10.2.89
python==3.6.9
torch==1.5.0
torch_geometric==1.4.3
```

## prepare

Run `split.py` to create random train/val/test splits for each dataset.

## topology update

Run `topology_update.py` to create updated edges and save them.

Then run `topology_update_train.py` to train the models and save the results.

Use `stats.py --mode val_test_result` to analyze the results.

## training node augmentation

Run `training_node_augmentation.py` to train with TNA algorithm. Remember to save the outputs to a text file.

Use `tna_parser.py` to analyze the above generated outputs.

Run `tna_ensemble.py` and `tna_distilled.py` to get the ensemble and distillation results, respectively.

## combined

Run `combined_models.py` to run the combined version. The topology update result filename and the training node augmentation outputs template needs to be specified to get the corresponding threshold when running this script.

## Bibtex

```bibtex
@inproceedings{yang2021selfenhanced,
    title={Self-Enhanced GNN: Improving Graph Neural Networks Using Model Outputs}, 
    author={Han Yang and Xiao Yan and Xinyan Dai and Yongqiang Chen and James Cheng},
    year={2021},
    booktitle={International Joint Conference on Neural Networks}
}
```

