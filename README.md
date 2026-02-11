# Self-*Soup*ervision: Making and seasoning model soups with self-supervised learning!

arXiv paper: https://arxiv.org/abs/2602.02890 

mini-VTAB: https://huggingface.co/datasets/antofuller/mini-VTAB

mini-VTAB-C: https://huggingface.co/datasets/antofuller/mini-VTAB-corruptions


## Usage

We use MAE's encoder (for the stock) and decoder (when doing MAE inter-training). So you must download these pre-trained weights (e.g. from: https://github.com/facebookresearch/mae/issues/8) and store them in seperate files called `mae_encoder.pt` and `mae_decoder.pt` inside your path: `/YOUR_PATH/saved_models`

Make ingredients via inter-training on mini-VTAB training data (without labels), then self-season (i.e. learn a mixture of ingredients *without* labels) for a classifier.
```bash
python inter_train.py --models_dir /YOUR_PATH/saved_models --exp_name configs/inter_train_vtab_for_seasoning --gpu_id 0
python self_seasoning.py --models_dir /YOUR_PATH/saved_models --save_dir /YOUR_PATH/outputs  --gpu_id 0
```

Make ingredients via inter-training on mini-VTAB training data (without labels), then fine-tune and mix.

```bash
python inter_train.py --models_dir /YOUR_PATH/saved_models --exp_name configs/inter_train_vtab_for_ft --gpu_id 0
python fine_tune.py --models_dir /YOUR_PATH/saved_models --gpu_id 0
python soup_and_eval.py --models_dir /YOUR_PATH/saved_models --save_dir /YOUR_PATH/outputs --gpu_id 0
```

Note: In our paper, we inter-train on all training samples from VTAB (not just the 1K training samples from mini-VTAB). We do everything else (i.e. season, fine-tune, and test) on mini-VTAB. Since the inter-training in this repo uses *mini*-VTAB, running these commands won't exactly reproduce our results, but it will be close.