# mixedXLM

The Original Code in [XLM] (https://github.com/facebookresearch/XLM)
In this work, we train our model use code-mixing corpus
Currently, only available in Malaya, English, Chinese.

## Dependence
- Python3
- NumPy
- PyTorch
- Some Tools you can follow this [link] (https://github.com/facebookresearch/XLM/tree/master/tools)

## Code-Mixing Language Model Pretraining
### 1. Perparing the data
First prepare your code-mixing corpus and learn BPE vocabulary
2nd, applay BPE token to your data files:
```
$FASTBPE applybpe $OUTPATH/train.mix $ORIGINAL_DATA/mix.train $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/test.mix $ORIGINAL_DATA/mix.test $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/valid.mix $ORIGINAL_DATA/mix.valid $OUTPATH/codes &
```
3rd, get the post-BPE vocabulary:
```
cat $OUTPATH/train.mix | $FASTBPE getvocab - > $OUTPATH/vocab &
```

4th, Binarize the data to limit the size of the data we load in memory:
```
python preprocess.py $OUTPATH/vocab $OUTPATH/train.mix &
python preprocess.py $OUTPATH/vocab $OUTPATH/test.mix &
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.mix &
```
### 2. Training The Language Model
for example:
```
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py  # Train in multiple GPU
--exp_name large_mix 				# experiment name
--dump_path ./new_dumped 			# where to store the experiment
--data_path $OUTPATH 				# data location
--lgs 'mix' 
--clm_steps '' 
--mlm_steps 'mix' 
--emb_dim 512 					# embeddings / model dimension
--n_layers 6                             	# number of layers
--n_heads 8 
--dropout 0.1 
--attention_dropout 0.1 
--gelu_activation true 
--batch_size 8 					# sequences per batch
--bptt 16 					# sequences length  (streams of 256 tokens)
--optimizer adam_inverse_sqrt,lr=0.00010,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001 
--epoch_size 50000 				# number of sentences per epoch
--max_epoch 600 				# max number of epochs (~infinite here)
--validation_metrics _valid_mix_mlm_ppl 
--stopping_criterion _valid_mix_mlm_ppl,25 
--fp16 true --word_mask_keep_rand '0.8,0.1,0.1' 
--word_pred '0.15' 
--amp 1 
--exp_id 1
```

### 3. Fine-tune in Sentiment Analysis Task
Prepare your dataset use same codes and same vocab file, like in section 1
do this command to fine-tune
```
python do_sentifit.py 
--exp_name FT_MixedXLM_SA 		# experiment name 
--dump_path ./new_dumped 		# where to store the experiment
--model_path best-valid_mlm_ppl.pth 	# model location
--data_path $OUTPATH 			# data location
--n_epochs 50 
--epoch_size 20 
--exp_id 001
```


## Reference
Please cite [1] if you found the resources in this repository useful.

### Cross-lingual Language Model Pretraining
[1] G. Lample *, A. Conneau * Cross-lingual Language Model Pretraining

* Equal contribution. Order has been determined with a coin flip.

```
@article{lample2019cross,
  title={Cross-lingual Language Model Pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```



