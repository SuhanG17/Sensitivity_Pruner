# Sensitivity_Pruner
This repo is dedicated to include code for Sensitivity Pruner (SP), a kernel-level pruning method

## There are three network structure used in this project and each is pruned using 1~3 datasets:
  + ResNet-56
    - CIFAR10
    - CIFAR100
    - ILSVRC2012 (imagenet)
  + VGG-16
    - CIFAR10
    - CIFAR100
    - ILSVRC2012 (imagenet)
  + MobileNetV2
    - ILSVRC2012 (imagenet)

## Using imagenet as an example, code can be used following the comannds below:
1. Download Data:  make sure that Imagenet data is downloaded and parsed into `train` and `val` folders, each containing 1000 directories and each directory contains the images of this category accordingly.
2. Generate Data: run `get_data.sh` script to create dataloader for both training and validation set. Be careful that the `--batch_size` parameter only controls the batch size of pruning stage, 64 is the default batch size for fine-tuning stage
  ```shell
  python get_data.sh
  ```

3. Download pretrained model: Imagenet, ResNet-56, MobileNetV2, VGG-16 available at  [pytorch repo](https://pytorch.org/vision/stable/models.html)
4. Pruning, run the following command to prune
  ```shell
  python run_all.sh
  ```

5. Finetuning, run the following command to finetune the pruned model
  ```shell
  python finetune.sh
  ```

## Further notes:
1. The pruning process is implemented block by block, which means that after the pruning strategy is fully trained, the channel with 0 in indicator matrix will be removed, hence the `compressing` of network. Therefore, in pruning stage, each iteration in loop includes two parts, strategy generation and compression. If the pruning process is interrupted during the loop, we suggest to resume by compressing the last block/layer and restart pruning for current block.
  + The `_by_layer` shell scripts are used under this circumstance  
2. If beta initiation is to be modified, modify the following command in `main.py` by changing 1 to your initiation, but if range needs to be changed as well, the `--beta_range` has to be modified accordingly
  ```python
  tmp = np.linspace(1, 100, int(args.num_epochs * len(train_loader) / args.beta_range))
  ```
3. The `evaluate.py` and `evaluate.sh` is used to calculated the FLOPs in each network, which is used to measure FLOPs drop.




## Parameter explained 
+ MobileNetV2

| Parameter  | Meaning | Default |
| ---------- | --------|---------|
| block_id  | There are 7 blocks in MobileNetV2, hence, choose among 1-7 | 0 |
| channel_dim_path | path to the final channel index for all blocks, should be stored in block_7/ | SSNet_mobilev2/logs/block_7/channel_dim.pth |
| pretrained_model | In evaluate.py to indicate if measure FLOPs for pretrained or pruned model|  False |

+ ResNet-56, VGG-16

| Parameter  | Meaning | Default |
| ---------- | --------|---------|
| group_id  | There are four groups in ResNet-56, hence, choose among 0,1,2,3 | 0 |
| compression_rate  | percent of channels to be retained, indicated using float number with range [0, 1] | 0.4 |
| layersort         | if not specified, global sort is used, else layer sort | False |
