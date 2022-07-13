def set_up_datasets(args):
    if args.dataset == 'miniimagenet':
        args.num_class = 64
        if args.deepemd == 'fcn':
            from Models.dataloader.miniimagenet.fcn.mini_imagenet import MiniImageNet as Dataset
        elif args.deepemd == 'ic':
            from Models.dataloader.miniimagenet.ic.mini_imagenet import MiniImageNet as Dataset
    
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
        if args.deepemd == 'fcn':
            from Models.dataloader.tieredimagenet.fcn.tiered_imagenet import tieredImageNet as Dataset
        elif args.deepemd == 'ic':
            from Models.dataloader.tieredimagenet.ic.tiered_imagenet import tieredImageNet as Dataset
    
    else:
        raise ValueError('Unkown Dataset')
    return Dataset
