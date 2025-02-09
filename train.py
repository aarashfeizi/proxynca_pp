import logging
import dataset
import utils
import loss

import os

import torch
import numpy as np
import matplotlib

matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
import time
import argparse
import json
import random
import h5py
from utils import JSONEncoder, json_dumps

def load_h5(data_description, path):
     data = None
     with h5py.File(path, 'r') as hf:
         data = hf[data_description][:]
     return data

def save_best_checkpoint(model):
    print(f'saving to results/' + args.log_filename + '.pt')
    torch.save(model.state_dict(), 'results/' + args.log_filename + '.pt')


def load_best_checkpoint(model):
    model.load_state_dict(torch.load('results/' + args.log_filename + '.pt'))
    model = model.cuda()
    return model


def batch_lbl_stats(y):
    print(torch.unique(y))
    kk = torch.unique(y)
    kk_c = torch.zeros(kk.size(0))
    for kx in range(kk.size(0)):
        for jx in range(y.size(0)):
            if y[jx] == kk[kx]:
                kk_c[kx] += 1


def get_centers(dl_tr):
    c_centers = torch.zeros(dl_tr.dataset.nb_classes(), args.sz_embedding).cuda()
    n_centers = torch.zeros(dl_tr.dataset.nb_classes()).cuda()
    for ct, (x, y, _) in enumerate(dl_tr):
        with torch.no_grad():
            m = model(x.cuda())
        for ix in range(m.size(0)):
            c_centers[y] += m[ix]
            n_centers[y] += 1
    for ix in range(n_centers.size(0)):
        c_centers[ix] = c_centers[ix] / n_centers[ix]

    return c_centers


parser = argparse.ArgumentParser(description='Training ProxyNCA++')
parser.add_argument('--dataset', default='cub')
parser.add_argument('--config', default='config.json')
parser.add_argument('--embedding-size', default=2048, type=int, dest='sz_embedding')
parser.add_argument('--batch-size', default=32, type=int, dest='sz_batch')
parser.add_argument('--epochs', default=40, type=int, dest='nb_epochs')
parser.add_argument('--log-filename', default='example')
parser.add_argument('--workers', default=16, type=int, dest='nb_workers')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mode', default='train', choices=['train', 'trainval', 'test'],
                    help='train with train data or train with trainval')
parser.add_argument('--testset', default='test1', type=str)
parser.add_argument('--valset', default='val1', type=str)
parser.add_argument('--lr_steps', default=[1000], nargs='+', type=int)
parser.add_argument('--source_dir', default='', type=str)
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--small_dataset', default=False, action='store_true')
parser.add_argument('--eval_nmi', default=False, action='store_true')
parser.add_argument('--recall', default=[1, 2, 4, 8, 16, 32], nargs='+', type=int)
parser.add_argument('--init_eval', default=False, action='store_true')
parser.add_argument('--no_warmup', default=False, action='store_true')
parser.add_argument('--apex', default=False, action='store_true')
parser.add_argument('--warmup_k', default=5, type=int)
parser.add_argument('--x_path', default='', type=str)
parser.add_argument('--t_path', default='', type=str)

parser.add_argument('--xname', default='', type=str)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # set random seed for all gpus

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('log'):
    os.makedirs('log')

curr_fn = os.path.basename(args.config).split(".")[0]


config = utils.load_config(args.config)

dataset_config = utils.load_config('dataset/config.json')

if args.source_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['source'])
    dataset_config['dataset'][args.dataset]['source'] = os.path.join(args.source_dir, bs_name)
if args.root_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['root'])
    dataset_config['dataset'][args.dataset]['root'] = os.path.join(args.root_dir, bs_name)

# if args.apex:
#     from apex import amp

# set NMI or recall accordingly depending on dataset. note for cub and cars R=1,2,4,8
if (args.mode == 'trainval' or args.mode == 'test'):
    if args.dataset == 'sop' or args.dataset == 'sop_h5':
        args.recall = [1, 10, 100, 1000]
    elif 'cub' in args.dataset or 'cars' in args.dataset:
        args.eval_nmi = True

args.nb_epochs = config['nb_epochs']
args.sz_batch = config['sz_batch']
args.sz_embedding = config['sz_embedding']
if 'warmup_k' in config:
    args.warmup_k = config['warmup_k']

transform_key = 'transform_parameters'
if 'transform_key' in config.keys():
    transform_key = config['transform_key']

if args.xname != '':
    xname = args.xname + '_'
else:
    xname = ''

out_results_fn = f'log/res_{args.dataset}_{curr_fn}_{args.mode}_{xname}{args.nb_epochs}ep_{args.sz_embedding}_bs{args.sz_batch}_nc{config["num_class_per_batch"]}_baseLR{config["opt"]["args"]["base"]["lr"]}_pncaLR{config["opt"]["args"]["proxynca"]["lr"]}.txt'

args.log_filename = f'{args.dataset}_{curr_fn}_{args.mode}_{xname}{args.nb_epochs}ep_{args.sz_embedding}_bs{args.sz_batch}_nc{config["num_class_per_batch"]}_baseLR{config["opt"]["args"]["base"]["lr"]}_pncaLR{config["opt"]["args"]["proxynca"]["lr"]}'

# out_results_fn = f'log/{args.dataset}_{curr_fn}_{args.mode}_{args.xname}_{args.sz_embedding}_bs{args.sz_batch}_nc{config["num_class_per_batch"]}_baseLR{config["opt"]["args"]["base"]["lr"]}_pncaLR{config["opt"]["args"]["proxynca"]["lr"]}.txt' % (,
#                                                        curr_fn,
#                                                        args.mode,
#                                                        args.xname,
#                                                        args.sz_embedding,
#                                                        args.sz_batch,
#                                                        config['num_class_per_batch'],
#                                                        round(config['opt']['args']['base']['lr'], 10),
#                                                        round(config['opt']['args']['proxynca']['lr'], 10))
# args.log_filename = '%s_%s_%s_%s_%d_bs%d_nc%d_baseLR%f_pncaLR%f' % (args.dataset,
#                                                                     curr_fn,
#                                                                     args.mode,
#                                                                     args.xname,
#                                                                     args.sz_embedding,
#                                                                     args.sz_batch,
#                                                                     config['num_class_per_batch'],
#                                                                     round(config['opt']['args']['base']['lr'], 10),
#                                                                     round(config['opt']['args']['proxynca']['lr'], 10))

if args.mode == 'test':
    args.log_filename = args.log_filename.replace('test', 'trainval')

best_epoch = args.nb_epochs

feat = config['model']['type']()
feat.eval()
in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
feat.train()
emb = torch.nn.Linear(in_sz, args.sz_embedding)
model = torch.nn.Sequential(feat, emb)

if not args.apex:
    model = torch.nn.DataParallel(model)
model = model.cuda()

if args.mode == 'trainval':
    train_results_fn = f'log/res_{args.dataset}_{curr_fn}_train_{xname}{args.nb_epochs}ep_{args.sz_embedding}_bs{args.sz_batch}_nc{config["num_class_per_batch"]}_baseLR{config["opt"]["args"]["base"]["lr"]}_pncaLR{config["opt"]["args"]["proxynca"]["lr"]}.txt'
    # train_results_fn = 'log/%s_%s_%s_%s_%d_bs%d_nc%d_baseLR%f_pncaLR%f' % (args.dataset,
    #                                                                        curr_fn,
    #                                                                        '',
    #                                                                        args.xname,
    #                                                                        args.sz_embedding,
    #                                                                        args.sz_batch,
    #                                                                        config['num_class_per_batch'],
    #                                                                        round(config['opt']['args']['base']['lr'], 10),
    #                                                                        round(config['opt']['args']['proxynca']['lr'], 10))
    if os.path.exists(train_results_fn):
        with open(train_results_fn, 'r') as f:
            train_results = json.load(f)
        args.lr_steps = train_results['lr_steps']
        best_epoch = train_results['best_epoch']

train_transform = dataset.utils.make_transform(
    **dataset_config[transform_key]
)
print('best_epoch', best_epoch)

results = {}

if ('inshop' not in args.dataset) and ('hotel' not in args.dataset):
    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        # pin_memory = True
    )
elif 'hotel' in args.dataset:
    if args.small_dataset:
        to_add_to_name = '_small'
    else:
        to_add_to_name = ''

    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset][f'root_{args.testset + to_add_to_name}'],
            source=dataset_config['dataset'][args.dataset][f'source_{args.testset + to_add_to_name}'],
            classes=dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        # pin_memory = True
    )
else:
    # inshop trainval mode
    dl_query = torch.utils.data.DataLoader(
        dataset.load_inshop(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            ),
            dset_type='query'
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        # pin_memory = True
    )
    dl_gallery = torch.utils.data.DataLoader(
        dataset.load_inshop(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            ),
            dset_type='gallery'
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        # pin_memory = True
    )

    import pdb
    pdb.set_trace()

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
        logging.StreamHandler()
    ]
)

if 'hotel' not in args.dataset:
    if args.mode == 'train':
        tr_dataset = dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['train'],
            transform=train_transform
        )
    elif args.mode == 'trainval' or args.mode == 'test':
        tr_dataset = dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['trainval'],
            transform=train_transform
        )
else:
    if args.small_dataset:
        to_add_to_name = '_small'
    else:
        to_add_to_name = ''

    if args.mode == 'train':
        tr_dataset = dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root_train' + to_add_to_name],
            source=dataset_config['dataset'][args.dataset]['source_train' + to_add_to_name],
            classes=dataset_config['dataset'][args.dataset]['classes']['train'],
            transform=train_transform
        )
    elif args.mode == 'trainval' or args.mode == 'test':
        tr_dataset = dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root_trainval' + to_add_to_name],
            source=dataset_config['dataset'][args.dataset]['source_trainval' + to_add_to_name],
            classes=dataset_config['dataset'][args.dataset]['classes']['trainval'],
            transform=train_transform
        )
num_class_per_batch = config['num_class_per_batch']
num_gradcum = config['num_gradcum']
is_random_sampler = config['is_random_sampler']
if is_random_sampler:
    batch_sampler = dataset.utils.RandomBatchSampler(tr_dataset.ys, args.sz_batch, True, num_class_per_batch,
                                                     num_gradcum)
else:

    batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_class_per_batch,
                                                       int(args.sz_batch / num_class_per_batch))

dl_tr = torch.utils.data.DataLoader(
    tr_dataset,
    batch_sampler=batch_sampler,
    num_workers=args.nb_workers,
    # pin_memory = True
)

print("===")
# if args.mode == 'train':
if 'hotel' not in args.dataset:
    dl_val = torch.utils.data.DataLoader(
        dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['val'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        # drop_last=True
        # pin_memory = True
    )
else:
    if args.small_dataset:
        to_add_to_name = '_small'
    else:
        to_add_to_name = ''

    dl_val = torch.utils.data.DataLoader(
        dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset][f'root_{args.valset + to_add_to_name}'],
            source=dataset_config['dataset'][args.dataset][f'source_{args.valset + to_add_to_name}'],
            classes=dataset_config['dataset'][args.dataset]['classes']['val'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        # drop_last=True
        # pin_memory = True
    )

criterion = config['criterion']['type'](
    nb_classes=dl_tr.dataset.nb_classes(),
    sz_embed=args.sz_embedding,
    **config['criterion']['args']
).cuda()

opt_warmup = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            'lr': 0
        },
        {
            **{'params': list(emb.parameters()
                              )
               },
            **config['opt']['args']['embedding']

        },

        {
            **{'params': criterion.parameters()}
            ,
            **config['opt']['args']['proxynca']

        },

    ],
    **config['opt']['args']['base']
)

opt = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            **config['opt']['args']['backbone']
        },
        {
            **{'params': list(emb.parameters()
                              )
               },
            **config['opt']['args']['embedding']
        },

        {
            **{'params': criterion.parameters()},
            **config['opt']['args']['proxynca']
        },

    ],
    **config['opt']['args']['base']
)

# if args.apex:
#     [model, criterion], [opt, opt_warmup] = amp.initialize([model, criterion], [opt, opt_warmup], opt_level='O1')
#     model = torch.nn.DataParallel(model)

if args.x_path != '':
    feats = load_h5(f'cub_eval_val_feats', args.x_path)
    labels = load_h5(f'cub_eval_val_classes', args.t_path)
else:
    feats, labels = None, None

if args.mode == 'test':
    with torch.no_grad():
        logging.info("**Evaluating...(test mode)**")
        if feats is None:
            model = load_best_checkpoint(model)
        else:
            model = None
        if 'inshop' in args.dataset:
            utils.evaluate_qi(model, dl_query, dl_gallery)
        else:
            val_nmi, val_recall, val_auroc = utils.evaluate(model, dl_val, args.eval_nmi, args.recall, x=feats, t=labels, save_name=f'val_{args.dataset}_{args.valset}')
            test_nmi, test_recall, test_auroc = utils.evaluate(model, dl_ev, args.eval_nmi, args.recall, x=feats, t=labels, save_name=f'test_{args.dataset}_{args.testset}')
            result_str = '*' * 50
            result_str += '\n'
            result_str += f'{args.valset} nmi: {val_nmi}\n'
            result_str += f'{args.valset} recall at {args.recall}: {val_recall}\n'
            result_str += f'{args.valset} AUROC: {val_auroc}\n'
            result_str += '*' * 50
            result_str += f'{args.testset} nmi: {test_nmi}\n'
            result_str += f'{args.testset} recall at {args.recall}: {test_recall}\n'
            result_str += f'{args.testset} AUROC: {test_auroc}\n'

            with open('results/' + args.log_filename + f'_results_{args.dataset}_{args.valset}_{args.testset}.txt', 'w') as f:
                f.write(result_str)


    exit()

if args.mode == 'train':
    scheduler = config['lr_scheduler']['type'](
        opt, **config['lr_scheduler']['args']
    )
elif args.mode == 'trainval':
    scheduler = config['lr_scheduler2']['type'](
        opt,
        milestones=args.lr_steps,
        gamma=0.1
        # opt, **config['lr_scheduler2']['args']
    )

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []

t1 = time.time()

if args.init_eval:
    logging.info("**Evaluating initial model...**")
    with torch.no_grad():
        if args.mode == 'train' or args.mode == 'trainval':
            c_dl = dl_val
        else:
            c_dl = dl_ev

        utils.evaluate(model, c_dl, args.eval_nmi, args.recall)  # dl_val

it = 0

best_val_hmean = 0
best_val_nmi = 0
best_val_auroc = 0
best_val_epoch = 0
best_val_r1 = 0
best_test_nmi = 0
best_test_auroc = 0
best_test_r1 = 0
best_test_r2 = 0
best_test_r5 = 0
best_test_r8 = 0
best_tnmi = 0

prev_lr = opt.param_groups[0]['lr']
lr_steps = []

print(len(dl_tr))

if not args.no_warmup:
    # warm up training for 5 epochs
    logging.info("**warm up for %d epochs.**" % args.warmup_k)
    for e in range(0, args.warmup_k):
        for ct, (x, y, _) in enumerate(dl_tr):
            opt_warmup.zero_grad()
            m = model(x.cuda())
            loss = criterion(m, y.cuda())
            # if args.apex:
            #     with amp.scale_loss(loss, opt_warmup) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt_warmup.step()
        logging.info('warm up ends in %d epochs' % (args.warmup_k - e))

from tqdm import tqdm
print('total epochs:', args.nb_epochs)
for e in range(1, args.nb_epochs + 1):
    # if args.mode == 'trainval':
    #    scheduler.step(e)

    if args.mode == 'train' or args.mode == 'trainval':
        curr_lr = opt.param_groups[0]['lr']
        print(prev_lr, curr_lr)
        if curr_lr != prev_lr:
            prev_lr = curr_lr
            lr_steps.append(e)

    time_per_epoch_1 = time.time()
    losses_per_epoch = []
    tnmi = []

    opt.zero_grad()
    loss_total = 0
    with tqdm(total=len(dl_tr), desc=f'Epoch {e}/{args.nb_epochs}') as t:
        for ct, (x, y, _) in enumerate(dl_tr):
            it += 1

            m = model(x.cuda())

            loss1 = criterion(m, y.cuda())
            loss = loss1
            loss_total += loss.item()

            t.set_postfix(loss=(loss_total / (ct + 1)))

            # if args.apex:
            #     with amp.scale_loss(loss, opt) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)

            losses_per_epoch.append(loss.data.cpu().numpy())

            # if (ct + 1) % 1 == 0:
            opt.step()
            opt.zero_grad()

            t.update()

    time_per_epoch_2 = time.time()
    losses.append(np.mean(losses_per_epoch[-20:]))
    print('it: {}'.format(it))
    print(opt)
    logging.info(
        "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
            e,
            losses[-1],
            time_per_epoch_2 - time_per_epoch_1
        )
    )

    model.losses = losses
    model.current_epoch = e

    if e == best_epoch:
        break

    if args.mode == 'train' or args.mode == 'trainval':
        with torch.no_grad():
            logging.info("**Validation...**")
            nmi, recall, auroc = utils.evaluate(model, dl_val, args.eval_nmi, args.recall)

        chmean = (2 * nmi * recall[0]) / (nmi + recall[0])

        if args.mode == 'trainval':
            scheduler.step(e)
        else:
            scheduler.step(chmean)

        # if chmean > best_val_hmean:
        if auroc > best_val_auroc:
            best_val_hmean = chmean
            best_val_nmi = nmi
            best_val_r1 = recall[0]
            best_val_r2 = recall[1]
            best_val_r4 = recall[2]
            best_val_r8 = recall[3]
            best_val_auroc = auroc
            best_val_epoch = e
            best_tnmi = torch.Tensor(tnmi).mean()

            save_best_checkpoint(model)
            print('Model saved!!')

        if e == (args.nb_epochs - 1):
            # saving last epoch
            results['last_NMI'] = nmi
            results['last_hmean'] = chmean
            results['last_auroc'] = auroc
            results['best_epoch'] = best_val_epoch
            results['last_R1'] = recall[0]
            results['last_R2'] = recall[1]
            results['last_R4'] = recall[2]
            results['last_R8'] = recall[3]

            # saving best epoch
            results['best_NMI'] = best_val_nmi
            results['best_hmean'] = best_val_hmean
            results['best_auroc'] = best_val_auroc
            results['best_R1'] = best_val_r1
            results['best_R2'] = best_val_r2
            results['best_R4'] = best_val_r4
            results['best_R8'] = best_val_r8

        logging.info('Best val epoch: %s', str(best_val_epoch))
        logging.info('Best val hmean: %s', str(best_val_hmean))
        logging.info('Best val nmi: %s', str(best_val_nmi))
        logging.info('Best val r1: %s', str(best_val_r1))
        logging.info('Best val auroc: %s', str(best_val_auroc))
        logging.info(str(lr_steps))


if args.mode == 'trainval':

    with torch.no_grad():
        logging.info("**Evaluating...**")
        model = load_best_checkpoint(model)
        if 'inshop' in args.dataset:
            best_test_nmi, (best_test_r1, best_test_r10, best_test_r20, best_test_r30, best_test_r40,
                            best_test_r50) = utils.evaluate_qi(model, dl_query, dl_gallery)
        else:
            best_test_nmi, (best_test_r1, best_test_r2, best_test_r4, best_test_r8, best_test_r16,
                            best_test_r32), best_test_auroc = utils.evaluate(model, dl_val,
                                                                             args.eval_nmi,
                                                                             args.recall)
            # best_test_nmi, (best_test_r1, best_test_r2, best_test_r4, best_test_r8, best_test_r16, best_test_r32), best_test_auroc = utils.evaluate(model, dl_ev,
            #                                                                                          args.eval_nmi,
            #                                                                                          args.recall)
        # logging.info('Best test r8: %s', str(best_test_r8))
    if 'inshop' in args.dataset:
        results['NMI'] = best_test_nmi
        results['auroc'] = best_test_auroc
        results['R1'] = best_test_r1
        results['R10'] = best_test_r10
        results['R20'] = best_test_r20
        results['R30'] = best_test_r30
        results['R40'] = best_test_r40
        results['R50'] = best_test_r50
    else:
        results['NMI'] = best_test_nmi
        results['auroc'] = best_test_auroc
        results['R1'] = best_test_r1
        results['R2'] = best_test_r2
        results['R4'] = best_test_r4
        results['R8'] = best_test_r8
        results['R16'] = best_test_r4
        results['R32'] = best_test_r8

if args.mode == 'train':
    print('lr_steps', lr_steps)
    results['lr_steps'] = lr_steps

with open(out_results_fn, 'w') as outfile:
    json.dump(results, outfile)

t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
