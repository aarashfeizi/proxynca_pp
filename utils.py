
from __future__ import print_function
from __future__ import division

import copy

from sklearn.metrics.pairwise import cosine_similarity

import evaluation
import numpy as np
import torch
import logging
import loss
import json
import networks
import time
#import margin_net
import similarity

# __repr__ may contain `\n`, json replaces it by `\\n` + indent

json_dumps = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def load_config(config_name = 'config.json'):
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)
    return config

def predict_batchwise(model, dataloader):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # extract batches (A becomes list of samples)
        for batch in dataloader:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).cpu()


                for j in J:
                    #if i == 1: print(j)
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]

def predict_batchwise_inshop(model, dataloader):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # use tqdm when the dataset is large (SOProducts)
        is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in dataloader:#, desc='predict', disable=not is_verbose:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).data.cpu().numpy()
                    # take only subset of resulting embedding w.r.t dataset
                for j in J:
                    A[i].append(np.asarray(j))
        result = [np.stack(A[i]) for i in range(len(A))]
    model.train()
    model.train(model_is_training) # revert to previous training state
    return result

def evaluate(model, dataloader, eval_nmi=True, recall_list=[1,2,4,8], x=None, t=None, save_name=''):
    eval_time = time.time()
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    if x is None:
        X, T, *_ = predict_batchwise(model, dataloader)
    else:
        print('USING GIVEN X AND T, NOT CALCULATING!!!!!!!!!!!!!!!!!!!!!!!!')
        X = x
        T = t

    if save_name != '':
        import pickle
        with open(f'X_{save_name}.pkl', 'wb') as f:
            pickle.dump(X, f)

        with open(f'T_{save_name}.pkl', 'wb') as f:
            pickle.dump(T, f)

    output_str = ''
    print('done collecting prediction')

    #eval_time = time.time() - eval_time
    #logging.info('Eval time: %.2f' % eval_time)

    if eval_nmi:
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(
            T,
            evaluation.cluster_by_kmeans(
                X, nb_classes
            )
        )
    else:
        nmi = 1

    logging.info("NMI: {:.3f}".format(nmi * 100))
    output_str += "NMI: {:.3f}\n".format(nmi * 100)

    # get predictions by assigning nearest 8 neighbors with euclidian
    max_dist = max(recall_list)
    Y = evaluation.assign_by_euclidian_at_k(X, T, max_dist)
    Y = torch.from_numpy(Y)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in recall_list:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
        output_str += "R@{} : {:.3f}\n".format(k, 100 * r_at_k)

    chmean = (2*nmi*recall[0]) / (nmi + recall[0])
    logging.info("hmean: %s", str(chmean))
    # output_str += "hmean: %s\n", str(chmean)

    eval_time = time.time() - eval_time
    logging.info('Eval time: %.2f' % eval_time)

    auroc = calc_auroc(X, T)
    logging.info("AUROC: {:.3f}".format(auroc * 100))
    # output_str += "AUROC: {:.3f}\n".format(auroc * 100)

    return nmi, recall, auroc


def evaluate_qi(model, dl_query, dl_gallery,
                K = [1, 10, 20, 30, 40, 50], with_nmi = False):

    # calculate embeddings with model and get targets
    X_query, T_query, *_ = predict_batchwise_inshop(
        model, dl_query)
    X_gallery, T_gallery, *_ = predict_batchwise_inshop(
        model, dl_gallery)

    

    nb_classes = dl_query.dataset.nb_classes()

    assert nb_classes == len(set(T_query))
    #assert nb_classes == len(T_query.unique())

    # calculate full similarity matrix, choose only first `len(X_query)` rows
    # and only last columns corresponding to the column
    T_eval = torch.cat(
        [torch.from_numpy(T_query), torch.from_numpy(T_gallery)])
    X_eval = torch.cat(
        [torch.from_numpy(X_query), torch.from_numpy(X_gallery)])
    D = similarity.pairwise_distance(X_eval)[:len(X_query), len(X_query):]

    #D = torch.from_numpy(D)
    # get top k labels with smallest (`largest = False`) distance
    Y = T_gallery[D.topk(k = max(K), dim = 1, largest = False)[1]]

    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T_query, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if with_nmi:
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(
            T_eval.numpy(),
            evaluation.cluster_by_kmeans(
                X_eval.numpy(), nb_classes
            )
        )
    else:
        nmi = 1

    logging.info("NMI: {:.3f}".format(nmi * 100))

    return nmi, recall


def make_batch_bce_labels(labels):
    """
    :param labels: e.g. tensor of size (N,1)
    :return: binary matrix of labels of size (N, N)
    """

    l_ = labels.repeat(len(labels)).reshape(-1, len(labels))
    l__ = labels.repeat_interleave(len(labels)).reshape(-1, len(labels))

    final_bce_labels = (l_ == l__).type(torch.float32)

    # final_bce_labels.fill_diagonal_(0)

    return final_bce_labels

def get_samples(l, k):
    if len(l) < k:
        to_ret = np.random.choice(l, k, replace=True)
    else:
        to_ret = np.random.choice(l, k, replace=False)

    return to_ret

def get_xs_ys(bce_labels, k=1):
    """

    :param bce_labels: tensor of (N, N) with 0s and 1s
    :param k: number of pos and neg samples per anch
    :return:

    """
    xs = []
    ys = []
    bce_labels_copy = copy.deepcopy(bce_labels)
    bce_labels_copy.fill_diagonal_(-1)
    for i, row in enumerate(bce_labels_copy):
        neg_idx = torch.where(row == 0)[0]
        pos_idx = torch.where(row == 1)[0]

        ys.extend(get_samples(neg_idx, k))
        ys.extend(get_samples(pos_idx, k))
        xs.extend(get_samples([i], 2 * k))

    return xs, ys

def calc_auroc(embeddings, labels):
    from sklearn.metrics import roc_auc_score
    bce_labels = make_batch_bce_labels(labels)
    similarities = cosine_similarity(embeddings)

    xs, ys = get_xs_ys(bce_labels, k=1)

    true_labels = bce_labels[xs, ys]
    predicted_labels = similarities[xs, ys]

    return roc_auc_score(true_labels, predicted_labels)

