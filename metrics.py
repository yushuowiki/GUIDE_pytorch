from sklearn.metrics import roc_auc_score, average_precision_score,auc,\
                precision_recall_curve,roc_curve
import scipy as sp   

def auc_roc(scores, target):
    # print('scores',scores)
    # print('target',target)
    return roc_auc_score(target, scores)

def auc_pr(scores, target):
    # print('scores',scores)
    precision, recall, _ = precision_recall_curve(target, scores)
    auc_pr = auc(recall, precision)
    return auc_pr

def precision(scores, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    pred=scores.argsort()[-maxk:][::-1]

    pred=pred.reshape(-1)
    target=target.reshape(-1)

    correct = target[pred]
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype(float).sum(0)
        res.append(correct_k*(1.0 / k))
    return res

def recall(scores, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    anomaly_size = target.sum()

#    _, pred = scores.view(1,-1).topk(maxk, 1, True, True)
    pred=scores.argsort()[-maxk:][::-1]
    
    pred=pred.reshape(-1)
    target=target.reshape(-1)
    correct = target[pred]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype(float).sum(0)
        # print(correct_k)
        res.append(correct_k*(1.0 / anomaly_size))
    return res

