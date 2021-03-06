import numpy as np


def calculate_candidates_ranking(prediction, ground_truth,
                                 eval_candidates_num=10):
    total_num_split = len(ground_truth) // eval_candidates_num  # batch size

    prediction_split = np.split(prediction, total_num_split)
    ground_truth_split = np.split(np.array(ground_truth), total_num_split)
    origin_rank_split = np.split(
        np.tile(np.arange(0, eval_candidates_num), total_num_split),
        total_num_split
    )
    stack_scores = np.stack(
        (ground_truth_split, prediction_split, origin_rank_split),
        axis=-1
    )

    rank_by_pred_l = []
    for i, stack_score in enumerate(stack_scores):
        rank_by_pred = sorted(stack_score, key=lambda x: x[1], reverse=True)
        rank_by_pred = np.stack(rank_by_pred, axis=-1)
        rank_by_pred_l.append(rank_by_pred[0])
    rank_by_pred = np.array(rank_by_pred_l)

    pos_index = []
    for sorted_score in rank_by_pred:
        pos_index.append([
            p_i for p_i, score in enumerate(sorted_score) if int(score) == 1
        ])

    return rank_by_pred, pos_index, stack_scores


def logits_recall_at_k(pos_index, k_list=None):
    if k_list is None:
        k_list = [1, 2, 5, 10]
    num_correct = np.zeros([len(pos_index), len(k_list)])
    index_dict = dict()
    for i, p_i in enumerate(pos_index):
        index_dict[i] = p_i

    for i, p_i in enumerate(pos_index):
        if len(p_i) == 1 and p_i[0] >= 0:
            for j, k in enumerate(k_list):
                if p_i[0] + 1 <= k:
                    num_correct[i][j] += 1
        elif len(p_i) > 1:
            for j, k in enumerate(k_list):
                all_recall_at_k = [
                    1 if cand_p_i + 1 <= k else 0 for cand_p_i in p_i
                ]
                num_correct[i][j] += np.mean(all_recall_at_k)

    return np.sum(num_correct, axis=0)


def logits_mrr(pos_index):
    mrr = [
        1 / (p_i[0] + 1) if len(p_i) > 0 and p_i[0] >= 0 else 0
        for i, p_i in enumerate(pos_index)
    ]

    return np.sum(mrr)


def precision_at_one(rank_by_pred):
    num_correct = [0] * rank_by_pred.shape[0]
    for i, sorted_score in enumerate(rank_by_pred):
        if len(sorted_score) > 0 and int(sorted_score[0]) == 1:
            num_correct[i] = 1

    return np.sum(num_correct)


def mean_average_precision_fn(pos_index):
    map = []
    for i, p_i in enumerate(pos_index):
        if len(p_i) > 0:
            all_precision = [
                (j + 1) / (cand_p_i + 1) for j, cand_p_i in enumerate(p_i)
            ]
            map.append(np.mean(all_precision))
        elif len(p_i) == 0:
            map.append(0)  # no answer

    return np.sum(map)
