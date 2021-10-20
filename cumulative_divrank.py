# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:26:16 2021
@author: Dr. Deepak Kumar
"""
from typing import Dict
from operator import itemgetter
import logging
LOGGER = logging.getLogger(__name__)
import networkx as nx
import numpy as np
import configparser

def com_divrank(graph, lambda_, alpha: float) -> Dict[str, float]:
    """
    Cumulative DivRank  is ranking algorithm that attempts to balance
    between node centrality and diversity.
    Args:
        graph
        r: The "personalization vector"; by default, ``r = ones(1, n)/n``
        lambda_: Float in [0.0, 1.0]
        alpha: Float in [0.0, 1.0] that controls the strength of self-links.
    Returns:
        Mapping of node to score ordered by descending divrank score
    """
    # check function arguments
    config_file = "config.cfg"
    config = configparser.ConfigParser()
    config.read(config_file)
    # Specify some constants
    max_iter = config["divrank_param"].getint("max_iter")
    diff = config["divrank_param"].getint("diff")
    tol = config["divrank_param"].getfloat("tol")
    personalization_vect = config["divrank_param"].get("r")

    if len(graph) == 0:
        LOGGER.error("`graph` is empty")
        return {}
    # specify the order of nodes to use in creating the matrix
    # and then later recovering the values from the order index
    nodes_list = [node for node in graph]
    # create adjacency matrix, i.e.
    # n x n matrix where entry W_ij is the weight of the edge from V_i to V_j
    W = nx.to_numpy_matrix(graph, nodelist=nodes_list, weight="weight").A
    n = W.shape[1]
    # create flat prior personalization vector if none given
    if personalization_vect:
        per_vect = np.array([n * [1 / float(n)]])
    process = np.array([n * [1 / float(n)]])
    pr_pastall = process
    # Get p0(v -> u), i.e. transition probability prior to reinforcement
    tmp = np.reshape(np.sum(W, axis=1), (n, 1))
    idx_nan = np.flatnonzero(tmp == 0)
    W0 = W / np.tile(tmp, (1, n))
    W0[idx_nan, :] = 0
    del W

    # Comulative DivRank algorithm
    current_iter = 0
    while current_iter < max_iter and diff > tol:
        W1 = alpha * W0 * np.tile(pr_pastall, (n, 1))
        W1 = W1 - np.diag(W1[:, 0]) + (1 - alpha) * np.diag(pr_pastall[0, :])
        tmp1 = np.reshape(np.sum(W1, axis=1), (n, 1))
        prob = W1 / np.tile(tmp1, (1, n))
        prob = ((1 - lambda_) * prob) + (lambda_ * np.tile(per_vect, (n, 1)))
        pr_new = np.dot(process, prob)
        current_iter += 1
        diff = np.sum(np.abs(pr_new - process)) / np.sum(process)
        process = pr_new
        pr_pastall = pr_pastall + process
        
    # sort nodes by Commulative divrank score
    results = sorted(((i, score) for i, score in enumerate(process.flatten().tolist())),
                     key=itemgetter(1), reverse=True, )
    # replace node number by node value
    divranks = {nodes_list[result[0]]: result[1] for result in results}
    return divranks
