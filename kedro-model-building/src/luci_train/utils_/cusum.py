import numpy as np
import math


def get_arl(prob, cl, R1=2, Ractual=1, scaling=1000):
    """
    Find the average run length (ARL) of a one-sided RA-CUSUM chart with binary outcomes.

    :param prob: Predicted probabilities of negative outcome
    :param R1: the odds' ratio multiplier for the alternative hypothesis (>1 for a chart that looks for increases).
    R1 =2 is a common choice, set chart to detecting a doubling of the odds of mortality.
    :param Ractual: the true odds' ratio multiplier (Ractual=1 gives in-control ARL),
    by changing Ractual you can explore
    the ARL out-of-control
    :param cl: Control limit of the RA-CUSUM chart with binary outcomes
    :param scaling: scaling controls the quality of the approximation,
    larger values are better but increase the computation burden (suggested 600-1000)
    :return: ARL(average run length)
    """
    # find the average run length (ARL) of a one-sided RA-CUSUM chart with binary outcomes
    # prob
    # second column gives the corresponding mortality rate
    # R1 is the odds ratio multiplier for the alternative hupothesis (>1 for a chart that looks for increases)
    # R1 =2 is a common choice, set chart to detecting a doubling of the odds of mortality
    # Ractual is the true odds ratio multiplier (Ractual=1 gives in-control ARL), by changing Ractual you can explore
    # the ARL out-of-control
    # CL is the control limit (>0)
    # calculate the distribution of probabilities
    unique, counts = np.unique(prob, return_counts=True)
    rel_counts = counts / len(prob)
    # scaling controls the quality of the approximation, larger values are better but increase the computation burden
    mp = rel_counts  # Parsonnet distribution
    mr = unique  # model
    z = []  # initialize
    p = []

    for i in range(len(mp)):
        pi = mr[i]
        wnf = math.log(1 / (1 + (R1 - 1) * pi))  # score for success (non death) case
        wf = math.log(R1 / (1 + (R1 - 1) * pi))  # failure (death) case
        # translate given pi into actual pi based on odds' multiplier hactual
        pf = (Ractual * pi) / (1 - pi + Ractual * pi)
        z.extend((wnf, wf))
        p.extend(((1 - pf) * mp[i], pf * mp[i]))

    # scale weights and boundary to give all integers, so we can use a Markov chain to approx. ARL
    z = np.around([x * scaling for x in z])
    p = np.array(p)
    cl = np.around(cl * scaling)
    order_z = z.argsort()
    z_sort = z[order_z]
    p_sort = p[order_z]

    # combine weights that are not different
    z_now = z_sort[0]
    z = [z_sort[0]]
    p = [p_sort[0]]
    now = 0
    for i in range(1, len(z_sort)):
        if abs(z_sort[i]) - z_now < 0.001:  # same weight as previous one, collapse
            p[now] = p[now] + p_sort[i]
        else:
            z_now = z_sort[i]
            now = now + 1
            z.append(z_sort[i])
            p.append(p_sort[i])

    z = np.array(z)
    p = np.array(p)
    # checked

    # find the run length properties of a variables based CUSUM using the Markov chain approach.
    lz = len(z)
    # define transition matrix without absorbing state
    r = np.zeros((int(cl), int(cl)))  # initialize
    for state in range(int(cl)):  # see where we can go from this state
        now = z + state  # possible new states
        zz = np.where(now <= 0)  # find all new states that have negative scores
        zz = np.array(zz)
        r[state, 0] = p[zz].sum()  # lead back to zero
        absorb = np.where(now >= cl)  # lead to absorption
        absorb = np.array(absorb)
        if zz.size == 0:
            zz_max = -1
        else:
            zz_max = np.amax(zz)
        if absorb.size == 0:
            absorb_min = lz
        else:
            absorb_min = np.amin(absorb)
        for j in range(zz_max + 1, absorb_min):  # all other states
            col_index = int(now[j])
            r[state, col_index] = p[j]
    # find the average run length
    id_matrix = np.identity(int(cl))
    b = np.ones((int(cl), 1))
    avg = np.linalg.solve(id_matrix - r, b)
    arl_value = avg.item(0)
    return arl_value
