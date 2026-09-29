"""Experimental integer kernels with the reference engine's exact tie/seed rules."""

import numpy as np
from numba import njit


def packed_postings(index):
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    positions = [[] for _ in range(21**3)]
    for word, postings in index.postings.items():
        code = 0
        for letter in word:
            code = code * 21 + alphabet.index(letter)
        positions[code] = postings
    offsets = np.array([0, *np.cumsum([len(p) for p in positions])], dtype=np.int64)
    packed = np.array([pair for p in positions for pair in p], dtype=np.int64).reshape(-1, 2)
    return offsets, packed


@njit(cache=True)
def support_kernel(query, mask, postings_offset, postings, max_target_length, window=64):
    qlength = len(query)
    span = qlength + max_target_length + 1
    capacity = 0
    for i in range(qlength - 2):
        if mask[i] and mask[i + 1] and mask[i + 2]:
            word = query[i] * 441 + query[i + 1] * 21 + query[i + 2]
            capacity += postings_offset[word + 1] - postings_offset[word]
    keys = np.empty(capacity, dtype=np.int64)
    used = 0
    for i in range(qlength - 2):
        if mask[i] and mask[i + 1] and mask[i + 2]:
            word = query[i] * 441 + query[i + 1] * 21 + query[i + 2]
            for j in range(postings_offset[word], postings_offset[word + 1]):
                tid, tpos = postings[j]
                keys[used] = (tid * span + tpos - i + qlength) * qlength + i
                used += 1
    keys.sort()
    tids, diagonals = np.empty(capacity, dtype=np.int64), np.empty(capacity, dtype=np.int64)
    begin, count = 0, 0
    while begin < used:
        group = keys[begin] // qlength
        end = begin + 1
        while end < used and keys[end] // qlength == group:
            end += 1
        left = begin
        passed = False
        for right in range(begin, end):
            position = keys[right] % qlength
            while left < right and position - keys[left] % qlength > window:
                left += 1
            if left < right and position - keys[left] % qlength >= 3:
                passed = True
                break
        if passed:
            tids[count] = group // span
            diagonals[count] = group % span - qlength
            count += 1
        begin = end
    return tids[:count], diagonals[:count]


@njit(cache=True)
def ungapped_kernel(query, flat, offsets, tids, diagonals, matrix, threshold=20):
    ids = np.empty(len(tids), dtype=np.int64)
    bests = np.empty(len(tids), dtype=np.int64)
    count, begin = 0, 0
    while begin < len(tids):
        tid = tids[begin]
        best, end = 0, begin
        while end < len(tids) and tids[end] == tid:
            diagonal = diagonals[end]
            current = 0
            length = offsets[tid + 1] - offsets[tid]
            for i in range(max(0, -diagonal), min(len(query), length - diagonal)):
                current = max(0, current + matrix[query[i], flat[offsets[tid] + i + diagonal]])
                best = max(best, current)
            end += 1
        if best >= threshold:
            ids[count], bests[count] = tid, best
            count += 1
        begin = end
    return ids[:count], bests[:count]


@njit(cache=True)
def traceback_kernel(query, target, matrix, gap_open, gap_extend):
    n, m = len(query), len(target)
    h = np.zeros((n + 1, m + 1), dtype=np.int64)
    e = np.full((n + 1, m + 1), -(1 << 62), dtype=np.int64)
    f = np.full((n + 1, m + 1), -(1 << 62), dtype=np.int64)
    hp = np.zeros((n + 1, m + 1), dtype=np.uint8)
    ep = np.zeros((n + 1, m + 1), dtype=np.uint8)
    fp = np.zeros((n + 1, m + 1), dtype=np.uint8)
    best, end_i, end_j = 0, 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opened, extended = h[i - 1, j] - gap_open, e[i - 1, j] - gap_extend
            e[i, j], ep[i, j] = max(opened, extended), int(extended > opened)
            opened, extended = h[i, j - 1] - gap_open, f[i, j - 1] - gap_extend
            f[i, j], fp[i, j] = max(opened, extended), int(extended > opened)
            diagonal = h[i - 1, j - 1] + matrix[query[i - 1], target[j - 1]]
            value = max(0, diagonal, e[i, j], f[i, j])
            h[i, j] = value
            if value > 0:
                hp[i, j] = 1 if value == diagonal else 2 if value == e[i, j] else 3
            if value > best:
                best, end_i, end_j = value, i, j
    i, j, state, used = end_i, end_j, 0, 0
    ops = np.empty(n + m, dtype=np.uint8)
    while True:
        if state == 0:
            if h[i, j] == 0:
                break
            if hp[i, j] == 1:
                i, j = i - 1, j - 1
                ops[used], used = 1, used + 1
            else:
                state = hp[i, j]
        elif state == 2:
            extend = ep[i, j]
            i -= 1
            ops[used], used = 2, used + 1
            state = 2 if extend else 0
        else:
            extend = fp[i, j]
            j -= 1
            ops[used], used = 3, used + 1
            state = 3 if extend else 0
    return best, i, end_i, j, end_j, ops[:used][::-1]


def decoded_alignment(query, target, raw):
    from itertools import groupby

    from mini3di_search.align_reference import Alignment

    score, qs, qe, ts, te, operations = raw
    if not score:
        return Alignment()
    ops = ["?MID"[int(op)] for op in operations]
    qi, ti, aq, at = int(qs), int(ts), [], []
    for op in ops:
        aq.append(query[qi] if op in "MI" else "-")
        at.append(target[ti] if op in "MD" else "-")
        qi += op in "MI"
        ti += op in "MD"
    cigar = "".join(f"{sum(1 for _ in group)}{op}" for op, group in groupby(ops))
    return Alignment(
        int(score), int(qs), int(qe), int(ts), int(te), cigar, "".join(aq), "".join(at)
    )
