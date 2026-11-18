"""Trip-local geometric context for ranking directed road candidates."""

import numpy as np


def relative_cost(cost):
    return cost - cost.min(axis=-1, keepdims=True)


def path_context(unary, transition):
    """Minimum path cost on either side of each point, conditional on its arc."""
    forward = np.zeros_like(unary)
    backward = np.zeros_like(unary)
    for i in range(1, len(unary)):
        before = forward[i - 1] + unary[i - 1]
        forward[i] = relative_cost((before[:, None] + transition[i - 1]).min(axis=0))
    for i in range(len(unary) - 2, -1, -1):
        after = backward[i + 1] + unary[i + 1]
        backward[i] = relative_cost((transition[i] + after[None, :]).min(axis=1))
    return forward, backward


def sequence_costs(ids, edges, x, times):
    distance = np.expm1(x[:, :, 0])
    length = np.expm1(x[:, :, 3])
    position = x[:, :, 2] * length
    dt = np.diff(times)
    motion = (np.expm1(x[1:, 0, 7]) * dt)[:, None, None]
    previous = position[:-1, :, None]
    current = position[1:, None, :]
    same = ids[:-1, :, None] == ids[1:, None, :]
    connected = edges[:-1, :, None, 1] == edges[1:, None, :, 0]
    travel = length[:-1, :, None] - previous + current
    # A missing connector in the candidate list is possible. Keep a finite cost.
    penalty = np.where(connected, 0.0, 5.0)
    transition = penalty + np.minimum(np.abs(travel - motion) / 12.0, 9.0)
    delta = current - previous
    stay_cost = np.abs(delta - motion) / 12.0 + np.maximum(-delta - 2.0, 0.0) / 5.0
    transition = np.where(same, np.minimum(stay_cost, 14.0), transition)
    transition[(dt > 10) | (dt <= 0)] = 0.0
    return distance, transition.astype(np.float32)


def temporal_features(ids, edges, x, times):
    """Only geometry, timing and ID equality enter the features."""
    distance, transition = sequence_costs(ids, edges, x, times)
    extra = []
    for noise, heading_weight in [(5.0, 0.25), (12.0, 1.0)]:
        unary = relative_cost(
            np.minimum(distance / noise, 24.0) + heading_weight * (1 - x[:, :, 6])
        )
        left, right = path_context(unary, transition)
        extra.extend([left, right, relative_cost(unary + left + right)])
    # Repeated evidence from the same directed arc, matched independently of rank.
    for lag in [-12, -4, -1, 1, 4, 12]:
        idx = np.clip(np.arange(len(x)) + lag, 0, len(x) - 1)
        match = ids[:, :, None] == ids[idx, None, :]
        found = (
            match.any(axis=2) & (np.abs(times[idx] - times) <= 4 * abs(lag))[:, None]
        )
        j = match.argmax(axis=2)
        for field in [0, 2, 6]:
            other = np.take_along_axis(x[idx, :, field], j, axis=1)
            value = other - x[:, :, field] if field == 2 else other
            extra.append(np.where(found, value, np.nan))
    return np.concatenate((x, np.stack(extra, axis=-1)), axis=-1).astype(np.float32)


def extended_context(ids, edges, x, times, gps):
    """Look past stationary samples using travelled distance and directed endpoints."""
    lat = np.radians(gps[:, 1].mean())
    xy = (gps - gps[0]) * (np.pi / 180 * 6371008.8) * np.array([np.cos(lat), 1.0])
    steps = np.diff(xy, axis=0)
    norm = np.linalg.norm(steps, axis=1)
    heading = steps / np.maximum(norm[:, None], 1e-12)
    # Recover each segment's direction from the observed GPS/segment cosines.
    # Least squares also handles straight tracks with just one identifiable axis.
    unique, index = np.unique(ids, return_inverse=True)
    index = index.reshape(ids.shape)
    systems = np.zeros((len(unique), 2, 2))
    rhs = np.zeros((len(unique), 2))
    for rows, field in [(slice(1, None), 4), (slice(None, -1), 5)]:
        inverse = index[rows].ravel()
        for dim in range(2):
            weight = heading[:, dim, None]
            rhs[:, dim] += np.bincount(
                inverse, (weight * x[rows, :, field]).ravel(), minlength=len(unique)
            )
            for other in range(2):
                pair = weight * heading[:, other, None]
                systems[:, dim, other] += np.bincount(
                    inverse,
                    np.broadcast_to(pair, index[rows].shape).ravel(),
                    minlength=len(unique),
                )
    directions = (np.linalg.pinv(systems, rcond=1e-5) @ rhs[..., None]).squeeze(-1)[
        index
    ]
    path_length = np.r_[0.0, np.cumsum(norm)]
    length = np.expm1(x[:, :, 3])
    extra = []
    for metres in (12.0, 35.0, 90.0):
        for sign in (-1, 1):
            idx = np.searchsorted(path_length, path_length + sign * metres).clip(
                0, len(x) - 1
            )
            delta = (xy[idx] - xy) * sign
            motion = np.linalg.norm(delta, axis=1)
            bearing = delta / np.maximum(motion[:, None], 1e-10)
            alignment = (directions * bearing[:, None]).sum(axis=-1).clip(-1, 1)
            available = length * (x[:, :, 2] if sign < 0 else 1 - x[:, :, 2])
            match = ids[:, :, None] == ids[idx, None, :]
            extra += [
                alignment,
                np.log1p(np.maximum(motion[:, None] * alignment - available, 0)),
                np.where(
                    match.any(axis=2),
                    np.take_along_axis(x[idx, :, 2], match.argmax(axis=2), axis=1)
                    - x[:, :, 2],
                    np.nan,
                ),
            ]
    for metres in (3.0, 12.0):
        dist, tr = sequence_costs(ids, edges, x, times)
        cost = relative_cost(np.minimum((dist / metres) ** 2, 40.0))
        left, right = path_context(cost, tr)
        extra.extend([left, right, relative_cost(cost + left + right)])
    return np.concatenate([x, np.stack(extra, axis=-1)], axis=-1).astype(np.float32)


def road_travel(ids, edges, x, times):
    """Directed path-length evidence on the graph induced by a trip's candidates."""
    import heapq

    nodes, keys = np.unique(edges, return_inverse=True)
    keys = keys.reshape(edges.shape)
    length = np.expm1(x[:, :, 3])
    _, first = np.unique(ids, return_index=True)
    adjacency = [[] for _ in nodes]
    for (a, b), metres in zip(keys.reshape(-1, 2)[first], length.ravel()[first]):
        adjacency[a].append((b, float(metres)))

    pairs = keys[:-1, :, None, 1] * len(nodes) + keys[1:, None, :, 0]
    unique, lookup = np.unique(pairs, return_inverse=True)
    source, target = np.divmod(unique, len(nodes))
    middle = np.full(len(unique), np.inf, np.float32)
    for group in np.split(np.arange(len(unique)), np.flatnonzero(np.diff(source)) + 1):
        if not len(group):
            continue
        start = source[group[0]]
        best = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > best[node]:
                continue
            for nxt, step in adjacency[node]:
                value = cost + step
                if value <= 250 and value < best.get(nxt, np.inf):
                    best[nxt] = value
                    heapq.heappush(queue, (value, nxt))
        middle[group] = [best.get(n, np.inf) for n in target[group]]
    middle = middle[lookup].reshape(pairs.shape)
    progress = length * x[:, :, 2]
    remaining = length[:-1] - progress[:-1]
    travel = remaining[:, :, None] + middle + progress[1:, None, :]
    dt = np.diff(times)
    motion = (np.expm1(x[1:, 0, 7]) * dt)[:, None, None]
    same = ids[:-1, :, None] == ids[1:, None, :]
    delta = progress[1:, None, :] - progress[:-1, :, None]
    travel = np.where(
        same & (delta >= -12), np.minimum(travel, np.maximum(delta, 0)), travel
    )
    return travel, motion, same, dt


def road_context(ids, edges, x, times):
    """Accumulate geometric path evidence at three spatial scales."""
    travel, motion, same, dt = road_travel(ids, edges, x, times)
    dist = np.expm1(x[:, :, 0])
    extra = []
    for error_scale, spatial_scale, switch_cost in [
        (5.0, 8.0, 0.4),
        (12.0, 4.0, 1.0),
        (3.0, 16.0, 0.1),
    ]:
        transition = np.minimum(
            abs(travel - motion) / error_scale, 20.0
        ) + switch_cost * (~same)
        transition[(dt > 15) | (dt <= 0)] = 0.0
        unary = relative_cost(
            np.minimum((dist / spatial_scale) ** 2, 60.0) + 0.25 * (1 - x[:, :, 6])
        )
        left, right = path_context(unary, transition.astype(np.float32))
        extra.extend([left, right, relative_cost(unary + left + right)])
    return np.concatenate((x, np.stack(extra, axis=-1)), axis=-1).astype(np.float32)


def align_prediction(ids, distances, logits, initial, times, strength):
    """Monotonically align GPS observations to a locally decoded arc sequence."""
    out = initial.copy()
    for obs in np.split(
        np.arange(len(ids)),
        np.flatnonzero((np.diff(times) > 10) | (np.diff(times) <= 0)) + 1,
    ):
        chosen = initial[obs]
        changes = np.r_[True, chosen[1:] != chosen[:-1]]
        route = chosen[changes]
        position = np.cumsum(changes) - 1
        offsets = np.arange(-8, 9)
        band = np.clip(position[:, None] + offsets, 0, len(route) - 1)
        matches = ids[obs, :, None] == route[band][:, None, :]
        rank = matches.argmax(axis=1)
        geometric = np.take_along_axis(distances[obs], rank, axis=1)
        learned = np.take_along_axis(logits[obs], rank, axis=1)
        emission = np.where(
            matches.any(axis=1),
            np.minimum(geometric**2, 2500.0) / 16.0 - strength * learned,
            np.inf,
        )
        previous = emission[0]
        back = np.zeros(band.shape, np.int16)
        for i in range(1, len(obs)):
            prefix = np.minimum.accumulate(previous)
            best = np.maximum.accumulate(
                np.where(previous == prefix, np.arange(len(offsets)), 0)
            )
            limit = np.searchsorted(band[i - 1], band[i], side="right") - 1
            valid = limit >= 0
            back[i] = best[limit.clip(0)]
            previous = np.where(valid, previous[back[i]] + emission[i], np.inf)
        j = int(previous.argmin())
        for i in range(len(obs) - 1, -1, -1):
            out[obs[i]] = route[band[i, j]]
            j = back[i, j]
    return out


def score_context(ids, edges, x, times, logits):
    """Context from a first-stage ranker that has not seen the trip labels."""
    _, tr = sequence_costs(ids, edges, x, times)
    cost = relative_cost(-logits).astype(np.float32)
    extras = [cost]
    for weight in (0.35, 1.5, 4.0):
        left, right = path_context(cost, weight * tr)
        extras.extend([left, right, relative_cost(left + right + cost)])
    # Within a window, aggregate evidence only across occurrences of the same arc.
    for direction in (-1, 1):
        total = np.zeros_like(cost)
        count = np.zeros_like(cost)
        best = np.full_like(cost, np.inf)
        for lag in (1, 2, 4, 8, 16, 32, 64):
            idx = np.clip(np.arange(len(x)) + direction * lag, 0, len(x) - 1)
            match = ids[:, :, None] == ids[idx, None, :]
            valid = match.any(axis=2) & (np.abs(times[idx] - times) <= 4 * lag)[:, None]
            scores = np.take_along_axis(cost[idx], match.argmax(axis=2), axis=1)
            total += np.where(valid, scores, 0)
            count += valid
            best = np.minimum(best, np.where(valid, scores, np.inf))
        extras.extend(
            [total / np.maximum(count, 1), count, np.where(count, best, np.nan)]
        )
    return np.concatenate([x, np.stack(extras, axis=-1)], axis=-1).astype(np.float32)


def geometry_features(ids, edges, base, times, gps):
    local = temporal_features(ids, edges, base, times)
    extended = extended_context(ids, edges, local, times, gps)
    return road_context(ids, edges, extended, times)


def rank_scores(model, x):
    return model.decision_function(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])


def choose_arcs(bundle, ids, edges, x, times):
    """Apply the frozen ranking and path settings to one complete trip."""
    from scipy.special import log_softmax

    scores = rank_scores(bundle["model"], x)
    if "second_stage" in bundle:
        x = score_context(ids, edges, x, times, scores)
        scores = rank_scores(bundle["second_stage"], x)
    distance, transition = sequence_costs(ids, edges, x, times)
    cost = -log_softmax(scores, axis=1)
    if bundle["weight"]:
        left, right = path_context(cost, bundle["weight"] * transition)
        cost = cost + left + right
    chosen = ids[np.arange(len(ids)), cost.argmin(axis=1)]
    if bundle["alignment"] is not None:
        chosen = align_prediction(
            ids, distance, scores, chosen, times, bundle["alignment"]
        )
    return chosen
