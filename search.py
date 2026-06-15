"""
The classic search algorithms, in a generic form.
They know nothing about chemistry: we hand them a "problem" (an initial state,
successors, a goal test, a heuristic) and they search.

We have Dijkstra (A* with h=0), A* and IDA*.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Optional


class Problem:
    """Interface of a search problem. A state must be hashable."""

    def initial_state(self) -> Hashable:
        raise NotImplementedError

    def is_goal(self, state: Hashable) -> bool:
        raise NotImplementedError

    def successors(self, state: Hashable) -> Iterable[tuple[Any, Hashable, float]]:
        # returns triples (action, next state, step cost)
        raise NotImplementedError

    def heuristic(self, state: Hashable) -> float:
        return 0.0


@dataclass
class SearchResult:
    """What a search returns, so we can compare the algorithms against each other."""
    found: bool
    cost: float
    plan: list[Any] = field(default_factory=list)
    nodes_expanded: int = 0
    nodes_generated: int = 0
    elapsed: float = 0.0
    algo: str = ""

    def __str__(self) -> str:
        status = "SUCCESS" if self.found else "FAILURE"
        return (f"[{self.algo:<13}] {status} | cost={self.cost:g} "
                f"| expanded={self.nodes_expanded:>7} "
                f"| generated={self.nodes_generated:>7} "
                f"| time={self.elapsed * 1000:8.2f} ms")


def astar(problem: Problem,
          use_heuristic: bool = True,
          name: Optional[str] = None) -> SearchResult:
    """Plain A*. With use_heuristic=False we set h=0, so it becomes Dijkstra."""
    algo = name or ("A*" if use_heuristic else "Dijkstra")
    h = problem.heuristic if use_heuristic else (lambda s: 0.0)

    start = problem.initial_state()
    t0 = time.perf_counter()

    # priority queue ordered on f = g + h, the counter breaks ties
    counter = itertools.count()
    open_list = [(h(start), next(counter), 0.0, start)]
    best_g = {start: 0.0}
    parent: dict[Hashable, tuple[Hashable, Any]] = {}
    closed: set[Hashable] = set()

    generated, expanded = 1, 0

    while open_list:
        f, _, g, state = heapq.heappop(open_list)

        # the same state may have been pushed several times, skip stale copies
        if state in closed or g > best_g.get(state, float("inf")):
            continue

        expanded += 1
        if problem.is_goal(state):
            return SearchResult(True, g, _reconstruct(parent, state),
                                expanded, generated, time.perf_counter() - t0, algo)
        closed.add(state)

        for action, succ, cost in problem.successors(state):
            ng = g + cost
            if ng < best_g.get(succ, float("inf")):
                best_g[succ] = ng
                parent[succ] = (state, action)
                hf = h(succ)
                if hf == float("inf"):
                    continue  # unreachable molecule, prune this branch
                heapq.heappush(open_list, (ng + hf, next(counter), ng, succ))
                generated += 1

    return SearchResult(False, float("inf"), [], expanded, generated,
                        time.perf_counter() - t0, algo)


def _reconstruct(parent: dict, goal: Hashable) -> list[Any]:
    """Walk the parent pointers back up to recover the sequence of actions."""
    actions = []
    state = goal
    while state in parent:
        prev, action = parent[state]
        actions.append(action)
        state = prev
    actions.reverse()
    return actions


def ida_star(problem: Problem, name: str = "IDA*") -> SearchResult:
    """
    IDA*: repeated depth-first searches that cut off as soon as f exceeds a
    threshold. The threshold starts at h(root) then becomes the smallest f we cut.
    """
    start = problem.initial_state()
    t0 = time.perf_counter()

    stats = {"exp": 0, "gen": 1}
    path: list[Any] = []
    on_branch = {start}  # to avoid looping on the current branch

    def dfs(state: Hashable, g: float, threshold: float) -> float:
        # returns -1 if the goal is found, else the smallest f that went over the threshold
        stats["exp"] += 1
        f = g + problem.heuristic(state)
        if f > threshold:
            return f
        if problem.is_goal(state):
            return -1.0

        best = float("inf")
        for action, succ, cost in problem.successors(state):
            if succ in on_branch:
                continue
            on_branch.add(succ)
            path.append(action)
            stats["gen"] += 1

            t = dfs(succ, g + cost, threshold)
            if t == -1.0:
                return -1.0
            best = min(best, t)

            path.pop()
            on_branch.discard(succ)
        return best

    threshold = problem.heuristic(start)
    while True:
        if threshold == float("inf"):
            return SearchResult(False, float("inf"), [], stats["exp"], stats["gen"],
                                time.perf_counter() - t0, name)
        t = dfs(start, 0.0, threshold)
        if t == -1.0:
            return SearchResult(True, float(len(path)), list(path),
                                stats["exp"], stats["gen"],
                                time.perf_counter() - t0, name)
        threshold = t
