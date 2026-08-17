"""PUCT Monte-Carlo Tree Search, in the AlphaZero formulation.

The search never enumerates moves itself: it always asks
:func:`app.engine.encoding.legal_move_indices` for the legal move set, so the
rules of chess constrain the search structurally. The network only supplies a
*prior* over that legal set plus a value estimate.

Two knobs make this the same code path for both play styles the project needs:

``simulations = 1``
    The root is evaluated once and the highest-prior legal move is played.
    This is "pure policy network, no search".

``simulations = N``
    Full PUCT search: ``N`` evaluations, visit counts as the improved policy.

Leaves are gathered in batches (with virtual loss to keep the collectors from
piling onto the same branch) so one forward pass serves many simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np

from app.config import SearchConfig
from app.engine.encoding import PositionHistory, legal_move_indices
from app.engine.evaluator import Evaluator, masked_softmax
from app.engine.rules import Termination, terminal_state


class Node:
    """One position in the search tree."""

    __slots__ = (
        "move",
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "virtual_loss",
        "terminal",
    )

    def __init__(self, move: Optional[chess.Move], prior: float) -> None:
        self.move = move
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Optional[Dict[int, "Node"]] = None
        self.virtual_loss = 0
        self.terminal: Optional[Termination] = None

    @property
    def is_expanded(self) -> bool:
        return self.children is not None

    @property
    def q(self) -> float:
        """Mean value from the point of view of the side to move *at this node*.

        Pending virtual losses count as wins *for this node's mover*, i.e. as
        losses for the parent that is choosing between siblings. That is what
        steers concurrent leaf collectors away from the branch.
        """
        visits = self.visit_count + self.virtual_loss
        if visits == 0:
            return 0.0
        return (self.value_sum + self.virtual_loss) / visits

    @property
    def effective_visits(self) -> float:
        return self.visit_count + self.virtual_loss


@dataclass
class SearchResult:
    """Everything a caller (self-play, arena, or the API) might want."""

    best_move: Optional[chess.Move]
    best_index: Optional[int]
    visit_counts: Dict[int, int] = field(default_factory=dict)
    policy_target: Dict[int, float] = field(default_factory=dict)
    move_by_index: Dict[int, chess.Move] = field(default_factory=dict)
    root_value: float = 0.0
    simulations: int = 0
    nodes_created: int = 0
    principal_variation: List[chess.Move] = field(default_factory=list)
    terminal: Optional[Termination] = None
    used_search: bool = True

    def top_moves(self, limit: int = 5) -> List[Tuple[chess.Move, float, int]]:
        ranked = sorted(
            self.policy_target.items(), key=lambda kv: kv[1], reverse=True
        )[:limit]
        return [
            (self.move_by_index[i], p, self.visit_counts.get(i, 0)) for i, p in ranked
        ]


class MCTS:
    """A reusable searcher. One instance per worker / per game is fine."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: Optional[SearchConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = rng or np.random.default_rng()
        self._nodes_created = 0

    # -- public API -------------------------------------------------------- #

    def search(
        self,
        state: PositionHistory,
        *,
        add_noise: bool = False,
        simulations: Optional[int] = None,
    ) -> SearchResult:
        """Run the search on ``state`` without mutating it."""
        budget = simulations if simulations is not None else self.config.simulations
        budget = max(1, int(budget))
        self._nodes_created = 0

        termination = terminal_state(state)
        if termination is not None:
            return SearchResult(
                best_move=None,
                best_index=None,
                root_value=termination.value,
                simulations=0,
                terminal=termination,
                used_search=False,
            )

        root = Node(move=None, prior=1.0)
        legal_map = legal_move_indices(state.board)
        if not legal_map:  # defensive: should be covered by terminal_state
            return SearchResult(None, None, root_value=0.0, used_search=False)

        planes = state.encode()
        logits, root_value = self.evaluator.evaluate(planes)
        self._expand(root, legal_map, logits)
        root.visit_count = 1
        root.value_sum = root_value
        simulations_done = 1

        if add_noise:
            self._apply_dirichlet_noise(root)

        while simulations_done < budget:
            remaining = budget - simulations_done
            batch_target = min(self.config.max_batch_size, remaining)
            leaves: List[Tuple[List[Node], Dict[int, chess.Move]]] = []
            planes_batch: List[np.ndarray] = []

            for _ in range(batch_target):
                path = self._select(root, state)
                leaf = path[-1]

                leaf_termination = terminal_state(state)
                if leaf_termination is not None:
                    leaf.terminal = leaf_termination
                    self._backup(path, leaf_termination.value)
                    self._unwind(state, len(path) - 1)
                    simulations_done += 1
                    continue

                leaf_legal = legal_move_indices(state.board)
                planes_batch.append(state.encode())
                leaves.append((path, leaf_legal))
                self._unwind(state, len(path) - 1)

            if planes_batch:
                evaluations = self.evaluator.evaluate_many(planes_batch)
                for (path, leaf_legal), (logits, value) in zip(leaves, evaluations):
                    leaf = path[-1]
                    if not leaf.is_expanded:
                        self._expand(leaf, leaf_legal, logits)
                    self._backup(path, value)
                    simulations_done += 1

            if not planes_batch and batch_target == 0:
                break

        return self._make_result(root, state, simulations_done, root_value)

    def select_move(
        self,
        state: PositionHistory,
        *,
        add_noise: bool = False,
        temperature: Optional[float] = None,
        simulations: Optional[int] = None,
    ) -> Tuple[Optional[chess.Move], SearchResult]:
        """Search, then pick a move (sampling if ``temperature > 0``)."""
        result = self.search(state, add_noise=add_noise, simulations=simulations)
        if result.best_move is None:
            return None, result

        if temperature is None:
            temperature = (
                self.config.temperature
                if state.ply < self.config.temperature_moves
                else 0.0
            )
        if temperature <= 1e-6:
            return result.best_move, result

        indices = list(result.policy_target.keys())
        weights = np.array([result.policy_target[i] for i in indices], dtype=np.float64)
        weights = np.power(np.maximum(weights, 1e-12), 1.0 / temperature)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return result.best_move, result
        weights /= total
        chosen = int(self.rng.choice(len(indices), p=weights))
        return result.move_by_index[indices[chosen]], result

    # -- internals --------------------------------------------------------- #

    def _expand(
        self,
        node: Node,
        legal_map: Dict[int, chess.Move],
        logits: np.ndarray,
    ) -> None:
        indices = list(legal_map.keys())
        priors = masked_softmax(logits, indices)
        node.children = {
            index: Node(move=legal_map[index], prior=float(prior))
            for index, prior in zip(indices, priors)
        }
        self._nodes_created += len(node.children)

    def _apply_dirichlet_noise(self, root: Node) -> None:
        if root.children is None or not root.children:
            return
        epsilon = self.config.dirichlet_epsilon
        if epsilon <= 0.0:
            return
        alpha = self.config.dirichlet_alpha
        noise = self.rng.dirichlet([alpha] * len(root.children))
        for child, sample in zip(root.children.values(), noise):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * float(sample)

    def _c_puct(self, parent_visits: int) -> float:
        base = self.config.c_puct_base
        if base and base > 0:
            return (
                math.log((parent_visits + base + 1.0) / base) + self.config.c_puct
            )
        return self.config.c_puct

    def _select(self, root: Node, state: PositionHistory) -> List[Node]:
        """Descend to an unexpanded leaf, pushing moves onto ``state``.

        Virtual loss is applied on the way down so that concurrently collected
        leaves in the same batch explore different branches.
        """
        path: List[Node] = [root]
        node = root
        virtual = self.config.virtual_loss

        while node.is_expanded and node.children:
            index = self._best_child_index(node)
            child = node.children[index]
            state.push(child.move)
            node.virtual_loss += virtual
            node = child
            path.append(node)
            if node.terminal is not None:
                break

        node.virtual_loss += virtual
        return path

    def _best_child_index(self, node: Node) -> int:
        assert node.children is not None
        total_visits = max(node.effective_visits, 1)
        sqrt_total = math.sqrt(total_visits)
        c_puct = self._c_puct(total_visits)

        # Scores below are all in the *parent's* frame. An unvisited child is
        # optimistically assumed to be worth what the parent is already worth,
        # minus a pessimism margin (First Play Urgency reduction).
        parent_q = node.q
        fpu = parent_q - self.config.fpu_reduction

        best_index = -1
        best_score = -float("inf")
        for index, child in node.children.items():
            visits = child.effective_visits
            if visits == 0:
                q_value = fpu
            else:
                # child.q is from the child's mover perspective; negate it.
                q_value = -child.q
            score = q_value + c_puct * child.prior * sqrt_total / (1 + visits)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _backup(self, path: Sequence[Node], value: float) -> None:
        """Propagate ``value`` (leaf's perspective) up the path, flipping sign."""
        virtual = self.config.virtual_loss
        signed = value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += signed
            node.virtual_loss -= virtual
            signed = -signed

    @staticmethod
    def _unwind(state: PositionHistory, count: int) -> None:
        for _ in range(count):
            state.pop()

    def _make_result(
        self,
        root: Node,
        state: PositionHistory,
        simulations_done: int,
        root_value: float,
    ) -> SearchResult:
        assert root.children is not None
        visit_counts = {i: c.visit_count for i, c in root.children.items()}
        move_by_index = {i: c.move for i, c in root.children.items()}
        total_visits = sum(visit_counts.values())

        if total_visits > 0:
            policy_target = {i: v / total_visits for i, v in visit_counts.items()}
            best_index = max(
                visit_counts,
                key=lambda i: (visit_counts[i], -root.children[i].q),  # type: ignore[index]
            )
            used_search = True
        else:
            # simulations == 1: no child was ever visited, so the network's raw
            # prior *is* the policy. This is the "policy-only" mode.
            policy_target = {i: c.prior for i, c in root.children.items()}
            best_index = max(policy_target, key=policy_target.get)  # type: ignore[arg-type]
            used_search = False

        return SearchResult(
            best_move=move_by_index[best_index],
            best_index=best_index,
            visit_counts=visit_counts,
            policy_target=policy_target,
            move_by_index=move_by_index,
            root_value=root.q if total_visits > 0 else root_value,
            simulations=simulations_done,
            nodes_created=self._nodes_created,
            principal_variation=self._principal_variation(root),
            used_search=used_search,
        )

    @staticmethod
    def _principal_variation(root: Node, max_depth: int = 12) -> List[chess.Move]:
        line: List[chess.Move] = []
        node = root
        for _ in range(max_depth):
            if not node.children:
                break
            best = max(node.children.values(), key=lambda c: c.visit_count)
            if best.visit_count == 0 or best.move is None:
                break
            line.append(best.move)
            node = best
        return line
