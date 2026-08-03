# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import logging
import math
import multiprocessing as mp
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.dp_cost_model import (
    DeepSeekV4PrefillCostModel,
)
from sglang.srt.managers.dp_prefix_cache import DPPrefixCacheTracker
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    ClearHiCacheReqInput,
    ElasticScaleUpdateReq,
    FlushCacheReqInput,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
    sock_send,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.runtime_context import get_exec, publish
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

SCHEDULER_PIDS_ARG = "scheduler_pids"


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()
    COST_AWARE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


@dataclass(frozen=True, slots=True)
class PendingDispatch:
    seq: int
    estimated_tokens: int
    estimated_cost_s: float


def waterfill_cost_aware_batch(
    *,
    base_loads: Sequence[float],
    request_costs: Sequence[Sequence[float]],
    request_counts: Sequence[int],
    eligible_ranks: Sequence[int],
    max_request_skew: int,
    min_fanout: int,
    target_slack: float,
) -> list[int]:
    """Assign a micro-batch while bounding count and projected-finish skew."""
    dp_size = len(base_loads)
    if len(request_counts) != dp_size:
        raise ValueError("request_counts must match base_loads")
    ranks = sorted(set(eligible_ranks))
    if not ranks or any(rank < 0 or rank >= dp_size for rank in ranks):
        raise ValueError("eligible_ranks must contain valid DP ranks")
    if max_request_skew < 1:
        raise ValueError("max_request_skew must be at least 1")
    if min_fanout < 0:
        raise ValueError("min_fanout cannot be negative")
    if target_slack < 0:
        raise ValueError("target_slack cannot be negative")
    for costs in request_costs:
        if len(costs) != dp_size:
            raise ValueError("every request cost row must match base_loads")

    assignments = [-1] * len(request_costs)
    if not assignments:
        return assignments

    shadow_loads = list(base_loads)
    shadow_counts = list(request_counts)
    min_cost_sum = sum(min(costs[rank] for rank in ranks) for costs in request_costs)
    target = max(
        max(shadow_loads[rank] for rank in ranks),
        (
            sum(shadow_loads[rank] for rank in ranks) + min_cost_sum
        )
        / len(ranks),
    )
    target *= 1.0 + target_slack

    fanout_goal = min(
        len(assignments),
        len(ranks),
        min_fanout if min_fanout > 0 else len(ranks),
    )
    used_ranks: set[int] = set()
    request_order = sorted(
        range(len(request_costs)),
        key=lambda request_index: (
            max(request_costs[request_index][rank] for rank in ranks)
            - min(request_costs[request_index][rank] for rank in ranks),
            max(request_costs[request_index][rank] for rank in ranks),
            -request_index,
        ),
        reverse=True,
    )

    for request_index in request_order:
        costs = request_costs[request_index]
        min_count = min(shadow_counts[rank] for rank in ranks)
        candidates = [
            rank
            for rank in ranks
            if shadow_counts[rank] + 1 <= min_count + max_request_skew
        ]
        if not candidates:
            candidates = [
                rank for rank in ranks if shadow_counts[rank] == min_count
            ]

        if len(used_ranks) < fanout_goal:
            unseen = [rank for rank in candidates if rank not in used_ranks]
            if unseen:
                candidates = unseen

        fitting = [
            rank
            for rank in candidates
            if shadow_loads[rank] + costs[rank] <= target
        ]
        pool = fitting or candidates
        target_rank = min(
            pool,
            key=lambda rank: (
                shadow_loads[rank] + costs[rank],
                shadow_counts[rank],
                rank,
            ),
        )
        assignments[request_index] = target_rank
        shadow_loads[target_rank] += costs[target_rank]
        shadow_counts[target_rank] += 1
        used_ranks.add(target_rank)
        if not fitting:
            target = max(target, shadow_loads[target_rank])

    return assignments


class DPBudget:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        self.prefill_cost = [0.0] * dp_size
        self._reported_total_requests = [0] * dp_size
        self._reported_total_tokens = [0] * dp_size
        self._reported_prefill_cost = [0.0] * dp_size
        self.unacked_dispatches: list[deque[PendingDispatch]] = [
            deque() for _ in range(dp_size)
        ]
        self.last_dispatch_seq = [0] * dp_size
        self.last_timestamp = [0.0] * dp_size

    def update_budget(self, loads):
        """Reconcile scheduler snapshots without dropping unacknowledged work."""
        for load in loads:
            rank = load.dp_rank
            if rank < 0 or rank >= self.dp_size:
                continue
            if load.timestamp <= self.last_timestamp[rank]:
                continue
            self.last_timestamp[rank] = load.timestamp
            self._reported_total_requests[rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self._reported_total_tokens[rank] = load.num_total_tokens
            self._reported_prefill_cost[rank] = load.prefill_cost_s

            accepted_seq = max(0, load.last_accepted_dispatch_seq)
            pending = self.unacked_dispatches[rank]
            while pending and pending[0].seq <= accepted_seq:
                pending.popleft()

            self.total_requests[rank] = self._reported_total_requests[rank] + len(
                pending
            )
            self.total_tokens[rank] = self._reported_total_tokens[rank] + sum(
                item.estimated_tokens for item in pending
            )
            self.prefill_cost[rank] = self._reported_prefill_cost[rank] + sum(
                item.estimated_cost_s for item in pending
            )

    def record_dispatch(
        self,
        rank: int,
        *,
        estimated_tokens: int = 0,
        estimated_cost_s: float = 0.0,
    ) -> int:
        if rank < 0 or rank >= self.dp_size:
            raise ValueError(f"DP rank {rank} is outside [0, {self.dp_size})")
        self.last_dispatch_seq[rank] += 1
        self.unacked_dispatches[rank].append(
            PendingDispatch(
                seq=self.last_dispatch_seq[rank],
                estimated_tokens=estimated_tokens,
                estimated_cost_s=estimated_cost_s,
            )
        )
        self.total_requests[rank] += 1
        self.total_tokens[rank] += estimated_tokens
        self.prefill_cost[rank] += estimated_cost_s
        return self.last_dispatch_seq[rank]

    def dispatch(
        self,
        method: LoadBalanceMethod,
        *,
        estimated_tokens: int = 0,
        estimated_cost_s: float = 0.0,
        estimated_costs_s: Sequence[float] | None = None,
    ):
        selected_cost_s = estimated_cost_s
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        elif method == LoadBalanceMethod.COST_AWARE:
            if estimated_costs_s is None:
                estimated_costs_s = [estimated_cost_s] * self.dp_size
            if len(estimated_costs_s) != self.dp_size:
                raise ValueError(
                    "Rank-specific costs must match dp_size: "
                    f"{len(estimated_costs_s)} != {self.dp_size}"
                )
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (
                    self.prefill_cost[i] + estimated_costs_s[i],
                    self.total_requests[i],
                ),
            )
            selected_cost_s = estimated_costs_s[target_rank]
        else:
            return None

        # Keep speculative load until a scheduler snapshot acknowledges it.
        self.record_dispatch(
            target_rank,
            estimated_tokens=estimated_tokens,
            estimated_cost_s=selected_cost_s,
        )
        return target_rank


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func
        self.dp_cost_dispatch_window_s = (
            server_args.dp_cost_dispatch_window_ms / 1000.0
        )
        self.dp_cost_target_slack = server_args.dp_cost_target_slack
        self.dp_cost_max_request_skew = server_args.dp_cost_max_request_skew
        self.dp_cost_min_fanout = server_args.dp_cost_min_fanout
        self.dp_cost_promised_cache_discount = (
            server_args.dp_cost_promised_cache_discount
        )

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
            LoadBalanceMethod.COST_AWARE: self.cost_aware_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
            LoadBalanceMethod.COST_AWARE,
        )

        self.launch_dp_size: int = server_args.dp_size
        self.max_dp_size: int = server_args.max_ep_size or server_args.dp_size
        assert self.max_dp_size >= self.launch_dp_size, (
            f"--max-ep-size ({self.max_dp_size}) must be >= "
            f"--dp ({self.launch_dp_size})."
        )

        self.dp_active: List[bool] = [True] * self.launch_dp_size + [False] * (
            self.max_dp_size - self.launch_dp_size
        )

        self.dp_budget = DPBudget(server_args.dp_size)
        self.load_snapshot_reader = create_load_snapshot_reader(
            server_args,
            port_args,
            caller="DataParallelController",
        )
        self._last_refresh_time = 0.0

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.dp_cost_model_spec = None
        self.dp_prefix_cache = None
        self.workers: List[Optional[zmq.Socket]] = [None] * self.max_dp_size
        self.status: List[bool] = list(self.dp_active)
        self._active_workers: List[int] = list(range(self.launch_dp_size))
        self._active_count_cache: int = self.launch_dp_size

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            # When local control broadcast is enabled, send control messages to
            # every DP group leader (attn_tp_rank=0) so each leader broadcasts
            # within its own attn_tp_group instead of the full tp_group.
            # Otherwise fall back to the original behaviour: send to only the
            # first leader, which then broadcasts over the full tp_group.
            local_ctrl = server_args.enable_dp_attention_local_control_broadcast
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.dp_cost_model = None
        if self.load_balance_method == LoadBalanceMethod.COST_AWARE:
            if self.dp_cost_model_spec is None:
                raise RuntimeError(
                    "Cost-aware DP dispatch did not receive a cost model from "
                    "the scheduler"
                )
            self.dp_cost_model = DeepSeekV4PrefillCostModel.from_spec(
                self.dp_cost_model_spec
            )
            page_size = server_args.page_size
            device_pages = max(1, self.max_total_num_tokens // page_size)
            host_pages = 0
            if server_args.enable_hierarchical_cache:
                if server_args.hicache_size > 0:
                    bytes_per_token = (
                        self.dp_cost_model.full_cache_bytes_per_input_token
                        + self.dp_cost_model.swa_cache_bytes_per_input_token
                    )
                    host_tokens = int(server_args.hicache_size * 1e9 / bytes_per_token)
                    host_pages = max(1, host_tokens // page_size)
                else:
                    host_pages = max(1, int(device_pages * server_args.hicache_ratio))
            storage_pages = (
                host_pages if server_args.hicache_storage_backend is not None else 0
            )
            self.dp_prefix_cache = DPPrefixCacheTracker(
                dp_size=server_args.dp_size,
                page_size=page_size,
                device_pages_per_rank=device_pages,
                host_pages_per_rank=host_pages,
                storage_pages_per_rank=storage_pages,
                host_write_through=(
                    server_args.hicache_write_policy == "write_through"
                ),
            )

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        for i, worker in enumerate(self.workers):
            if worker is not None and self.status[i]:
                sock_send(worker, obj)

    def send_control_message(self, obj):
        if self.dp_prefix_cache is not None and isinstance(
            obj, (FlushCacheReqInput, ClearHiCacheReqInput)
        ):
            self.dp_prefix_cache.clear()
        for i in self._active_workers[:: self.control_message_step]:
            worker = self.workers[i]
            if worker is not None:
                sock_send(worker, obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        if get_exec().moe.elastic_ep_backend is not None:
            if len(ranks.status) != self.max_dp_size:
                logger.warning(
                    "[Elastic EP][DPC] active rank status len=%d != max_dp_size=%d; "
                    "ignoring update",
                    len(ranks.status),
                    self.max_dp_size,
                )
                return
            self.status = [
                self.dp_active[i] and bool(ranks.status[i])
                for i in range(self.max_dp_size)
            ]
            self._refresh_active_workers()
            return
        if len(ranks.status) != self.max_dp_size:
            logger.warning(
                "[DPC] update_active_ranks: status len=%d != max_dp_size=%d; "
                "ignoring update",
                len(ranks.status),
                self.max_dp_size,
            )
            return
        self.status = list(ranks.status)
        self._refresh_active_workers()

    def add_elastic_workers(self, slot_offset: int, slot_count: int):
        """Activate a range of pre-bound worker slots."""
        end = slot_offset + slot_count
        if end > self.max_dp_size:
            raise ValueError(
                f"[Elastic EP] add_elastic_workers: slot_offset={slot_offset} + "
                f"slot_count={slot_count} exceeds max_dp_size={self.max_dp_size}. "
                f"Restart with a larger --max-ep-size."
            )

        for slot in range(slot_offset, end):
            if self.dp_active[slot]:
                logger.debug(
                    "[Elastic EP] add_elastic_workers: slot %d already active; "
                    "skipping",
                    slot,
                )
                continue
            assert self.workers[slot] is not None, (
                f"[Elastic EP] add_elastic_workers: slot {slot} was not "
                f"pre-bound at launch; expected a primary-bound PUSH socket."
            )
            self.dp_active[slot] = True
            self.status[slot] = True

        self._refresh_active_workers()
        logger.debug(
            "[Elastic EP] DataParallelController activated slots %s "
            "(active=%d / max=%d)",
            list(range(slot_offset, end)),
            self._active_count_cache,
            self.max_dp_size,
        )

    def _refresh_active_workers(self) -> None:
        self._active_workers = [
            i for i, active in enumerate(self.dp_active) if active and self.status[i]
        ]
        self._active_count_cache = len(self._active_workers)

    def refresh_load_budget(self):
        # Limit SHM/ZMQ polling overhead. DPBudget reconciles every accepted
        # snapshot with its unacknowledged dispatch ledger, so refreshes cannot
        # erase requests that have not reached the scheduler yet.
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.02:
            return
        self._last_refresh_time = now
        self.dp_budget.update_budget(self.load_snapshot_reader.read_all())

    def dispatching_with_trace(self, req: Req, refresh_load_budget: bool = True):
        if refresh_load_budget and self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()

        time_stats = DPControllerReqTimeStats.new_from_obj(
            unwrap_from_pickle(req.time_stats)
        )

        time_stats.set_dp_dispatch_time()
        req.time_stats = wrap_as_pickle(time_stats)
        self.dispatching(req)
        req.time_stats = time_stats
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (ActiveRanksOutput, self.update_active_ranks),
                (
                    ElasticScaleUpdateReq,
                    lambda msg: self.add_elastic_workers(
                        msg.slot_offset, msg.slot_count
                    ),
                ),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            tmp_port_args.instance_id = port_args.instance_id

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        is_joiner = server_args.is_ep_scale_joiner
        if server_args.dist_init_addr is None or is_joiner:
            na = NetworkAddress(
                server_args.host or "127.0.0.1",
                server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(server_args.dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = sock_recv(rep_socket)
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                sock_send(rep_socket, wrap_as_pickle(worker_ports))
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            if get_exec().moe.elastic_ep_backend is None:
                rep_socket.close()
            else:
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: List[int]):
        """Background thread: serve the pre-bound worker-port list to
        late-arriving elastic joiners. Publishes port numbers only; the primary
        keeps ownership of every socket."""
        while True:
            try:
                client_rank = sock_recv(rep_socket)
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            # Send worker ports to client
            sock_send(rep_socket, wrap_as_pickle(worker_ports))
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            sock_send(req_socket, wrap_as_pickle(str(node_rank)))

            # Receive worker ports
            worker_ports = sock_recv(req_socket)
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def _joiner_local_tp_span(self, server_args: ServerArgs) -> int:
        return server_args.tp_size

    def _joiner_slot_offset(self, server_args: ServerArgs) -> int:
        return server_args.ep_join_rank_offset

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        if server_args.dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(server_args.dist_init_addr).host

        worker_ports = []
        if server_args.is_ep_scale_joiner:
            # Scale joiners connect to their pre-bound primary worker sockets.
            primary = NetworkAddress.parse(server_args.dist_init_addr)
            primary_endpoint = NetworkAddress(
                primary.host, primary.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA
            ).to_tcp()
            all_ports = self._receive_ports_as_client(
                primary_endpoint, server_args.node_rank
            )
            offset = self._joiner_slot_offset(server_args)
            local_tp_span = self._joiner_local_tp_span(server_args)
            broadcasted_ports = all_ports[offset : offset + local_tp_span]
        elif server_args.node_rank == 0:
            # Elastic primaries reserve sockets for the maximum DP size.
            bind_count = (
                self.max_dp_size
                if server_args.elastic_ep_backend is not None
                else server_args.dp_size
            )
            for slot in range(bind_count):
                worker_port, worker_socket = get_zmq_socket_on_host(
                    self.context, zmq.PUSH, host=bind_host
                )
                worker_ports.append(worker_port)
                self.workers[slot] = worker_socket
                logger.debug(
                    "Assigned port %s to worker slot %s on host %s",
                    worker_port,
                    slot,
                    bind_host,
                )
            broadcasted_ports = self._broadcast_worker_ports(server_args, worker_ports)
        else:
            broadcasted_ports = self._broadcast_worker_ports(server_args, None)

        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        if server_args.is_ep_scale_joiner:
            # Scale joiners enumerate their full local TP span.
            tp_rank_range = range(server_args.tp_size)
            tp_size_per_node = server_args.tp_size
        else:
            tp_rank_range = range(
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
            )

        attn_cp_rank = 0
        moe_dp_rank = 0
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank, _ = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    if server_args.is_ep_scale_joiner:
                        # Scale-joiner outputs return through the primary tokenizer.
                        primary_addr = NetworkAddress.parse(server_args.dist_init_addr)
                        primary_port_base = primary_addr.port + 1
                        rank_port_args.tokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base
                        ).to_tcp()
                        rank_port_args.detokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base + 1
                        ).to_tcp()
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port
                    rank_port_args.instance_id = port_args.instance_id

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_dp_size = (
                    server_args.dp_size if server_args.enable_dp_attention else 1
                )

                # Parallelism hierarchy (outermost to innermost):
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                # Scheduler internals use local ranks; logs use global ranks.
                offset = server_args.ep_join_rank_offset
                display_tp_rank = tp_rank + offset
                display_moe_ep_rank = moe_ep_rank + offset
                display_dp_rank = dp_rank + offset if dp_rank is not None else None

                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                            display_tp_rank,
                            display_dp_rank,
                            display_moe_ep_rank,
                        ),
                    )
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]
        cost_model_spec = scheduler_info[0].get("dp_cost_model_spec")
        if cost_model_spec is not None:
            if (
                self.dp_cost_model_spec is not None
                and self.dp_cost_model_spec != cost_model_spec
            ):
                raise RuntimeError(
                    "DP schedulers returned inconsistent cost-model specs"
                )
            self.dp_cost_model_spec = cost_model_spec

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.routed_dp_rank is not None:
            rank = req.routed_dp_rank
            if (
                rank < 0
                or rank >= len(self.workers)
                or rank not in self._active_workers
                or self.workers[rank] is None
            ):
                raise ValueError(f"DP rank {rank} is not active.")
            logger.debug(f"Direct routing to DP rank {rank}")
            sock_send(self.workers[rank], req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        active = self._active_workers
        if not active:
            raise RuntimeError("No active DP workers are available for routing.")
        attempts = 0
        while attempts < len(active):
            slot = active[self.round_robin_counter % len(active)]
            self.round_robin_counter = (self.round_robin_counter + 1) % len(active)
            if self.status[slot]:
                logger.debug(f"Choose worker {slot}")
                sock_send(self.workers[slot], req)
                return
            attempts += 1
        raise RuntimeError(
            f"Cannot route request: all {len(active)} active DP workers "
            "are unavailable."
        )

    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        sock_send(self.workers[target_rank], req)

    def total_requests_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_REQUESTS,
            estimated_tokens=len(req.input_ids),
        )
        self._set_dispatch_metadata(req, target_worker)
        sock_send(self.workers[target_worker], req)

    def total_tokens_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        estimated_tokens = len(req.input_ids)
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
        )
        self._set_dispatch_metadata(req, target_worker)
        sock_send(self.workers[target_worker], req)

    def _set_dispatch_metadata(
        self,
        req: Req,
        target_rank: int,
        estimated_cost_s: float = 0.0,
    ) -> None:
        req.dp_dispatch_seq = self.dp_budget.last_dispatch_seq[target_rank]
        req.dp_dispatch_cost_s = estimated_cost_s

    @staticmethod
    def _prefix_cache_namespace(req: Req) -> bytes:
        return repr(
            (
                getattr(req, "lora_id", None),
                getattr(req, "extra_key", None),
            )
        ).encode()

    def _estimate_cost_aware_request(self, req: Req) -> list[float]:
        assert self.dp_cost_model is not None
        assert self.dp_prefix_cache is not None
        input_tokens = len(req.input_ids)
        namespace = self._prefix_cache_namespace(req)
        rank_hits = self.dp_prefix_cache.estimate(
            req.input_ids,
            namespace=namespace,
        )
        max_cacheable_tokens = max(0, input_tokens - 1)
        estimated_costs_s = []
        for hit in rank_hits:
            promised_tokens = int(
                hit.promised_tokens * self.dp_cost_promised_cache_discount
            )
            cached_context_tokens = min(
                max_cacheable_tokens,
                hit.cached_context_tokens + promised_tokens,
            )
            estimated_costs_s.append(
                self.dp_cost_model.estimate(
                    input_tokens=input_tokens,
                    cached_context_tokens=cached_context_tokens,
                    host_cache_tokens=hit.host_transfer_tokens,
                    storage_cache_tokens=hit.storage_tokens,
                    swa_host_cache_tokens=min(
                        hit.host_transfer_tokens,
                        self.dp_cost_model.window_size,
                    ),
                ).total_seconds
            )
        return estimated_costs_s

    def _dispatch_cost_aware_requests(self, reqs: Sequence[Req]) -> None:
        assert self.dp_prefix_cache is not None
        routable_reqs = []
        request_costs = []
        namespaces = []
        for req in reqs:
            namespace = self._prefix_cache_namespace(req)
            if self.maybe_external_dp_rank_routing(req):
                self.dp_prefix_cache.promise(
                    req.routed_dp_rank,
                    req.input_ids,
                    namespace=namespace,
                )
                continue
            routable_reqs.append(req)
            request_costs.append(self._estimate_cost_aware_request(req))
            namespaces.append(namespace)

        if not routable_reqs:
            return
        assignments = waterfill_cost_aware_batch(
            base_loads=self.dp_budget.prefill_cost,
            request_costs=request_costs,
            request_counts=self.dp_budget.total_requests,
            eligible_ranks=self._active_workers,
            max_request_skew=self.dp_cost_max_request_skew,
            min_fanout=self.dp_cost_min_fanout,
            target_slack=self.dp_cost_target_slack,
        )
        for req, costs, namespace, target_rank in zip(
            routable_reqs,
            request_costs,
            namespaces,
            assignments,
        ):
            estimated_tokens = len(req.input_ids)
            selected_cost_s = costs[target_rank]
            self.dp_budget.record_dispatch(
                target_rank,
                estimated_tokens=estimated_tokens,
                estimated_cost_s=selected_cost_s,
            )
            self._set_dispatch_metadata(
                req,
                target_rank,
                estimated_cost_s=selected_cost_s,
            )
            sock_send(self.workers[target_rank], req)
            self.dp_prefix_cache.promise(
                target_rank,
                req.input_ids,
                namespace=namespace,
            )

    def cost_aware_scheduler(self, req: Req):
        self._dispatch_cost_aware_requests([req])

    def _dispatch_cost_aware_microbatch(self, reqs: Sequence[Req]) -> None:
        self.refresh_load_budget()
        time_stats_list = []
        for req in reqs:
            time_stats = DPControllerReqTimeStats.new_from_obj(
                unwrap_from_pickle(req.time_stats)
            )
            time_stats.set_dp_dispatch_time()
            req.time_stats = wrap_as_pickle(time_stats)
            time_stats_list.append(time_stats)

        self._dispatch_cost_aware_requests(reqs)

        for req, time_stats in zip(reqs, time_stats_list):
            req.time_stats = time_stats
            req.time_stats.set_dp_dispatch_finish_time()

    @staticmethod
    def _cost_aware_request_list(obj) -> Optional[list[Req]]:
        if isinstance(obj, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
            return [obj]
        if isinstance(
            obj,
            (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
        ):
            return list(obj)
        return None

    def event_loop(self):
        if self.load_balance_method != LoadBalanceMethod.COST_AWARE:
            while True:
                while True:
                    self.soft_watchdog.feed()
                    try:
                        recv_req = sock_recv(
                            self.recv_from_tokenizer,
                            flags=zmq.NOBLOCK,
                        )
                    except zmq.ZMQError:
                        break
                    self._request_dispatcher(recv_req)

        pending: list[Req] = []
        deadline = 0.0
        while True:
            self.soft_watchdog.feed()
            try:
                recv_req = sock_recv(self.recv_from_tokenizer, flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                if not pending:
                    continue
                remaining_s = deadline - time.perf_counter()
                if remaining_s > 0:
                    self.recv_from_tokenizer.poll(
                        timeout=max(1, math.ceil(remaining_s * 1000))
                    )
                    continue
                self._dispatch_cost_aware_microbatch(pending)
                pending.clear()
                continue

            cost_reqs = self._cost_aware_request_list(recv_req)
            if cost_reqs is None:
                if pending:
                    self._dispatch_cost_aware_microbatch(pending)
                    pending.clear()
                self._request_dispatcher(recv_req)
                continue

            if not pending:
                deadline = time.perf_counter() + self.dp_cost_dispatch_window_s
            pending.extend(cost_reqs)
            max_batch_size = max(1, 2 * len(self._active_workers))
            while len(pending) >= max_batch_size:
                microbatch = pending[:max_batch_size]
                del pending[:max_batch_size]
                self._dispatch_cost_aware_microbatch(microbatch)
                if pending:
                    deadline = (
                        time.perf_counter() + self.dp_cost_dispatch_window_s
                    )


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # This process reads the config namespaces before spawning schedulers.
    publish(server_args, role="dp_controller")
    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        # The primary owns routing for the expanded scheduler set.
        if server_args.node_rank == 0 and not server_args.is_ep_scale_joiner:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
