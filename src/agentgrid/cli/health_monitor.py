#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

"""
Health monitor CLI for Agent Grid. Connects to the DHT and displays
the status of all peers serving model blocks.

Usage:
    python -m agentgrid.cli.health_monitor --model Qwen/Qwen2.5-Coder-32B-Instruct
    python -m agentgrid.cli.health_monitor --model Qwen/Qwen2.5-Coder-32B-Instruct --refresh 5
    python -m agentgrid.cli.health_monitor --model Qwen/Qwen2.5-Coder-32B-Instruct --json
"""

import argparse
import json
import sys
import time

from agentgrid.constants import PUBLIC_INITIAL_PEERS
from agentgrid.data_structures import ServerState, UID_DELIMITER


def _load_model_config(model_name: str, token: str = None):
    """Load model config to get dht_prefix and num_hidden_layers."""
    # Import here to trigger model class registration
    import agentgrid.models  # noqa: F401
    from agentgrid.utils.auto_config import AutoDistributedConfig

    kwargs = {}
    if token:
        kwargs["token"] = token
    config = AutoDistributedConfig.from_pretrained(model_name, **kwargs)
    return config


def _query_grid(dht, block_uids):
    """Query the DHT for all block UIDs and return module infos + spans."""
    from agentgrid.utils.dht import compute_spans, get_remote_module_infos

    module_infos = get_remote_module_infos(dht, block_uids, latest=True)
    spans = compute_spans(module_infos, min_state=ServerState.OFFLINE)
    return module_infos, spans


def _build_peers_data(spans):
    """Convert spans dict to a list of peer data dicts."""
    peers = []
    for peer_id, span in sorted(spans.items(), key=lambda x: x[1].start):
        info = span.server_info
        peers.append({
            "peer_id": peer_id.to_base58()[:16] + "...",
            "peer_id_full": peer_id.to_base58(),
            "public_name": info.public_name or "",
            "state": info.state.name,
            "blocks": f"{span.start}-{span.end - 1}",
            "num_blocks": span.length,
            "throughput": round(info.throughput, 2) if info.throughput else 0,
            "cache_tokens_left": info.cache_tokens_left,
            "video_card": info.video_card or "",
            "operating_system": info.operating_system or "",
            "quant_type": info.quant_type or "",
            "torch_dtype": info.torch_dtype or "",
            "using_relay": info.using_relay if info.using_relay is not None else False,
        })
    return peers


def _compute_coverage(module_infos, total_blocks):
    """Compute which blocks have at least one online server."""
    covered = set()
    for info in module_infos:
        for peer_id, server_info in info.servers.items():
            if server_info.state == ServerState.ONLINE:
                # Extract block index from uid
                try:
                    _, block_idx = info.uid.rsplit(UID_DELIMITER, 1)
                    covered.add(int(block_idx))
                except (ValueError, AttributeError):
                    pass
    return covered


def _print_table(peers, coverage, total_blocks, model_name):
    """Print a rich table of peer status."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title=f"Agent Grid Health: {model_name}")
        table.add_column("Peer ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("State", no_wrap=True)
        table.add_column("Blocks", justify="center")
        table.add_column("RPS", justify="right")
        table.add_column("Cache Left", justify="right")
        table.add_column("GPU", style="magenta")
        table.add_column("OS")
        table.add_column("Quant")
        table.add_column("Dtype")
        table.add_column("Relay")

        for p in peers:
            state = p["state"]
            if state == "ONLINE":
                state_str = f"[bold green]{state}[/bold green]"
            elif state == "JOINING":
                state_str = f"[bold yellow]{state}[/bold yellow]"
            else:
                state_str = f"[bold red]{state}[/bold red]"

            cache = str(p["cache_tokens_left"]) if p["cache_tokens_left"] is not None else "-"
            relay = "yes" if p["using_relay"] else "no"

            table.add_row(
                p["peer_id"],
                p["public_name"],
                state_str,
                p["blocks"],
                str(p["throughput"]),
                cache,
                p["video_card"],
                p["operating_system"],
                p["quant_type"],
                p["torch_dtype"],
                relay,
            )

        console.print()
        console.print(table)

        # Summary
        online = sum(1 for p in peers if p["state"] == "ONLINE")
        joining = sum(1 for p in peers if p["state"] == "JOINING")
        covered_pct = len(coverage) / total_blocks * 100 if total_blocks > 0 else 0

        console.print(
            f"\n  Peers: [green]{online} online[/green], [yellow]{joining} joining[/yellow], "
            f"{len(peers)} total"
        )
        console.print(
            f"  Block coverage: {len(coverage)}/{total_blocks} ({covered_pct:.0f}%)"
        )

        if len(coverage) < total_blocks:
            missing = sorted(set(range(total_blocks)) - coverage)
            if len(missing) <= 20:
                console.print(f"  [red]Missing blocks: {missing}[/red]")
            else:
                console.print(f"  [red]Missing blocks: {missing[:10]}...({len(missing)} total)[/red]")
        console.print()

    except ImportError:
        # Fallback without rich
        _print_table_plain(peers, coverage, total_blocks, model_name)


def _print_table_plain(peers, coverage, total_blocks, model_name):
    """Plain text fallback when rich is not installed."""
    print(f"\nAgent Grid Health: {model_name}")
    print("-" * 120)
    header = f"{'Peer ID':<20} {'Name':<15} {'State':<10} {'Blocks':<10} {'RPS':>6} {'Cache':>10} {'GPU':<20} {'Quant':<18}"
    print(header)
    print("-" * 120)
    for p in peers:
        cache = str(p["cache_tokens_left"]) if p["cache_tokens_left"] is not None else "-"
        print(
            f"{p['peer_id']:<20} {p['public_name']:<15} {p['state']:<10} {p['blocks']:<10} "
            f"{p['throughput']:>6} {cache:>10} {p['video_card']:<20} {p['quant_type']:<18}"
        )
    print("-" * 120)

    online = sum(1 for p in peers if p["state"] == "ONLINE")
    joining = sum(1 for p in peers if p["state"] == "JOINING")
    covered_pct = len(coverage) / total_blocks * 100 if total_blocks > 0 else 0
    print(f"Peers: {online} online, {joining} joining, {len(peers)} total")
    print(f"Block coverage: {len(coverage)}/{total_blocks} ({covered_pct:.0f}%)")
    print()


def _print_json(peers, coverage, total_blocks, model_name):
    """Print JSON output."""
    data = {
        "model": model_name,
        "total_blocks": total_blocks,
        "covered_blocks": len(coverage),
        "peers_online": sum(1 for p in peers if p["state"] == "ONLINE"),
        "peers_total": len(peers),
        "peers": peers,
    }
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Agent Grid Health Monitor")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS,
        help="DHT initial peer multiaddrs"
    )
    parser.add_argument("--token", type=str, default=None, help="HuggingFace auth token")
    parser.add_argument("--refresh", type=int, default=0, help="Refresh interval in seconds (0 = one-shot)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    # Load model config
    print(f"Loading model config for {args.model}...")
    try:
        config = _load_model_config(args.model, token=args.token)
    except Exception as e:
        print(f"Error loading model config: {e}", file=sys.stderr)
        sys.exit(1)

    dht_prefix = config.dht_prefix
    num_blocks = config.num_hidden_layers
    print(f"Model has {num_blocks} blocks (DHT prefix: {dht_prefix})")

    # Build block UIDs
    block_uids = [f"{dht_prefix}{UID_DELIMITER}{i}" for i in range(num_blocks)]

    # Connect to DHT
    print(f"Connecting to DHT...")
    from hivemind import DHT

    dht = DHT(initial_peers=args.initial_peers, start=True, client_mode=True)
    print(f"Connected. Peer ID: {dht.peer_id}")

    try:
        if args.refresh > 0 and not args.json:
            # Auto-refresh mode with rich.live
            try:
                from rich.live import Live
                from rich.console import Console

                console = Console()
                with Live(console=console, refresh_per_second=1) as live:
                    while True:
                        module_infos, spans = _query_grid(dht, block_uids)
                        peers = _build_peers_data(spans)
                        coverage = _compute_coverage(module_infos, num_blocks)

                        # Build output by capturing console
                        from io import StringIO
                        from rich.console import Console as StrConsole

                        buf = StringIO()
                        str_console = StrConsole(file=buf, force_terminal=True)

                        from rich.table import Table

                        table = Table(title=f"Agent Grid Health: {args.model} (refreshing every {args.refresh}s)")
                        table.add_column("Peer ID", style="cyan", no_wrap=True)
                        table.add_column("Name", style="green")
                        table.add_column("State", no_wrap=True)
                        table.add_column("Blocks", justify="center")
                        table.add_column("RPS", justify="right")
                        table.add_column("Cache Left", justify="right")
                        table.add_column("GPU", style="magenta")
                        table.add_column("Quant")
                        table.add_column("Relay")

                        for p in peers:
                            state = p["state"]
                            if state == "ONLINE":
                                state_str = f"[bold green]{state}[/bold green]"
                            elif state == "JOINING":
                                state_str = f"[bold yellow]{state}[/bold yellow]"
                            else:
                                state_str = f"[bold red]{state}[/bold red]"
                            cache = str(p["cache_tokens_left"]) if p["cache_tokens_left"] is not None else "-"
                            relay = "yes" if p["using_relay"] else "no"
                            table.add_row(
                                p["peer_id"], p["public_name"], state_str, p["blocks"],
                                str(p["throughput"]), cache, p["video_card"], p["quant_type"], relay,
                            )

                        online = sum(1 for p in peers if p["state"] == "ONLINE")
                        joining = sum(1 for p in peers if p["state"] == "JOINING")
                        covered_pct = len(coverage) / num_blocks * 100 if num_blocks > 0 else 0

                        live.update(table)
                        time.sleep(args.refresh)

            except ImportError:
                # Fallback: clear screen and reprint
                while True:
                    print("\033[2J\033[H", end="")  # Clear screen
                    module_infos, spans = _query_grid(dht, block_uids)
                    peers = _build_peers_data(spans)
                    coverage = _compute_coverage(module_infos, num_blocks)
                    _print_table_plain(peers, coverage, num_blocks, args.model)
                    time.sleep(args.refresh)
        else:
            # One-shot mode
            module_infos, spans = _query_grid(dht, block_uids)
            peers = _build_peers_data(spans)
            coverage = _compute_coverage(module_infos, num_blocks)

            if args.json:
                _print_json(peers, coverage, num_blocks, args.model)
            else:
                _print_table(peers, coverage, num_blocks, args.model)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        dht.shutdown()


if __name__ == "__main__":
    main()
