#!/usr/bin/env python3
"""
Main entry point for running Teleavatar robot with OpenPI policy.

This script uses the standard openpi_client.runtime framework for clean,
modular robot control with remote policy inference.

Usage:
    # Start policy server first (in another terminal):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_teleavatar_v2 \
        --policy.dir=checkpoints/pi05_teleavatar_v2/my_experiment/20000

    # Then run this script:
    python examples/teleavatar_v2/main.py --remote-host 127.0.0.1
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from examples.teleavatar_v2 import env as _env


@dataclasses.dataclass
class Args:
    """Command-line arguments for Teleavatar deployment."""

    # Remote policy server settings
    remote_host: str = "127.0.0.1"
    """IP address of the policy server (e.g., '192.168.1.100')"""

    remote_port: int = 8000
    """Port of the policy server"""

    # Control settings
    control_frequency: float = 30.0
    """Control loop frequency in Hz (rate at which the policy command is stepped)"""

    interp_frequency: float = 200.0
    """Rate (Hz) to republish des_q, interpolated from the command stream (~200 Hz)."""

    interpolate: bool = True
    """Interpolate des_q up to interp_frequency; --no-interpolate publishes raw (ZOH)."""

    open_loop_horizon: int = 16
    """Number of actions to execute before querying the policy again. Must not
    exceed the action chunk length of the trained model (30 for the teleavatar
    configs)."""

    # Task settings
    prompt: str = "Stack the three blocks"
    """Language instruction for the robot"""

    # Episode settings
    num_episodes: int = 100
    """Number of episodes to run"""

    max_episode_steps: int = 0
    """Maximum steps per episode (0 = unlimited)"""


def main(args: Args) -> None:
    """Main function to run Teleavatar with policy inference."""

    logging.info("=" * 60)
    logging.info("Teleavatar OpenPI Deployment")
    logging.info("=" * 60)
    logging.info(f"Policy server: ws://{args.remote_host}:{args.remote_port}")
    logging.info(f"Control frequency: {args.control_frequency} Hz")
    if args.interpolate:
        logging.info(f"des_q interp publish: {args.interp_frequency} Hz (ZOH-staircase fix)")
    else:
        logging.info("des_q interpolation: OFF (raw commands, original ZOH behavior)")
    logging.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    logging.info(f"Prompt: '{args.prompt}'")
    logging.info("=" * 60)

    # Create WebSocket client policy
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )

    # Get and log server metadata
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Connected to policy server. Metadata: {metadata}")

    # Create Teleavatar environment
    # Note: Images are kept at original resolution to match training data
    environment = _env.TeleavatarEnvironment(
        prompt=args.prompt,
        control_frequency=args.control_frequency,
        interp_frequency=args.interp_frequency,
        interpolate=args.interpolate,
    )

    # Create policy agent with action chunking
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.open_loop_horizon,  # Execute this many actions before querying
        )
    )

    # Create runtime
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],  # Could add video recording, logging, etc.
        max_hz=args.control_frequency,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # Run!
    logging.info("\nStarting robot control loop...")
    logging.info("Press Ctrl+C to stop\n")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n\nStopping robot (Ctrl+C pressed)...")
    finally:
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )

    # Parse arguments and run
    args: Args = tyro.cli(Args)
    main(args)
