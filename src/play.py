"""CLI for playing Turtle Soup puzzle games."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from enum import Enum
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent))

from game.engine import GameEngine
from game.session_runner import GameSessionRunner, GameResponse
from game.domain.entities import GameState


def setup_logging(verbose: bool = False, agent_mode: bool = False) -> None:
    """Configure logging based on verbosity and mode.
    
    Args:
        verbose: Enable debug logging for all modules
        agent_mode: Enable info logging for game-related modules (agent output visible)
    """
    root_level = logging.WARNING
    
    if verbose:
        root_level = logging.DEBUG
    elif agent_mode:
        root_level = logging.INFO
    
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    for noisy_logger in ["httpx", "httpcore", "urllib3", "openai", "ollama"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    game_logger = logging.getLogger("game")
    if agent_mode or verbose:
        game_logger.setLevel(logging.INFO)


logger = logging.getLogger(__name__)


class GameOutputHandler:
    """Handler for game output with configurable formatting."""
    
    def __init__(self, show_thinking: bool = False, color_output: bool = True):
        self.show_thinking = show_thinking
        self.color_output = color_output
        self._turn_count = 0
    
    def format_player_message(self, message: str) -> str:
        """Format player agent message for display."""
        self._turn_count += 1
        prefix = f"[Turn {self._turn_count}]" if self.show_thinking else ""
        return f"\n{prefix} 🎮 Player Agent: {message}"
    
    def format_dm_message(self, message: str, verdict: Optional[str] = None, verdict_only: bool = False) -> str:
        """Format DM response for display.
        
        Args:
            message: Full DM response message (may include explanation)
            verdict: The verdict string (YES/NO/etc)
            verdict_only: If True, display only the verdict without explanation
                         (for human player mode where they should only see verdict)
        """
        verdict_emoji = ""
        if verdict:
            verdict_map = {
                "YES": "✅", "NO": "❌", 
                "YES_AND_NO": "⚖️", "IRRELEVANT": "🔄",
                "correct": "🎉", "partial": "🤔", "incorrect": "💭"
            }
            verdict_emoji = verdict_map.get(verdict.upper(), "")
        
        if verdict_only and verdict:
            # For human player mode: show only the verdict (no explanation)
            verdict_display = {
                "YES": "**YES**",
                "NO": "**NO**",
                "YES_AND_NO": "**YES and NO**",
                "IRRELEVANT": "**IRRELEVANT**",
            }
            display_text = verdict_display.get(verdict.upper(), verdict)
            return f"   🎲 DM: {verdict_emoji} {display_text}"
        else:
            # For player agent mode: show full message with explanation (for observation)
            return f"   🎲 DM: {verdict_emoji} {message}"
    
    def format_game_status(self, runner: "GameSessionRunner") -> str:
        """Format current game status."""
        session = runner.session
        return (
            f"\n📊 Status: Turn {session.turn_count} | "
            f"Hints: {session.hint_count}/{runner.puzzle.constraints.max_hints}"
        )


class AgentMode(str, Enum):
    HUMAN = "human"
    PLAYER_AGENT = "player_agent"
    DM_AGENT = "dm_agent"
    FULL_AGENT = "full_agent"


class GamePlayCLI:
    def __init__(self):
        self._engine = GameEngine()

    async def list_puzzles(self) -> None:
        print("\n=== Available Puzzles ===\n")
        puzzles = self._engine.list_puzzles()

        if not puzzles:
            print("No puzzles found. Run 'python cli.py init' first.")
            return

        for i, puzzle in enumerate(puzzles, 1):
            tags = ", ".join(puzzle.tags) if puzzle.tags else "none"
            print(f"{i}. {puzzle.id}")
            print(f"   Title: {puzzle.title}")
            print(f"   Difficulty: {puzzle.difficulty or 'unspecified'}")
            print(f"   Tags: {tags}")
            print()

    async def start_game(self, puzzle_id: str, player_name: str, agent_mode: AgentMode = AgentMode.HUMAN) -> None:
        print(f"\n=== Starting Game ===")
        print(f"Puzzle: {puzzle_id}")
        print(f"Player: {player_name}")
        print(f"Mode: {agent_mode.value}")
        print()

        try:
            session = await self._engine.create_session(puzzle_id, player_name)
        except ValueError as e:
            print(f"Error: {e}")
            return

        puzzle = self._engine.get_puzzle(puzzle_id)

        player_agent_mode = agent_mode in (AgentMode.PLAYER_AGENT, AgentMode.FULL_AGENT)
        dm_agent_mode = agent_mode in (AgentMode.DM_AGENT, AgentMode.FULL_AGENT)

        runner = GameSessionRunner(
            session=session,
            puzzle=puzzle,
            kb_manager=self._engine.kb_manager,
            memory_manager=self._engine.memory_manager,
            session_store=self._engine.session_store,
            llm_client=self._engine.model_registry.get_llm_client(),
            agents_config=self._engine.agents_config,
            player_agent_mode=player_agent_mode,
            dm_agent_mode=dm_agent_mode,
        )

        response = runner.start_game()
        print(response.message)
        print()

        if agent_mode == AgentMode.FULL_AGENT:
            await self._auto_play_loop(runner)
        elif agent_mode == AgentMode.PLAYER_AGENT:
            await self._player_agent_loop(runner)
        else:
            await self._game_loop(runner)

    async def resume_game(self, session_id: str, agent_mode: AgentMode = AgentMode.HUMAN) -> None:
        print(f"\n=== Resuming Game ===")
        print(f"Session: {session_id}")
        print(f"Mode: {agent_mode.value}")
        print()

        try:
            session = self._engine.get_session(session_id)
        except ValueError as e:
            print(f"Error: {e}")
            return

        if session.state != GameState.IN_PROGRESS:
            print(f"Session is not active (state: {session.state.value})")
            return

        puzzle = self._engine.get_puzzle(session.puzzle_id)

        player_agent_mode = agent_mode in (AgentMode.PLAYER_AGENT, AgentMode.FULL_AGENT)
        dm_agent_mode = agent_mode in (AgentMode.DM_AGENT, AgentMode.FULL_AGENT)

        runner = GameSessionRunner(
            session=session,
            puzzle=puzzle,
            kb_manager=self._engine.kb_manager,
            memory_manager=self._engine.memory_manager,
            session_store=self._engine.session_store,
            llm_client=self._engine.model_registry.get_llm_client(),
            agents_config=self._engine.agents_config,
            player_agent_mode=player_agent_mode,
            dm_agent_mode=dm_agent_mode,
        )

        print("Game resumed. Type /help for commands.")
        print()

        if agent_mode == AgentMode.FULL_AGENT:
            await self._auto_play_loop(runner)
        elif agent_mode == AgentMode.PLAYER_AGENT:
            await self._player_agent_loop(runner)
        else:
            await self._game_loop(runner)

    async def _game_loop(self, runner: GameSessionRunner) -> None:
        """Human player mode - player only sees verdict, no explanation."""
        output = GameOutputHandler(show_thinking=False)
        
        while runner.is_active:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGame interrupted.")
                break

            if not user_input:
                continue

            response = await runner.process_player_input(user_input)
            
            # Human player only sees the verdict, not the explanation
            # The full explanation is logged for review but not shown to player
            if response.verdict:
                print(output.format_dm_message(response.message, response.verdict, verdict_only=True))
            else:
                # For non-verdict responses (commands, etc.), show full message
                print(f"\n{response.message}")

            if response.game_over:
                print("\n=== Game Over ===")
                break

        print("\nThanks for playing!")

    async def _player_agent_loop(self, runner: GameSessionRunner) -> None:
        """Player agent mode - AI asks questions, DM responds."""
        print("╔══════════════════════════════════════════════════════╗")
        print("║  Player Agent Mode - AI will ask questions for you  ║")
        print("║  Press Ctrl+C to interrupt                          ║")
        print("╚══════════════════════════════════════════════════════╝\n")
        
        output = GameOutputHandler(show_thinking=True)
        turn = 0

        while runner.is_active:
            try:
                turn += 1
                print(f"\n{'─' * 50}")
                print(f"📍 Turn {turn}")
                
                response = await runner.run_player_agent_turn()
                
                player_msg = response.metadata.get('player_message', '')
                if player_msg:
                    print(output.format_player_message(player_msg))
                
                # In player agent mode, show full DM response (including explanation) for observation
                # Note: The player agent's prompt only receives the clean verdict, not this display
                verdict_display = response.verdict or ""
                print(output.format_dm_message(response.message, verdict_display, verdict_only=False))

                if response.game_over:
                    print("\n" + "═" * 50)
                    print("🏁 GAME OVER")
                    print("═" * 50)
                    break

                # Small delay for readability
                await asyncio.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Player agent interrupted.")
                break

        print("\n🎮 Thanks for playing!")

    async def _auto_play_loop(self, runner: GameSessionRunner) -> None:
        """Full agent mode - both Player and DM are AI agents."""
        print("╔══════════════════════════════════════════════════════╗")
        print("║  Full Agent Mode - AI vs AI demonstration           ║")
        print("║  Both Player and DM are controlled by AI            ║")
        print("║  Press Ctrl+C to interrupt                          ║")
        print("╚══════════════════════════════════════════════════════╝\n")

        max_turns = self._engine.game_config.game.max_turn_count
        output = GameOutputHandler(show_thinking=True)
        turn = 0

        try:
            # Use streaming mode for real-time output
            async for response in runner.run_auto_play_stream(max_turns=max_turns):
                turn += 1
                
                # Skip intro message formatting
                if turn == 1 and not response.metadata.get('player_message'):
                    print(f"\n📖 {response.message}")
                    continue
                
                print(f"\n{'─' * 50}")
                print(f"📍 Turn {turn}")
                
                player_msg = response.metadata.get('player_message', '')
                if player_msg:
                    print(output.format_player_message(player_msg))
                
                # In auto play mode, show full DM response (including explanation) for observation
                # Note: The player agent's prompt only receives the clean verdict, not this display
                verdict_display = response.verdict or ""
                print(output.format_dm_message(response.message, verdict_display, verdict_only=False))

                if response.game_over:
                    print("\n" + "═" * 50)
                    print("🏁 GAME OVER")
                    print("═" * 50)
                    break
                
                # Small delay for readability
                await asyncio.sleep(0.3)
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Auto play interrupted.")

        print("\n🎮 Thanks for playing!")

    async def list_sessions(self, state: str | None = None) -> None:
        print("\n=== Game Sessions ===\n")

        state_filter = None
        if state:
            try:
                state_filter = GameState(state)
            except ValueError:
                print(f"Invalid state: {state}")
                return

        sessions = self._engine.list_sessions(state_filter=state_filter)

        if not sessions:
            print("No sessions found.")
            return

        for session in sessions:
            print(f"Session: {session.session_id}")
            print(f"  Puzzle: {session.puzzle_id}")
            print(f"  State: {session.state.value}")
            print(f"  Turns: {session.turn_count}")
            print(f"  Hints: {session.hint_count}")
            print(f"  Created: {session.created_at}")
            print()

    async def close(self) -> None:
        await self._engine.close()


def main():
    parser = argparse.ArgumentParser(
        description="Turtle Soup Puzzle Game CLI"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("puzzles", help="List available puzzles")

    play_parser = subparsers.add_parser("play", help="Start a new game")
    play_parser.add_argument("puzzle_id", help="Puzzle ID to play")
    play_parser.add_argument(
        "--player", "-p",
        default="player",
        help="Player name (default: player)",
    )
    play_parser.add_argument(
        "--mode", "-m",
        choices=["human", "player_agent", "dm_agent", "full_agent"],
        default="human",
        help="Game mode: human (default), player_agent (AI player), dm_agent (AI DM), full_agent (both AI)",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume an existing game")
    resume_parser.add_argument("session_id", help="Session ID to resume")
    resume_parser.add_argument(
        "--mode", "-m",
        choices=["human", "player_agent", "dm_agent", "full_agent"],
        default="human",
        help="Game mode: human (default), player_agent (AI player), dm_agent (AI DM), full_agent (both AI)",
    )

    sessions_parser = subparsers.add_parser("sessions", help="List game sessions")
    sessions_parser.add_argument(
        "--state", "-s",
        choices=["lobby", "in_progress", "completed", "aborted"],
        help="Filter by state",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Configure logging based on mode
    agent_mode_str = getattr(args, 'mode', 'human')
    is_agent_mode = agent_mode_str in ('player_agent', 'dm_agent', 'full_agent')
    setup_logging(verbose=args.verbose, agent_mode=is_agent_mode)

    cli = GamePlayCLI()

    try:
        if args.command == "puzzles":
            asyncio.run(cli.list_puzzles())
        elif args.command == "play":
            agent_mode = AgentMode(args.mode)
            asyncio.run(cli.start_game(args.puzzle_id, args.player, agent_mode))
        elif args.command == "resume":
            agent_mode = AgentMode(args.mode)
            asyncio.run(cli.resume_game(args.session_id, agent_mode))
        elif args.command == "sessions":
            asyncio.run(cli.list_sessions(args.state))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        asyncio.run(cli.close())


if __name__ == "__main__":
    main()
