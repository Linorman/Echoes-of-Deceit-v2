"""Game Session Runner for managing active game sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, AsyncIterator

from config import AgentsConfig
from game.domain.entities import (
    AgentRole,
    GameSession,
    GameState,
    Puzzle,
    SessionEvent,
)
from game.kb_manager import KnowledgeBaseManager
from game.memory.entities import EventTag, SessionEventRecord
from game.memory.manager import MemoryManager
from game.storage.session_store import GameSessionStore
from models.base import LLMClient
from rag.base_provider import QueryResult

logger = logging.getLogger(__name__)

# Game event logger for agent visibility
game_logger = logging.getLogger("game.session")


class MessageType(str, Enum):
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    COMMAND = "command"
    UNKNOWN = "unknown"


class DMVerdict(str, Enum):
    YES = "YES"
    NO = "NO"
    YES_AND_NO = "YES_AND_NO"
    IRRELEVANT = "IRRELEVANT"


class HypothesisVerdict(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


@dataclass
class GameResponse:
    message: str
    verdict: Optional[str] = None
    game_over: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class GameSessionRunner:
    COMMAND_PREFIXES = ("/", "!", "\\")
    HYPOTHESIS_KEYWORDS = [
        "i think", "my guess", "the answer is", "hypothesis",
        "the solution is", "my theory", "i believe",
        "我认为", "我猜", "答案是", "我的猜测", "谜底是",
    ]

    def __init__(
        self,
        session: GameSession,
        puzzle: Puzzle,
        kb_manager: KnowledgeBaseManager,
        memory_manager: MemoryManager,
        session_store: GameSessionStore,
        llm_client: LLMClient,
        agents_config: AgentsConfig,
        player_agent_mode: bool = False,
        dm_agent_mode: bool = False,
    ):
        self._session = session
        self._puzzle = puzzle
        self._kb_manager = kb_manager
        self._memory_manager = memory_manager
        self._session_store = session_store
        self._llm_client = llm_client
        self._agents_config = agents_config
        self._player_agent_mode = player_agent_mode
        self._dm_agent_mode = dm_agent_mode
        self._player_agent_question_count = 0

    @property
    def session(self) -> GameSession:
        return self._session

    @property
    def puzzle(self) -> Puzzle:
        return self._puzzle

    @property
    def is_active(self) -> bool:
        return self._session.is_active

    @property
    def turn_count(self) -> int:
        return self._session.turn_count

    @property
    def question_count(self) -> int:
        """Returns the actual number of questions asked (actual turns)."""
        return self._count_questions()

    def start_game(self) -> GameResponse:
        if self._session.state != GameState.LOBBY:
            return GameResponse(
                message="Game has already started or is completed.",
                metadata={"error": True},
            )

        self._session.start()
        self._save_session()

        intro_message = self._generate_intro()
        self._append_event(AgentRole.DM, intro_message, [EventTag.INTRO, EventTag.NARRATION])

        return GameResponse(message=intro_message)

    def _generate_intro(self) -> str:
        dm_config = self._agents_config.dm
        persona = dm_config.persona

        intro_parts = []

        if persona.tone == "mysterious":
            intro_parts.append("*The air grows thick with mystery as a strange tale unfolds...*\n")
        elif persona.tone == "friendly":
            intro_parts.append("Welcome to our puzzle game! Let me share a curious story with you.\n")
        else:
            intro_parts.append("A puzzle awaits you.\n")

        intro_parts.append(f"**{self._puzzle.title}**\n")
        intro_parts.append(self._puzzle.puzzle_statement)
        intro_parts.append("\n\nAsk yes/no questions to uncover the truth, or propose your hypothesis when ready.")
        intro_parts.append(f"\n\nCommands: /hint, /status, /history, /quit")

        return "\n".join(intro_parts)

    async def process_player_input(self, message: str) -> GameResponse:
        if not self._session.is_active:
            if self._session.state == GameState.LOBBY:
                return self.start_game()
            return GameResponse(
                message="This game session is not active.",
                metadata={"error": True},
            )

        message = message.strip()
        if not message:
            return GameResponse(message="Please enter a message.")

        message_type = self._classify_input(message)

        if message_type == MessageType.COMMAND:
            return self._handle_command(message)
        elif message_type == MessageType.HYPOTHESIS:
            return await self._handle_hypothesis(message)
        else:
            return await self._handle_question(message)

    def _classify_input(self, message: str) -> MessageType:
        if any(message.startswith(p) for p in self.COMMAND_PREFIXES):
            return MessageType.COMMAND

        message_lower = message.lower()
        if any(kw in message_lower for kw in self.HYPOTHESIS_KEYWORDS):
            return MessageType.HYPOTHESIS

        return MessageType.QUESTION

    def _handle_command(self, message: str) -> GameResponse:
        cmd = message.lstrip("/!\\").lower().split()[0]

        if cmd in ("hint", "h"):
            return self._handle_hint_command()
        elif cmd in ("status", "s"):
            return self._handle_status_command()
        elif cmd in ("history", "hist"):
            return self._handle_history_command()
        elif cmd in ("quit", "exit", "q"):
            return self._handle_quit_command()
        elif cmd in ("help", "?"):
            return self._handle_help_command()
        else:
            return GameResponse(
                message=f"Unknown command: {cmd}. Type /help for available commands."
            )

    def _handle_hint_command(self) -> GameResponse:
        hint_config = self._agents_config.hint
        max_hints = self._puzzle.constraints.max_hints

        if self._session.hint_count >= max_hints:
            return GameResponse(
                message=f"You've used all {max_hints} hints available for this puzzle."
            )

        if not self._puzzle.has_hints or self._session.hint_count >= len(self._puzzle.hints):
            return GameResponse(
                message="No more hints available. Try a different approach!"
            )

        hint = self._puzzle.hints[self._session.hint_count]
        self._session.hint_count += 1
        self._save_session()

        if hint_config.strategy.initial_vagueness == "high" and self._session.hint_count <= 2:
            hint_message = f"*A subtle nudge:* {hint}"
        else:
            hint_message = f"*Hint {self._session.hint_count}/{max_hints}:* {hint}"

        self._append_event(AgentRole.DM, hint_message, [EventTag.HINT])

        return GameResponse(
            message=hint_message,
            metadata={"hints_used": self._session.hint_count, "hints_remaining": max_hints - self._session.hint_count},
        )

    def _handle_status_command(self) -> GameResponse:
        status_lines = [
            f"**Game Status**",
            f"Puzzle: {self._puzzle.title}",
            f"State: {self._session.state.value}",
            f"Questions asked: {self._count_questions()}",
            f"Hints used: {self._session.hint_count}/{self._puzzle.constraints.max_hints}",
        ]
        return GameResponse(message="\n".join(status_lines))

    def _handle_history_command(self) -> GameResponse:
        recent_events = self._get_recent_qa_pairs(limit=10)
        if not recent_events:
            return GameResponse(message="No question history yet.")

        lines = ["**Recent Q&A:**"]
        for i, (q, a) in enumerate(recent_events, 1):
            lines.append(f"{i}. Q: {q[:50]}{'...' if len(q) > 50 else ''}")
            lines.append(f"   A: {a}")

        return GameResponse(message="\n".join(lines))

    def _handle_quit_command(self) -> GameResponse:
        self._session.state = GameState.ABORTED
        self._session.updated_at = datetime.now()
        self._save_session()

        return GameResponse(
            message="Game session ended. Thanks for playing!",
            game_over=True,
        )

    def _handle_help_command(self) -> GameResponse:
        help_text = """**Available Commands:**
/hint (h) - Request a hint
/status (s) - View game status
/history - View recent Q&A
/quit (q) - End the game
/help (?) - Show this help

**How to Play:**
- Ask yes/no questions to gather clues
- When ready, state your hypothesis (start with "I think..." or "My guess is...")
"""
        return GameResponse(message=help_text)

    async def run_player_agent_turn(self) -> GameResponse:
        if not self._player_agent_mode:
            return GameResponse(
                message="Player agent mode is not enabled.",
                metadata={"error": True},
            )

        if not self._session.is_active:
            return GameResponse(
                message="Game session is not active.",
                metadata={"error": True},
            )

        player_config = self._agents_config.player_agent
        form_hypothesis_after = player_config.behavior.form_hypothesis_after_questions
        max_questions = player_config.behavior.max_questions_before_guess

        should_hypothesize = self._player_agent_question_count >= form_hypothesis_after
        if should_hypothesize and self._should_player_agent_hypothesize():
            game_logger.info("🧠 Player agent preparing hypothesis (questions asked: %d)", 
                           self._player_agent_question_count)
            message = await self._generate_player_agent_hypothesis()
            response = await self._handle_hypothesis(message)
            response.metadata["player_message"] = message
            return response
        else:
            game_logger.info("🤔 Player agent preparing question #%d", 
                           self._player_agent_question_count + 1)
            message = await self._generate_player_agent_question()
            response = await self._handle_question(message)
            response.metadata["player_message"] = message
            return response

    def _should_player_agent_hypothesize(self) -> bool:
        qa_pairs = self._get_recent_qa_pairs(limit=50, verdict_only=True)
        yes_count = sum(1 for _, a in qa_pairs if "YES" in a.upper() and "NO" not in a.upper())
        total = len(qa_pairs)
        if total > 0 and yes_count / total >= 0.4:
            game_logger.debug("Player agent deciding to hypothesize (yes_ratio=%.2f)", yes_count / total)
            return True
        return False

    async def _generate_player_agent_question(self) -> str:
        player_config = self._agents_config.player_agent
        persona_name = player_config.persona.name
        strategies = player_config.behavior.question_strategies

        game_logger.debug("Generating player agent question (count=%d)", self._player_agent_question_count + 1)

        rag_context = ""
        if self._session.kb_id:
            try:
                result = await self._kb_manager.query_public(
                    self._session.kb_id,
                    self._puzzle.puzzle_statement
                )
                if result and result.answer:
                    rag_context = result.answer[:500]
                    game_logger.debug("RAG context retrieved for player agent")
            except Exception as e:
                logger.warning("Player agent RAG query failed: %s", e)

        recent_qa = self._format_qa_for_prompt()
        prompt = self._build_player_question_prompt(
            puzzle_statement=self._puzzle.puzzle_statement,
            recent_qa=recent_qa,
            rag_context=rag_context,
            persona_name=persona_name,
            strategies=strategies,
        )

        try:
            response = await self._llm_client.agenerate(prompt)
            question = response.strip()
            if not question.endswith("?"):
                question += "?"
            self._player_agent_question_count += 1
            return question
        except Exception as e:
            logger.error("Player agent question generation failed: %s", e)
            return "Is there something unusual about this situation?"

    async def _generate_player_agent_hypothesis(self) -> str:
        player_config = self._agents_config.player_agent
        persona_name = player_config.persona.name

        recent_qa = self._format_qa_for_prompt(limit=20)
        prompt = self._build_player_hypothesis_prompt(
            puzzle_statement=self._puzzle.puzzle_statement,
            recent_qa=recent_qa,
            persona_name=persona_name,
        )

        try:
            game_logger.debug("Calling LLM for hypothesis generation...")
            response = await self._llm_client.agenerate(prompt)
            hypothesis = response.strip()
            if not hypothesis.lower().startswith("i think"):
                hypothesis = f"I think {hypothesis}"
            game_logger.info("💡 Player agent formed hypothesis")
            return hypothesis
        except Exception as e:
            logger.error("Player agent hypothesis generation failed: %s", e)
            return "I think there's something unusual about this situation."

    def _format_qa_for_prompt(self, limit: int = 10) -> str:
        """Format Q&A pairs for player agent prompt - uses verdict only, no explanations."""
        qa_pairs = self._get_recent_qa_pairs(limit=limit, verdict_only=True)
        if not qa_pairs:
            return "No questions asked yet."
        return "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

    def _build_player_question_prompt(
        self,
        puzzle_statement: str,
        recent_qa: str,
        rag_context: str,
        persona_name: str,
        strategies: List[str],
    ) -> str:
        strategy_hints = {
            "binary_elimination": "Ask questions that divide possibilities in half",
            "detail_probing": "Focus on specific details that seem unusual",
            "scenario_testing": "Test specific scenarios or interpretations",
        }
        strategy_text = "\n".join([f"- {strategy_hints.get(s, s)}" for s in strategies])

        return f"""You are {persona_name}, an AI player in a Turtle Soup puzzle game.
Your goal is to solve the puzzle by asking yes/no questions.

PUZZLE:
{puzzle_statement}

{f"BACKGROUND INFO: {rag_context}" if rag_context else ""}

PREVIOUS Q&A:
{recent_qa}

QUESTION STRATEGIES:
{strategy_text}

INSTRUCTIONS:
1. Analyze the puzzle and previous answers carefully
2. Form a strategic question that helps narrow down the solution
3. Ask only ONE yes/no question
4. Do not repeat questions already asked
5. Focus on understanding WHY or HOW the situation occurred

Respond with only your question, nothing else."""

    def _build_player_hypothesis_prompt(
        self,
        puzzle_statement: str,
        recent_qa: str,
        persona_name: str,
    ) -> str:
        return f"""You are {persona_name}, an AI player in a Turtle Soup puzzle game.
Based on your investigation, it's time to propose your hypothesis.

PUZZLE:
{puzzle_statement}

INVESTIGATION HISTORY:
{recent_qa}

INSTRUCTIONS:
1. Analyze all the YES/NO answers you've received
2. Form a coherent explanation that fits all the confirmed facts
3. State your hypothesis clearly, starting with "I think..."

Respond with your hypothesis, starting with "I think..."."""

    async def run_auto_play(
        self, 
        max_turns: int = 50,
        on_response: Optional[Callable[[GameResponse], None]] = None,
    ) -> List[GameResponse]:
        """Run automatic play with optional real-time callback.
        
        Args:
            max_turns: Maximum number of turns to play
            on_response: Optional callback function called for each response
        
        Returns:
            List of all game responses
        """
        responses = []
        
        if self._session.state == GameState.LOBBY:
            start_response = self.start_game()
            responses.append(start_response)
            if on_response:
                on_response(start_response)
            game_logger.info("🎮 Game started: %s", self._puzzle.title)

        turn = 0
        while self._session.is_active and turn < max_turns:
            game_logger.info("🔄 Turn %d: Player agent thinking...", turn + 1)
            
            response = await self.run_player_agent_turn()
            responses.append(response)
            
            # Log the agent interaction for visibility
            player_msg = response.metadata.get('player_message', '')
            if player_msg:
                game_logger.info("🎮 Player: %s", player_msg[:100] + ("..." if len(player_msg) > 100 else ""))
            game_logger.info("🎲 DM [%s]: %s", 
                           response.verdict or "N/A", 
                           response.message[:100] + ("..." if len(response.message) > 100 else ""))
            
            if on_response:
                on_response(response)
            
            if response.game_over:
                game_logger.info("🏁 Game ended!")
                break
            
            turn += 1

        return responses

    async def run_auto_play_stream(self, max_turns: int = 50) -> AsyncIterator[GameResponse]:
        """Run automatic play as an async generator for streaming output.
        
        Args:
            max_turns: Maximum number of turns to play
            
        Yields:
            GameResponse for each turn
        """
        if self._session.state == GameState.LOBBY:
            start_response = self.start_game()
            game_logger.info("🎮 Game started: %s", self._puzzle.title)
            yield start_response

        turn = 0
        while self._session.is_active and turn < max_turns:
            game_logger.info("🔄 Turn %d: Player agent thinking...", turn + 1)
            
            response = await self.run_player_agent_turn()
            
            # Log the agent interaction for visibility
            player_msg = response.metadata.get('player_message', '')
            if player_msg:
                game_logger.info("🎮 Player: %s", player_msg[:100] + ("..." if len(player_msg) > 100 else ""))
            game_logger.info("🎲 DM [%s]: %s", 
                           response.verdict or "N/A", 
                           response.message[:100] + ("..." if len(response.message) > 100 else ""))
            
            yield response
            
            if response.game_over:
                game_logger.info("🏁 Game ended!")
                break
            
            turn += 1

    async def _handle_question(self, message: str) -> GameResponse:
        self._append_event(AgentRole.PLAYER, message, [EventTag.QUESTION])
        game_logger.debug("Processing question: %s", message[:50] + ("..." if len(message) > 50 else ""))

        kb_id = self._session.kb_id
        if kb_id:
            game_logger.debug("Querying knowledge base: %s", kb_id)
        rag_context = await self._kb_manager.query_full(kb_id, message) if kb_id else None

        verdict, explanation = await self._evaluate_question(message, rag_context)
        game_logger.debug("Question evaluated: verdict=%s", verdict.value)

        response_text = self._format_question_response(verdict, explanation)

        self._append_event(AgentRole.DM, response_text, [EventTag.ANSWER], verdict=verdict.value)

        return GameResponse(
            message=response_text,
            verdict=verdict.value,
        )

    async def _evaluate_question(
        self,
        question: str,
        rag_context: Optional[QueryResult],
    ) -> tuple[DMVerdict, str]:
        judge_config = self._agents_config.judge

        context_text = ""
        if rag_context and rag_context.answer:
            context_text = rag_context.answer

        prompt = self._build_question_evaluation_prompt(
            question=question,
            puzzle_statement=self._puzzle.puzzle_statement,
            puzzle_answer=self._puzzle.answer,
            rag_context=context_text,
            strictness=judge_config.strictness,
        )

        try:
            response = await self._llm_client.agenerate(prompt)
            verdict, explanation = self._parse_verdict_response(response)
        except Exception as e:
            logger.error("LLM evaluation failed: %s", e)
            verdict = DMVerdict.IRRELEVANT
            explanation = "I'm having trouble processing that question. Try rephrasing it."

        return verdict, explanation

    def _build_question_evaluation_prompt(
        self,
        question: str,
        puzzle_statement: str,
        puzzle_answer: str,
        rag_context: str,
        strictness: str,
    ) -> str:
        strictness_instruction = ""
        if strictness == "strict":
            strictness_instruction = "Be very precise. Only say YES if it directly follows from the answer."
        elif strictness == "lenient":
            strictness_instruction = "Be generous in interpretation. If the question is roughly on the right track, lean towards YES."
        else:
            strictness_instruction = "Use reasonable judgment to evaluate the question."

        prompt = f"""You are the Judge in a Turtle Soup (situation puzzle) game. Your role is to evaluate player questions and respond with a verdict.

PUZZLE STATEMENT (what the player sees):
{puzzle_statement}

HIDDEN ANSWER (only you know this):
{puzzle_answer}

{f"ADDITIONAL CONTEXT: {rag_context}" if rag_context else ""}

PLAYER'S QUESTION:
{question}

INSTRUCTIONS:
1. Evaluate whether the player's question is relevant to solving the puzzle.
2. {strictness_instruction}
3. Respond with one of: YES, NO, YES_AND_NO, or IRRELEVANT
4. Provide a brief explanation (1-2 sentences max) without revealing the answer.

RESPONSE FORMAT:
VERDICT: [YES/NO/YES_AND_NO/IRRELEVANT]
EXPLANATION: [Brief explanation without spoilers]

Your response:"""

        return prompt

    def _parse_verdict_response(self, response: str) -> tuple[DMVerdict, str]:
        response = response.strip()
        lines = response.split("\n")

        verdict = DMVerdict.IRRELEVANT
        explanation = ""

        for line in lines:
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                verdict_text = line.split(":", 1)[1].strip().upper()
                if "YES_AND_NO" in verdict_text or "YES AND NO" in verdict_text:
                    verdict = DMVerdict.YES_AND_NO
                elif "YES" in verdict_text and "NO" not in verdict_text:
                    verdict = DMVerdict.YES
                elif "NO" in verdict_text:
                    verdict = DMVerdict.NO
                else:
                    verdict = DMVerdict.IRRELEVANT
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()

        if not explanation:
            explanation = response[:200] if len(response) > 200 else response

        return verdict, explanation

    def _format_question_response(self, verdict: DMVerdict, explanation: str) -> str:
        judge_config = self._agents_config.judge
        dm_config = self._agents_config.dm

        verdict_display = {
            DMVerdict.YES: "✓ **YES**",
            DMVerdict.NO: "✗ **NO**",
            DMVerdict.YES_AND_NO: "◐ **YES and NO**",
            DMVerdict.IRRELEVANT: "○ **IRRELEVANT**",
        }

        response = verdict_display.get(verdict, verdict.value)

        if judge_config.response_format.include_explanation and explanation:
            max_len = judge_config.response_format.max_explanation_length
            if len(explanation) > max_len:
                explanation = explanation[:max_len-3] + "..."
            response += f"\n{explanation}"

        if dm_config.behavior.encourage_player and verdict == DMVerdict.YES:
            response += "\n*You're on the right track!*"

        return response

    async def _handle_hypothesis(self, message: str) -> GameResponse:
        self._append_event(AgentRole.PLAYER, message, [EventTag.HYPOTHESIS])
        game_logger.info("📋 Evaluating hypothesis...")

        kb_id = self._session.kb_id
        rag_context = await self._kb_manager.query_full(kb_id, message) if kb_id else None

        verdict, explanation = await self._evaluate_hypothesis(message, rag_context)
        game_logger.info("⚖️ Hypothesis verdict: %s", verdict.value.upper())

        if verdict == HypothesisVerdict.CORRECT:
            response_text = self._format_correct_hypothesis(explanation)
            self._complete_game(score=self._calculate_score())
            game_over = True
            game_logger.info("🎉 Player solved the puzzle!")
        else:
            response_text = self._format_incorrect_hypothesis(verdict, explanation)
            game_over = False
            game_logger.info("💭 Hypothesis was %s, game continues", verdict.value)

        self._append_event(AgentRole.DM, response_text, [EventTag.FINAL_VERDICT])

        return GameResponse(
            message=response_text,
            verdict=verdict.value,
            game_over=game_over,
        )

        return GameResponse(
            message=response_text,
            verdict=verdict.value,
            game_over=game_over,
        )

    async def _evaluate_hypothesis(
        self,
        hypothesis: str,
        rag_context: Optional[QueryResult],
    ) -> tuple[HypothesisVerdict, str]:
        context_text = ""
        if rag_context and rag_context.answer:
            context_text = rag_context.answer

        prompt = self._build_hypothesis_evaluation_prompt(
            hypothesis=hypothesis,
            puzzle_statement=self._puzzle.puzzle_statement,
            puzzle_answer=self._puzzle.answer,
            rag_context=context_text,
        )

        try:
            response = await self._llm_client.agenerate(prompt)
            verdict, explanation = self._parse_hypothesis_response(response)
        except Exception as e:
            logger.error("LLM hypothesis evaluation failed: %s", e)
            verdict = HypothesisVerdict.INCORRECT
            explanation = "Unable to evaluate the hypothesis. Please try again."

        return verdict, explanation

    def _build_hypothesis_evaluation_prompt(
        self,
        hypothesis: str,
        puzzle_statement: str,
        puzzle_answer: str,
        rag_context: str,
    ) -> str:
        prompt = f"""You are the Judge in a Turtle Soup (situation puzzle) game. The player has proposed a hypothesis for the solution.

PUZZLE STATEMENT:
{puzzle_statement}

CANONICAL ANSWER:
{puzzle_answer}

{f"ADDITIONAL CONTEXT: {rag_context}" if rag_context else ""}

PLAYER'S HYPOTHESIS:
{hypothesis}

INSTRUCTIONS:
1. Compare the player's hypothesis with the canonical answer.
2. Determine if the hypothesis captures the essential truth of the puzzle.
3. A hypothesis is CORRECT if it identifies the core mechanism/reason, even if wording differs.
4. A hypothesis is PARTIAL if it captures some key elements but misses crucial parts.
5. A hypothesis is INCORRECT if it misses the main point entirely.

RESPONSE FORMAT:
VERDICT: [CORRECT/PARTIAL/INCORRECT]
EXPLANATION: [Explain why, and if correct, congratulate the player]

Your response:"""

        return prompt

    def _parse_hypothesis_response(self, response: str) -> tuple[HypothesisVerdict, str]:
        response = response.strip()
        lines = response.split("\n")

        verdict = HypothesisVerdict.INCORRECT
        explanation = ""

        for line in lines:
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                verdict_text = line.split(":", 1)[1].strip().upper()
                if "CORRECT" in verdict_text and "INCORRECT" not in verdict_text and "PARTIAL" not in verdict_text:
                    verdict = HypothesisVerdict.CORRECT
                elif "PARTIAL" in verdict_text:
                    verdict = HypothesisVerdict.PARTIAL
                else:
                    verdict = HypothesisVerdict.INCORRECT
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()

        if not explanation:
            explanation = response[:300] if len(response) > 300 else response

        return verdict, explanation

    def _format_correct_hypothesis(self, explanation: str) -> str:
        lines = [
            "🎉 **CORRECT!**",
            "",
            explanation,
            "",
            "**The Full Answer:**",
            self._puzzle.answer,
            "",
            f"Questions asked: {self._count_questions()}",
            f"Hints used: {self._session.hint_count}",
        ]
        return "\n".join(lines)

    def _format_incorrect_hypothesis(self, verdict: HypothesisVerdict, explanation: str) -> str:
        if verdict == HypothesisVerdict.PARTIAL:
            prefix = "🔶 **Partially Correct**\n\nYou're getting closer!"
        else:
            prefix = "❌ **Not Quite**\n\nThat's not the solution."

        return f"{prefix}\n\n{explanation}\n\nKeep investigating!"

    def _complete_game(self, score: Optional[int] = None) -> None:
        self._session.complete(score=score)
        self._save_session()

        if self._session.player_ids:
            player_id = self._session.player_ids[0]
            self._memory_manager.summarize_session(
                session_id=self._session.session_id,
                player_id=player_id,
                puzzle_id=self._session.puzzle_id,
            )

    def _calculate_score(self) -> int:
        base_score = 1000
        question_penalty = self._count_questions() * 10
        hint_penalty = self._session.hint_count * 50
        return max(100, base_score - question_penalty - hint_penalty)

    def _count_questions(self) -> int:
        return sum(
            1 for event in self._session.turn_history
            if EventTag.QUESTION.value in event.tags
        )

    def _get_recent_qa_pairs(self, limit: int = 10, verdict_only: bool = False) -> List[tuple[str, str]]:
        """Get recent question-answer pairs.
        
        Args:
            limit: Maximum number of pairs to return
            verdict_only: If True, return only the verdict (YES/NO/etc) for answers,
                         otherwise return the full message including explanation.
        """
        pairs = []
        events = self._session.turn_history

        question = None
        for event in events:
            if EventTag.QUESTION.value in event.tags:
                question = event.message
            elif EventTag.ANSWER.value in event.tags and question:
                if verdict_only and event.verdict:
                    answer = event.verdict
                else:
                    answer = event.message
                pairs.append((question, answer))
                question = None

        return pairs[-limit:]

    def _append_event(
        self,
        role: AgentRole,
        message: str,
        tags: List[EventTag],
        verdict: Optional[str] = None,
    ) -> None:
        turn_index = len(self._session.turn_history)

        event = SessionEvent(
            session_id=self._session.session_id,
            turn_index=turn_index,
            role=role,
            message=message,
            tags=[t.value for t in tags],
            verdict=verdict,
        )
        self._session.turn_history.append(event)
        self._session.updated_at = datetime.now()
        self._save_session()

        event_record = SessionEventRecord(
            session_id=self._session.session_id,
            turn_index=turn_index,
            role=role.value,
            message=message,
            tags=[t.value for t in tags],
        )
        self._memory_manager.append_session_event(self._session.session_id, event_record)

    def _save_session(self) -> None:
        self._session_store.save_session(self._session)
