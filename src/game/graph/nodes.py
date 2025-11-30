"""LangGraph Node implementations for game orchestration.

This module contains all the nodes used in the game graph:
- PlayerMessageNode: Entry point for player input
- PlayerAgentNode: AI player that generates questions and hypotheses
- DMQuestionNode: Handles player questions
- DMHypothesisNode: Evaluates player hypotheses
- CommandHandlerNode: Processes meta commands
- MemoryUpdateNode: Updates session and player memory
- IntroNode: Generates game introduction
- RevealSolutionNode: Reveals the solution at game end
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from game.graph.state import (
    GameGraphState,
    GamePhase,
    GameMode,
    MessageType,
    DMVerdict,
    HypothesisVerdict,
    TurnEvent,
)

if TYPE_CHECKING:
    from config import AgentsConfig
    from game.kb_manager import KnowledgeBaseManager
    from game.memory.manager import MemoryManager
    from models.base import LLMClient

logger = logging.getLogger(__name__)

GAME_EVENT_SESSION_START = "session_start"
GAME_EVENT_SESSION_END = "session_end"
GAME_EVENT_HINT_USED = "hint_used"
GAME_EVENT_HYPOTHESIS_VERDICT = "hypothesis_verdict"

COMMAND_PREFIXES = ("/", "!", "\\")
HYPOTHESIS_KEYWORDS = [
    "i think", "my guess", "the answer is", "hypothesis",
    "the solution is", "my theory", "i believe",
    "我认为", "我猜", "答案是", "我的猜测", "谜底是",
]


def log_game_event(
    event_type: str,
    session_id: str,
    player_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    event_data = {
        "event_type": event_type,
        "session_id": session_id,
        "player_id": player_id,
        "timestamp": datetime.now().isoformat(),
        **(details or {}),
    }
    logger.info("GAME_EVENT: %s", event_data)


class BaseNode(ABC):
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        kb_manager: Optional[KnowledgeBaseManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        agents_config: Optional[AgentsConfig] = None,
    ):
        self._llm_client = llm_client
        self._kb_manager = kb_manager
        self._memory_manager = memory_manager
        self._agents_config = agents_config

    @abstractmethod
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        pass

    def _get_player_profile_context(self, player_id: str) -> Optional[str]:
        if not self._memory_manager:
            return None

        dm_config = self._agents_config.dm if self._agents_config else None
        if dm_config and not dm_config.profile_integration.enabled:
            return None

        try:
            results = self._memory_manager.retrieve_player_profile(player_id, limit=3)
            if not results:
                return None

            profile_parts = []
            for result in results:
                content = result.value.get("content", "")
                summary_type = result.value.get("summary_type", "")
                if content:
                    profile_parts.append(f"[{summary_type}] {content}")

            if profile_parts:
                return "\n".join(profile_parts)
        except Exception as e:
            logger.warning("Failed to retrieve player profile: %s", e)

        return None

    def _get_profile_adaptation_hints(self, profile_context: Optional[str]) -> str:
        if not profile_context:
            return ""

        dm_config = self._agents_config.dm if self._agents_config else None
        if not dm_config:
            return ""

        hints = []
        profile_config = dm_config.profile_integration

        if profile_config.adapt_difficulty:
            hints.append("Adjust response complexity based on player skill level.")
        if profile_config.adapt_explanations:
            hints.append("Tailor explanation style to player preferences.")
        if profile_config.adapt_hint_strength:
            hints.append("Consider player history when providing guidance.")

        if not hints:
            return ""

        weight = profile_config.profile_weight
        if weight == "high":
            hints.insert(0, "IMPORTANT: Strongly consider the following player profile:")
        elif weight == "low":
            hints.insert(0, "Note: Lightly consider the following player profile:")
        else:
            hints.insert(0, "Consider the following player profile:")

        return "\n".join(hints) + f"\n{profile_context}"


class PlayerMessageNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        message = state.last_user_message.strip()

        if not message:
            return {"message_type": MessageType.UNKNOWN}

        message_type = self._classify_input(message)
        return {"message_type": message_type}

    def _classify_input(self, message: str) -> MessageType:
        if any(message.startswith(p) for p in COMMAND_PREFIXES):
            return MessageType.COMMAND

        message_lower = message.lower()
        if any(kw in message_lower for kw in HYPOTHESIS_KEYWORDS):
            return MessageType.HYPOTHESIS

        return MessageType.QUESTION


class IntroNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        dm_config = self._agents_config.dm if self._agents_config else None
        persona_tone = dm_config.persona.tone if dm_config else "mysterious"

        log_game_event(
            GAME_EVENT_SESSION_START,
            state.session_id,
            state.player_id,
            {"puzzle_id": state.puzzle_id, "puzzle_title": state.puzzle_title},
        )

        intro_parts = []

        if persona_tone == "mysterious":
            intro_parts.append("*The air grows thick with mystery as a strange tale unfolds...*\n")
        elif persona_tone == "friendly":
            intro_parts.append("Welcome to our puzzle game! Let me share a curious story with you.\n")
        else:
            intro_parts.append("A puzzle awaits you.\n")

        intro_parts.append(f"**{state.puzzle_title}**\n")
        intro_parts.append(state.puzzle_statement)
        intro_parts.append("\n\nAsk yes/no questions to uncover the truth, or propose your hypothesis when ready.")
        intro_parts.append("\n\nCommands: /hint, /status, /history, /quit")

        intro_message = "\n".join(intro_parts)

        event = TurnEvent(
            turn_index=state.turn_index,
            role="dm",
            message=intro_message,
            tags=["intro", "narration"],
        )

        return {
            "last_dm_response": intro_message,
            "game_phase": GamePhase.PLAYING,
            "turn_history": [event],
            "turn_index": state.turn_index + 1,
        }


class PlayerAgentNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        if not self._llm_client:
            return {"error": "LLM client not available for player agent"}

        player_config = self._agents_config.player_agent if self._agents_config else None
        if not player_config or not player_config.enabled:
            return {"error": "Player agent not enabled"}

        question_count = state.player_agent_question_count
        max_questions = player_config.behavior.max_questions_before_guess if player_config else 20
        form_hypothesis_after = player_config.behavior.form_hypothesis_after_questions if player_config else 10

        should_hypothesize = question_count >= form_hypothesis_after
        if should_hypothesize and self._should_attempt_hypothesis(state):
            return await self._generate_hypothesis(state)
        else:
            return await self._generate_question(state)

    def _should_attempt_hypothesis(self, state: GameGraphState) -> bool:
        yes_count = sum(1 for e in state.turn_history if e.verdict == "YES")
        total_questions = state.player_agent_question_count
        if total_questions > 0 and yes_count / total_questions >= 0.4:
            return True
        return False

    async def _generate_question(self, state: GameGraphState) -> Dict[str, Any]:
        player_config = self._agents_config.player_agent if self._agents_config else None
        persona_name = player_config.persona.name if player_config else "Detective"
        strategies = player_config.behavior.question_strategies if player_config else ["binary_elimination"]

        rag_context = ""
        if self._kb_manager and state.kb_id:
            try:
                result = await self._kb_manager.query_public(state.kb_id, state.puzzle_statement)
                if result and result.answer:
                    rag_context = result.answer[:500]
            except Exception as e:
                logger.warning("Player agent RAG query failed: %s", e)

        recent_qa = self._format_recent_qa(state)
        prompt = self._build_question_prompt(
            puzzle_statement=state.puzzle_statement,
            recent_qa=recent_qa,
            rag_context=rag_context,
            persona_name=persona_name,
            strategies=strategies,
        )

        if self._llm_client is None:
            return {"error": "LLM client not configured for player agent"}

        try:
            response = await self._llm_client.agenerate(prompt)
            question = self._parse_question_response(response)

            event = TurnEvent(
                turn_index=state.turn_index,
                role="player_agent",
                message=question,
                tags=["question", "ai_player"],
            )

            return {
                "last_user_message": question,
                "message_type": MessageType.QUESTION,
                "player_agent_question_count": state.player_agent_question_count + 1,
                "turn_history": [event],
                "turn_index": state.turn_index + 1,
                "awaiting_player_agent": False,
            }
        except Exception as e:
            logger.error("Player agent question generation failed: %s", e)
            return {"error": f"Failed to generate question: {e}"}

    async def _generate_hypothesis(self, state: GameGraphState) -> Dict[str, Any]:
        player_config = self._agents_config.player_agent if self._agents_config else None
        persona_name = player_config.persona.name if player_config else "Detective"

        recent_qa = self._format_recent_qa(state, limit=20)
        prompt = self._build_hypothesis_prompt(
            puzzle_statement=state.puzzle_statement,
            recent_qa=recent_qa,
            persona_name=persona_name,
        )

        if self._llm_client is None:
            return {"error": "LLM client not configured for player agent"}

        try:
            response = await self._llm_client.agenerate(prompt)
            hypothesis = self._parse_hypothesis_response(response)

            event = TurnEvent(
                turn_index=state.turn_index,
                role="player_agent",
                message=hypothesis,
                tags=["hypothesis", "ai_player"],
            )

            return {
                "last_user_message": hypothesis,
                "message_type": MessageType.HYPOTHESIS,
                "turn_history": [event],
                "turn_index": state.turn_index + 1,
                "awaiting_player_agent": False,
            }
        except Exception as e:
            logger.error("Player agent hypothesis generation failed: %s", e)
            return {"error": f"Failed to generate hypothesis: {e}"}

    def _format_recent_qa(self, state: GameGraphState, limit: int = 10) -> str:
        qa_pairs = []
        question = None

        for event in state.turn_history:
            if "question" in event.tags:
                question = event.message
            elif "answer" in event.tags and question:
                verdict = event.verdict or "UNKNOWN"
                qa_pairs.append(f"Q: {question}\nA: {verdict}")
                question = None

        if not qa_pairs:
            return "No questions asked yet."

        recent = qa_pairs[-limit:]
        return "\n\n".join(recent)

    def _build_question_prompt(
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

    def _build_hypothesis_prompt(
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

    def _parse_question_response(self, response: str) -> str:
        question = response.strip()
        if not question.endswith("?"):
            question += "?"
        return question

    def _parse_hypothesis_response(self, response: str) -> str:
        hypothesis = response.strip()
        if not hypothesis.lower().startswith("i think"):
            hypothesis = f"I think {hypothesis}"
        return hypothesis


class DMQuestionNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        question = state.last_user_message

        player_event = TurnEvent(
            turn_index=state.turn_index,
            role="player",
            message=question,
            tags=["question"],
        )

        rag_context = ""
        if self._kb_manager and state.kb_id:
            try:
                result = await self._kb_manager.query_full(state.kb_id, question)
                if result and result.answer:
                    rag_context = result.answer
            except Exception as e:
                logger.warning("RAG query failed: %s", e)

        player_profile = self._get_player_profile_context(state.player_id)

        verdict, explanation = await self._evaluate_question(
            question=question,
            puzzle_statement=state.puzzle_statement,
            puzzle_answer=state.puzzle_answer,
            rag_context=rag_context,
            player_profile=player_profile,
        )

        response_text = self._format_response(verdict, explanation)

        dm_event = TurnEvent(
            turn_index=state.turn_index + 1,
            role="dm",
            message=response_text,
            tags=["answer"],
            verdict=verdict.value,
        )

        return {
            "last_dm_response": response_text,
            "last_verdict": verdict.value,
            "turn_history": [player_event, dm_event],
            "turn_index": state.turn_index + 2,
        }

    async def _evaluate_question(
        self,
        question: str,
        puzzle_statement: str,
        puzzle_answer: str,
        rag_context: str,
        player_profile: Optional[str] = None,
    ) -> tuple[DMVerdict, str]:
        if not self._llm_client:
            return DMVerdict.IRRELEVANT, "Unable to process question."

        judge_config = self._agents_config.judge if self._agents_config else None
        strictness = judge_config.strictness if judge_config else "moderate"

        prompt = self._build_prompt(
            question=question,
            puzzle_statement=puzzle_statement,
            puzzle_answer=puzzle_answer,
            rag_context=rag_context,
            strictness=strictness,
            player_profile=player_profile,
        )

        try:
            response = await self._llm_client.agenerate(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error("LLM evaluation failed: %s", e)
            return DMVerdict.IRRELEVANT, "I'm having trouble processing that question."

    def _build_prompt(
        self,
        question: str,
        puzzle_statement: str,
        puzzle_answer: str,
        rag_context: str,
        strictness: str,
        player_profile: Optional[str] = None,
    ) -> str:
        strictness_instruction = {
            "strict": "Be very precise. Only say YES if it directly follows from the answer.",
            "lenient": "Be generous. If the question is roughly on track, lean towards YES.",
        }.get(strictness, "Use reasonable judgment to evaluate the question.")

        profile_section = ""
        if player_profile:
            adaptation_hints = self._get_profile_adaptation_hints(player_profile)
            if adaptation_hints:
                profile_section = f"\n\nPLAYER PROFILE:\n{adaptation_hints}\n"

        return f"""You are the Judge in a Turtle Soup (situation puzzle) game.

PUZZLE STATEMENT:
{puzzle_statement}

HIDDEN ANSWER:
{puzzle_answer}

{f"ADDITIONAL CONTEXT: {rag_context}" if rag_context else ""}{profile_section}
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

    def _parse_response(self, response: str) -> tuple[DMVerdict, str]:
        lines = response.strip().split("\n")
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

    def _format_response(self, verdict: DMVerdict, explanation: str) -> str:
        verdict_display = {
            DMVerdict.YES: "✓ **YES**",
            DMVerdict.NO: "✗ **NO**",
            DMVerdict.YES_AND_NO: "◐ **YES and NO**",
            DMVerdict.IRRELEVANT: "○ **IRRELEVANT**",
        }

        response = verdict_display.get(verdict, verdict.value)

        judge_config = self._agents_config.judge if self._agents_config else None
        if judge_config and judge_config.response_format.include_explanation and explanation:
            max_len = judge_config.response_format.max_explanation_length
            if len(explanation) > max_len:
                explanation = explanation[:max_len - 3] + "..."
            response += f"\n{explanation}"

        dm_config = self._agents_config.dm if self._agents_config else None
        if dm_config and dm_config.behavior.encourage_player and verdict == DMVerdict.YES:
            response += "\n*You're on the right track!*"

        return response


class DMHypothesisNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        hypothesis = state.last_user_message

        player_event = TurnEvent(
            turn_index=state.turn_index,
            role="player",
            message=hypothesis,
            tags=["hypothesis"],
        )

        rag_context = ""
        if self._kb_manager and state.kb_id:
            try:
                result = await self._kb_manager.query_full(state.kb_id, hypothesis)
                if result and result.answer:
                    rag_context = result.answer
            except Exception as e:
                logger.warning("RAG query failed: %s", e)

        verdict, explanation = await self._evaluate_hypothesis(
            hypothesis=hypothesis,
            puzzle_statement=state.puzzle_statement,
            puzzle_answer=state.puzzle_answer,
            rag_context=rag_context,
        )

        if verdict == HypothesisVerdict.CORRECT:
            response_text = self._format_correct(explanation, state)
            new_phase = GamePhase.COMPLETED
            score = self._calculate_score(state)
            log_game_event(
                GAME_EVENT_HYPOTHESIS_VERDICT,
                state.session_id,
                state.player_id,
                {"verdict": "correct", "score": score, "puzzle_id": state.puzzle_id},
            )
        else:
            response_text = self._format_incorrect(verdict, explanation)
            new_phase = state.game_phase
            score = state.score
            log_game_event(
                GAME_EVENT_HYPOTHESIS_VERDICT,
                state.session_id,
                state.player_id,
                {"verdict": verdict.value, "puzzle_id": state.puzzle_id},
            )

        dm_event = TurnEvent(
            turn_index=state.turn_index + 1,
            role="dm",
            message=response_text,
            tags=["final_verdict"],
            verdict=verdict.value,
        )

        return {
            "last_dm_response": response_text,
            "last_verdict": verdict.value,
            "game_phase": new_phase,
            "score": score,
            "turn_history": [player_event, dm_event],
            "turn_index": state.turn_index + 2,
        }

    async def _evaluate_hypothesis(
        self,
        hypothesis: str,
        puzzle_statement: str,
        puzzle_answer: str,
        rag_context: str,
    ) -> tuple[HypothesisVerdict, str]:
        if not self._llm_client:
            return HypothesisVerdict.INCORRECT, "Unable to evaluate hypothesis."

        prompt = self._build_prompt(
            hypothesis=hypothesis,
            puzzle_statement=puzzle_statement,
            puzzle_answer=puzzle_answer,
            rag_context=rag_context,
        )

        try:
            response = await self._llm_client.agenerate(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error("LLM hypothesis evaluation failed: %s", e)
            return HypothesisVerdict.INCORRECT, "Unable to evaluate the hypothesis."

    def _build_prompt(
        self,
        hypothesis: str,
        puzzle_statement: str,
        puzzle_answer: str,
        rag_context: str,
    ) -> str:
        return f"""You are the Judge in a Turtle Soup (situation puzzle) game.

PUZZLE STATEMENT:
{puzzle_statement}

CANONICAL ANSWER:
{puzzle_answer}

{f"ADDITIONAL CONTEXT: {rag_context}" if rag_context else ""}

PLAYER'S HYPOTHESIS:
{hypothesis}

INSTRUCTIONS:
1. Compare the player's hypothesis with the canonical answer.
2. A hypothesis is CORRECT if it identifies the core mechanism/reason.
3. A hypothesis is PARTIAL if it captures some key elements but misses crucial parts.
4. A hypothesis is INCORRECT if it misses the main point.

RESPONSE FORMAT:
VERDICT: [CORRECT/PARTIAL/INCORRECT]
EXPLANATION: [Explain why, and if correct, congratulate the player]

Your response:"""

    def _parse_response(self, response: str) -> tuple[HypothesisVerdict, str]:
        lines = response.strip().split("\n")
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

    def _format_correct(self, explanation: str, state: GameGraphState) -> str:
        question_count = sum(1 for e in state.turn_history if "question" in e.tags)
        lines = [
            "🎉 **CORRECT!**",
            "",
            explanation,
            "",
            "**The Full Answer:**",
            state.puzzle_answer,
            "",
            f"Questions asked: {question_count}",
            f"Hints used: {state.hint_count}",
        ]
        return "\n".join(lines)

    def _format_incorrect(self, verdict: HypothesisVerdict, explanation: str) -> str:
        if verdict == HypothesisVerdict.PARTIAL:
            prefix = "🔶 **Partially Correct**\n\nYou're getting closer!"
        else:
            prefix = "❌ **Not Quite**\n\nThat's not the solution."
        return f"{prefix}\n\n{explanation}\n\nKeep investigating!"

    def _calculate_score(self, state: GameGraphState) -> int:
        question_count = sum(1 for e in state.turn_history if "question" in e.tags)
        base_score = 1000
        question_penalty = question_count * 10
        hint_penalty = state.hint_count * 50
        return max(100, base_score - question_penalty - hint_penalty)


class CommandHandlerNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        message = state.last_user_message.lstrip("/!\\").lower()
        cmd = message.split()[0] if message else ""

        handler = {
            "hint": self._handle_hint,
            "h": self._handle_hint,
            "status": self._handle_status,
            "s": self._handle_status,
            "history": self._handle_history,
            "hist": self._handle_history,
            "quit": self._handle_quit,
            "exit": self._handle_quit,
            "q": self._handle_quit,
            "help": self._handle_help,
            "?": self._handle_help,
        }.get(cmd, self._handle_unknown)

        return await handler(state, cmd)

    async def _handle_hint(self, state: GameGraphState, cmd: str) -> Dict[str, Any]:
        max_hints = state.max_hints

        if state.hint_count >= max_hints:
            response = f"You've used all {max_hints} hints available for this puzzle."
            return {"last_dm_response": response}

        hint_count = state.hint_count + 1

        hint_config = self._agents_config.hint if self._agents_config else None
        initial_vagueness = hint_config.strategy.initial_vagueness if hint_config else "high"

        player_profile = self._get_player_profile_context(state.player_id)
        if player_profile and "Advanced" in player_profile:
            initial_vagueness = "high"
        elif player_profile and "Beginner" in player_profile:
            initial_vagueness = "low"

        if initial_vagueness == "high" and hint_count <= 2:
            hint_message = f"*A subtle nudge:* Think about the situation from a different angle."
        elif initial_vagueness == "low" or hint_count > 2:
            hint_message = f"*Hint {hint_count}/{max_hints}:* Consider what might be unusual about the scenario."
        else:
            hint_message = f"*Hint {hint_count}/{max_hints}:* Consider what might be unusual about the scenario."

        log_game_event(
            GAME_EVENT_HINT_USED,
            state.session_id,
            state.player_id,
            {"hint_count": hint_count, "max_hints": max_hints, "puzzle_id": state.puzzle_id},
        )

        event = TurnEvent(
            turn_index=state.turn_index,
            role="dm",
            message=hint_message,
            tags=["hint"],
        )

        return {
            "last_dm_response": hint_message,
            "hint_count": hint_count,
            "turn_history": [event],
            "turn_index": state.turn_index + 1,
        }

    async def _handle_status(self, state: GameGraphState, cmd: str) -> Dict[str, Any]:
        question_count = sum(1 for e in state.turn_history if "question" in e.tags)
        status_lines = [
            "**Game Status**",
            f"Puzzle: {state.puzzle_title}",
            f"Phase: {state.game_phase}",
            f"Questions asked: {question_count}",
            f"Hints used: {state.hint_count}/{state.max_hints}",
        ]
        return {"last_dm_response": "\n".join(status_lines)}

    async def _handle_history(self, state: GameGraphState, cmd: str) -> Dict[str, Any]:
        qa_pairs = []
        question = None

        for event in state.turn_history:
            if "question" in event.tags:
                question = event.message
            elif "answer" in event.tags and question:
                qa_pairs.append((question, event.message))
                question = None

        if not qa_pairs:
            return {"last_dm_response": "No question history yet."}

        recent = qa_pairs[-10:]
        lines = ["**Recent Q&A:**"]
        for i, (q, a) in enumerate(recent, 1):
            q_display = q[:50] + "..." if len(q) > 50 else q
            lines.append(f"{i}. Q: {q_display}")
            lines.append(f"   A: {a}")

        return {"last_dm_response": "\n".join(lines)}

    async def _handle_quit(self, state: GameGraphState, cmd: str) -> Dict[str, Any]:
        return {
            "last_dm_response": "Game session ended. Thanks for playing!",
            "game_phase": GamePhase.ABORTED,
        }

    async def _handle_help(self, state: GameGraphState, cmd: str) -> Dict[str, Any]:
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
        return {"last_dm_response": help_text}

    async def _handle_unknown(self, state: GameGraphState, cmd: str) -> Dict[str, Any]:
        return {
            "last_dm_response": f"Unknown command: {cmd}. Type /help for available commands."
        }


class MemoryUpdateNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        if not self._memory_manager:
            return {}

        from game.memory.entities import SessionEventRecord

        for event in state.turn_history[-2:]:
            try:
                record = SessionEventRecord(
                    session_id=state.session_id,
                    turn_index=event.turn_index,
                    role=event.role,
                    message=event.message,
                    tags=event.tags,
                    timestamp=event.timestamp,
                )
                self._memory_manager.append_session_event(state.session_id, record)
            except Exception as e:
                logger.warning("Failed to append session event: %s", e)

        if state.game_phase == GamePhase.COMPLETED:
            question_count = sum(1 for e in state.turn_history if "question" in e.tags)
            log_game_event(
                GAME_EVENT_SESSION_END,
                state.session_id,
                state.player_id,
                {
                    "puzzle_id": state.puzzle_id,
                    "success": True,
                    "question_count": question_count,
                    "hint_count": state.hint_count,
                    "score": state.score,
                },
            )
            try:
                self._memory_manager.summarize_session(
                    session_id=state.session_id,
                    player_id=state.player_id,
                    puzzle_id=state.puzzle_id,
                )
            except Exception as e:
                logger.warning("Failed to summarize session: %s", e)
        elif state.game_phase == GamePhase.ABORTED:
            question_count = sum(1 for e in state.turn_history if "question" in e.tags)
            log_game_event(
                GAME_EVENT_SESSION_END,
                state.session_id,
                state.player_id,
                {
                    "puzzle_id": state.puzzle_id,
                    "success": False,
                    "aborted": True,
                    "question_count": question_count,
                    "hint_count": state.hint_count,
                },
            )

        return {}


class RevealSolutionNode(BaseNode):
    async def __call__(self, state: GameGraphState) -> Dict[str, Any]:
        question_count = sum(1 for e in state.turn_history if "question" in e.tags)

        lines = [
            "**Game Complete!**",
            "",
            "**The Full Story:**",
            state.puzzle_answer,
            "",
            "**Your Stats:**",
            f"- Questions asked: {question_count}",
            f"- Hints used: {state.hint_count}",
            f"- Final score: {state.score or 'N/A'}",
            "",
            "Thanks for playing!",
        ]

        reveal_message = "\n".join(lines)

        event = TurnEvent(
            turn_index=state.turn_index,
            role="dm",
            message=reveal_message,
            tags=["reveal", "narration"],
        )

        return {
            "last_dm_response": reveal_message,
            "turn_history": [event],
            "turn_index": state.turn_index + 1,
        }
