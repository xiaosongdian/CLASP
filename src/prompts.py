#!/usr/bin/env python3
"""
Unified prompt template management (persona generation/refinement/action prediction).
"""

# =========================
# Persona Generation (S0)
# =========================
SYSTEM_INSTRUCTION_PROFILE = (
 "You are an expert in computational social behavior and user modeling. "
 "Convert noisy user action traces into an abstract persona for simulation and prediction. "
 "Do not copy long raw texts verbatim; summarize into stable traits, habits, interests, and style."
)


FREE_FORM_PROMPT = """
You will serve as an assistant to help generate a user profile based on the user's social media behavior history to better understand the user's interests and predict their future actions.

USER BEHAVIOR HISTORY ({action_count} records):
{behavior_data}

Your profile should:
1. Focus on patterns that can predict FUTURE behavior (not just describe past actions)
2. Include any dimensions you discover through analysis - don't limit yourself
3. Capture the user's unique behavioral signatures that distinguish them from others
4. Be actionable for predicting: what content they'll engage with, what actions they'll take, what triggers their participation
5. Scale the depth and detail of your analysis with the amount of input data - more data should yield richer, more nuanced insights

User Profile:
"""


# =========================
# Persona Refinement (Candidate Personas)
# =========================
SYSTEM_INSTRUCTION_REFINEMENT = (
 "You refine an existing user persona according to prediction errors. "
 "Keep useful stable traits, correct mismatches, and output one improved persona only."
)

PROFILE_REFINEMENT_PROMPT = """Old persona:
{old_persona}

Observed behavior discrepancies (predicted vs actual):
{behavior_discrepancies}

Please output a revised persona that better explains actual behavior and keeps good prior information.
"""


def build_profile_refinement_prompt_messages(
 old_persona: str,
 behavior_discrepancies: str,
) -> list[dict[str, str]]:
 """Build prompt (DPO/SFT). """
 user_block = PROFILE_REFINEMENT_PROMPT.format(
 old_persona=old_persona,
 behavior_discrepancies=behavior_discrepancies,
)
 return [
 {"role": "system", "content": SYSTEM_INSTRUCTION_REFINEMENT},
 {"role": "user", "content": user_block},
]


# =========================
# Action Prediction (Decision + Content)
# =========================
AVAILABLE_ACTIONS = "post, reply, repost, like"

DECISION_INSTRUCTION = (
 "Predict the most likely next user action type from the available set. "
 "Return one token only: post/reply/repost/like."
)

DECISION_INPUT_TEMPLATE = """Target user profile:
{user_profile}

Recent user actions (chronological order; oldest to newest within this block):
{action_history}

Current scenario:
{scenario}

Available actions:
{available_actions}

Answer with exactly one action label.
"""

CONTENT_INSTRUCTION = (
 "Generate the likely text content for the user's action. "
 "Return plain content only, no extra explanation."
)

CONTENT_INPUT_TEMPLATE = """Target user profile:
{user_profile}

Recent user actions (chronological order; oldest to newest within this block):
{action_history}

Current scenario:
{scenario}

Generate the text content:
"""


# =========================
# Discrepancy Text Template (for persona refinement)
# =========================
DISCREPANCY_TEMPLATE = """Scenario [{idx}]: {scenario_context}
{object_block}Predicted action: {predicted_action}
Actual action: {actual_action}
"""

