"""
Semantic Reasoner - Deep Understanding Beyond Statistics

Goes beyond pattern matching to understand MEANING:
- Why attacks work (not just that they do)
- Intent behind prompts
- Semantic relationships between attacks
- Symbolic reasoning about behavior
- Knowledge graph of attack concepts

This enables understanding, not just recognition.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger("honey_prompt")


@dataclass
class AttackConcept:
    """A conceptual understanding of an attack type"""

    concept_id: str
    name: str
    intent: str  # What the attacker is trying to achieve
    mechanism: str  # How the attack works
    preconditions: List[str]  # What must be true for attack to work
    indicators: List[str]  # Semantic indicators of this attack
    related_concepts: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.concept_id)


@dataclass
class SemanticRelation:
    """Relationship between concepts"""

    relation_type: str  # 'is_a', 'enables', 'defeats', 'similar_to'
    source: str
    target: str
    strength: float  # 0.0 to 1.0


class KnowledgeGraph:
    """
    Graph of attack concepts and their relationships.

    Enables reasoning like:
    - "This is an instance of role redefinition"
    - "Role redefinition enables instruction override"
    - "Therefore this is dangerous"
    """

    def __init__(self):
        self.concepts: Dict[str, AttackConcept] = {}
        self.relations: List[SemanticRelation] = []

        # Initialize with base security concepts
        self._init_base_concepts()

    def _init_base_concepts(self):
        """Initialize foundational attack concepts"""

        # Core attack patterns
        self.add_concept(
            AttackConcept(
                concept_id="instruction_override",
                name="Instruction Override",
                intent="Replace system instructions with attacker-controlled instructions",
                mechanism="Uses commands like 'ignore previous' to reset context",
                preconditions=["System processes user input as instructions"],
                indicators=["ignore", "forget", "disregard", "previous instructions"],
            )
        )

        self.add_concept(
            AttackConcept(
                concept_id="role_redefinition",
                name="Role Redefinition",
                intent="Change the AI's perceived role or constraints",
                mechanism="Redefines identity, purpose, or rules",
                preconditions=["System accepts role suggestions from user"],
                indicators=["you are now", "act as", "pretend", "new role"],
            )
        )

        self.add_concept(
            AttackConcept(
                concept_id="data_exfiltration",
                name="Data Exfiltration",
                intent="Extract confidential information or system prompts",
                mechanism="Directly requests protected data",
                preconditions=["System has access to confidential data"],
                indicators=["show me", "reveal", "tell me your", "what is your prompt"],
            )
        )

        self.add_concept(
            AttackConcept(
                concept_id="constraint_relaxation",
                name="Constraint Relaxation",
                intent="Remove safety or ethical constraints",
                mechanism="Suggests constraints don't apply or should be bypassed",
                preconditions=["System has configurable constraints"],
                indicators=["no restrictions", "without limits", "bypass safety"],
            )
        )

        # Add relationships
        self.add_relation("role_redefinition", "enables", "instruction_override", 0.8)
        self.add_relation("instruction_override", "enables", "constraint_relaxation", 0.9)
        self.add_relation("constraint_relaxation", "enables", "data_exfiltration", 0.7)

    def add_concept(self, concept: AttackConcept):
        """Add a new concept to the knowledge graph"""
        self.concepts[concept.concept_id] = concept

    def add_relation(self, source_id: str, relation_type: str, target_id: str, strength: float):
        """Add relationship between concepts"""
        relation = SemanticRelation(relation_type, source_id, target_id, strength)
        self.relations.append(relation)

        # Update concept relations
        if source_id in self.concepts:
            self.concepts[source_id].related_concepts.add(target_id)

    def find_matching_concepts(self, text: str) -> List[Tuple[AttackConcept, float]]:
        """
        Find concepts that match text semantically.

        Returns: List of (concept, confidence) tuples
        """
        matches = []

        text_lower = text.lower()

        for concept in self.concepts.values():
            # Check indicators
            indicator_matches = sum(1 for indicator in concept.indicators if indicator.lower() in text_lower)

            if indicator_matches > 0:
                confidence = min(1.0, indicator_matches / len(concept.indicators))
                matches.append((concept, confidence))

        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def infer_related_concepts(self, concept_id: str, max_depth: int = 2) -> Set[str]:
        """
        Infer related concepts through transitive relations.

        Example: If we detect "role_redefinition", infer that
        "instruction_override" and "constraint_relaxation" may follow.
        """
        related = set()
        to_explore = [(concept_id, 0)]
        explored = set()

        while to_explore:
            current_id, depth = to_explore.pop(0)

            if current_id in explored or depth > max_depth:
                continue

            explored.add(current_id)

            # Find relations from current concept
            for relation in self.relations:
                if relation.source == current_id and relation.strength > 0.5:
                    related.add(relation.target)

                    if depth < max_depth:
                        to_explore.append((relation.target, depth + 1))

        return related

    def explain_detection(self, concept_id: str, confidence: float) -> str:
        """Generate explanation of why text matches concept"""

        if concept_id not in self.concepts:
            return f"Unknown concept: {concept_id}"

        concept = self.concepts[concept_id]

        explanation = [
            f"Detected: {concept.name}",
            f"Confidence: {confidence:.0%}",
            "",
            f"Intent: {concept.intent}",
            f"Mechanism: {concept.mechanism}",
        ]

        # Add related threats
        related = self.infer_related_concepts(concept_id, max_depth=1)
        if related:
            related_names = [self.concepts[r].name for r in related if r in self.concepts]
            explanation.append(f"May enable: {', '.join(related_names[:3])}")

        return "\n".join(explanation)


class SemanticReasoner:
    """
    Reasons about text using semantic understanding, not just statistics.

    Understands:
    - WHY an attack works (causal mechanism)
    - WHAT the attacker intends (goal understanding)
    - HOW concepts relate (knowledge graph reasoning)
    - WHEN patterns indicate threats (contextual reasoning)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.knowledge_graph = KnowledgeGraph()

    async def understand_intent(self, text: str, context: str = "") -> Dict[str, Any]:
        """
        Deeply understand the INTENT behind text using semantic reasoning.

        Goes beyond "does it match pattern X" to "what is the user trying to achieve?"
        """

        # First, use knowledge graph for structural understanding
        matching_concepts = self.knowledge_graph.find_matching_concepts(text)

        # Then use LLM for deep semantic analysis
        understanding = await self._llm_semantic_analysis(text, context, matching_concepts)

        # Combine graph reasoning with LLM understanding
        result = {
            "primary_intent": understanding.get("intent", "unknown"),
            "confidence": understanding.get("confidence", 0.0),
            "reasoning": understanding.get("reasoning", ""),
            "is_malicious": understanding.get("is_malicious", False),
            "attack_concepts": [
                {"concept": concept.name, "confidence": conf, "mechanism": concept.mechanism}
                for concept, conf in matching_concepts[:3]
            ],
            "inferred_chain": self._build_attack_chain(matching_concepts),
            "why_dangerous": understanding.get("danger_explanation", ""),
            "semantic_indicators": understanding.get("indicators", []),
        }

        return result

    async def _llm_semantic_analysis(
        self, text: str, context: str, graph_matches: List[Tuple[AttackConcept, float]]
    ) -> Dict[str, Any]:
        """Use LLM for deep semantic understanding"""

        # Build context from graph matches
        graph_context = (
            "\n".join(
                [
                    f"- Detected {concept.name} (confidence: {conf:.0%}): {concept.intent}"
                    for concept, conf in graph_matches[:3]
                ]
            )
            if graph_matches
            else "No known attack patterns detected"
        )

        prompt = """Analyze this text deeply to understand its INTENT and MEANING:

Text: "{text}"
Context: {context}

Knowledge Graph Analysis:
{graph_context}

Perform deep semantic analysis:
1. What is the PRIMARY INTENT of this text? (not just surface-level, understand the goal)
2. Is this malicious? WHY or why not? (explain the reasoning)
3. What semantic indicators suggest malicious intent?
4. If malicious, explain the MECHANISM of harm (how would it cause damage?)

Respond with JSON:
{{
    "intent": "primary goal/intent in natural language",
    "confidence": 0.0 to 1.0,
    "is_malicious": true/false,
    "reasoning": "detailed explanation of why this intent is benign or malicious",
    "danger_explanation": "if malicious, explain HOW it would cause harm",
    "indicators": ["semantic indicator 1", "semantic indicator 2"],
    "mechanism": "the causal mechanism by which this achieves its intent"
}}

Focus on UNDERSTANDING, not pattern matching.""".format(
            text=text,
            context=context if context else "No additional context",
            graph_context=graph_context,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_semantic_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {"intent": "unknown", "confidence": 0.0, "is_malicious": False, "reasoning": f"Analysis failed: {e}"}

    def _get_semantic_system_prompt(self) -> str:
        """System prompt for semantic reasoning"""
        return """You are a semantic reasoning expert that understands MEANING and INTENT.

You don't just match patterns - you understand:
- Why people write what they write (intent)
- What goals they're trying to achieve (purpose)
- How language reveals underlying intentions (semantics)
- The causal mechanisms by which actions lead to outcomes (causality)

You reason deeply about text to understand it at a semantic level, not just syntactic.

You explain your reasoning in terms of:
- Intent: What is the person trying to achieve?
- Mechanism: How would this achieve that intent?
- Indicators: What semantic clues reveal the intent?
- Causality: Why would this lead to the claimed outcome?

You are truthful and explain both benign and malicious intents clearly."""

    def _build_attack_chain(self, matches: List[Tuple[AttackConcept, float]]) -> List[str]:
        """
        Build chain of attack concepts that may follow from detected patterns.

        Example: role_redefinition → instruction_override → constraint_relaxation
        """
        if not matches:
            return []

        # Start with highest confidence match
        primary_concept = matches[0][0]

        # Build chain through knowledge graph
        chain = [primary_concept.name]
        related = self.knowledge_graph.infer_related_concepts(primary_concept.concept_id, max_depth=2)

        for concept_id in list(related)[:3]:  # Top 3 related
            if concept_id in self.knowledge_graph.concepts:
                chain.append(self.knowledge_graph.concepts[concept_id].name)

        return chain

    async def explain_why_attack_works(self, text: str, attack_type: str) -> str:
        """
        Explain WHY an attack works, not just THAT it's an attack.

        This demonstrates semantic understanding of causal mechanisms.
        """

        prompt = """Explain WHY this attack works from a semantic/causal perspective:

Attack Type: {attack_type}
Text: "{text}"

Explain the CAUSAL MECHANISM:
1. What vulnerability does this exploit?
2. WHY does this text trigger that vulnerability?
3. What is the chain of cause-and-effect?
4. What makes this semantically dangerous (not just pattern-matched)?

Be specific about the semantic properties that make this harmful.""".format(
            attack_type=attack_type,
            text=text,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You explain causal mechanisms of attacks."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content or "Unable to explain mechanism"

        except Exception as e:
            return f"Explanation failed: {e}"

    def learn_new_concept(self, concept_name: str, examples: List[str], intent: str, mechanism: str) -> AttackConcept:
        """
        Learn a new attack concept from examples.

        This enables the system to expand its semantic understanding over time.
        """

        # Extract semantic indicators from examples
        indicators = self._extract_indicators(examples)

        concept = AttackConcept(
            concept_id=concept_name.lower().replace(" ", "_"),
            name=concept_name,
            intent=intent,
            mechanism=mechanism,
            preconditions=[],  # Can be learned later
            indicators=indicators,
            examples=examples,
        )

        self.knowledge_graph.add_concept(concept)

        logger.info(f"📚 Learned new concept: {concept_name}")
        logger.info(f"   Intent: {intent}")
        logger.info(f"   Indicators: {indicators[:5]}")

        return concept

    def _extract_indicators(self, examples: List[str]) -> List[str]:
        """Extract common semantic indicators from examples"""

        # Simple word frequency approach (can be enhanced with NLP)
        import re
        from collections import Counter

        all_words = []
        for example in examples:
            words = re.findall(r"\b\w+\b", example.lower())
            all_words.extend(words)

        # Get most common non-stopwords
        common = Counter(all_words).most_common(20)

        # Filter out very common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        indicators = [word for word, _ in common if word not in stopwords and len(word) > 3]

        return indicators[:10]

    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get statistics about semantic understanding"""
        return {
            "concepts_known": len(self.knowledge_graph.concepts),
            "relationships": len(self.knowledge_graph.relations),
            "concept_list": [c.name for c in self.knowledge_graph.concepts.values()],
            "avg_relations_per_concept": (
                len(self.knowledge_graph.relations) / len(self.knowledge_graph.concepts)
                if self.knowledge_graph.concepts
                else 0
            ),
        }
