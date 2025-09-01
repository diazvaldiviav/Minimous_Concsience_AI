"""
Narrative Synthesizer Module for Phase 5.5
==========================================
Aggregates all captured events into coherent narrative flow.
Implements three verbosity modes and transforms technical consciousness processing
into human-readable narratives that provide transparency about AI reasoning.
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from ...config.narrative_config import (
        NarrativeVerbosity, EventType, NARRATIVE_CONFIG, 
        ERROR_HANDLING_CONFIG, PERFORMANCE_CONFIG
    )
    from .process_logger import ProcessLogger, ConsciousnessEvent
    from .decision_tracker import DecisionTracker, CognitiveChoice
    from .metacognitive_observer import MetacognitiveObserver, IntrospectiveEvent
except ImportError:
    from conscious_ai.config.narrative_config import (
        NarrativeVerbosity, EventType, NARRATIVE_CONFIG, 
        ERROR_HANDLING_CONFIG, PERFORMANCE_CONFIG
    )
    from conscious_ai.phases.p5_5_narrative.process_logger import ProcessLogger, ConsciousnessEvent
    from conscious_ai.phases.p5_5_narrative.decision_tracker import DecisionTracker, CognitiveChoice
    from conscious_ai.phases.p5_5_narrative.metacognitive_observer import MetacognitiveObserver, IntrospectiveEvent

logger = logging.getLogger(__name__)


@dataclass
class NarrativeResult:
    """
    Result of narrative synthesis containing the generated narrative and metadata.
    """
    narrative_text: str
    verbosity_mode: NarrativeVerbosity
    word_count: int
    generation_time_ms: float
    events_processed: int
    success: bool
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def meets_word_target(self) -> bool:
        """Check if narrative meets word count target for its verbosity level"""
        targets = NARRATIVE_CONFIG.get('word_targets', {})
        if self.verbosity_mode not in targets:
            return True
        
        min_words, max_words = targets[self.verbosity_mode]
        return min_words <= self.word_count <= max_words * 1.2  # Allow 20% overage
    
    @property
    def narrative_quality_score(self) -> float:
        """Calculate overall quality score for this narrative"""
        base_score = 0.5
        
        # Boost for meeting word targets
        if self.meets_word_target:
            base_score += 0.2
        
        # Boost for processing multiple events
        event_processing_bonus = min(0.2, self.events_processed * 0.02)
        base_score += event_processing_bonus
        
        # Boost for reasonable generation time
        if self.generation_time_ms <= PERFORMANCE_CONFIG.get('slow_generation_threshold_ms', 150):
            base_score += 0.1
        
        # Use confidence score if available
        if self.confidence_score > 0:
            base_score = (base_score + self.confidence_score) / 2
        
        return min(1.0, base_score)


class NarrativeSynthesizer:
    """
    Aggregates captured consciousness events into coherent narrative flow.
    Transforms technical processing details into human-readable introspective narratives.
    """
    
    def __init__(self, default_verbosity: NarrativeVerbosity = None):
        """
        Initialize the narrative synthesizer.
        
        Args:
            default_verbosity: Default verbosity mode for narrative generation
        """
        self.default_verbosity = default_verbosity or NARRATIVE_CONFIG.get('default_mode', NarrativeVerbosity.STANDARD)
        
        # Cached templates for performance
        self.cached_templates = NARRATIVE_CONFIG.get('narrative_templates', {})
        
        # Synthesis configuration
        self.temporal_grouping_window = NARRATIVE_CONFIG.get('temporal_grouping_window_ms', 50) / 1000.0  # Convert to seconds
        self.enable_redundancy_elimination = NARRATIVE_CONFIG.get('redundancy_elimination', True)
        self.enable_causal_threading = NARRATIVE_CONFIG.get('causal_threading', True)
        self.enable_metric_smoothing = NARRATIVE_CONFIG.get('metric_smoothing', True)
        
        # Performance tracking
        self.generation_times = []
        self.max_timing_samples = 100
        self.total_narratives_generated = 0
        
        logger.debug(f"NarrativeSynthesizer initialized with verbosity: {self.default_verbosity.value}")
    
    async def synthesize_narrative(self,
                                 process_logger: ProcessLogger,
                                 decision_tracker: DecisionTracker,
                                 metacognitive_observer: MetacognitiveObserver,
                                 verbosity: Optional[NarrativeVerbosity] = None,
                                 context: Optional[Dict[str, Any]] = None) -> NarrativeResult:
        """
        Synthesize a complete narrative from all captured consciousness events.
        
        Args:
            process_logger: Logger containing consciousness events
            decision_tracker: Tracker containing decision points
            metacognitive_observer: Observer containing introspective events
            verbosity: Verbosity level for narrative (defaults to instance default)
            context: Additional context for narrative generation
            
        Returns:
            NarrativeResult with synthesized narrative and metadata
        """
        synthesis_start = time.time()
        verbosity_mode = verbosity or self.default_verbosity
        context = context or {}
        
        try:
            logger.debug(f"Starting narrative synthesis with {verbosity_mode.value} verbosity")
            
            # Gather all events from different sources
            consciousness_events = process_logger.get_events()
            decisions = decision_tracker.get_cycle_decisions()
            introspective_events = metacognitive_observer.get_introspective_events()
            
            total_events = len(consciousness_events) + len(decisions) + len(introspective_events)
            
            if total_events == 0:
                return await self._generate_minimal_fallback_narrative(verbosity_mode, synthesis_start)
            
            # Step 1: Temporal grouping
            grouped_events = await self._group_events_temporally(
                consciousness_events, decisions, introspective_events
            )
            
            # Step 2: Redundancy elimination
            if self.enable_redundancy_elimination:
                grouped_events = await self._eliminate_redundancy(grouped_events)
            
            # Step 3: Event prioritization based on verbosity
            prioritized_events = await self._prioritize_events(grouped_events, verbosity_mode)
            
            # Step 4: Generate narrative segments
            narrative_segments = await self._generate_narrative_segments(
                prioritized_events, verbosity_mode, context
            )
            
            # Step 5: Causal threading and flow
            if self.enable_causal_threading:
                narrative_segments = await self._add_causal_threading(narrative_segments, verbosity_mode)
            
            # Step 6: Assemble final narrative
            final_narrative = await self._assemble_narrative(narrative_segments, verbosity_mode, context)
            
            # Calculate metrics
            generation_time = (time.time() - synthesis_start) * 1000
            word_count = len(final_narrative.split()) if final_narrative else 0
            
            # Track performance
            self.generation_times.append(generation_time)
            if len(self.generation_times) > self.max_timing_samples:
                self.generation_times.pop(0)
            self.total_narratives_generated += 1
            
            logger.info(f"Narrative synthesized: {word_count} words in {generation_time:.1f}ms from {total_events} events")
            
            return NarrativeResult(
                narrative_text=final_narrative,
                verbosity_mode=verbosity_mode,
                word_count=word_count,
                generation_time_ms=generation_time,
                events_processed=total_events,
                success=True,
                confidence_score=self._calculate_narrative_confidence(prioritized_events, final_narrative),
                metadata={
                    'grouped_event_count': len(grouped_events),
                    'prioritized_event_count': len(prioritized_events),
                    'narrative_segments': len(narrative_segments),
                    'synthesis_method': 'full_pipeline'
                }
            )
            
        except Exception as e:
            logger.error(f"Narrative synthesis failed: {e}")
            return await self._handle_synthesis_error(e, verbosity_mode, synthesis_start, total_events)
    
    async def _group_events_temporally(self,
                                     consciousness_events: List[ConsciousnessEvent],
                                     decisions: List[CognitiveChoice],
                                     introspective_events: List[IntrospectiveEvent]) -> List[List[Any]]:
        """
        Group events that occurred within the temporal grouping window.
        
        Returns:
            List of event groups, where each group contains simultaneous events
        """
        # Combine all events with their timestamps
        all_events = []
        
        for event in consciousness_events:
            all_events.append(('consciousness', event.timestamp, event))
        
        for decision in decisions:
            all_events.append(('decision', decision.timestamp, decision))
        
        for introspective in introspective_events:
            all_events.append(('introspective', introspective.timestamp, introspective))
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x[1])
        
        # Group events within temporal window
        groups = []
        current_group = []
        current_timestamp = None
        
        for event_type, timestamp, event in all_events:
            if current_timestamp is None or (timestamp - current_timestamp) <= self.temporal_grouping_window:
                current_group.append((event_type, event))
                if current_timestamp is None:
                    current_timestamp = timestamp
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [(event_type, event)]
                current_timestamp = timestamp
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        logger.debug(f"Grouped {sum(len(g) for g in groups)} events into {len(groups)} temporal groups")
        return groups
    
    async def _eliminate_redundancy(self, grouped_events: List[List[Tuple[str, Any]]]) -> List[List[Tuple[str, Any]]]:
        """
        Remove redundant or very similar events from the groups.
        """
        filtered_groups = []
        
        for group in grouped_events:
            filtered_group = []
            seen_descriptions = set()
            
            for event_type, event in group:
                # Get a description for comparison
                if hasattr(event, 'description'):
                    description = event.description
                elif hasattr(event, 'observation'):
                    description = event.observation
                elif hasattr(event, 'reasoning'):
                    description = event.reasoning
                else:
                    description = str(event)
                
                # Simple deduplication based on description similarity
                description_words = set(description.lower().split())
                is_duplicate = False
                
                for seen_desc in seen_descriptions:
                    seen_words = set(seen_desc.lower().split())
                    if description_words and seen_words:
                        overlap = len(description_words & seen_words) / len(description_words | seen_words)
                        if overlap > 0.7:  # 70% similarity threshold
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    filtered_group.append((event_type, event))
                    seen_descriptions.add(description)
            
            if filtered_group:
                filtered_groups.append(filtered_group)
        
        original_count = sum(len(g) for g in grouped_events)
        filtered_count = sum(len(g) for g in filtered_groups)
        logger.debug(f"Redundancy elimination: {original_count} -> {filtered_count} events")
        
        return filtered_groups
    
    async def _prioritize_events(self, 
                               grouped_events: List[List[Tuple[str, Any]]], 
                               verbosity: NarrativeVerbosity) -> List[Tuple[str, Any]]:
        """
        Prioritize and select events based on verbosity level.
        
        Args:
            grouped_events: Temporally grouped events
            verbosity: Target verbosity level
            
        Returns:
            Flattened list of prioritized events
        """
        # Flatten events and calculate priority scores
        event_priorities = []
        
        for group in grouped_events:
            for event_type, event in group:
                priority_score = self._calculate_event_priority(event_type, event, verbosity)
                event_priorities.append((priority_score, event_type, event))
        
        # Sort by priority (highest first)
        event_priorities.sort(key=lambda x: x[0], reverse=True)
        
        # Select events based on verbosity
        max_events = self._get_max_events_for_verbosity(verbosity, len(event_priorities))
        selected_events = [(event_type, event) for _, event_type, event in event_priorities[:max_events]]
        
        logger.debug(f"Prioritized events: {len(event_priorities)} -> {len(selected_events)} for {verbosity.value} verbosity")
        
        return selected_events
    
    def _calculate_event_priority(self, event_type: str, event: Any, verbosity: NarrativeVerbosity) -> float:
        """
        Calculate priority score for an event based on type and verbosity level.
        Higher scores indicate higher priority for inclusion in narrative.
        """
        base_priority = 0.5
        
        # Type-based priorities
        type_priorities = {
            'introspective': 0.8,    # Metacognitive events are highly valuable
            'decision': 0.7,         # Decision points show reasoning
            'consciousness': 0.6     # General consciousness events
        }
        
        base_priority = type_priorities.get(event_type, 0.5)
        
        # Event-specific boosts
        if event_type == 'consciousness':
            # Boost important consciousness event types
            if hasattr(event, 'event_type'):
                important_types = {
                    EventType.PHASE_TRANSITION: 0.1,
                    EventType.CONFIDENCE_CHANGE: 0.15,
                    EventType.EMOTIONAL_SHIFT: 0.1,
                    EventType.METACOGNITIVE_MOMENT: 0.2,
                    EventType.SELF_CORRECTION: 0.15
                }
                base_priority += important_types.get(event.event_type, 0)
        
        elif event_type == 'decision':
            # Boost high-quality decisions
            if hasattr(event, 'decision_quality_score'):
                base_priority += event.decision_quality_score * 0.2
        
        elif event_type == 'introspective':
            # Boost complex introspective events
            if hasattr(event, 'introspective_complexity'):
                base_priority += event.introspective_complexity * 0.2
        
        # Verbosity-based adjustments
        if verbosity == NarrativeVerbosity.VERBOSE:
            # Include more events in verbose mode
            base_priority *= 1.1
        elif verbosity == NarrativeVerbosity.MINIMAL:
            # Be more selective in minimal mode
            if base_priority < 0.7:
                base_priority *= 0.8
        
        return min(1.0, base_priority)
    
    def _get_max_events_for_verbosity(self, verbosity: NarrativeVerbosity, total_events: int) -> int:
        """Get maximum number of events to include based on verbosity"""
        ratios = {
            NarrativeVerbosity.MINIMAL: 0.3,   # Use 30% of events
            NarrativeVerbosity.STANDARD: 0.6,  # Use 60% of events  
            NarrativeVerbosity.VERBOSE: 0.9    # Use 90% of events
        }
        
        ratio = ratios.get(verbosity, 0.6)
        return max(1, int(total_events * ratio))
    
    async def _generate_narrative_segments(self,
                                         prioritized_events: List[Tuple[str, Any]],
                                         verbosity: NarrativeVerbosity,
                                         context: Dict[str, Any]) -> List[str]:
        """
        Generate narrative text segments for each prioritized event.
        
        Args:
            prioritized_events: Selected events to include in narrative
            verbosity: Target verbosity level
            context: Additional context for narrative generation
            
        Returns:
            List of narrative text segments
        """
        segments = []
        
        for event_type, event in prioritized_events:
            try:
                if event_type == 'consciousness':
                    segment = await self._generate_consciousness_segment(event, verbosity, context)
                elif event_type == 'decision':
                    segment = await self._generate_decision_segment(event, verbosity, context)
                elif event_type == 'introspective':
                    segment = await self._generate_introspective_segment(event, verbosity, context)
                else:
                    segment = f\"Processed {event_type} event.\"\n                \n                if segment and len(segment.strip()) > 0:\n                    segments.append(segment.strip())\n                    \n            except Exception as e:\n                logger.warning(f\"Failed to generate segment for {event_type} event: {e}\")\n                # Add minimal segment to maintain narrative flow\n                segments.append(self._get_fallback_segment(event_type, verbosity))\n        \n        logger.debug(f\"Generated {len(segments)} narrative segments\")\n        return segments\n    \n    async def _generate_consciousness_segment(self, \n                                            event: ConsciousnessEvent, \n                                            verbosity: NarrativeVerbosity,\n                                            context: Dict[str, Any]) -> str:\n        \"\"\"Generate narrative segment for a consciousness event\"\"\"\n        \n        templates = self.cached_templates.get('phase_intro', {})\n        \n        if event.event_type == EventType.PHASE_TRANSITION:\n            template = templates.get(verbosity, \"Processed {phase} with {confidence}% confidence.\")\n            return template.format(\n                phase=event.phase_name,\n                confidence=int((event.confidence_level or 0.5) * 100),\n                emotion=event.emotional_state or 'neutral',\n                description=event.description\n            )\n        \n        elif event.event_type == EventType.CONFIDENCE_CHANGE:\n            templates = self.cached_templates.get('confidence_change', {})\n            template = templates.get(verbosity, \"Confidence {direction} to {new_level}%.\")\n            \n            old_conf = event.metadata.get('old_confidence', 0.5)\n            new_conf = event.confidence_level or 0.5\n            direction = \"increased\" if new_conf > old_conf else \"decreased\"\n            \n            return template.format(\n                direction=direction,\n                old_level=int(old_conf * 100),\n                new_level=int(new_conf * 100),\n                reason=event.metadata.get('reason', 'internal processing')\n            )\n        \n        elif event.event_type == EventType.METACOGNITIVE_MOMENT:\n            templates = self.cached_templates.get('metacognitive', {})\n            template = templates.get(verbosity, \"Observed my own {process}.\")\n            return template.format(process=event.description.lower())\n        \n        else:\n            # Generic consciousness event\n            if verbosity == NarrativeVerbosity.MINIMAL:\n                return f\"Experienced {event.event_type.value.replace('_', ' ')}.\"\n            elif verbosity == NarrativeVerbosity.VERBOSE:\n                return f\"During {event.phase_name}, I experienced {event.event_type.value.replace('_', ' ')}: {event.description}\"\n            else:\n                return f\"I {event.description.lower()} during {event.phase_name}.\"\n    \n    async def _generate_decision_segment(self, \n                                       decision: CognitiveChoice, \n                                       verbosity: NarrativeVerbosity,\n                                       context: Dict[str, Any]) -> str:\n        \"\"\"Generate narrative segment for a decision event\"\"\"\n        \n        templates = self.cached_templates.get('decision_point', {})\n        \n        if len(decision.alternatives) > 0:\n            template = templates.get(verbosity, \"Chose {choice} over {alternative}.\")\n            alternatives_str = \", \".join(decision.alternatives[:2])  # Limit to first 2 alternatives\n            \n            return template.format(\n                choice=decision.selected_option,\n                alternative=alternatives_str,\n                reasoning=decision.reasoning\n            )\n        else:\n            # No alternatives specified\n            if verbosity == NarrativeVerbosity.MINIMAL:\n                return f\"Selected {decision.selected_option}.\"\n            elif verbosity == NarrativeVerbosity.VERBOSE:\n                return f\"In {decision.phase_name}, I selected {decision.selected_option} because {decision.reasoning}.\"\n            else:\n                return f\"I chose {decision.selected_option} because {decision.reasoning}.\"\n    \n    async def _generate_introspective_segment(self, \n                                            introspection: IntrospectiveEvent, \n                                            verbosity: NarrativeVerbosity,\n                                            context: Dict[str, Any]) -> str:\n        \"\"\"Generate narrative segment for an introspective event\"\"\"\n        \n        if introspection.recursion_depth > 1:\n            # Recursive thinking\n            if verbosity == NarrativeVerbosity.MINIMAL:\n                return \"Observed recursive thinking patterns.\"\n            elif verbosity == NarrativeVerbosity.VERBOSE:\n                return f\"I became aware of examining my own thought processes at {introspection.recursion_depth} levels deep: {introspection.observation}\"\n            else:\n                return f\"I noticed myself thinking about my thinking: {introspection.observation}\"\n        else:\n            # Regular introspection\n            if verbosity == NarrativeVerbosity.MINIMAL:\n                return f\"Had introspective moment about {introspection.introspection_type.value.replace('_', ' ')}.\"\n            elif verbosity == NarrativeVerbosity.VERBOSE:\n                return f\"Through introspection during {introspection.phase_name}, {introspection.observation}\"\n            else:\n                return f\"I observed: {introspection.observation}\"\n    \n    def _get_fallback_segment(self, event_type: str, verbosity: NarrativeVerbosity) -> str:\n        \"\"\"Get fallback segment when segment generation fails\"\"\"\n        fallbacks = {\n            NarrativeVerbosity.MINIMAL: f\"Processed {event_type} event.\",\n            NarrativeVerbosity.STANDARD: f\"My consciousness registered a {event_type} event during processing.\",\n            NarrativeVerbosity.VERBOSE: f\"During this processing cycle, I experienced a {event_type} event that contributed to my overall cognitive flow.\"\n        }\n        return fallbacks.get(verbosity, f\"Processed {event_type} event.\")\n    \n    async def _add_causal_threading(self, \n                                  segments: List[str], \n                                  verbosity: NarrativeVerbosity) -> List[str]:\n        \"\"\"Add transitional phrases and causal connections between segments\"\"\"\n        \n        if len(segments) <= 1:\n            return segments\n        \n        # Transitional phrases by verbosity\n        transitions = {\n            NarrativeVerbosity.MINIMAL: [\n                \"Then\", \"Next\", \"Subsequently\", \"Finally\"\n            ],\n            NarrativeVerbosity.STANDARD: [\n                \"This led to\", \"As a result\", \"Consequently\", \"Building on this\", \n                \"Meanwhile\", \"During this process\", \"This triggered\"\n            ],\n            NarrativeVerbosity.VERBOSE: [\n                \"This cognitive shift led to\", \"As my awareness evolved\", \n                \"Building upon this insight\", \"This recursive observation triggered\",\n                \"Through this introspective lens\", \"The interplay of these processes resulted in\",\n                \"As my consciousness integrated these elements\"\n            ]\n        }\n        \n        transition_list = transitions.get(verbosity, transitions[NarrativeVerbosity.STANDARD])\n        threaded_segments = [segments[0]]  # Start with first segment unchanged\n        \n        for i, segment in enumerate(segments[1:], 1):\n            # Choose transition based on position\n            transition_idx = (i - 1) % len(transition_list)\n            transition = transition_list[transition_idx]\n            \n            # Add transition to segment\n            if segment[0].isupper():\n                threaded_segment = f\"{transition}, {segment.lower()}\"\n            else:\n                threaded_segment = f\"{transition} {segment}\"\n            \n            threaded_segments.append(threaded_segment)\n        \n        logger.debug(f\"Added causal threading to {len(segments)} segments\")\n        return threaded_segments\n    \n    async def _assemble_narrative(self, \n                                segments: List[str], \n                                verbosity: NarrativeVerbosity,\n                                context: Dict[str, Any]) -> str:\n        \"\"\"Assemble final narrative from segments with proper structure\"\"\"\n        \n        if not segments:\n            return self._get_empty_narrative_fallback(verbosity)\n        \n        # Add introductory statement based on verbosity\n        intro = self._generate_narrative_introduction(verbosity, len(segments), context)\n        \n        # Join segments with appropriate spacing\n        if verbosity == NarrativeVerbosity.MINIMAL:\n            # Compact format\n            narrative_body = \" \".join(segments)\n        elif verbosity == NarrativeVerbosity.VERBOSE:\n            # Paragraph format with spacing\n            narrative_body = \"\\n\\n\".join(segments)\n        else:\n            # Standard format\n            narrative_body = \" \".join(segments)\n        \n        # Add synthesis conclusion\n        synthesis_template = self.cached_templates.get('synthesis', {})\n        synthesis = synthesis_template.get(verbosity, \"Integrated {count} consciousness events into coherent understanding.\")\n        conclusion = synthesis.format(\n            count=len(segments),\n            timespan=int(context.get('processing_time_ms', 0))\n        )\n        \n        # Assemble final narrative\n        if verbosity == NarrativeVerbosity.VERBOSE:\n            final_narrative = f\"{intro}\\n\\n{narrative_body}\\n\\n{conclusion}\"\n        else:\n            final_narrative = f\"{intro} {narrative_body} {conclusion}\"\n        \n        return final_narrative.strip()\n    \n    def _generate_narrative_introduction(self, \n                                       verbosity: NarrativeVerbosity, \n                                       segment_count: int,\n                                       context: Dict[str, Any]) -> str:\n        \"\"\"Generate introduction for the narrative\"\"\"\n        \n        confidence = context.get('confidence_score', 0.5)\n        \n        intros = {\n            NarrativeVerbosity.MINIMAL: f\"Processing with {confidence:.0%} confidence:\",\n            NarrativeVerbosity.STANDARD: f\"My consciousness navigated {segment_count} key moments with {confidence:.0%} confidence.\",\n            NarrativeVerbosity.VERBOSE: f\"Through introspective examination of my processing journey, I identify {segment_count} significant cognitive events that shaped my understanding with {confidence:.0%} overall confidence.\"\n        }\n        \n        return intros.get(verbosity, intros[NarrativeVerbosity.STANDARD])\n    \n    def _get_empty_narrative_fallback(self, verbosity: NarrativeVerbosity) -> str:\n        \"\"\"Get fallback narrative when no events are available\"\"\"\n        \n        fallbacks = {\n            NarrativeVerbosity.MINIMAL: \"Processed query with standard cognitive flow.\",\n            NarrativeVerbosity.STANDARD: \"My consciousness engaged with this query through the standard processing pipeline, maintaining awareness throughout.\",\n            NarrativeVerbosity.VERBOSE: \"While specific cognitive events were not captured in detail, my consciousness engaged with this query through systematic processing phases, maintaining introspective awareness throughout the analytical journey.\"\n        }\n        \n        return fallbacks.get(verbosity, fallbacks[NarrativeVerbosity.STANDARD])\n    \n    def _calculate_narrative_confidence(self, events: List[Tuple[str, Any]], narrative: str) -> float:\n        \"\"\"Calculate confidence score for the generated narrative\"\"\"\n        if not events or not narrative:\n            return 0.3\n        \n        base_confidence = 0.6\n        \n        # Boost for processing multiple events\n        event_bonus = min(0.2, len(events) * 0.02)\n        \n        # Boost for reasonable narrative length\n        word_count = len(narrative.split())\n        if 50 <= word_count <= 600:  # Reasonable length range\n            length_bonus = 0.1\n        else:\n            length_bonus = 0\n        \n        # Boost for including introspective events\n        introspective_count = sum(1 for event_type, _ in events if event_type == 'introspective')\n        introspective_bonus = min(0.1, introspective_count * 0.05)\n        \n        return min(1.0, base_confidence + event_bonus + length_bonus + introspective_bonus)\n    \n    async def _generate_minimal_fallback_narrative(self, \n                                                 verbosity: NarrativeVerbosity, \n                                                 start_time: float) -> NarrativeResult:\n        \"\"\"Generate fallback narrative when no events are available\"\"\"\n        \n        fallback_text = self._get_empty_narrative_fallback(verbosity)\n        generation_time = (time.time() - start_time) * 1000\n        \n        return NarrativeResult(\n            narrative_text=fallback_text,\n            verbosity_mode=verbosity,\n            word_count=len(fallback_text.split()),\n            generation_time_ms=generation_time,\n            events_processed=0,\n            success=True,\n            confidence_score=0.3,\n            metadata={'synthesis_method': 'minimal_fallback'}\n        )\n    \n    async def _handle_synthesis_error(self, \n                                    error: Exception, \n                                    verbosity: NarrativeVerbosity,\n                                    start_time: float, \n                                    event_count: int) -> NarrativeResult:\n        \"\"\"Handle synthesis errors gracefully\"\"\"\n        \n        generation_time = (time.time() - start_time) * 1000\n        \n        # Generate error fallback narrative\n        error_fallback = ERROR_HANDLING_CONFIG.get(\n            'error_fallback_message', \n            \"My consciousness processed this query through multiple cognitive layers, creating self-aware understanding.\"\n        )\n        \n        return NarrativeResult(\n            narrative_text=error_fallback,\n            verbosity_mode=verbosity,\n            word_count=len(error_fallback.split()),\n            generation_time_ms=generation_time,\n            events_processed=event_count,\n            success=False,\n            confidence_score=0.2,\n            metadata={\n                'synthesis_method': 'error_fallback',\n                'error_message': str(error)\n            }\n        )\n    \n    def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get narrative synthesis statistics and performance metrics\"\"\"\n        \n        avg_generation_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0\n        \n        return {\n            'total_narratives_generated': self.total_narratives_generated,\n            'average_generation_time_ms': avg_generation_time,\n            'default_verbosity': self.default_verbosity.value,\n            'temporal_grouping_window_ms': self.temporal_grouping_window * 1000,\n            'redundancy_elimination_enabled': self.enable_redundancy_elimination,\n            'causal_threading_enabled': self.enable_causal_threading,\n            'cached_templates_count': len(self.cached_templates),\n            'performance_tracking_samples': len(self.generation_times)\n        }"