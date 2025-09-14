"""
SC Memory System - Streamlit Comparison Demo
Demonstrates token savings by comparing standard context vs SC Memory compressed approach
"""

import streamlit as st
import openai
from openai import OpenAI
import requests
import json
import time
from datetime import datetime
import tiktoken
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="SC Memory System - Comparison Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context_tokens" not in st.session_state:
        st.session_state.context_tokens = 0
    if "total_tokens_before" not in st.session_state:
        st.session_state.total_tokens_before = 0
    if "total_tokens_after" not in st.session_state:
        st.session_state.total_tokens_after = 0
    if "consolidated" not in st.session_state:
        st.session_state.consolidated = False
    if "adapter_id" not in st.session_state:
        st.session_state.adapter_id = None
    if "comparison_data" not in st.session_state:
        st.session_state.comparison_data = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = f"chat_{int(time.time())}"
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

init_session_state()

# Token counting function
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens for given text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4

# Calculate total context tokens
def calculate_context_tokens() -> int:
    """Calculate total tokens in current message context."""
    total = 0
    for msg in st.session_state.messages:
        total += count_tokens(msg["content"])
    return total

# Extract key facts from conversation
def extract_key_facts(messages: List[Dict]) -> List[Dict]:
    """Extract key facts from conversation messages."""
    facts = []

    # Simple extraction: look for statements that seem factual
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            # Split into sentences
            sentences = msg["content"].split('.')
            for sentence in sentences:
                # Skip short sentences
                if len(sentence.strip()) > 20:
                    facts.append({
                        "claim": sentence.strip() + ".",
                        "importance": 0.8 - (i * 0.05),  # Decrease importance for older messages
                        "confidence": 0.75,
                        "category": "conversation"
                    })

    # Limit to 10 most important facts
    return facts[:10] if facts else [
        {
            "claim": "No specific facts extracted",
            "importance": 0.5,
            "confidence": 0.5,
            "category": "general"
        }
    ]

# Chat with GPT-4o-mini using standard context
def chat_with_standard_context(user_input: str) -> Tuple[str, int]:
    """
    Chat with GPT-4o-mini using standard context approach.
    Returns: (response_text, tokens_used)
    """
    client = OpenAI(api_key=st.session_state.openai_api_key)

    # Add user message to context
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # Call OpenAI with FULL message history (standard approach)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
            temperature=0.7,
            max_tokens=500
        )

        assistant_response = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Add assistant response to context
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Update token counts
        st.session_state.context_tokens = calculate_context_tokens()
        st.session_state.total_tokens_before += tokens_used

        return assistant_response, tokens_used

    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        return "Error: Could not get response", 0

# Consolidate memory to SC Memory System
def consolidate_to_sc_memory() -> bool:
    """
    Send current conversation to MEP for consolidation.
    Returns: success status
    """
    # Build conversation summary
    conversation_lines = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_lines.append(f"{role}: {msg['content']}")

    summary_text = " ".join(conversation_lines)[:1500]  # Limit to 1500 chars

    # Extract facts
    facts = extract_key_facts(st.session_state.messages)

    # Build MEP proposal
    proposal = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "external_user_id": st.session_state.user_id,
        "external_chat_id": st.session_state.chat_id,
        "event_id": f"evt_{int(time.time())}",
        "trigger": "manual",
        "context_fill": min(st.session_state.context_tokens / 4096, 1.0),
        "token_usage": {
            "window_tokens": 4096,
            "used_tokens": st.session_state.context_tokens,
            "max_tokens": 8192
        },
        "message_span": {
            "from_turn": 0,
            "to_turn": len(st.session_state.messages) - 1
        },
        "summary_text": summary_text,
        "key_facts": facts
    }

    # Send to MEP API
    try:
        response = requests.post(
            "http://localhost:8000/mep/v1/proposals",
            json=proposal,
            headers={"Authorization": "Bearer test_token"}
        )

        if response.ok:
            result = response.json()
            st.session_state.adapter_id = result.get("proposal_id")
            st.session_state.consolidated = True

            # IMPORTANT: Clear context after consolidation
            st.session_state.messages = []
            st.session_state.context_tokens = 0

            return True
        else:
            st.error(f"Consolidation failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        st.error(f"MEP API error: {str(e)}")
        return False

# Query with SC Memory
def chat_with_sc_memory(user_input: str) -> Tuple[str, int, Optional[Dict]]:
    """
    Chat using SC Memory compressed context.
    Returns: (response_text, tokens_used, memory_context)
    """
    if not st.session_state.consolidated:
        return "No memory consolidated yet. Please consolidate first.", 0, None

    # Query MAP API for compressed context
    try:
        map_response = requests.get(
            "http://localhost:8000/map/v1/context",
            params={
                "provider": "openai",
                "external_user_id": st.session_state.user_id,
                "query": user_input,
                "token_budget": 320,
                "min_truth": 0.75,
                "format": "json"
            },
            headers={"Authorization": "Bearer test_token"}
        )

        if not map_response.ok:
            return f"MAP API error: {map_response.status_code}", 0, None

        memory_context = map_response.json()

        if not memory_context.get("has_memory"):
            return "No relevant memory found", 0, memory_context

        # Build compressed prompt
        compressed_prompt = f"""Based on our previous conversation memory:

SUMMARY: {memory_context.get('gist', 'No summary available')}

KEY POINTS:
{chr(10).join([f"- {turn.get('t', turn.get('content', ''))}" for turn in memory_context.get('turns', [])[:5]])}

FACTS:
{chr(10).join([f"- {fact.get('c', fact.get('claim', ''))}" for fact in memory_context.get('facts', [])[:5]])}

User Question: {user_input}

Please answer based on the compressed memory context above."""

        # Call GPT with compressed context
        client = OpenAI(api_key=st.session_state.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are continuing a conversation using compressed memory context."},
                {"role": "user", "content": compressed_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Track tokens for comparison
        st.session_state.total_tokens_after += tokens_used

        return answer, tokens_used, memory_context

    except Exception as e:
        st.error(f"SC Memory error: {str(e)}")
        return "Error accessing memory", 0, None

# Main UI
def main():
    st.title("🧠 SC Memory System - Comparison Demo")
    st.markdown("**Compare standard context memory vs. SC Memory compression**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # OpenAI API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for GPT-4o-mini",
            value=st.session_state.openai_api_key
        )
        if api_key:
            st.session_state.openai_api_key = api_key

        st.divider()

        # Token metrics
        st.header("📊 Token Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Current Context",
                f"{st.session_state.context_tokens} tokens",
                help="Tokens in current conversation"
            )
        with col2:
            context_fill = (st.session_state.context_tokens / 4096) * 100 if st.session_state.context_tokens > 0 else 0
            st.metric(
                "Context Fill",
                f"{context_fill:.1f}%",
                help="Percentage of context window used"
            )

        st.divider()

        # Consolidation button
        st.header("🔄 Memory Consolidation")

        if not st.session_state.consolidated:
            st.info("Chat normally until you want to consolidate memory")

            if st.button(
                "🚀 Consolidate to SC Memory",
                type="primary",
                disabled=len(st.session_state.messages) < 2,
                help="Send conversation to LoRA training"
            ):
                with st.spinner("Consolidating memory... (15-20 seconds)"):
                    if consolidate_to_sc_memory():
                        st.success("✅ Memory consolidated! Context cleared.")
                        time.sleep(15)  # Wait for consolidation
                        st.rerun()
                    else:
                        st.error("❌ Consolidation failed")
        else:
            st.success("✅ Memory Consolidated")
            if st.session_state.adapter_id:
                st.caption(f"Adapter ID: {st.session_state.adapter_id[:8]}...")

        st.divider()

        # Comparison metrics
        if st.session_state.consolidated:
            st.header("💰 Comparison")

            before_tokens = st.session_state.total_tokens_before
            after_tokens = st.session_state.total_tokens_after

            if before_tokens > 0 and after_tokens > 0:
                savings = ((before_tokens - after_tokens) / before_tokens) * 100
                st.metric(
                    "Token Savings",
                    f"{savings:.1f}%",
                    f"-{before_tokens - after_tokens} tokens"
                )

    # Main chat interface
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
        return

    # Mode indicator
    if st.session_state.consolidated:
        st.info("🧠 **Mode: SC Memory** - Using compressed context from LoRA adapter")
    else:
        st.info("💬 **Mode: Standard Context** - Using full conversation history")

    # Display conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get response based on mode
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.consolidated:
                    # Use SC Memory
                    response, tokens, memory_ctx = chat_with_sc_memory(prompt)

                    # Show memory context in expander
                    if memory_ctx:
                        with st.expander("🔍 Memory Context Used"):
                            st.json(memory_ctx)
                else:
                    # Use standard context
                    response, tokens = chat_with_standard_context(prompt)

                st.write(response)
                st.caption(f"Tokens used: {tokens}")

    # Token comparison chart
    if st.session_state.total_tokens_before > 0 or st.session_state.total_tokens_after > 0:
        with st.expander("📊 Token Usage Comparison"):
            fig = go.Figure(data=[
                go.Bar(
                    name='Standard Context',
                    x=['Total Tokens'],
                    y=[st.session_state.total_tokens_before],
                    marker_color='red'
                ),
                go.Bar(
                    name='SC Memory',
                    x=['Total Tokens'],
                    y=[st.session_state.total_tokens_after],
                    marker_color='green'
                )
            ])
            fig.update_layout(
                title="Token Usage: Standard vs SC Memory",
                yaxis_title="Tokens",
                barmode='group',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show savings summary
            if st.session_state.total_tokens_before > 0 and st.session_state.total_tokens_after > 0:
                savings_pct = ((st.session_state.total_tokens_before - st.session_state.total_tokens_after) / st.session_state.total_tokens_before) * 100
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Standard Tokens", st.session_state.total_tokens_before)
                with col2:
                    st.metric("SC Memory Tokens", st.session_state.total_tokens_after)
                with col3:
                    st.metric("Savings", f"{savings_pct:.1f}%", f"-{st.session_state.total_tokens_before - st.session_state.total_tokens_after}")

if __name__ == "__main__":
    main()