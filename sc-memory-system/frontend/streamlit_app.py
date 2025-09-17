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
    # Use persistent user/chat IDs to maintain memory across sessions
    if "user_id" not in st.session_state:
        # Generate a stable user ID based on a deterministic seed
        # In production, this would be based on actual user authentication
        st.session_state.user_id = "demo_user_001"  # Fixed demo user
    if "chat_id" not in st.session_state:
        # Use a session-specific but stable chat ID
        # This allows for session persistence while supporting multiple chats per user
        # In production, would be managed by proper session/chat management
        st.session_state.chat_id = "demo_chat_001"  # Fixed demo chat
    if "provider" not in st.session_state:
        # Consistent provider for both consolidation and retrieval
        st.session_state.provider = "openai"  # Can be made configurable
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = "your-api-key-here"  # Replace with your actual API key

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

# Monitor training progress
def check_training_status(proposal_id: str) -> Dict:
    """Check training status from MEP API."""
    try:
        response = requests.get(
            f"http://localhost:8000/mep/v1/proposals/{proposal_id}/status",
            headers={"Authorization": "Bearer dev-bearer-token"}
        )
        if response.ok:
            return response.json()
        return {"status": "unknown", "error": f"Status check failed: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

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
        "provider": st.session_state.provider,  # Use consistent provider from session state
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
            headers={"Authorization": "Bearer dev-bearer-token"}
        )

        if response.ok:
            result = response.json()
            proposal_id = result.get("proposal_id")
            st.session_state.adapter_id = proposal_id
            st.session_state.consolidated = True

            # Monitor training progress
            progress_placeholder = st.empty()
            logs_placeholder = st.empty()

            # Training progress monitoring
            max_wait_time = 120  # 2 minutes max
            check_interval = 2   # Check every 2 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                status_info = check_training_status(proposal_id)

                # Update progress display
                with progress_placeholder.container():
                    st.info(f"🔄 Training Status: {status_info.get('status', 'unknown').upper()}")
                    st.progress(min(elapsed_time / max_wait_time, 1.0))
                    st.caption(f"Elapsed: {elapsed_time}s / {max_wait_time}s")

                # Update logs display
                with logs_placeholder.container():
                    if "logs" in status_info and status_info["logs"]:
                        st.text_area("📝 Training Logs:", value=status_info["logs"], height=200, disabled=True)
                    elif "error" in status_info:
                        st.error(f"❌ Training Error: {status_info['error']}")
                    else:
                        st.info("⏳ Waiting for training logs...")

                # Check if training completed
                if status_info.get("status") in ["completed", "failed", "error"]:
                    break

                time.sleep(check_interval)
                elapsed_time += check_interval

            # Clear progress displays
            progress_placeholder.empty()
            logs_placeholder.empty()

            # IMPORTANT: Clear context after consolidation
            st.session_state.messages = []
            st.session_state.context_tokens = 0

            # Final status
            final_status = check_training_status(proposal_id)
            if final_status.get("status") == "completed":
                st.success("✅ Training completed successfully!")
                return True
            else:
                st.error(f"❌ Training failed: {final_status.get('error', 'Unknown error')}")
                return False

        else:
            error_details = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            st.error(f"Consolidation failed: {response.status_code} - {error_details}")
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
                "provider": st.session_state.provider,  # Use consistent provider from session state
                "external_user_id": st.session_state.user_id,
                "external_chat_id": st.session_state.chat_id,  # CRITICAL FIX: Include chat_id
                "query": user_input,
                "token_budget": 320,
                "min_truth": 0.75,
                "format": "json"
            },
            headers={"Authorization": "Bearer dev-bearer-token"}
        )

        if not map_response.ok:
            try:
                error_details = map_response.json()
                error_msg = f"MAP API error {map_response.status_code}: {error_details.get('error', {}).get('message', 'Unknown error')}"
                if error_details.get('error', {}).get('details'):
                    error_msg += f" | Details: {error_details['error']['details']}"
                return error_msg, 0, None
            except:
                return f"MAP API error: {map_response.status_code} - {map_response.text}", 0, None

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

        # Add user message to NEW context (post-consolidation)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Build messages: system prompt with compressed memory + new conversation
        messages = [
            {"role": "system", "content": f"You are an AI assistant. Here's compressed memory from our previous conversation:\n\n{compressed_prompt}\n\nNow continue our conversation naturally, building on this memory."}
        ]

        # Add NEW conversation context (accumulated since consolidation)
        messages.extend(st.session_state.messages)

        # Call GPT with compressed memory + new context
        client = OpenAI(api_key=st.session_state.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Add assistant response to context for continuity
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Update token counts
        st.session_state.context_tokens = calculate_context_tokens()
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
                with st.spinner("Consolidating memory... (may take 1-2 minutes)"):
                    if consolidate_to_sc_memory():
                        st.success("✅ Memory consolidated! Context cleared.")
                        st.rerun()
                    else:
                        st.error("❌ Consolidation failed")
        else:
            st.success("✅ Memory Consolidated")
            if st.session_state.adapter_id:
                st.caption(f"Adapter ID: {st.session_state.adapter_id[:8]}...")

        st.divider()

        # Debug section
        st.header("🔧 Debug Info")

        # Session state information
        with st.expander("📋 Session State"):
            st.text(f"User ID: {st.session_state.user_id}")
            st.text(f"Chat ID: {st.session_state.chat_id}")
            st.text(f"Provider: {st.session_state.provider}")
            st.text(f"Consolidated: {st.session_state.consolidated}")
            if st.session_state.adapter_id:
                st.text(f"Adapter ID: {st.session_state.adapter_id}")
            st.caption("These IDs are used for memory consolidation and retrieval")

        # Check backend health
        try:
            health_response = requests.get("http://localhost:8000/health", timeout=2)
            if health_response.ok:
                st.success("✅ Backend Online")
                health_data = health_response.json()
                st.caption(f"Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
            else:
                st.error(f"❌ Backend Error: {health_response.status_code}")
        except Exception as e:
            st.error(f"❌ Backend Offline: {str(e)}")

        # Check MAP API specifically
        if st.session_state.consolidated:
            try:
                test_response = requests.get(
                    "http://localhost:8000/map/v1/context",
                    params={
                        "provider": st.session_state.provider,  # Use consistent provider
                        "external_user_id": st.session_state.user_id,
                        "external_chat_id": st.session_state.chat_id,  # Include chat_id
                        "query": "test",
                        "token_budget": 100
                    },
                    headers={"Authorization": "Bearer dev-bearer-token"},
                    timeout=3
                )
                if test_response.ok:
                    st.success("✅ MAP API Working")
                else:
                    error_details = test_response.json() if test_response.headers.get('content-type') == 'application/json' else test_response.text
                    st.error(f"❌ MAP API Error: {error_details}")
            except Exception as e:
                st.error(f"❌ MAP API Failed: {str(e)}")

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