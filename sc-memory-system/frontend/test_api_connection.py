"""
Simple test script to validate API connections for SC Memory System frontend.
Tests both MEP and MAP endpoints to ensure backend is running correctly.
"""

import requests
import json
import time

# Test configuration
BACKEND_URL = "http://localhost:8001"
AUTH_TOKEN = "dev-bearer-token"
TEST_USER_ID = f"test_user_{int(time.time())}"
TEST_CHAT_ID = f"test_chat_{int(time.time())}"

def test_health_check():
    """Test if the backend is running."""
    print("Testing backend health...")
    try:
        response = requests.get(f"{BACKEND_URL}/mep/v1/health")
        if response.ok:
            print("+ Backend is running!")
            return True
        else:
            print(f"- Health check failed: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("X Cannot connect to backend. Is it running on port 8000?")
        return False

def test_mep_endpoint():
    """Test MEP proposal submission."""
    print("\nTesting MEP endpoint...")

    # Sample proposal
    proposal = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "external_user_id": TEST_USER_ID,
        "external_chat_id": TEST_CHAT_ID,
        "event_id": f"evt_{int(time.time())}",
        "trigger": "manual",
        "context_fill": 0.5,
        "token_usage": {
            "window_tokens": 4096,
            "used_tokens": 2000,
            "max_tokens": 8192
        },
        "message_span": {
            "from_turn": 0,
            "to_turn": 5
        },
        "summary_text": "Test conversation about physics and mathematics",
        "key_facts": [
            {
                "claim": "Einstein's equation E=mc² shows mass-energy equivalence",
                "importance": 0.9,
                "confidence": 0.95,
                "category": "physics"
            }
        ]
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/mep/v1/proposals",
            json=proposal,
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
        )

        if response.ok:
            result = response.json()
            print(f"+ MEP endpoint working! Proposal ID: {result.get('proposal_id', 'unknown')}")
            return result.get('proposal_id')
        else:
            print(f"- MEP endpoint failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"- MEP endpoint error: {e}")
        return None

def test_map_endpoint():
    """Test MAP context retrieval."""
    print("\nTesting MAP endpoint...")

    try:
        response = requests.get(
            f"{BACKEND_URL}/map/v1/context",
            params={
                "provider": "openai",
                "external_user_id": TEST_USER_ID,
                "query": "What did we discuss about physics?",
                "token_budget": 320,
                "min_truth": 0.75,
                "format": "json"
            },
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
        )

        if response.ok:
            result = response.json()
            has_memory = result.get("has_memory", False)
            print(f"+ MAP endpoint working! Has memory: {has_memory}")
            if has_memory:
                print(f"GIST: {result.get('gist', 'No gist')[:100]}...")
            return True
        else:
            print(f"- MAP endpoint failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"- MAP endpoint error: {e}")
        return False

def main():
    """Run all tests."""
    print("SC Memory System - API Connection Test")
    print("=" * 50)

    # Test backend connection
    if not test_health_check():
        print("\n- Backend not available. Please start it with:")
        print("   cd sc-memory-system")
        print("   uvicorn src.api.main:app --reload --port 8000")
        return

    # Test MEP endpoint
    proposal_id = test_mep_endpoint()

    # Test MAP endpoint
    test_map_endpoint()

    print("\n" + "=" * 50)
    print("Test Summary:")
    print("- Health check: +")
    print(f"- MEP endpoint: {'+' if proposal_id else '-'}")
    print("- MAP endpoint: +")

    if proposal_id:
        print("\nReady to run Streamlit frontend!")
        print("   cd frontend")
        print("   streamlit run streamlit_app.py")
    else:
        print("\nSome endpoints failed. Check backend logs.")

if __name__ == "__main__":
    main()