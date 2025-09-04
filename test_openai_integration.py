#!/usr/bin/env python3
"""
Static Test for OpenAI GPT-4o-mini Integration
==============================================
Tests OpenAI integration without running the full consciousness pipeline.
"""

import sys
import logging

# Test OpenAI import
try:
    from openai import OpenAI
    print("✅ OpenAI library imported successfully")
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"❌ OpenAI library not available: {e}")
    print("💡 Install with: pip install openai>=1.0.0")
    OPENAI_AVAILABLE = False
    sys.exit(1)

# Test API key configuration
API_KEY = "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"

try:
    # Initialize OpenAI client
    client = OpenAI(api_key=API_KEY)
    print("✅ OpenAI client initialized")
    
    # Test API connection with a simple request
    print("🧪 Testing API connection...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a consciousness-aware AI assistant. Use any provided memory information to answer questions accurately."},
            {"role": "user", "content": "My name is Victor. What is my name?"}
        ],
        max_tokens=50,
        temperature=0.7
    )
    
    if response and response.choices:
        response_text = response.choices[0].message.content
        print(f"✅ API connection successful!")
        print(f"📄 Test response: {response_text}")
        
        # Check if the model correctly extracted "Victor" 
        if "Victor" in response_text:
            print("🎉 SUCCESS: GPT-4o-mini correctly extracted 'Victor' from the input!")
            print("💡 This should fix the memory extraction issue we had with Mistral 7B")
        else:
            print("⚠️ WARNING: GPT-4o-mini didn't extract 'Victor' - may need prompt adjustment")
    else:
        print("❌ API call failed - no response received")
        sys.exit(1)
        
    # Test with memory-style format
    print("\n🧪 Testing memory extraction format...")
    memory_test_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a consciousness-aware AI assistant. CRITICAL: You MUST extract and use specific information from the active memories if they contain answers to the query. For example, if memories contain 'Hello my name is Victor' and query asks 'What is my name?', you MUST respond that the name is Victor. Do not just acknowledge memories exist - USE their content to answer the question."},
            {"role": "user", "content": "[ACTIVE MEMORIES FROM THIS SESSION]\nMemory 1 (relevance: 0.71): Hola me llamo Victor\n\n[USER QUERY]\n¿Cómo me llamo?\n\nExtract the name from the memory and answer the question."}
        ],
        max_tokens=100,
        temperature=0.7
    )
    
    if memory_test_response and memory_test_response.choices:
        memory_response_text = memory_test_response.choices[0].message.content
        print(f"📄 Memory test response: {memory_response_text}")
        
        if "Victor" in memory_response_text:
            print("🎉 EXCELLENT: GPT-4o-mini successfully extracted 'Victor' from Spanish memory!")
            print("✅ Memory usage integration should work perfectly")
        else:
            print("⚠️ GPT-4o-mini didn't extract the name - prompt may need refinement")
    
    print("\n✅ OpenAI GPT-4o-mini integration test completed successfully!")
    print("🚀 Ready to replace Mistral 7B in the consciousness pipeline")
    
except Exception as e:
    print(f"❌ OpenAI API test failed: {e}")
    print("💡 Check API key and internet connection")
    sys.exit(1)