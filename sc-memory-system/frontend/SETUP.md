# SC Memory System Frontend - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- SC Memory System backend running

### 1. Start the Backend

```bash
# Navigate to project root
cd sc-memory-system

# Install backend dependencies (if not done already)
pip install -e .

# Start the backend server
uvicorn src.api.main:app --reload --port 8000
```

### 2. Test Backend Connection

```bash
# Navigate to frontend directory
cd frontend

# Test API endpoints
python test_api_connection.py
```

Expected output:
```
🧠 SC Memory System - API Connection Test
==================================================
🔍 Testing backend health...
✅ Backend is running!

🔍 Testing MEP endpoint...
✅ MEP endpoint working! Proposal ID: abc123...

🔍 Testing MAP endpoint...
✅ MAP endpoint working! Has memory: False

==================================================
🎯 Test Summary:
- Health check: ✅
- MEP endpoint: ✅
- MAP endpoint: ✅

🚀 Ready to run Streamlit frontend!
```

### 3. Install Frontend Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

### 4. Run the Frontend

```bash
# Option 1: Use run script
run.bat          # Windows
bash run.sh      # Linux/Mac

# Option 2: Run directly
streamlit run streamlit_app.py --server.port 8501
```

### 5. Open in Browser

Navigate to: `http://localhost:8501`

## 🎯 Demo Workflow

### Phase 1: Standard Context Mode
1. Enter your OpenAI API key in the sidebar
2. Start chatting about any topic (e.g., "Explain quantum physics")
3. Continue for 5-10 messages to build up context
4. Watch the token count grow in the sidebar

### Phase 2: Memory Consolidation
1. Click "🚀 Consolidate to SC Memory" button
2. Wait 15-20 seconds for LoRA training to complete
3. Context is automatically cleared after consolidation

### Phase 3: SC Memory Mode
1. Continue the conversation about the same topic
2. System now uses compressed memory instead of full context
3. Compare token usage in the visualization chart
4. Typical savings: 70-90% token reduction

## 🐛 Troubleshooting

### Backend Connection Issues
```bash
# Check if backend is running
curl http://localhost:8000/mep/v1/health

# Expected response:
{"status": "healthy", ...}
```

### OpenAI API Issues
- Ensure your API key is valid and has credits
- Check rate limits if requests fail
- gpt-4o-mini should be available in your account

### Frontend Issues
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Run with debug mode
streamlit run streamlit_app.py --logger.level debug
```

### Memory Consolidation Issues
- Ensure you have at least 2 message exchanges before consolidating
- Wait 15+ seconds after clicking consolidate
- Check backend logs for LoRA training errors

## 📊 Expected Results

**Token Savings Example:**
- Standard context: 2,500 tokens
- SC Memory: 450 tokens
- Savings: 82% reduction

**Performance:**
- Consolidation time: 15-20 seconds
- Response time: 2-5 seconds
- Memory retrieval: <200ms

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set in .env file
OPENAI_API_KEY=your_key_here
STREAMLIT_SERVER_PORT=8501
SC_BACKEND_URL=http://localhost:8000
```

### Token Budget Settings
- Default: 320 tokens for compressed context
- Adjustable in MAP API calls
- Recommended range: 200-500 tokens

## 📝 API Endpoints Used

### MEP (Memory Exchange Protocol)
- **URL**: `POST /mep/v1/proposals`
- **Purpose**: Submit conversation for LoRA training
- **Auth**: Bearer token required

### MAP (Memory Access Protocol)
- **URL**: `GET /map/v1/context`
- **Purpose**: Retrieve compressed memory context
- **Auth**: Bearer token required

## ✅ Success Criteria

The demo is working correctly when:
- [ ] Backend health check passes
- [ ] Standard chat mode accumulates tokens
- [ ] Consolidation button triggers MEP successfully
- [ ] Context clears after consolidation
- [ ] SC Memory mode uses compressed context
- [ ] Token savings visualization shows reduction
- [ ] All UI text is in English

## 🆘 Support

If you encounter issues:

1. **Check the console** for error messages
2. **Verify backend logs** for API errors
3. **Test API endpoints** manually with curl
4. **Check OpenAI account** for API limits
5. **Restart both backend and frontend** if needed

---

**Happy testing! 🚀**