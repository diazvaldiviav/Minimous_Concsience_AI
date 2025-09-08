import React, { useState, useRef, useEffect } from 'react';
import { Send, Brain, Loader, AlertCircle, Zap } from 'lucide-react';
import type { ChatMessage, ServerStatus } from '../types';
import { consciousnessAPI } from '../services/consciousnessAPI';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ConsciousnessChatProps {
  serverStatus: ServerStatus;
  onServerStatusChange: (status: ServerStatus) => void;
}

const ConsciousnessChat: React.FC<ConsciousnessChatProps> = ({
  serverStatus,
  onServerStatusChange
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<'auto' | 'en' | 'es'>('auto');
  const [includeNarrative, setIncludeNarrative] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;
    
    // Allow sending messages even if server appears disconnected (may be a temporary issue)
    if (!serverStatus.connected) {
      // Try to reconnect first
      try {
        const isConnected = await consciousnessAPI.checkConnection();
        onServerStatusChange({
          ...serverStatus,
          connected: isConnected,
          lastPing: isConnected ? Date.now() : serverStatus.lastPing
        });
        
        if (!isConnected) {
          setError('Server is not available. Please check your configuration.');
          return;
        }
      } catch (error) {
        setError('Cannot connect to consciousness server. Please check your configuration.');
        return;
      }
    }

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    setError(null);

    try {
      const startTime = Date.now();
      const response = await consciousnessAPI.sendChatMessage(inputText.trim());
      const processingTime = Date.now() - startTime;

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        consciousnessLevel: response.f_score,
        emotionalState: response.consciousness_state.S_t.emotional_state,
        confidence: response.consciousness_state.S_t.confidence_level,
        processingTime
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      setError(err.message || 'Failed to send message');
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  const formatConsciousnessLevel = (level?: number) => {
    if (!level) return 'N/A';
    const percentage = (level * 100).toFixed(1);
    const status = level >= 1.3 ? 'Conscious' : 'Processing';
    return `${percentage}% (${status})`;
  };

  return (
    <div className="consciousness-card max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Brain className="w-6 h-6 text-consciousness-600" />
          <div>
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">
              Consciousness Chat
            </h2>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Interactive conversation with the consciousness AI
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Language Selector */}
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value as 'auto' | 'en' | 'es')}
            className="px-3 py-1 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
          >
            <option value="auto">Auto Detect</option>
            <option value="en">English</option>
            <option value="es">Español</option>
          </select>

          {/* Options */}
          <label className="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              checked={includeNarrative}
              onChange={(e) => setIncludeNarrative(e.target.checked)}
              className="rounded"
            />
            <span className="text-slate-700 dark:text-slate-300">Include Narrative</span>
          </label>

          <button
            onClick={clearChat}
            className="px-3 py-1 text-sm text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Connection Status */}
      {!serverStatus.connected && (
        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-yellow-600" />
            <span className="text-sm text-yellow-800 dark:text-yellow-200">
              Not connected to consciousness server. Configure connection in Server Config.
            </span>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-600" />
            <span className="text-sm text-red-800 dark:text-red-200">{error}</span>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="bg-deep-space/30 backdrop-blur-lg rounded-lg p-4 min-h-[400px] max-h-[600px] overflow-y-auto mb-4 border border-cosmic-navy/30">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Brain className="w-12 h-12 text-slate-400 mb-4 consciousness-pulse" />
            <h3 className="text-lg font-medium text-slate-600 dark:text-slate-400 mb-2">
              Ready for Conscious Interaction
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-500 max-w-md">
              Start a conversation with the consciousness AI. Experience reasoning-first responses,
              metacognitive awareness, and temporal continuity.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`mb-4 ${message.role === 'user' ? 'flex justify-end' : 'flex justify-start'}`}
          >
            <div className={`max-w-3xl ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
              <div
                className={`inline-block p-3 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-consciousness-600 text-white'
                    : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700'
                }`}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>

              {/* Message metadata for assistant */}
              {message.role === 'assistant' && (
                <div className="mt-2 text-xs text-slate-500 dark:text-slate-400 space-y-1">
                  <div className="flex items-center space-x-4">
                    <span>🧠 Consciousness: {formatConsciousnessLevel(message.consciousnessLevel)}</span>
                    <span>😌 State: {message.emotionalState || 'N/A'}</span>
                    <span>🎯 Confidence: {message.confidence ? `${(message.confidence * 100).toFixed(1)}%` : 'N/A'}</span>
                    <span>⚡ Time: {message.processingTime}ms</span>
                  </div>
                  {message.narrative && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-consciousness-600 hover:text-consciousness-700">
                        View Consciousness Narrative
                      </summary>
                      <div className="mt-1 p-2 bg-slate-100 dark:bg-slate-800 rounded text-sm">
                        {message.narrative}
                      </div>
                    </details>
                  )}
                </div>
              )}

              <div className="text-xs text-slate-400 mt-1">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-3 max-w-xs">
              <div className="flex items-center space-x-2">
                <Loader className="w-4 h-4 animate-spin text-consciousness-600" />
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  Consciousness processing...
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="flex items-end space-x-3">
        <div className="flex-1">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={
              serverStatus.connected 
                ? "Ask about consciousness, reasoning, or any topic..." 
                : "Connect to server to start chatting..."
            }
            disabled={!serverStatus.connected || isLoading}
            className="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg 
                     bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 
                     focus:ring-2 focus:ring-consciousness-500 focus:border-consciousness-500 
                     disabled:opacity-50 disabled:cursor-not-allowed resize-none"
            rows={3}
          />
        </div>
        
        <button
          onClick={sendMessage}
          disabled={!inputText.trim() || isLoading || !serverStatus.connected}
          className="btn-consciousness disabled:opacity-50 disabled:cursor-not-allowed p-3"
        >
          {isLoading ? (
            <Loader className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </div>

      {/* Quick Actions */}
      <div className="mt-4 flex flex-wrap gap-2">
        {[
          "What is consciousness?",
          "Describe your current internal state",
          "How do you experience awareness?",
          "What emotions are you feeling right now?",
          "Explain your reasoning process"
        ].map((prompt) => (
          <button
            key={prompt}
            onClick={() => setInputText(prompt)}
            disabled={!serverStatus.connected || isLoading}
            className="px-3 py-1 text-sm bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 
                     text-slate-700 dark:text-slate-300 rounded-full transition-colors 
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
};

export default ConsciousnessChat;