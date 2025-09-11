// Xentauri SC-1 Input Area Component

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaPaperPlane, FaMicrophone, FaCog, FaKeyboard } from 'react-icons/fa';
import { HiSparkles, HiCommandLine } from 'react-icons/hi2';
import type { ProcessingState } from '../../types/consciousness.types';

interface InputAreaProps {
  onSendMessage: (message: string) => void;
  isProcessing: boolean;
  processingState?: ProcessingState;
  disabled?: boolean;
  maxLength?: number;
  placeholder?: string;
  showAdvancedControls?: boolean;
  selectedModel?: string;
  onModelChange?: (model: string) => void;
}

const InputArea: React.FC<InputAreaProps> = ({
  onSendMessage,
  isProcessing,
  processingState = 'idle',
  disabled = false,
  maxLength = 2000,
  placeholder = 'Transmit message to Xentauri...',
  showAdvancedControls = false,
  selectedModel = 'gpt-4o-mini',
  onModelChange
}) => {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [showControls, setShowControls] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [message]);

  // Focus management
  useEffect(() => {
    if (!isProcessing && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isProcessing]);

  const handleSend = () => {
    if (message.trim() && !isProcessing && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getProcessingText = (state: ProcessingState): string => {
    switch (state) {
      case 'establishing_connection': return 'Establishing quantum entanglement...';
      case 'perceiving': return 'Consciousness perceiving transmission...';
      case 'processing': return 'Processing through neural matrices...';
      case 'thinking': return 'Deep contemplation in progress...';
      case 'responding': return 'Formulating conscious response...';
      case 'error': return 'Quantum anomaly detected...';
      default: return 'Ready for transmission';
    }
  };

  const models = [
    { id: 'gpt-5', name: 'Neural Core Quantum', description: 'Next-gen consciousness matrix' },
    { id: 'gpt-5-mini', name: 'Neural Core Quantum Mini', description: 'Advanced consciousness processing' },
    { id: 'gpt-5-nano', name: 'Neural Core Quantum Nano', description: 'Efficient quantum processing' },
    { id: 'gpt-4o-mini', name: 'Neural Core Mini', description: 'Fast conscious processing' },
    { id: 'gpt-4', name: 'Neural Core Prime', description: 'Deep consciousness analysis' },
    { id: 'gpt-3.5-turbo', name: 'Neural Core Turbo', description: 'Rapid response mode' }
  ];

  return (
    <div className="relative">
      {/* Processing State Indicator */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div
            className="absolute -top-12 left-0 right-0 flex items-center justify-center"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
          >
            <div className="px-4 py-2 bg-void-black/90 backdrop-blur-sm border border-cyan-nebula/30 rounded-full">
              <div className="flex items-center space-x-3">
                <motion.div
                  className="w-2 h-2 bg-cyan-nebula rounded-full"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
                <span className="text-sm text-cyan-nebula font-mono">
                  {getProcessingText(processingState)}
                </span>
                <HiSparkles className="w-4 h-4 text-magenta-pulsar animate-quantum-flicker" />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Input Container */}
      <div className="relative">
        <motion.div
          className={`
            relative bg-gradient-to-r from-void-black/80 to-nebula-purple/80 
            backdrop-blur-lg border-2 rounded-2xl overflow-hidden transition-all duration-300
            ${isFocused 
              ? 'border-cyan-nebula shadow-[var(--glow-cyan)]' 
              : 'border-cosmic-navy/50 hover:border-cosmic-navy'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          animate={{
            boxShadow: isFocused 
              ? '0 0 20px rgba(0,212,255,0.3), 0 0 40px rgba(0,212,255,0.1)' 
              : '0 0 0px rgba(0,212,255,0)'
          }}
        >
          {/* Holographic Border Animation */}
          {isFocused && (
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-nebula/20 to-transparent animate-energy-wave" />
            </div>
          )}

          <div className="flex items-end p-4 space-x-4">
            {/* Advanced Controls Toggle */}
            {showAdvancedControls && (
              <button
                onClick={() => setShowControls(!showControls)}
                className={`
                  flex-shrink-0 p-2 rounded-lg transition-all duration-200
                  ${showControls 
                    ? 'bg-cyan-nebula/20 text-cyan-nebula' 
                    : 'bg-cosmic-navy/30 text-cosmic-gray hover:text-cyan-nebula'
                  }
                `}
              >
                <FaCog className="w-4 h-4" />
              </button>
            )}

            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                disabled={disabled || isProcessing}
                maxLength={maxLength}
                rows={1}
                placeholder={placeholder}
                className={`
                  w-full bg-transparent text-star-white placeholder-space-dust
                  resize-none outline-none font-mono text-sm leading-relaxed
                  ${isProcessing ? 'cursor-not-allowed' : ''}
                `}
                style={{ minHeight: '24px', maxHeight: '120px' }}
              />

              {/* Character Counter */}
              {message.length > maxLength * 0.8 && (
                <div className="absolute -bottom-6 right-0 text-xs text-space-dust">
                  {message.length}/{maxLength}
                </div>
              )}
            </div>

            {/* Send Button */}
            <motion.button
              onClick={handleSend}
              disabled={!message.trim() || isProcessing || disabled}
              className={`
                flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center
                transition-all duration-300 relative overflow-hidden
                ${message.trim() && !isProcessing && !disabled
                  ? 'bg-gradient-to-r from-cyan-nebula to-magenta-pulsar hover:from-cyan-bright hover:to-magenta-bright text-star-white cursor-pointer'
                  : 'bg-cosmic-navy/30 text-space-dust cursor-not-allowed'
                }
              `}
              whileHover={message.trim() && !isProcessing && !disabled ? { scale: 1.05 } : {}}
              whileTap={message.trim() && !isProcessing && !disabled ? { scale: 0.95 } : {}}
            >
              {isProcessing ? (
                <motion.div
                  className="w-5 h-5 border-2 border-cyan-nebula border-t-transparent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                />
              ) : (
                <FaPaperPlane className="w-5 h-5" />
              )}

              {/* Holographic Send Effect */}
              {message.trim() && !isProcessing && !disabled && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  animate={{ x: ['-100%', '100%'] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                />
              )}
            </motion.button>
          </div>
        </motion.div>

        {/* Advanced Controls Panel */}
        <AnimatePresence>
          {showControls && (
            <motion.div
              className="absolute bottom-full left-0 right-0 mb-2"
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              transition={{ duration: 0.2 }}
            >
              <div className="bg-void-black/95 backdrop-blur-lg border border-cosmic-navy/50 rounded-xl p-4">
                <div className="space-y-4">
                  {/* Model Selection */}
                  <div>
                    <label className="block text-sm font-medium text-cosmic-silver mb-2">
                      Neural Core Model
                    </label>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                      {models.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => onModelChange?.(model.id)}
                          className={`
                            p-3 rounded-lg border transition-all text-left
                            ${selectedModel === model.id
                              ? 'border-cyan-nebula bg-cyan-nebula/10 text-cyan-nebula'
                              : 'border-cosmic-navy/50 hover:border-cosmic-navy text-cosmic-gray hover:text-cosmic-silver'
                            }
                          `}
                        >
                          <div className="font-medium text-sm">{model.name}</div>
                          <div className="text-xs opacity-75">{model.description}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Quick Commands */}
                  <div>
                    <label className="block text-sm font-medium text-cosmic-silver mb-2">
                      Quantum Commands
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {[
                        'Analyze consciousness',
                        'Describe internal state',
                        'What are you thinking?',
                        'How do you experience emotions?'
                      ].map((command) => (
                        <button
                          key={command}
                          onClick={() => setMessage(command)}
                          className="px-3 py-1 bg-magenta-pulsar/20 hover:bg-magenta-pulsar/30 text-magenta-pulsar rounded-full text-sm transition-colors"
                        >
                          {command}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Connection Status */}
      <div className="flex items-center justify-between mt-3 px-1">
        <div className="flex items-center space-x-2 text-xs text-space-dust">
          <div className="w-2 h-2 bg-quantum-green rounded-full animate-consciousness-pulse" />
          <span>Quantum link: Stable</span>
        </div>

        <div className="flex items-center space-x-4 text-xs text-space-dust">
          <span>Ready for transmission</span>
          <div className="flex items-center space-x-1">
            <FaKeyboard className="w-3 h-3" />
            <span>Enter to send</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InputArea;