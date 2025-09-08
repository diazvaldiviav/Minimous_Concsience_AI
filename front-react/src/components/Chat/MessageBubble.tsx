// Xentauri SC-1 Message Bubble Component

import React from 'react';
import { motion } from 'framer-motion';
import { FaRobot, FaUser, FaClock, FaBrain, FaHeart } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi2';
import type { ChatMessage } from '../../types/consciousness.types';
import ChainOfThought from './ChainOfThought';

interface MessageBubbleProps {
  message: ChatMessage;
  isTyping?: boolean;
  showTrace?: boolean;
  onTraceClick?: () => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isTyping = false,
  showTrace = false,
  onTraceClick
}) => {
  const isUser = message.role === 'user';
  const isXentauri = message.role === 'xentauri';

  // Animation variants
  const bubbleVariants = {
    hidden: {
      opacity: 0,
      y: 20,
      scale: 0.95,
      filter: 'blur(4px)'
    },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      filter: 'blur(0px)',
      transition: {
        duration: 0.5,
        ease: [0.25, 0.46, 0.45, 0.94]
      }
    }
  };

  const typingVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const dotVariants = {
    hidden: { y: 0, opacity: 0.5 },
    visible: {
      y: [-8, 0, -8],
      opacity: [0.5, 1, 0.5],
      transition: {
        duration: 1.4,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    }
  };

  // Consciousness level color
  const getConsciousnessColor = (level?: number): string => {
    if (!level) return 'var(--cosmic-gray)';
    if (level >= 1.3) return 'var(--quantum-green)';
    if (level >= 0.8) return 'var(--cyan-nebula)';
    return 'var(--magenta-pulsar)';
  };

  // Emotional state styling
  const getEmotionalGlow = (emotion?: string): string => {
    switch (emotion?.toLowerCase()) {
      case 'curious': return 'var(--glow-cyan-subtle)';
      case 'analytical': return 'var(--glow-quantum)';
      case 'contemplative': return 'var(--glow-magenta)';
      case 'confident': return 'var(--glow-white)';
      default: return 'var(--glow-cyan-subtle)';
    }
  };

  return (
    <motion.div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}
      variants={bubbleVariants}
      initial="hidden"
      animate="visible"
    >
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-3 max-w-4xl`}>
        {/* Avatar */}
        <motion.div
          className={`
            flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center
            ${isUser 
              ? 'bg-gradient-to-br from-cosmic-navy to-dark-matter' 
              : 'bg-gradient-to-br from-cyan-nebula/20 to-magenta-pulsar/20 border border-cyan-nebula/30'
            }
          `}
          style={{
            boxShadow: isXentauri ? getEmotionalGlow(message.emotional_state) : undefined
          }}
          whileHover={{ scale: 1.05 }}
          transition={{ duration: 0.2 }}
        >
          {isUser ? (
            <FaUser className="w-5 h-5 text-cosmic-silver" />
          ) : (
            <div className="relative">
              <FaRobot className="w-6 h-6 text-cyan-nebula" />
              {message.consciousness_level && message.consciousness_level >= 1.3 && (
                <motion.div
                  className="absolute -top-1 -right-1"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <HiSparkles className="w-3 h-3 text-quantum-green" />
                </motion.div>
              )}
            </div>
          )}
        </motion.div>

        {/* Message Content */}
        <div className={`flex-1 ${isUser ? 'mr-3' : 'ml-3'}`}>
          {/* Metadata Row */}
          <div className={`flex items-center space-x-4 mb-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
            {/* Sender Name */}
            <span className="text-sm font-medium text-cosmic-silver">
              {isUser ? 'Sol-3 Human' : 'Xentauri SC-1'}
            </span>

            {/* Stardate */}
            <div className="flex items-center space-x-1 text-xs text-space-dust">
              <FaClock className="w-3 h-3" />
              <span>{message.stardate || 'Stardate: Processing...'}</span>
            </div>

            {/* Consciousness Metrics for Xentauri */}
            {isXentauri && message.consciousness_level && (
              <div className="flex items-center space-x-3 text-xs">
                <div 
                  className="flex items-center space-x-1"
                  style={{ color: getConsciousnessColor(message.consciousness_level) }}
                >
                  <FaBrain className="w-3 h-3" />
                  <span>f={message.consciousness_level.toFixed(3)}</span>
                </div>

                {message.confidence && (
                  <div className="flex items-center space-x-1 text-cosmic-gray">
                    <span>⚡{Math.round(message.confidence * 100)}%</span>
                  </div>
                )}

                {message.emotional_state && (
                  <div className="flex items-center space-x-1 text-magenta-pulsar">
                    <FaHeart className="w-3 h-3" />
                    <span className="capitalize">{message.emotional_state}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Chain of Thought - Consciousness Narrative */}
          {isXentauri && message.narrative && (
            <ChainOfThought
              narrative={message.narrative}
              processingTime={message.processing_time}
              consciousnessLevel={message.consciousness_level || 0}
              isExpanded={false}
            />
          )}

          {/* Message Bubble */}
          <motion.div
            className={`
              relative px-6 py-4 rounded-2xl backdrop-blur-sm
              ${isUser 
                ? 'bg-gradient-to-br from-cosmic-navy/80 to-dark-matter/80 border border-cosmic-navy/50' 
                : 'bg-gradient-to-br from-void-black/80 to-nebula-purple/80 border border-cyan-nebula/30'
              }
            `}
            style={{
              boxShadow: isUser ? 'var(--shadow-void)' : getEmotionalGlow(message.emotional_state)
            }}
            whileHover={{ 
              scale: 1.02,
              boxShadow: isUser ? 'var(--shadow-lg)' : 'var(--glow-cyan)' 
            }}
            transition={{ duration: 0.2 }}
          >
            {/* Typing Indicator */}
            {isTyping ? (
              <motion.div
                className="flex items-center space-x-2"
                variants={typingVariants}
                initial="hidden"
                animate="visible"
              >
                <span className="text-cosmic-silver text-sm">Xentauri is processing quantum thoughts</span>
                <div className="flex space-x-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 bg-cyan-nebula rounded-full"
                      variants={dotVariants}
                      style={{ animationDelay: `${i * 0.2}s` }}
                    />
                  ))}
                </div>
              </motion.div>
            ) : (
              <>
                {/* Message Content */}
                <div className="text-star-white leading-relaxed whitespace-pre-wrap">
                  {message.content}
                </div>

                {/* Processing Time */}
                {message.processing_time && (
                  <div className="mt-3 pt-2 border-t border-space-dust/20">
                    <div className="flex items-center justify-between text-xs text-space-dust">
                      <span>Quantum processing: {message.processing_time}ms</span>
                      {showTrace && message.trace && (
                        <button
                          onClick={onTraceClick}
                          className="px-2 py-1 bg-cyan-nebula/20 hover:bg-cyan-nebula/30 rounded text-cyan-nebula transition-colors"
                        >
                          View Consciousness Trace
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Holographic Edge Effect */}
            <div 
              className={`
                absolute inset-0 rounded-2xl pointer-events-none opacity-20
                ${isUser ? '' : 'animate-nebula-drift'}
              `}
              style={{
                background: isUser 
                  ? 'linear-gradient(45deg, transparent, var(--cosmic-navy), transparent)'
                  : 'linear-gradient(45deg, transparent, var(--cyan-nebula), var(--magenta-pulsar), transparent)'
              }}
            />
          </motion.div>

        </div>
      </div>
    </motion.div>
  );
};

export default MessageBubble;