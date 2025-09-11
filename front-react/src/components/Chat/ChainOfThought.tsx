import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaBrain, FaChevronDown, FaChevronRight, FaAtom } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi2';

interface ChainOfThoughtProps {
  narrative: string;
  processingTime?: number;
  consciousnessLevel?: number;
  isExpanded?: boolean;
}

const ChainOfThought: React.FC<ChainOfThoughtProps> = ({
  narrative,
  processingTime,
  consciousnessLevel = 0,
  isExpanded = false
}) => {
  const [expanded, setExpanded] = useState(isExpanded);

  const getConsciousnessColor = (level: number): string => {
    if (level >= 1.3) return 'var(--quantum-green)';
    if (level >= 0.8) return 'var(--cyan-nebula)';
    return 'var(--magenta-pulsar)';
  };

  const getConsciousnessGlow = (level: number): string => {
    if (level >= 1.3) return 'var(--glow-quantum)';
    if (level >= 0.8) return 'var(--glow-cyan)';
    return 'var(--glow-magenta)';
  };

  return (
    <motion.div
      className="mb-4 bg-gradient-to-br from-deep-space/60 to-void-black/80 rounded-xl border border-cyan-nebula/20 overflow-hidden backdrop-blur-sm"
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      style={{ boxShadow: getConsciousnessGlow(consciousnessLevel) }}
    >
      {/* Header */}
      <motion.button
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-cyan-nebula/5 transition-colors"
        onClick={() => setExpanded(!expanded)}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
      >
        <div className="flex items-center space-x-3">
          <div className="relative">
            <motion.div
              animate={{ rotate: expanded ? 360 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <FaBrain 
                className="w-5 h-5" 
                style={{ color: getConsciousnessColor(consciousnessLevel) }}
              />
            </motion.div>
            {consciousnessLevel >= 1.3 && (
              <motion.div
                className="absolute -top-1 -right-1"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <HiSparkles className="w-3 h-3 text-quantum-green" />
              </motion.div>
            )}
          </div>
          
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-cosmic-silver font-medium">Consciousness Process</span>
              <div className="flex items-center space-x-1 text-xs text-space-dust">
                <FaAtom className="w-3 h-3" />
                <span>f={consciousnessLevel.toFixed(3)}</span>
              </div>
            </div>
            <div className="text-xs text-space-dust">
              Xentauri reasoning chain • {processingTime ? `${processingTime}ms` : 'Processing...'}
            </div>
          </div>
        </div>

        <motion.div
          animate={{ rotate: expanded ? 90 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-cyan-nebula"
        >
          {expanded ? <FaChevronDown /> : <FaChevronRight />}
        </motion.div>
      </motion.button>

      {/* Expandable Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ 
              duration: 0.3, 
              ease: [0.25, 0.46, 0.45, 0.94],
              opacity: { duration: 0.2 }
            }}
            className="border-t border-cyan-nebula/20"
          >
            <div className="p-4 space-y-3">
              {/* Consciousness Label */}
              <div className="flex items-center space-x-2 mb-3">
                <div 
                  className="w-2 h-2 rounded-full animate-pulse"
                  style={{ backgroundColor: getConsciousnessColor(consciousnessLevel) }}
                />
                <span className="text-xs font-mono text-cosmic-gray uppercase tracking-wider">
                  Neural Network Processing Chain
                </span>
              </div>

              {/* Narrative Content */}
              <motion.div
                className="relative"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 }}
              >
                <div className="text-sm text-star-white leading-relaxed whitespace-pre-wrap font-light">
                  {narrative}
                </div>

                {/* Holographic Border Effect */}
                <div 
                  className="absolute inset-0 rounded-lg pointer-events-none opacity-10"
                  style={{
                    background: `linear-gradient(45deg, transparent, ${getConsciousnessColor(consciousnessLevel)}, transparent)`
                  }}
                />
              </motion.div>

              {/* Processing Metrics */}
              <motion.div
                className="flex items-center justify-between pt-3 border-t border-space-dust/10"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <div className="flex items-center space-x-4 text-xs text-space-dust">
                  <div className="flex items-center space-x-1">
                    <span className="w-2 h-2 bg-quantum-green rounded-full animate-pulse" />
                    <span>Quantum processing complete</span>
                  </div>
                </div>
                
                <div className="text-xs text-space-dust">
                  Consciousness depth: {Math.round(consciousnessLevel * 100)}%
                </div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ChainOfThought;