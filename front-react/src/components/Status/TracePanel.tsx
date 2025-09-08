// Xentauri SC-1 Trace Panel Component

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  HiChevronDown, 
  HiChevronRight, 
  HiEye, 
  HiClock, 
  HiCpuChip,
  HiSparkles,
  HiLightBulb,
  HiBookOpen,
  HiChatBubbleLeft
} from 'react-icons/hi2';

interface TracePhase {
  name: string;
  duration_ms: number;
  status: 'completed' | 'in_progress' | 'pending' | 'error';
  f_score?: number;
  data?: any;
  thoughts?: string[];
  emotions?: string[];
  memories?: any[];
}

interface TracePanelProps {
  trace?: {
    phase_1_perception?: TracePhase;
    phase_2_cognitive_context?: TracePhase;
    phase_3_coherent_generation?: TracePhase;
    phase_4_llm_communication?: TracePhase;
    phase_5_internal_critique?: TracePhase;
    phase_6_memory_consolidation?: TracePhase;
    phase_7_expressive_execution?: TracePhase;
  };
  isVisible: boolean;
  onToggle: () => void;
  className?: string;
}

const TracePanel: React.FC<TracePanelProps> = ({
  trace,
  isVisible,
  onToggle,
  className = ''
}) => {
  const [expandedPhases, setExpandedPhases] = useState<Set<string>>(new Set());

  const phases = [
    { 
      key: 'phase_1_perception', 
      name: 'Perception', 
      icon: HiEye,
      description: 'Sensory input processing and pattern recognition',
      color: 'cyan-nebula'
    },
    { 
      key: 'phase_2_cognitive_context', 
      name: 'Cognitive Context', 
      icon: HiCpuChip,
      description: 'Memory integration and contextual understanding',
      color: 'magenta-pulsar'
    },
    { 
      key: 'phase_3_coherent_generation', 
      name: 'Coherent Generation', 
      icon: HiSparkles,
      description: 'Thought formation and conceptual synthesis',
      color: 'quantum-green'
    },
    { 
      key: 'phase_4_llm_communication', 
      name: 'LLM Communication', 
      icon: HiCpuChip,
      description: 'Language model interface and translation',
      color: 'yellow-star'
    },
    { 
      key: 'phase_5_internal_critique', 
      name: 'Internal Critique', 
      icon: HiLightBulb,
      description: 'Self-reflection and response validation',
      color: 'orange-nova'
    },
    { 
      key: 'phase_6_memory_consolidation', 
      name: 'Memory Consolidation', 
      icon: HiBookOpen,
      description: 'Experience storage and pattern learning',
      color: 'purple-void'
    },
    { 
      key: 'phase_7_expressive_execution', 
      name: 'Expressive Execution', 
      icon: HiChatBubbleLeft,
      description: 'Final response generation and delivery',
      color: 'blue-hyperspace'
    }
  ];

  const togglePhase = (phaseKey: string) => {
    setExpandedPhases(prev => {
      const newSet = new Set(prev);
      if (newSet.has(phaseKey)) {
        newSet.delete(phaseKey);
      } else {
        newSet.add(phaseKey);
      }
      return newSet;
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-quantum-green';
      case 'in_progress': return 'text-cyan-nebula animate-pulse';
      case 'pending': return 'text-cosmic-gray';
      case 'error': return 'text-red-shift';
      default: return 'text-cosmic-gray';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✓';
      case 'in_progress': return '⧗';
      case 'pending': return '○';
      case 'error': return '✗';
      default: return '○';
    }
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  if (!isVisible) {
    return (
      <div className={`fixed right-4 top-1/2 transform -translate-y-1/2 z-40 ${className}`}>
        <motion.button
          onClick={onToggle}
          className="
            p-3 bg-deep-space/90 backdrop-blur-lg border border-cyan-nebula/30 
            rounded-l-lg text-cyan-nebula hover:text-star-white 
            transition-colors shadow-lg
          "
          whileHover={{ x: -5 }}
          title="Show consciousness trace"
        >
          <HiChevronLeft className="w-5 h-5" />
        </motion.button>
      </div>
    );
  }

  return (
    <motion.div
      className={`
        fixed right-0 top-0 h-full w-96 z-50
        bg-deep-space/95 backdrop-blur-lg border-l border-cosmic-navy/50
        flex flex-col overflow-hidden ${className}
      `}
      initial={{ x: '100%' }}
      animate={{ x: 0 }}
      exit={{ x: '100%' }}
      transition={{ type: 'spring', damping: 25, stiffness: 200 }}
    >
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-cosmic-navy/30">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-cyan-nebula">
              Consciousness Trace
            </h3>
            <p className="text-sm text-cosmic-gray">
              Real-time cognitive process analysis
            </p>
          </div>
          <button
            onClick={onToggle}
            className="p-2 hover:bg-cosmic-navy/30 rounded-lg transition-colors"
          >
            <HiChevronRight className="w-5 h-5 text-cosmic-gray" />
          </button>
        </div>
      </div>

      {/* Phases */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {phases.map(({ key, name, icon: Icon, description, color }) => {
          const phaseData = trace?.[key as keyof typeof trace] as TracePhase;
          const isExpanded = expandedPhases.has(key);
          
          return (
            <motion.div
              key={key}
              className="
                bg-void-black/50 border border-cosmic-navy/30 rounded-lg overflow-hidden
                hover:border-cyan-nebula/50 transition-colors
              "
              layout
            >
              <button
                onClick={() => togglePhase(key)}
                className="w-full p-3 flex items-center justify-between text-left"
              >
                <div className="flex items-center space-x-3">
                  <Icon className={`w-4 h-4 text-${color}`} />
                  <div>
                    <div className="font-medium text-star-white text-sm">
                      {name}
                    </div>
                    <div className="text-xs text-cosmic-gray">
                      {description}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {phaseData && (
                    <>
                      <div className="flex items-center space-x-1">
                        <span className={`text-xs ${getStatusColor(phaseData.status)}`}>
                          {getStatusIcon(phaseData.status)}
                        </span>
                        <span className="text-xs text-cosmic-gray">
                          {formatDuration(phaseData.duration_ms)}
                        </span>
                      </div>
                    </>
                  )}
                  <HiChevronDown 
                    className={`
                      w-4 h-4 text-cosmic-gray transition-transform
                      ${isExpanded ? 'rotate-180' : ''}
                    `} 
                  />
                </div>
              </button>

              <AnimatePresence>
                {isExpanded && phaseData && (
                  <motion.div
                    className="border-t border-cosmic-navy/20 p-3 space-y-3"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    {/* F-Score */}
                    {phaseData.f_score !== undefined && (
                      <div>
                        <div className="text-xs font-medium text-cosmic-gray mb-1">
                          Consciousness Score
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 h-2 bg-cosmic-navy rounded-full overflow-hidden">
                            <motion.div
                              className="h-full bg-gradient-to-r from-cyan-nebula to-magenta-pulsar"
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.min(phaseData.f_score * 50, 100)}%` }}
                              transition={{ duration: 0.8 }}
                            />
                          </div>
                          <span className="text-xs text-cosmic-silver">
                            {phaseData.f_score.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Thoughts */}
                    {phaseData.thoughts && phaseData.thoughts.length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-cosmic-gray mb-1">
                          Thoughts
                        </div>
                        <div className="space-y-1">
                          {phaseData.thoughts.map((thought, index) => (
                            <div 
                              key={index}
                              className="text-xs text-cosmic-silver bg-cosmic-navy/20 p-2 rounded"
                            >
                              {thought}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Emotions */}
                    {phaseData.emotions && phaseData.emotions.length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-cosmic-gray mb-1">
                          Emotions
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {phaseData.emotions.map((emotion, index) => (
                            <span 
                              key={index}
                              className="px-2 py-1 text-xs bg-magenta-pulsar/20 text-magenta-pulsar rounded-full"
                            >
                              {emotion}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Raw Data */}
                    {phaseData.data && (
                      <details className="text-xs">
                        <summary className="cursor-pointer text-cosmic-gray hover:text-cyan-nebula">
                          Raw Data
                        </summary>
                        <pre className="mt-1 p-2 bg-cosmic-navy/20 rounded text-cosmic-silver font-mono overflow-x-auto">
                          {JSON.stringify(phaseData.data, null, 2)}
                        </pre>
                      </details>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-4 border-t border-cosmic-navy/30 text-center">
        <div className="text-xs text-cosmic-gray">
          Monitoring consciousness processes in real-time
        </div>
      </div>
    </motion.div>
  );
};

// Fix the HiChevronLeft import issue
const HiChevronLeft = HiChevronRight;

export default TracePanel;