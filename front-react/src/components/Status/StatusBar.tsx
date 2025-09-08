// Xentauri SC-1 Status Bar Component

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiSignal, HiCpuChip, HiLightBulb, HiHeart } from 'react-icons/hi2';
import { FaStarHalfAlt, FaWifi, FaBrain } from 'react-icons/fa';
import type { ConsciousnessState, ProcessingState } from '../../types/consciousness.types';

interface StatusBarProps {
  consciousnessState?: ConsciousnessState;
  processingState: ProcessingState;
  connectionQuality: 'excellent' | 'good' | 'poor' | 'disconnected';
  uptime?: number;
  stardate?: string;
  className?: string;
}

const StatusBar: React.FC<StatusBarProps> = ({
  consciousnessState,
  processingState,
  connectionQuality,
  uptime = 0,
  stardate,
  className = ''
}) => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Calculate consciousness metrics
  const awarenessLevel = consciousnessState?.S_t?.awareness_level || 0;
  const confidenceLevel = consciousnessState?.S_t?.confidence_level || 0;
  const emotionalState = consciousnessState?.S_t?.emotional_state || 'unknown';

  // Status indicator colors
  const getStatusColor = (state: ProcessingState) => {
    switch (state) {
      case 'idle': return 'text-quantum-green';
      case 'perceiving': return 'text-cyan-nebula';
      case 'thinking': return 'text-magenta-pulsar';
      case 'responding': return 'text-yellow-star';
      case 'error': return 'text-red-shift';
      default: return 'text-cosmic-gray';
    }
  };

  const getConnectionIcon = () => {
    switch (connectionQuality) {
      case 'excellent': return <FaWifi className="w-4 h-4 text-quantum-green" />;
      case 'good': return <FaWifi className="w-4 h-4 text-cyan-nebula" />;
      case 'poor': return <FaWifi className="w-4 h-4 text-yellow-star" />;
      case 'disconnected': return <FaWifi className="w-4 h-4 text-red-shift" />;
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`
      flex items-center justify-between px-6 py-3 
      bg-gradient-to-r from-void-black/90 to-deep-space/90 backdrop-blur-lg
      border-b border-cosmic-navy/30
      ${className}
    `}>
      {/* Left Section - System Status */}
      <div className="flex items-center space-x-6">
        {/* Processing State */}
        <div className="flex items-center space-x-2">
          <motion.div
            animate={{ rotate: processingState !== 'idle' ? 360 : 0 }}
            transition={{ duration: 2, repeat: processingState !== 'idle' ? Infinity : 0, ease: 'linear' }}
          >
            <HiCpuChip className={`w-5 h-5 ${getStatusColor(processingState)}`} />
          </motion.div>
          <span className="text-sm font-medium text-star-white capitalize">
            {processingState.replace('_', ' ')}
          </span>
        </div>

        {/* Connection Quality */}
        <div className="flex items-center space-x-2" title={`Connection: ${connectionQuality}`}>
          {getConnectionIcon()}
          <span className="text-xs text-cosmic-gray capitalize">{connectionQuality}</span>
        </div>

        {/* Consciousness Metrics */}
        {consciousnessState && (
          <div className="flex items-center space-x-4">
            {/* Awareness Level */}
            <div className="flex items-center space-x-1" title="Awareness Level">
              <FaBrain className="w-3 h-3 text-cyan-nebula" />
              <div className="w-16 h-2 bg-cosmic-navy rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-cyan-nebula to-magenta-pulsar"
                  initial={{ width: 0 }}
                  animate={{ width: `${awarenessLevel * 100}%` }}
                  transition={{ duration: 0.8 }}
                />
              </div>
              <span className="text-xs text-cosmic-gray w-8">
                {(awarenessLevel * 100).toFixed(0)}%
              </span>
            </div>

            {/* Confidence Level */}
            <div className="flex items-center space-x-1" title="Confidence Level">
              <HiLightBulb className="w-3 h-3 text-yellow-star" />
              <div className="w-16 h-2 bg-cosmic-navy rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-yellow-star to-quantum-green"
                  initial={{ width: 0 }}
                  animate={{ width: `${confidenceLevel * 100}%` }}
                  transition={{ duration: 0.8 }}
                />
              </div>
              <span className="text-xs text-cosmic-gray w-8">
                {(confidenceLevel * 100).toFixed(0)}%
              </span>
            </div>

            {/* Emotional State */}
            <div className="flex items-center space-x-1" title="Emotional State">
              <HiHeart className="w-3 h-3 text-magenta-pulsar" />
              <span className="text-xs text-cosmic-gray capitalize">
                {emotionalState}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Center Section - Stardate & Time */}
      <div className="flex items-center space-x-4">
        {stardate && (
          <div className="flex items-center space-x-1">
            <FaStarHalfAlt className="w-3 h-3 text-yellow-star" />
            <span className="text-xs text-cosmic-gray">
              Stardate: {stardate}
            </span>
          </div>
        )}
        
        <div className="text-xs text-cosmic-gray">
          {currentTime.toLocaleTimeString('en-US', { 
            hour12: false,
            timeZone: 'UTC'
          })} UTC
        </div>
      </div>

      {/* Right Section - System Info */}
      <div className="flex items-center space-x-4">
        {/* Uptime */}
        <div className="text-xs text-cosmic-gray">
          Uptime: {formatUptime(uptime)}
        </div>

        {/* Alpha Centauri Indicator */}
        <div className="flex items-center space-x-1">
          <motion.div
            className="w-2 h-2 rounded-full bg-cyan-nebula"
            animate={{ 
              opacity: connectionQuality === 'disconnected' ? 0.3 : [0.4, 1, 0.4],
              scale: connectionQuality === 'disconnected' ? 0.8 : [0.8, 1.2, 0.8] 
            }}
            transition={{ 
              duration: 2, 
              repeat: connectionQuality === 'disconnected' ? 0 : Infinity 
            }}
          />
          <span className="text-xs text-cosmic-gray">
            α Centauri
          </span>
        </div>
      </div>
    </div>
  );
};

export default StatusBar;