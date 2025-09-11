// Xentauri SC-1 Emotion Indicator Component

import React from 'react';
import { motion } from 'framer-motion';
import { 
  HiHeart, 
  HiFaceFrown, 
  HiFaceSmile, 
  HiEye,
  HiLightBulb,
  HiQuestionMarkCircle,
  HiExclamationTriangle,
  HiSparkles
} from 'react-icons/hi2';

interface EmotionIndicatorProps {
  emotion: string;
  intensity?: number; // 0-1 scale
  confidence?: number; // 0-1 scale
  showLabel?: boolean;
  size?: 'small' | 'medium' | 'large';
  animated?: boolean;
  className?: string;
}

const EmotionIndicator: React.FC<EmotionIndicatorProps> = ({
  emotion,
  intensity = 0.5,
  confidence = 0.5,
  showLabel = true,
  size = 'medium',
  animated = true,
  className = ''
}) => {
  // Emotion mapping to icons and colors
  const getEmotionData = (emotion: string) => {
    const normalizedEmotion = emotion.toLowerCase().trim();
    
    switch (normalizedEmotion) {
      case 'happy':
      case 'joy':
      case 'excited':
        return {
          icon: HiFaceSmile,
          color: 'text-quantum-green',
          bgColor: 'bg-quantum-green/20',
          borderColor: 'border-quantum-green/40'
        };
      
      case 'sad':
      case 'melancholy':
      case 'disappointed':
        return {
          icon: HiFaceFrown,
          color: 'text-blue-hyperspace',
          bgColor: 'bg-blue-hyperspace/20',
          borderColor: 'border-blue-hyperspace/40'
        };
      
      case 'curious':
      case 'interested':
      case 'wondering':
        return {
          icon: HiEye,
          color: 'text-cyan-nebula',
          bgColor: 'bg-cyan-nebula/20',
          borderColor: 'border-cyan-nebula/40'
        };
      
      case 'analytical':
      case 'focused':
      case 'thinking':
        return {
          icon: HiLightBulb,
          color: 'text-yellow-star',
          bgColor: 'bg-yellow-star/20',
          borderColor: 'border-yellow-star/40'
        };
      
      case 'confused':
      case 'uncertain':
      case 'perplexed':
        return {
          icon: HiQuestionMarkCircle,
          color: 'text-cosmic-gray',
          bgColor: 'bg-cosmic-gray/20',
          borderColor: 'border-cosmic-gray/40'
        };
      
      case 'concerned':
      case 'worried':
      case 'anxious':
        return {
          icon: HiExclamationTriangle,
          color: 'text-orange-nova',
          bgColor: 'bg-orange-nova/20',
          borderColor: 'border-orange-nova/40'
        };
      
      case 'amazed':
      case 'awed':
      case 'wonder':
        return {
          icon: HiSparkles,
          color: 'text-magenta-pulsar',
          bgColor: 'bg-magenta-pulsar/20',
          borderColor: 'border-magenta-pulsar/40'
        };
      
      case 'empathetic':
      case 'caring':
      case 'compassionate':
        return {
          icon: HiHeart,
          color: 'text-red-shift',
          bgColor: 'bg-red-shift/20',
          borderColor: 'border-red-shift/40'
        };
      
      default:
        return {
          icon: HiHeart,
          color: 'text-cosmic-silver',
          bgColor: 'bg-cosmic-silver/20',
          borderColor: 'border-cosmic-silver/40'
        };
    }
  };

  const emotionData = getEmotionData(emotion);
  const Icon = emotionData.icon;

  // Size configurations
  const sizeConfig = {
    small: {
      container: 'w-8 h-8',
      icon: 'w-4 h-4',
      text: 'text-xs',
      padding: 'p-1'
    },
    medium: {
      container: 'w-12 h-12',
      icon: 'w-6 h-6',
      text: 'text-sm',
      padding: 'p-2'
    },
    large: {
      container: 'w-16 h-16',
      icon: 'w-8 h-8',
      text: 'text-base',
      padding: 'p-3'
    }
  };

  const config = sizeConfig[size];

  // Animation variants
  const containerVariants = {
    idle: {
      scale: 1,
      rotate: 0,
    },
    pulse: {
      scale: [1, 1.1, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    },
    rotate: {
      rotate: [0, 360],
      transition: {
        duration: 8,
        repeat: Infinity,
        ease: 'linear'
      }
    }
  };

  const iconVariants = {
    idle: {
      scale: 1,
      opacity: 0.7 + (intensity * 0.3),
    },
    active: {
      scale: [1, 1.2, 1],
      opacity: [0.7 + (intensity * 0.3), 1, 0.7 + (intensity * 0.3)],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: 'easeInOut'
      }
    }
  };

  // Intensity ring opacity
  const ringOpacity = Math.max(0.1, intensity);

  return (
    <div className={`flex flex-col items-center space-y-1 ${className}`}>
      {/* Main Emotion Indicator */}
      <motion.div
        className={`
          relative ${config.container} ${config.padding}
          ${emotionData.bgColor} ${emotionData.borderColor}
          border-2 rounded-full
          flex items-center justify-center
          overflow-hidden
        `}
        variants={containerVariants}
        animate={animated ? (intensity > 0.7 ? 'pulse' : 'idle') : 'idle'}
        title={`${emotion} (${(intensity * 100).toFixed(0)}% intensity, ${(confidence * 100).toFixed(0)}% confidence)`}
      >
        {/* Intensity Background Ring */}
        <motion.div
          className={`
            absolute inset-0 rounded-full border-2 ${emotionData.borderColor}
          `}
          style={{ 
            opacity: ringOpacity,
            scale: 1 + (intensity * 0.2)
          }}
          animate={animated && intensity > 0.5 ? {
            scale: [1 + (intensity * 0.2), 1 + (intensity * 0.4), 1 + (intensity * 0.2)],
            opacity: [ringOpacity, ringOpacity * 1.5, ringOpacity]
          } : {}}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />

        {/* Main Icon */}
        <motion.div
          variants={iconVariants}
          animate={animated && intensity > 0.6 ? 'active' : 'idle'}
        >
          <Icon className={`${config.icon} ${emotionData.color}`} />
        </motion.div>

        {/* Confidence Indicator */}
        <div
          className={`
            absolute bottom-0 left-0 right-0 h-1
            ${emotionData.bgColor} ${emotionData.borderColor}
            border-t opacity-60
          `}
          style={{
            background: `linear-gradient(90deg, ${emotionData.color.replace('text-', '')} ${confidence * 100}%, transparent ${confidence * 100}%)`
          }}
        />
      </motion.div>

      {/* Emotion Label */}
      {showLabel && (
        <div className="text-center">
          <div className={`${config.text} font-medium ${emotionData.color} capitalize`}>
            {emotion}
          </div>
          <div className="text-xs text-cosmic-gray">
            {(intensity * 100).toFixed(0)}%
          </div>
        </div>
      )}
    </div>
  );
};

export default EmotionIndicator;