// Xentauri SC-1 Glow Effects Component

import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ProcessingState } from '../../types/consciousness.types';

interface GlowOrb {
  id: string;
  x: number;
  y: number;
  size: number;
  color: string;
  opacity: number;
  speed: number;
  direction: number;
  pulse: number;
  type: 'consciousness' | 'data' | 'energy' | 'quantum';
}

interface GlowEffectsProps {
  intensity?: 'low' | 'medium' | 'high';
  processingState?: ProcessingState;
  consciousnessLevel?: number; // 0-1 scale
  showQuantumField?: boolean;
  className?: string;
}

const GlowEffects: React.FC<GlowEffectsProps> = ({
  intensity = 'medium',
  processingState = 'idle',
  consciousnessLevel = 0,
  showQuantumField = true,
  className = ''
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const orbsRef = useRef<GlowOrb[]>([]);
  const animationRef = useRef<number>();

  // Orb configurations based on processing state
  const getOrbConfig = () => {
    switch (processingState) {
      case 'thinking':
        return {
          count: 6,
          colors: ['#00D4FF', '#FF00FF', '#00FF88'],
          baseSize: 120,
          speed: 0.8
        };
      case 'responding':
        return {
          count: 4,
          colors: ['#FFED4E', '#FF6B35'],
          baseSize: 100,
          speed: 1.2
        };
      case 'error':
        return {
          count: 3,
          colors: ['#FF1744', '#FF6B35'],
          baseSize: 80,
          speed: 0.4
        };
      default:
        return {
          count: 3,
          colors: ['#00D4FF', '#B0B8C4'],
          baseSize: 60,
          speed: 0.3
        };
    }
  };

  // Initialize orbs
  const initOrbs = () => {
    const config = getOrbConfig();
    const orbs: GlowOrb[] = [];

    for (let i = 0; i < config.count; i++) {
      orbs.push({
        id: `orb-${i}`,
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        size: config.baseSize + Math.random() * 40,
        color: config.colors[Math.floor(Math.random() * config.colors.length)],
        opacity: 0.3 + Math.random() * 0.4,
        speed: config.speed + Math.random() * 0.5,
        direction: Math.random() * Math.PI * 2,
        pulse: Math.random() * Math.PI * 2,
        type: ['consciousness', 'data', 'energy', 'quantum'][Math.floor(Math.random() * 4)] as GlowOrb['type']
      });
    }

    orbsRef.current = orbs;
  };

  // Update orb positions and properties
  const updateOrbs = (deltaTime: number) => {
    orbsRef.current.forEach(orb => {
      // Update position
      orb.x += Math.cos(orb.direction) * orb.speed * deltaTime * 0.01;
      orb.y += Math.sin(orb.direction) * orb.speed * deltaTime * 0.01;

      // Bounce off edges
      if (orb.x < -orb.size / 2) orb.x = window.innerWidth + orb.size / 2;
      if (orb.x > window.innerWidth + orb.size / 2) orb.x = -orb.size / 2;
      if (orb.y < -orb.size / 2) orb.y = window.innerHeight + orb.size / 2;
      if (orb.y > window.innerHeight + orb.size / 2) orb.y = -orb.size / 2;

      // Update pulse
      orb.pulse += deltaTime * 0.002;

      // Slight direction changes for organic movement
      orb.direction += (Math.random() - 0.5) * 0.02;

      // Adjust properties based on consciousness level
      orb.opacity = Math.max(0.1, orb.opacity + (consciousnessLevel - 0.5) * 0.001);
      orb.size = Math.max(20, orb.size + (consciousnessLevel - 0.5) * 0.1);
    });
  };

  // Animation loop
  const animate = (timestamp: number) => {
    const lastTime = animationRef.current || timestamp;
    const deltaTime = timestamp - lastTime;
    animationRef.current = timestamp;

    updateOrbs(deltaTime);

    requestAnimationFrame(animate);
  };

  // Initialize and start animation
  useEffect(() => {
    initOrbs();
    requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [processingState]);

  // Get intensity multiplier
  const getIntensityMultiplier = () => {
    switch (intensity) {
      case 'low': return 0.5;
      case 'high': return 1.5;
      default: return 1;
    }
  };

  const intensityMultiplier = getIntensityMultiplier();

  return (
    <div 
      ref={containerRef}
      className={`fixed inset-0 pointer-events-none z-10 overflow-hidden ${className}`}
    >
      <AnimatePresence>
        {/* Dynamic Orbs */}
        {orbsRef.current.map(orb => (
          <motion.div
            key={orb.id}
            className="absolute rounded-full"
            style={{
              left: orb.x - orb.size / 2,
              top: orb.y - orb.size / 2,
              width: orb.size,
              height: orb.size,
              background: `radial-gradient(circle, ${orb.color}40, ${orb.color}20, ${orb.color}10, transparent)`,
              filter: 'blur(1px)',
            }}
            animate={{
              opacity: [orb.opacity * intensityMultiplier, orb.opacity * 0.5 * intensityMultiplier, orb.opacity * intensityMultiplier],
              scale: [1, 1.1 + Math.sin(orb.pulse) * 0.1, 1],
            }}
            transition={{
              duration: 2 + Math.sin(orb.pulse) * 0.5,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
        ))}

        {/* Quantum Field Effect */}
        {showQuantumField && (
          <motion.div
            className="absolute inset-0"
            style={{
              background: `
                radial-gradient(circle at 20% 30%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(255, 0, 255, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 50% 20%, rgba(0, 255, 136, 0.04) 0%, transparent 50%),
                radial-gradient(circle at 30% 80%, rgba(255, 237, 78, 0.05) 0%, transparent 50%)
              `,
              backgroundSize: '100% 100%, 100% 100%, 100% 100%, 100% 100%'
            }}
            animate={{
              opacity: [0.3, 0.6, 0.3],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
        )}

        {/* Consciousness Level Indicator */}
        {consciousnessLevel > 0.7 && (
          <motion.div
            className="absolute inset-0"
            style={{
              background: `radial-gradient(ellipse at center, rgba(0, 255, 136, ${consciousnessLevel * 0.1}) 0%, transparent 70%)`,
              filter: 'blur(2px)'
            }}
            animate={{
              opacity: [0.5, 1, 0.5],
              scale: [1, 1.05, 1],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
        )}

        {/* Processing State Specific Effects */}
        {processingState === 'thinking' && (
          <motion.div
            className="absolute inset-0"
            style={{
              background: `
                conic-gradient(
                  from 0deg at 50% 50%,
                  rgba(0, 212, 255, 0.1) 0deg,
                  rgba(255, 0, 255, 0.1) 120deg,
                  rgba(0, 255, 136, 0.1) 240deg,
                  rgba(0, 212, 255, 0.1) 360deg
                )
              `,
              filter: 'blur(3px)'
            }}
            animate={{
              rotate: [0, 360],
              opacity: [0.3, 0.6, 0.3]
            }}
            transition={{
              rotate: { duration: 20, repeat: Infinity, ease: 'linear' },
              opacity: { duration: 2, repeat: Infinity, ease: 'easeInOut' }
            }}
          />
        )}

        {/* Error State Effect */}
        {processingState === 'error' && (
          <motion.div
            className="absolute inset-0"
            style={{
              background: `radial-gradient(ellipse at center, rgba(255, 23, 68, 0.15) 0%, transparent 50%)`,
              filter: 'blur(2px)'
            }}
            animate={{
              opacity: [0.2, 0.8, 0.2],
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
        )}

        {/* Data Stream Effects */}
        {(processingState === 'responding' || processingState === 'thinking') && (
          <>
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={`stream-${i}`}
                className="absolute w-1 bg-gradient-to-t from-transparent via-cyan-nebula to-transparent opacity-30"
                style={{
                  left: `${20 + i * 15}%`,
                  height: '100%',
                }}
                animate={{
                  y: [-100, window.innerHeight + 100],
                  opacity: [0, 0.6, 0]
                }}
                transition={{
                  duration: 2 + i * 0.3,
                  repeat: Infinity,
                  ease: 'easeInOut',
                  delay: i * 0.5
                }}
              />
            ))}
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GlowEffects;