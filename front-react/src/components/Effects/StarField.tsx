// Xentauri SC-1 StarField Effect Component

import React, { useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';

interface Star {
  x: number;
  y: number;
  z: number;
  prevX: number;
  prevY: number;
  size: number;
  opacity: number;
  color: string;
  twinkleOffset: number;
}

interface StarFieldProps {
  starCount?: number;
  speed?: number;
  depth?: number;
  trailLength?: number;
  interactive?: boolean;
  colors?: string[];
  className?: string;
}

const StarField: React.FC<StarFieldProps> = ({
  starCount = 200,
  speed = 0.5,
  depth = 1000,
  trailLength = 0.8,
  interactive = true,
  colors = [
    '#00D4FF', // cyan-nebula
    '#FF00FF', // magenta-pulsar  
    '#00FF88', // quantum-green
    '#FFED4E', // yellow-star
    '#FF6B35', // orange-nova
    '#FFFFFF', // star-white
    '#B0B8C4', // cosmic-silver
  ],
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const starsRef = useRef<Star[]>([]);
  const mouseRef = useRef({ x: 0, y: 0, active: false });

  // Initialize stars
  const initStars = useCallback(() => {
    const stars: Star[] = [];
    for (let i = 0; i < starCount; i++) {
      stars.push({
        x: (Math.random() - 0.5) * depth,
        y: (Math.random() - 0.5) * depth,
        z: Math.random() * depth,
        prevX: 0,
        prevY: 0,
        size: Math.random() * 2 + 0.5,
        opacity: Math.random() * 0.8 + 0.2,
        color: colors[Math.floor(Math.random() * colors.length)],
        twinkleOffset: Math.random() * Math.PI * 2
      });
    }
    starsRef.current = stars;
  }, [starCount, depth, colors]);

  // Update stars position
  const updateStars = useCallback((canvas: HTMLCanvasElement, deltaTime: number) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const mouseInfluence = 50;

    starsRef.current.forEach(star => {
      // Store previous position for trails
      const screenX = (star.x / star.z) * centerX + centerX;
      const screenY = (star.y / star.z) * centerY + centerY;
      star.prevX = screenX;
      star.prevY = screenY;

      // Move star towards viewer
      star.z -= speed * deltaTime;

      // Mouse interaction
      if (interactive && mouseRef.current.active) {
        const mouseInfluenceX = (mouseRef.current.x - centerX) * mouseInfluence;
        const mouseInfluenceY = (mouseRef.current.y - centerY) * mouseInfluence;
        
        star.x += mouseInfluenceX / star.z * deltaTime * 0.1;
        star.y += mouseInfluenceY / star.z * deltaTime * 0.1;
      }

      // Reset star if it's too close
      if (star.z <= 1) {
        star.x = (Math.random() - 0.5) * depth;
        star.y = (Math.random() - 0.5) * depth;
        star.z = depth;
        star.color = colors[Math.floor(Math.random() * colors.length)];
      }
    });
  }, [speed, depth, interactive, colors]);

  // Draw stars
  const drawStars = useCallback((canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, time: number) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // Clear canvas with slight trail effect
    ctx.fillStyle = `rgba(10, 10, 20, ${1 - trailLength})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    starsRef.current.forEach(star => {
      const screenX = (star.x / star.z) * centerX + centerX;
      const screenY = (star.y / star.z) * centerY + centerY;
      
      // Skip if star is outside screen
      if (screenX < -50 || screenX > canvas.width + 50 || 
          screenY < -50 || screenY > canvas.height + 50) {
        return;
      }

      // Calculate star properties based on distance
      const scale = (1 - star.z / depth);
      const size = star.size * scale * (0.5 + Math.sin(time * 0.003 + star.twinkleOffset) * 0.5);
      const opacity = star.opacity * scale;
      const brightness = 0.5 + Math.sin(time * 0.002 + star.twinkleOffset) * 0.3;

      // Draw star trail if moving fast
      if (trailLength > 0 && scale > 0.1) {
        const dx = screenX - star.prevX;
        const dy = screenY - star.prevY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > 0.5) {
          ctx.save();
          ctx.globalAlpha = opacity * 0.3;
          ctx.strokeStyle = star.color;
          ctx.lineWidth = size * 0.5;
          ctx.beginPath();
          ctx.moveTo(star.prevX, star.prevY);
          ctx.lineTo(screenX, screenY);
          ctx.stroke();
          ctx.restore();
        }
      }

      // Draw main star
      ctx.save();
      ctx.globalAlpha = opacity * brightness;
      ctx.fillStyle = star.color;
      
      // Create glow effect for larger stars
      if (size > 1) {
        const gradient = ctx.createRadialGradient(screenX, screenY, 0, screenX, screenY, size * 2);
        gradient.addColorStop(0, star.color);
        gradient.addColorStop(0.5, star.color + '80');
        gradient.addColorStop(1, star.color + '00');
        ctx.fillStyle = gradient;
      }

      ctx.beginPath();
      ctx.arc(screenX, screenY, size, 0, Math.PI * 2);
      ctx.fill();

      // Add sparkle effect for very close stars
      if (scale > 0.8 && Math.random() > 0.98) {
        ctx.globalAlpha = opacity;
        ctx.strokeStyle = star.color;
        ctx.lineWidth = 1;
        const sparkleSize = size * 3;
        
        ctx.beginPath();
        ctx.moveTo(screenX - sparkleSize, screenY);
        ctx.lineTo(screenX + sparkleSize, screenY);
        ctx.moveTo(screenX, screenY - sparkleSize);
        ctx.lineTo(screenX, screenY + sparkleSize);
        ctx.stroke();
      }

      ctx.restore();
    });
  }, [depth, trailLength, colors]);

  // Animation loop
  const animate = useCallback((time: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const deltaTime = Math.min(time - (animationRef.current || time), 50); // Cap delta time
    animationRef.current = time;

    updateStars(canvas, deltaTime);
    drawStars(canvas, ctx, time);

    requestAnimationFrame(animate);
  }, [updateStars, drawStars]);

  // Handle mouse movement
  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!interactive) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    mouseRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      active: true
    };
  }, [interactive]);

  const handleMouseLeave = useCallback(() => {
    mouseRef.current.active = false;
  }, []);

  // Resize handler
  const handleResize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
  }, []);

  // Setup effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    initStars();
    handleResize();

    // Add event listeners
    if (interactive) {
      canvas.addEventListener('mousemove', handleMouseMove);
      canvas.addEventListener('mouseleave', handleMouseLeave);
    }
    window.addEventListener('resize', handleResize);

    // Start animation
    requestAnimationFrame(animate);

    return () => {
      if (interactive) {
        canvas.removeEventListener('mousemove', handleMouseMove);
        canvas.removeEventListener('mouseleave', handleMouseLeave);
      }
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [initStars, handleResize, handleMouseMove, handleMouseLeave, animate, interactive]);

  return (
    <motion.canvas
      ref={canvasRef}
      className={`
        fixed inset-0 pointer-events-none z-0
        ${interactive ? 'pointer-events-auto' : ''}
        ${className}
      `}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 2 }}
      style={{
        background: 'radial-gradient(ellipse at center, #1a1a2e 0%, #16213e 50%, #0f0f23 100%)'
      }}
    />
  );
};

export default StarField;