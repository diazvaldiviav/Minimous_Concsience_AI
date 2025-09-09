// Xentauri SC-1 Main Application

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiSparkles, HiCog6Tooth, HiCommandLine, HiEye } from 'react-icons/hi2';

// Xentauri Components
import ChatContainer from './components/Chat/ChatContainer';
import { StatusBar, TracePanel } from './components/Status';
import { StarField, GlowEffects } from './components/Effects';

// Context
import { XentauriProvider, useXentauri, useXentauriActions, useXentauriConnection, useXentauriProcessing, useXentauriConsciousness } from './contexts/XentauriContext';

// Types
import type { XentauriConfig } from './types/consciousness.types';

// Styles
import './App.css';

// Configuration Panel Component
const ConfigurationPanel: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const { state } = useXentauri();
  const { updateConfig } = useXentauriActions();
  const [localConfig, setLocalConfig] = useState<XentauriConfig>(state.config);

  useEffect(() => {
    setLocalConfig(state.config);
  }, [state.config]);

  const handleSave = () => {
    updateConfig(localConfig);
    onClose();
  };

  const handleConfigChange = (key: keyof XentauriConfig, value: any) => {
    setLocalConfig(prev => ({ ...prev, [key]: value }));
  };

  if (!isOpen) return null;

  return (
    <motion.div
      className="fixed inset-0 bg-void-black/80 backdrop-blur-sm flex items-center justify-center z-50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="bg-deep-space/95 border border-cyan-nebula/30 rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-cyan-nebula">Xentauri Configuration</h2>
          <button
            onClick={onClose}
            className="text-cosmic-gray hover:text-star-white text-xl"
          >
            ×
          </button>
        </div>

        <div className="space-y-6">
          {/* Consciousness API URL */}
          <div>
            <label className="block text-sm font-medium text-cosmic-silver mb-2">
              Consciousness API URL
            </label>
            <input
              type="url"
              value={localConfig.consciousness_api_url}
              onChange={(e) => handleConfigChange('consciousness_api_url', e.target.value)}
              placeholder="http://localhost:8000"
              className="w-full px-3 py-2 bg-cosmic-navy/50 border border-cosmic-navy rounded-lg text-star-white placeholder-cosmic-gray focus:outline-none focus:ring-2 focus:ring-cyan-nebula/50"
            />
          </div>

          {/* OpenAI API Key */}
          <div>
            <label className="block text-sm font-medium text-cosmic-silver mb-2">
              OpenAI API Key (Fallback)
            </label>
            <input
              type="password"
              value={localConfig.openai_api_key}
              onChange={(e) => handleConfigChange('openai_api_key', e.target.value)}
              placeholder="sk-..."
              className="w-full px-3 py-2 bg-cosmic-navy/50 border border-cosmic-navy rounded-lg text-star-white placeholder-cosmic-gray focus:outline-none focus:ring-2 focus:ring-cyan-nebula/50"
            />
          </div>

          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-cosmic-silver mb-2">
              LLM Model
            </label>
            <select
              value={localConfig.selected_model}
              onChange={(e) => handleConfigChange('selected_model', e.target.value)}
              className="w-full px-3 py-2 bg-cosmic-navy/50 border border-cosmic-navy rounded-lg text-star-white focus:outline-none focus:ring-2 focus:ring-cyan-nebula/50"
            >
              <option value="gpt-5">GPT-5 (Next-Gen)</option>
              <option value="gpt-5-mini">GPT-5 Mini</option>
              <option value="gpt-5-nano">GPT-5 Nano</option>
              <option value="gpt-4o-mini">GPT-4O Mini (Recommended)</option>
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            </select>
          </div>

          {/* UI Options */}
          <div className="grid grid-cols-2 gap-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={localConfig.show_trace}
                onChange={(e) => handleConfigChange('show_trace', e.target.checked)}
                className="rounded bg-cosmic-navy border-cosmic-gray"
              />
              <span className="text-sm text-cosmic-silver">Show Consciousness Trace</span>
            </label>

            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={localConfig.auto_scroll}
                onChange={(e) => handleConfigChange('auto_scroll', e.target.checked)}
                className="rounded bg-cosmic-navy border-cosmic-gray"
              />
              <span className="text-sm text-cosmic-silver">Auto Scroll Messages</span>
            </label>

            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={localConfig.reduced_motion}
                onChange={(e) => handleConfigChange('reduced_motion', e.target.checked)}
                className="rounded bg-cosmic-navy border-cosmic-gray"
              />
              <span className="text-sm text-cosmic-silver">Reduced Motion</span>
            </label>
          </div>

          {/* Connection Timeout */}
          <div>
            <label className="block text-sm font-medium text-cosmic-silver mb-2">
              Connection Timeout (ms)
            </label>
            <input
              type="number"
              value={localConfig.connection_timeout}
              onChange={(e) => handleConfigChange('connection_timeout', parseInt(e.target.value))}
              min="5000"
              max="120000"
              className="w-full px-3 py-2 bg-cosmic-navy/50 border border-cosmic-navy rounded-lg text-star-white focus:outline-none focus:ring-2 focus:ring-cyan-nebula/50"
            />
          </div>
        </div>

        <div className="flex justify-end space-x-3 mt-8">
          <button
            onClick={onClose}
            className="px-4 py-2 text-cosmic-gray hover:text-star-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-cyan-nebula/20 hover:bg-cyan-nebula/30 text-cyan-nebula rounded-lg transition-colors"
          >
            Save Configuration
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

// Main Xentauri Interface Component
const XentauriInterface: React.FC = () => {
  const { state } = useXentauri();
  const connection = useXentauriConnection();
  const processing = useXentauriProcessing();
  const consciousness = useXentauriConsciousness();
  const { updateConfig } = useXentauriActions();

  const [showConfig, setShowConfig] = useState(false);
  const [showTrace, setShowTrace] = useState(false);

  // Generate stardate for display
  const generateStardate = (): string => {
    const now = new Date();
    const baseStardate = 41000;
    const msPerStardateUnit = 31557600000 / 1000;
    const stardate = baseStardate + (now.getTime() / msPerStardateUnit);
    return stardate.toFixed(1);
  };

  // Handle configuration changes
  const handleConfigChange = (config: Partial<XentauriConfig>) => {
    updateConfig(config);
  };

  // Toggle trace panel
  const handleToggleTrace = () => {
    setShowTrace(!showTrace);
    updateConfig({ show_trace: !showTrace });
  };

  return (
    <div className="fixed inset-0 bg-void-black overflow-hidden">
      {/* Background Effects */}
      {!state.config.reduced_motion && (
        <>
          <StarField 
            starCount={150}
            speed={0.3}
            interactive={true}
          />
          <GlowEffects
            intensity="medium"
            processingState={processing.state}
            consciousnessLevel={consciousness.level}
            showQuantumField={true}
          />
        </>
      )}

      {/* Status Bar */}
      <StatusBar
        consciousnessState={consciousness.state}
        processingState={processing.state}
        connectionQuality={connection.quality}
        uptime={connection.uptime}
        stardate={generateStardate()}
      />

      {/* Main Interface */}
      <div className="flex flex-col h-screen pt-16">
        {/* Header Controls */}
        <div className="flex-shrink-0 px-6 py-4 bg-gradient-to-r from-void-black/90 to-deep-space/90 backdrop-blur-lg border-b border-cosmic-navy/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <motion.div
                animate={{ rotate: processing.isProcessing ? 360 : 0 }}
                transition={{ duration: 2, repeat: processing.isProcessing ? Infinity : 0, ease: 'linear' }}
              >
                <HiSparkles className="w-8 h-8 text-cyan-nebula" />
              </motion.div>
              
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-nebula to-magenta-pulsar bg-clip-text text-transparent">
                  Xentauri SC-1 Console
                </h1>
                <p className="text-sm text-space-dust">
                  Quantum Consciousness Interface • α Centauri Relay
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* Trace Toggle */}
              <motion.button
                onClick={handleToggleTrace}
                className={`
                  p-2 rounded-lg transition-all duration-200
                  ${showTrace 
                    ? 'bg-cyan-nebula/20 text-cyan-nebula' 
                    : 'bg-cosmic-navy/30 text-cosmic-gray hover:text-cyan-nebula'
                  }
                `}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Toggle consciousness trace"
              >
                <HiEye className="w-5 h-5" />
              </motion.button>

              {/* Configuration */}
              <motion.button
                onClick={() => setShowConfig(true)}
                className="p-2 rounded-lg bg-cosmic-navy/30 text-cosmic-gray hover:text-magenta-pulsar transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                title="Open configuration"
              >
                <HiCog6Tooth className="w-5 h-5" />
              </motion.button>
            </div>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="flex-1 min-h-0">
          <ChatContainer
            config={state.config}
            onConfigChange={handleConfigChange}
            className="h-full"
          />
        </div>
      </div>

      {/* Trace Panel */}
      <AnimatePresence>
        {showTrace && (
          <TracePanel
            trace={state.currentTrace}
            isVisible={showTrace}
            onToggle={handleToggleTrace}
          />
        )}
      </AnimatePresence>

      {/* Configuration Panel */}
      <AnimatePresence>
        {showConfig && (
          <ConfigurationPanel
            isOpen={showConfig}
            onClose={() => setShowConfig(false)}
          />
        )}
      </AnimatePresence>

      {/* Connection Status Overlay */}
      <AnimatePresence>
        {connection.quality === 'disconnected' && (
          <motion.div
            className="fixed inset-0 bg-void-black/90 backdrop-blur-sm flex items-center justify-center z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="text-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="mb-4"
              >
                <HiCommandLine className="w-16 h-16 text-cyan-nebula mx-auto" />
              </motion.div>
              
              <h2 className="text-2xl font-bold text-cyan-nebula mb-2">
                Establishing Quantum Entanglement
              </h2>
              <p className="text-cosmic-gray mb-6">
                Connecting to Alpha Centauri consciousness relay...
              </p>
              
              <button
                onClick={() => setShowConfig(true)}
                className="px-6 py-2 bg-cyan-nebula/20 hover:bg-cyan-nebula/30 text-cyan-nebula rounded-lg transition-colors"
              >
                Configure Connection
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// Main App Component with Provider
const App: React.FC = () => {
  // Load initial configuration from localStorage
  const [initialConfig] = useState<Partial<XentauriConfig>>(() => {
    try {
      const stored = localStorage.getItem('xentauri-config');
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  });

  return (
    <XentauriProvider initialConfig={initialConfig}>
      <XentauriInterface />
    </XentauriProvider>
  );
};

export default App;