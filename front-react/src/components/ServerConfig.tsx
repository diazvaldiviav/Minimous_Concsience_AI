import React, { useState, useEffect } from 'react';
import { Settings, Server, Wifi, WifiOff, Save, RefreshCw, AlertCircle, CheckCircle, ExternalLink, Globe, Lock, Clock } from 'lucide-react';
import type { ServerStatus, ServerConfig as IServerConfig } from '../types';
import { consciousnessAPI } from '../services/consciousnessAPI';

interface ServerConfigProps {
  serverStatus: ServerStatus;
  onServerStatusChange: (status: ServerStatus) => void;
}

const ServerConfig: React.FC<ServerConfigProps> = ({ serverStatus, onServerStatusChange }) => {
  const [config, setConfig] = useState<IServerConfig>({
    baseUrl: 'http://localhost:8000',
    timeout: 30000,
    retries: 3,
    defaultLanguage: 'auto',
  });
  
  const [tempUrl, setTempUrl] = useState(config.baseUrl);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
    responseTime?: number;
  } | null>(null);
  const [saved, setSaved] = useState(false);
  const [serverInfo, setServerInfo] = useState<any>(null);

  // Common server URLs for quick selection
  const presetUrls = [
    { name: 'Local Development', url: 'http://localhost:8000', description: 'Default local server' },
    { name: 'Google Colab (Example)', url: 'https://abc123.ngrok.io', description: 'ngrok tunnel from Colab' },
    { name: 'Custom Server', url: '', description: 'Enter your custom URL' },
  ];

  useEffect(() => {
    // Load saved configuration from localStorage
    const savedConfig = localStorage.getItem('consciousness-server-config');
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig);
        setConfig(parsed);
        setTempUrl(parsed.baseUrl);
        consciousnessAPI.setBaseUrl(parsed.baseUrl);
        consciousnessAPI.setTimeout(parsed.timeout);
      } catch (error) {
        console.error('Failed to load saved config:', error);
      }
    }
  }, []);

  const testConnection = async () => {
    setTesting(true);
    setTestResult(null);
    
    try {
      const startTime = Date.now();
      
      // Try to connect to the server
      consciousnessAPI.setBaseUrl(tempUrl);
      const health = await consciousnessAPI.checkConnection();
      
      const responseTime = Date.now() - startTime;
      
      if (health) {
        setTestResult({
          success: true,
          message: 'Connection successful',
          responseTime
        });
        
        // Try to get server info
        try {
          const stats = await consciousnessAPI.getServerStats();
          setServerInfo(stats);
        } catch (error) {
          console.warn('Could not fetch server stats:', error);
        }
        
        // Update server status
        onServerStatusChange({
          connected: true,
          url: tempUrl,
          lastPing: Date.now()
        });
      } else {
        throw new Error('Health check returned false');
      }
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : 'Connection failed'
      });
      
      onServerStatusChange({
        connected: false,
        url: tempUrl
      });
    } finally {
      setTesting(false);
    }
  };

  const saveConfig = () => {
    const newConfig = {
      ...config,
      baseUrl: tempUrl
    };
    
    setConfig(newConfig);
    
    // Save to localStorage
    localStorage.setItem('consciousness-server-config', JSON.stringify(newConfig));
    
    // Update API configuration
    consciousnessAPI.setBaseUrl(newConfig.baseUrl);
    consciousnessAPI.setTimeout(newConfig.timeout);
    
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handlePresetSelect = (url: string) => {
    setTempUrl(url);
    setTestResult(null);
    setServerInfo(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Settings className="w-8 h-8 text-consciousness-600" />
          <div>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
              Server Configuration
            </h2>
            <p className="text-slate-500 dark:text-slate-400">
              Configure connection to Consciousness AI server
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {serverStatus.connected ? (
            <div className="flex items-center space-x-2 px-3 py-1 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-full">
              <Wifi className="w-4 h-4 text-green-600 dark:text-green-400" />
              <span className="text-sm font-medium text-green-600 dark:text-green-400">Connected</span>
            </div>
          ) : (
            <div className="flex items-center space-x-2 px-3 py-1 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-full">
              <WifiOff className="w-4 h-4 text-red-600 dark:text-red-400" />
              <span className="text-sm font-medium text-red-600 dark:text-red-400">Disconnected</span>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Connection Settings */}
        <div className="xl:col-span-2 space-y-6">
          {/* Server URL Configuration */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center space-x-2">
              <Server className="w-5 h-5" />
              <span>Server Connection</span>
            </h3>

            {/* Preset URLs */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Quick Select
              </label>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {presetUrls.map((preset, index) => (
                  <button
                    key={index}
                    onClick={() => preset.url && handlePresetSelect(preset.url)}
                    disabled={preset.url === '' && preset.name === 'Custom Server'}
                    className={`p-3 text-left border rounded-lg transition-colors ${
                      tempUrl === preset.url && preset.url !== ''
                        ? 'border-consciousness-500 bg-consciousness-50 dark:bg-consciousness-900/20'
                        : 'border-slate-200 dark:border-slate-600 hover:border-slate-300 dark:hover:border-slate-500'
                    }`}
                  >
                    <div className="font-medium text-slate-900 dark:text-white text-sm">
                      {preset.name}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      {preset.description}
                    </div>
                    {preset.url && (
                      <div className="text-xs text-consciousness-600 dark:text-consciousness-400 mt-1 font-mono">
                        {preset.url}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom URL Input */}
            <div className="mb-4">
              <label htmlFor="server-url" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Server URL
              </label>
              <div className="flex space-x-3">
                <input
                  type="url"
                  id="server-url"
                  value={tempUrl}
                  onChange={(e) => setTempUrl(e.target.value)}
                  placeholder="https://your-server.ngrok.io"
                  className="flex-1 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-white placeholder-slate-500 dark:placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-consciousness-500 focus:border-consciousness-500"
                />
                <button
                  onClick={testConnection}
                  disabled={testing || !tempUrl.trim()}
                  className="px-4 py-2 bg-consciousness-600 hover:bg-consciousness-700 disabled:bg-slate-400 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
                >
                  {testing ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Wifi className="w-4 h-4" />
                  )}
                  <span>{testing ? 'Testing...' : 'Test'}</span>
                </button>
              </div>
            </div>

            {/* Test Result */}
            {testResult && (
              <div className={`p-4 rounded-lg border mb-4 ${
                testResult.success
                  ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                  : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
              }`}>
                <div className="flex items-center space-x-2">
                  {testResult.success ? (
                    <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                  )}
                  <span className={`font-medium ${
                    testResult.success
                      ? 'text-green-600 dark:text-green-400'
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    {testResult.message}
                  </span>
                  {testResult.responseTime && (
                    <span className="text-sm text-slate-500 dark:text-slate-400">
                      ({testResult.responseTime}ms)
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Advanced Settings */}
            <div className="space-y-4">
              <h4 className="font-medium text-slate-900 dark:text-white">Advanced Settings</h4>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label htmlFor="timeout" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Timeout (ms)
                  </label>
                  <input
                    type="number"
                    id="timeout"
                    value={config.timeout}
                    onChange={(e) => setConfig(prev => ({ ...prev, timeout: parseInt(e.target.value) || 30000 }))}
                    min="5000"
                    max="120000"
                    step="5000"
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="retries" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Max Retries
                  </label>
                  <input
                    type="number"
                    id="retries"
                    value={config.retries}
                    onChange={(e) => setConfig(prev => ({ ...prev, retries: parseInt(e.target.value) || 3 }))}
                    min="1"
                    max="10"
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="language" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Default Language
                  </label>
                  <select
                    id="language"
                    value={config.defaultLanguage}
                    onChange={(e) => setConfig(prev => ({ ...prev, defaultLanguage: e.target.value as 'auto' | 'en' | 'es' }))}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-white"
                  >
                    <option value="auto">Auto Detect</option>
                    <option value="en">English</option>
                    <option value="es">Español</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Save Configuration */}
            <div className="flex justify-end pt-4 border-t border-slate-200 dark:border-slate-700 mt-6">
              <button
                onClick={saveConfig}
                className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center space-x-2 ${
                  saved
                    ? 'bg-green-600 text-white'
                    : 'bg-consciousness-600 hover:bg-consciousness-700 text-white'
                }`}
              >
                {saved ? (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    <span>Saved!</span>
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    <span>Save Configuration</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Server Information & Status */}
        <div className="space-y-6">
          {/* Connection Status */}
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center space-x-2">
              <Globe className="w-5 h-5" />
              <span>Connection Status</span>
            </h3>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">Status</span>
                <span className={`text-sm font-medium ${
                  serverStatus.connected 
                    ? 'text-green-600 dark:text-green-400' 
                    : 'text-red-600 dark:text-red-400'
                }`}>
                  {serverStatus.connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">URL</span>
                <span className="text-sm text-slate-900 dark:text-white font-mono truncate max-w-32" title={serverStatus.url}>
                  {serverStatus.url || 'Not set'}
                </span>
              </div>

              {serverStatus.lastPing && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Last Ping</span>
                  <span className="text-sm text-slate-900 dark:text-white">
                    {new Date(serverStatus.lastPing).toLocaleTimeString()}
                  </span>
                </div>
              )}

              {testResult?.responseTime && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Response Time</span>
                  <span className="text-sm text-slate-900 dark:text-white">
                    {testResult.responseTime}ms
                  </span>
                </div>
              )}
            </div>

            {serverStatus.connected && (
              <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                <a
                  href={`${serverStatus.url}/docs`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 text-consciousness-600 dark:text-consciousness-400 hover:text-consciousness-700 dark:hover:text-consciousness-300 text-sm"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>View API Docs</span>
                </a>
              </div>
            )}
          </div>

          {/* Server Information */}
          {serverInfo && (
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center space-x-2">
                <Server className="w-5 h-5" />
                <span>Server Info</span>
              </h3>

              <div className="space-y-3">
                {serverInfo.version && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Version</span>
                    <span className="text-sm text-slate-900 dark:text-white font-medium">
                      {serverInfo.version}
                    </span>
                  </div>
                )}

                {serverInfo.total_requests !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Total Requests</span>
                    <span className="text-sm text-slate-900 dark:text-white">
                      {serverInfo.total_requests.toLocaleString()}
                    </span>
                  </div>
                )}

                {serverInfo.average_f_score && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Avg F-Score</span>
                    <span className="text-sm text-slate-900 dark:text-white">
                      {serverInfo.average_f_score.toFixed(3)}
                    </span>
                  </div>
                )}

                {serverInfo.uptime && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Uptime</span>
                    <span className="text-sm text-slate-900 dark:text-white">
                      {Math.floor(serverInfo.uptime / 3600)}h {Math.floor((serverInfo.uptime % 3600) / 60)}m
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Help & Documentation */}
          <div className="bg-gradient-to-br from-consciousness-50 to-purple-50 dark:from-consciousness-900/20 dark:to-purple-900/20 rounded-xl border border-consciousness-200 dark:border-consciousness-800 p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">
              📚 Getting Started
            </h3>

            <div className="space-y-3 text-sm">
              <div>
                <h4 className="font-medium text-slate-900 dark:text-white mb-1">
                  1. Google Colab Server
                </h4>
                <p className="text-slate-600 dark:text-slate-400">
                  Upload <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">servidor_consciences.ipynb</code> to Google Colab and get a public ngrok URL.
                </p>
              </div>

              <div>
                <h4 className="font-medium text-slate-900 dark:text-white mb-1">
                  2. Local Development
                </h4>
                <p className="text-slate-600 dark:text-slate-400">
                  Run <code className="bg-slate-200 dark:bg-slate-700 px-1 rounded">python conscious_ai/api/run_server.py</code> locally on port 8000.
                </p>
              </div>

              <div>
                <h4 className="font-medium text-slate-900 dark:text-white mb-1">
                  3. Custom Deployment
                </h4>
                <p className="text-slate-600 dark:text-slate-400">
                  Deploy the FastAPI server to any cloud platform and enter the URL above.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ServerConfig;