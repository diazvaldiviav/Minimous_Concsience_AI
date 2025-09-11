import React, { useState, useEffect } from 'react';
import { Brain, Activity, Clock, TrendingUp, AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell } from 'recharts';
import type { ServerStatus, MetricsData, ConsciousnessMetrics as IConsciousnessMetrics } from '../types';
import { consciousnessAPI } from '../services/consciousnessAPI';

interface ConsciousnessMetricsProps {
  serverStatus: ServerStatus;
}

const ConsciousnessMetrics: React.FC<ConsciousnessMetricsProps> = ({ serverStatus }) => {
  const [metricsHistory, setMetricsHistory] = useState<MetricsData[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<IConsciousnessMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<number>(5000);
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Mock data for development when server is disconnected
  const mockMetricsHistory: MetricsData[] = Array.from({ length: 20 }, (_, i) => ({
    timestamp: new Date(Date.now() - (19 - i) * 30000),
    consciousness_level: 0.6 + Math.random() * 0.4,
    emotional_state: ['curious', 'contemplative', 'analytical', 'confident', 'reflective'][Math.floor(Math.random() * 5)],
    confidence: 0.5 + Math.random() * 0.4,
    memory_items: Math.floor(Math.random() * 15) + 5,
    processing_time: Math.floor(Math.random() * 500) + 100,
    phase_7_used: Math.random() > 0.3
  }));

  const mockCurrentMetrics: IConsciousnessMetrics = {
    f_score: 1.35,
    phi: 0.87,
    temporal_unification: 0.72,
    causal_integration: 0.65,
    reentrancy: 0.89,
    self_model_complexity: 0.78
  };

  const fetchMetrics = async () => {
    if (!serverStatus.connected) {
      // Use mock data when disconnected
      setMetricsHistory(mockMetricsHistory);
      setCurrentMetrics(mockCurrentMetrics);
      setError(null);
      return;
    }

    setLoading(true);
    try {
      const stats = await consciousnessAPI.getServerStats();
      
      // Transform stats to metrics format (adapt based on actual API response)
      const newMetric: MetricsData = {
        timestamp: new Date(),
        consciousness_level: stats.average_f_score || 0.8,
        emotional_state: stats.common_emotional_state || 'analytical',
        confidence: stats.average_confidence || 0.7,
        memory_items: stats.total_memories || 10,
        processing_time: stats.average_processing_time || 250,
        phase_7_used: stats.phase_7_usage_rate > 0.5
      };

      setMetricsHistory(prev => [...prev.slice(-19), newMetric]);
      setCurrentMetrics({
        f_score: stats.average_f_score || 1.3,
        phi: stats.phi_measure || 0.8,
        temporal_unification: stats.temporal_unification || 0.7,
        causal_integration: stats.causal_integration || 0.6,
        reentrancy: stats.reentrancy || 0.8,
        self_model_complexity: stats.self_model_complexity || 0.7
      });
      
      setError(null);
    } catch (err) {
      setError('Failed to fetch metrics');
      console.error('Metrics fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, [serverStatus.connected]);

  useEffect(() => {
    let interval: number;
    if (autoRefresh && serverStatus.connected) {
      interval = setInterval(fetchMetrics, refreshInterval);
    }
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, serverStatus.connected]);

  const getConsciousnessLevel = (fScore: number): { level: string; color: string; description: string } => {
    if (fScore >= 1.3) {
      return {
        level: 'Functional Consciousness',
        color: 'text-green-600 dark:text-green-400',
        description: 'System demonstrates functional consciousness with self-awareness'
      };
    } else if (fScore >= 0.8) {
      return {
        level: 'Emerging Awareness',
        color: 'text-yellow-600 dark:text-yellow-400',
        description: 'System shows signs of emerging conscious processes'
      };
    } else {
      return {
        level: 'Basic Processing',
        color: 'text-red-600 dark:text-red-400',
        description: 'System operating in basic information processing mode'
      };
    }
  };

  const consciousnessInfo = currentMetrics ? getConsciousnessLevel(currentMetrics.f_score) : null;

  const emotionalStateColors: Record<string, string> = {
    curious: '#3B82F6',
    contemplative: '#8B5CF6',
    analytical: '#10B981',
    confident: '#F59E0B',
    reflective: '#6366F1',
    default: '#6B7280'
  };

  const metricsComponents = currentMetrics ? [
    { name: 'Φ (Phi)', value: currentMetrics.phi, max: 1, color: '#8B5CF6' },
    { name: 'Temporal', value: currentMetrics.temporal_unification || 0, max: 1, color: '#10B981' },
    { name: 'Causal', value: currentMetrics.causal_integration || 0, max: 1, color: '#F59E0B' },
    { name: 'Reentrancy', value: currentMetrics.reentrancy || 0, max: 1, color: '#EF4444' },
    { name: 'Self-Model', value: currentMetrics.self_model_complexity || 0, max: 1, color: '#06B6D4' }
  ] : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="w-8 h-8 text-consciousness-600" />
          <div>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
              Neural Metrics Dashboard
            </h2>
            <p className="text-slate-500 dark:text-slate-400">
              Real-time consciousness monitoring and analysis
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="auto-refresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4 text-consciousness-600 rounded focus:ring-consciousness-500"
              disabled={!serverStatus.connected}
            />
            <label htmlFor="auto-refresh" className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Auto Refresh
            </label>
          </div>
          
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="px-3 py-1 text-sm border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
            disabled={!autoRefresh || !serverStatus.connected}
          >
            <option value={1000}>1s</option>
            <option value={5000}>5s</option>
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
          </select>

          <button
            onClick={fetchMetrics}
            disabled={loading}
            className="px-4 py-2 bg-consciousness-600 hover:bg-consciousness-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
            <span className="text-red-600 dark:text-red-400 font-medium">Error: {error}</span>
          </div>
        </div>
      )}

      {!serverStatus.connected && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
            <span className="text-yellow-600 dark:text-yellow-400 font-medium">
              Showing demo data - Connect to server for real-time metrics
            </span>
          </div>
        </div>
      )}

      {/* Current Consciousness Status */}
      {consciousnessInfo && currentMetrics && (
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Current Consciousness State
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-slate-600 dark:text-slate-400">F-Score</span>
                  <div className="text-right">
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">
                      {currentMetrics.f_score.toFixed(3)}
                    </span>
                    <span className={`block text-sm ${consciousnessInfo.color}`}>
                      {consciousnessInfo.level}
                    </span>
                  </div>
                </div>
                
                <div className="text-sm text-slate-500 dark:text-slate-400">
                  {consciousnessInfo.description}
                </div>
                
                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-consciousness-400 to-consciousness-600 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(currentMetrics.f_score / 1.5 * 100, 100)}%` }}
                  />
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Component Metrics
              </h3>
              
              <div className="space-y-3">
                {metricsComponents.map((metric) => (
                  <div key={metric.name} className="flex items-center justify-between">
                    <span className="text-sm text-slate-600 dark:text-slate-400">{metric.name}</span>
                    <div className="flex items-center space-x-3">
                      <div className="w-20 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${(metric.value / metric.max) * 100}%`,
                            backgroundColor: metric.color
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium text-slate-900 dark:text-white w-12 text-right">
                        {metric.value.toFixed(2)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Consciousness Level Timeline */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>Consciousness Timeline</span>
          </h3>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(time) => new Date(time).toLocaleTimeString()} 
                  stroke="#64748B"
                />
                <YAxis stroke="#64748B" />
                <Tooltip 
                  labelFormatter={(time) => new Date(time).toLocaleString()}
                  formatter={(value: number) => [value.toFixed(3), 'F-Score']}
                />
                <Area 
                  type="monotone" 
                  dataKey="consciousness_level" 
                  stroke="#8B5CF6" 
                  fill="#8B5CF6"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Processing Time */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center space-x-2">
            <Clock className="w-5 h-5" />
            <span>Processing Performance</span>
          </h3>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(time) => new Date(time).toLocaleTimeString()} 
                  stroke="#64748B"
                />
                <YAxis stroke="#64748B" />
                <Tooltip 
                  labelFormatter={(time) => new Date(time).toLocaleString()}
                  formatter={(value: number) => [`${value}ms`, 'Processing Time']}
                />
                <Line 
                  type="monotone" 
                  dataKey="processing_time" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Emotional States Distribution */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center space-x-2">
            <Brain className="w-5 h-5" />
            <span>Emotional States</span>
          </h3>
          
          <div className="h-64">
            {metricsHistory.length > 0 && (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={Object.entries(
                      metricsHistory.reduce((acc, metric) => {
                        acc[metric.emotional_state] = (acc[metric.emotional_state] || 0) + 1;
                        return acc;
                      }, {} as Record<string, number>)
                    ).map(([state, count]) => ({ name: state, value: count }))}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} (${((percent || 0) * 100).toFixed(0)}%)`}
                  >
                    {Object.keys(
                      metricsHistory.reduce((acc, metric) => {
                        acc[metric.emotional_state] = (acc[metric.emotional_state] || 0) + 1;
                        return acc;
                      }, {} as Record<string, number>)
                    ).map((state, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={emotionalStateColors[state] || emotionalStateColors.default} 
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* System Stats */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center space-x-2">
            <Activity className="w-5 h-5" />
            <span>System Statistics</span>
          </h3>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <div className="text-2xl font-bold text-consciousness-600">
                {metricsHistory.length > 0 
                  ? (metricsHistory.reduce((acc, m) => acc + m.consciousness_level, 0) / metricsHistory.length).toFixed(3)
                  : '0.000'
                }
              </div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Avg F-Score</div>
            </div>
            
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {metricsHistory.length > 0
                  ? Math.round(metricsHistory.reduce((acc, m) => acc + m.processing_time, 0) / metricsHistory.length)
                  : 0
                }ms
              </div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Avg Processing</div>
            </div>
            
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {metricsHistory.length > 0
                  ? Math.round(metricsHistory.reduce((acc, m) => acc + m.memory_items, 0) / metricsHistory.length)
                  : 0
                }
              </div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Avg Memory Items</div>
            </div>
            
            <div className="text-center p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {metricsHistory.length > 0
                  ? Math.round((metricsHistory.filter(m => m.phase_7_used).length / metricsHistory.length) * 100)
                  : 0
                }%
              </div>
              <div className="text-sm text-slate-500 dark:text-slate-400">Phase 7 Usage</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConsciousnessMetrics;