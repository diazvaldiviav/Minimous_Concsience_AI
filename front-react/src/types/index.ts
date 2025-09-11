// Core consciousness AI types
export interface ServerStatus {
  connected: boolean;
  url: string;
  lastPing?: number;
}

export interface ConsciousnessState {
  E_t: Record<string, number>; // Environmental perception
  M_t: MemoryItem[];            // Active memory
  S_t: {                        // Emotional/confidence state
    emotional_state: string;
    confidence_level: number;
    awareness_level?: number;
    focus?: string;
    coherence?: number;
  };
  G_t: {                        // Goal structure
    primary_goal: string;
    secondary?: string;
    confidence: number;
  };
  A_t: string[];               // Automatic thoughts
  cycle?: number;
  timestamp?: string;
}

export interface MemoryItem {
  type: string;
  content: string | { text: string };
  relevance: number;
  timestamp?: string;
}

export interface ConsciousnessMetrics {
  f_score: number;
  phi: number;
  temporal_unification?: number;
  causal_integration?: number;
  reentrancy?: number;
  self_model_complexity?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  consciousnessLevel?: number;
  emotionalState?: string;
  confidence?: number;
  processingTime?: number;
  reasoning?: string[];
  narrative?: string;
}

export interface ConsciousnessResponse {
  response: string;
  consciousness_metrics: ConsciousnessMetrics;
  consciousness_state: ConsciousnessState;
  processing_time_ms: number;
  narrative?: string;
  reasoning_steps?: string[];
  phase_7_enhanced?: boolean;
  model_used?: string;
}

export interface ChatRequest {
  message: string;
  language?: 'auto' | 'en' | 'es';
  include_narrative?: boolean;
  include_memory?: boolean;
  model_override?: string;
}

export interface ServerConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  apiKey?: string;
  defaultLanguage: 'auto' | 'en' | 'es';
}

export interface MetricsData {
  timestamp: Date;
  consciousness_level: number;
  emotional_state: string;
  confidence: number;
  memory_items: number;
  processing_time: number;
  phase_7_used: boolean;
}