// Xentauri SC-1 Consciousness Interface Types

export interface ConsciousnessRequest {
  user_input: string;
  final_model: 'gpt-4' | 'gpt-4o-mini' | 'gpt-3.5-turbo' | 'gpt-5' | 'gpt-5-mini' | 'gpt-5-nano';
  include_trace?: boolean;
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

export interface ConsciousnessTrace {
  phase_1_perception?: any;
  phase_2_cognitive_context?: ConsciousnessState;
  phase_3_coherent_generation?: any;
  phase_4_llm_communication?: any;
  phase_5_internal_critique?: any;
  phase_6_memory_consolidation?: any;
  phase_7_expressive_execution?: any;
  processing_time_ms: number;
  narrative_text?: string;
  transparency_narrative?: string;
}

export interface ConsciousnessResponse {
  response: string;
  confidence: number;
  emotional_state: string;
  consciousness_state: ConsciousnessState;
  consciousness_trace?: ConsciousnessTrace;
  processing_time_ms: number;
  f_score?: number;
  model_used?: string;
  narrative?: string;
}

export interface OpenAIRequest {
  consciousness_state: ConsciousnessState;
  user_input: string;
  model: 'gpt-4' | 'gpt-4o-mini' | 'gpt-3.5-turbo' | 'gpt-5' | 'gpt-5-mini' | 'gpt-5-nano';
}

export interface OpenAIResponse {
  response: string;
  model_used: string;
  tokens_used: number;
  processing_time_ms: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'xentauri';
  content: string;
  timestamp: Date;
  stardate?: string;
  consciousness_level?: number;
  emotional_state?: string;
  confidence?: number;
  trace?: ConsciousnessTrace;
  processing_time?: number;
  narrative?: string;
}

export interface XentauriSession {
  id: string;
  messages: ChatMessage[];
  created_at: Date;
  last_activity: Date;
  total_transmissions: number;
}

// UI State Types
export type ProcessingState = 
  | 'idle' 
  | 'establishing_connection'
  | 'perceiving'
  | 'processing'
  | 'thinking'
  | 'responding'
  | 'error';

export interface SystemStatus {
  state: ProcessingState;
  progress: number; // 0-100
  message: string;
  connection_quality: 'excellent' | 'good' | 'poor' | 'disconnected';
}

export interface EmotionalState {
  primary: string;
  secondary?: string;
  intensity: number; // 0-1
  confidence: number; // 0-1
}

export interface XentauriConfig {
  consciousness_api_url: string;
  openai_api_key: string;
  selected_model: 'gpt-4' | 'gpt-4o-mini' | 'gpt-3.5-turbo' | 'gpt-5' | 'gpt-5-mini' | 'gpt-5-nano';
  show_trace: boolean;
  auto_scroll: boolean;
  reduced_motion: boolean;
  connection_timeout: number;
}

// Animation and Effect Types
export interface StarFieldConfig {
  star_count: number;
  speed: number;
  size_range: [number, number];
  opacity_range: [number, number];
  parallax_enabled: boolean;
}

export interface GlowEffect {
  color: string;
  intensity: number;
  radius: number;
  pulse_speed?: number;
}

// Error Types
export interface XentauriError {
  type: 'connection' | 'consciousness' | 'openai' | 'timeout' | 'unknown';
  message: string;
  code?: string;
  retry_after?: number;
  details?: any;
}

// Event Types for real-time updates
export interface ConsciousnessEvent {
  type: 'phase_start' | 'phase_complete' | 'error' | 'progress';
  phase?: string;
  data?: any;
  timestamp: number;
}

export interface MessageMetadata {
  tokens_used?: number;
  model_used?: string;
  consciousness_layers?: number;
  response_quality?: number;
  user_satisfaction?: number;
}

// Stardate calculation helper type
export interface StardateInfo {
  stardate: string;
  earth_equivalent: Date;
  sector: string;
  quadrant: string;
}