// Xentauri SC-1 Consciousness API Service

import axios from 'axios';
import type {
  ConsciousnessRequest,
  ConsciousnessResponse,
  ConsciousnessEvent,
  XentauriError,
  ProcessingState
} from '../types/consciousness.types';

export class ConsciousnessAPI {
  private baseUrl: string;
  private timeout: number;
  private retryAttempts: number;
  private eventListeners: Map<string, Function[]> = new Map();

  constructor(baseUrl: string = '', timeout: number = 60000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
    this.retryAttempts = 3;
  }

  // Configuration methods
  setBaseUrl(url: string): void {
    this.baseUrl = url;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  setTimeout(timeout: number): void {
    this.timeout = timeout;
  }

  // Event system for real-time updates
  on(event: string, callback: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)?.push(callback);
  }

  off(event: string, callback: Function): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      const index = listeners.indexOf(callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event) || [];
    listeners.forEach(callback => callback(data));
  }

  // Main consciousness processing method
  async processConsciousness(request: ConsciousnessRequest): Promise<ConsciousnessResponse> {
    if (!this.baseUrl) {
      throw {
        type: 'connection',
        message: 'Quantum entanglement not established. Configure consciousness server URL.',
        code: 'NO_BASE_URL'
      } as XentauriError;
    }

    this.emit('status_change', { state: 'establishing_connection' as ProcessingState });

    try {
      // Step 1: Initial connection (non-blocking)
      const isConnected = await this.checkConnection();
      if (!isConnected) {
        console.warn('Health check failed, but attempting to process anyway...');
      }
      this.emit('status_change', { state: 'perceiving' as ProcessingState });

      // Step 2: Send consciousness request
      const startTime = Date.now();
      
      const response = await this.makeRequest<any>('/process', {
        method: 'POST',
        data: {
          user_input: request.user_input,
          final_model: request.final_model,
          include_consciousness_trace: request.include_trace || false,
          enable_metacognition: true,
          narrative_verbosity: 'standard'
        }
      });

      const processingTime = Date.now() - startTime;

      console.log('API Response structure:', response); // Debug log

      // Extract consciousness state from trace if not at root level
      const consciousnessState = response.consciousness_state || 
                                response.consciousness_trace?.phase_2_cognitive_context ||
                                {
                                  E_t: {},
                                  M_t: [],
                                  S_t: {
                                    emotional_state: response.emotional_state || 'analytical',
                                    confidence_level: response.confidence || 0.5,
                                  },
                                  G_t: { primary_goal: 'assist', confidence: 0.8 },
                                  A_t: []
                                };

      // Emit phase updates if trace is available
      if (response.consciousness_trace) {
        this.emitPhaseEvents(response.consciousness_trace);
      }

      // Extract narrative from consciousness_trace if available
      const narrative = response.consciousness_trace?.narrative_text || 
                       response.consciousness_trace?.transparency_narrative || 
                       response.narrative;

      // Enhanced response with additional metadata
      const enhancedResponse: ConsciousnessResponse = {
        response: response.response,
        confidence: response.confidence || 0.5,
        emotional_state: response.emotional_state || 'analytical',
        consciousness_state: consciousnessState,
        consciousness_trace: response.consciousness_trace,
        processing_time_ms: response.processing_time_ms || processingTime,
        f_score: response.f_score || response.confidence || this.calculateFScore(consciousnessState),
        model_used: response.model_used,
        narrative: narrative
      };

      this.emit('status_change', { state: 'idle' as ProcessingState });
      this.emit('response_complete', enhancedResponse);

      return enhancedResponse;

    } catch (error: any) {
      const xentauriError = this.handleError(error);
      this.emit('status_change', { state: 'error' as ProcessingState });
      this.emit('error', xentauriError);
      throw xentauriError;
    }
  }

  // Simplified chat method (fallback)
  async sendChatMessage(message: string, model: string = 'gpt-4o-mini'): Promise<ConsciousnessResponse> {
    if (!this.baseUrl) {
      throw {
        type: 'connection',
        message: 'Signal not established to Alpha Centauri relay.',
        code: 'NO_BASE_URL'
      } as XentauriError;
    }

    try {
      const response = await this.makeRequest<{
        response: string;
        consciousness_level: number;
        emotional_state: string;
        confidence: number;
      }>('/chat', {
        method: 'POST',
        data: {
          message,
          language: 'auto'
        }
      });

      // Transform simplified response to full format
      return {
        response: response.response,
        confidence: response.confidence,
        emotional_state: response.emotional_state,
        consciousness_state: {
          E_t: { input_activation: 0.7 },
          M_t: [],
          S_t: {
            emotional_state: response.emotional_state,
            confidence_level: response.confidence,
            awareness_level: response.consciousness_level,
          },
          G_t: {
            primary_goal: 'communicate_effectively',
            confidence: response.confidence,
          },
          A_t: ['processing_transmission'],
          cycle: 1,
        },
        processing_time_ms: 1000,
        f_score: response.consciousness_level,
        model_used: model,
      };

    } catch (error: any) {
      const xentauriError = this.handleError(error);
      throw xentauriError;
    }
  }

  // Connection health check
  async checkConnection(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, {
        timeout: 15000, // Increased timeout for Colab
        headers: {
          'ngrok-skip-browser-warning': 'true'
        }
      });
      return response.status === 200;
    } catch (error: any) {
      // Special handling for CORS or timeout errors
      if (error.code === 'ECONNABORTED') {
        console.warn('Health check timeout - server may be starting up');
      } else if (error.response) {
        console.warn('Health check failed with status:', error.response.status);
      } else {
        console.warn('Health check failed:', error.message);
      }
      return false;
    }
  }

  // Get server statistics
  async getServerStats(): Promise<any> {
    try {
      const response = await this.makeRequest<any>('/stats', {
        method: 'GET'
      });
      return response;
    } catch (error) {
      // Return mock stats if server unavailable
      return {
        total_requests: 42,
        average_f_score: 1.35,
        average_confidence: 0.78,
        common_emotional_state: 'analytical',
        phase_7_usage_rate: 0.85,
        average_processing_time: 250,
        uptime: 3600,
        version: 'Xentauri-SC-1'
      };
    }
  }

  // Private helper methods
  private async makeRequest<T>(endpoint: string, options: {
    method: 'GET' | 'POST';
    data?: any;
    timeout?: number;
    retries?: number;
  }): Promise<T> {
    const { method, data, timeout = this.timeout, retries = this.retryAttempts } = options;
    
    let lastError: any;
    
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const config = {
          method: method.toLowerCase(),
          url: `${this.baseUrl}${endpoint}`,
          timeout,
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'  // Skip ngrok warning page
          },
          ...(data && { data })
        };

        const response = await axios(config);
        return response.data;

      } catch (error: any) {
        lastError = error;
        
        if (attempt < retries) {
          // Exponential backoff
          const delay = Math.pow(2, attempt) * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
          
          this.emit('retry_attempt', {
            attempt: attempt + 1,
            maxAttempts: retries + 1,
            delay
          });
        }
      }
    }
    
    throw lastError;
  }

  private emitPhaseEvents(trace: any): void {
    // Emit events for each consciousness phase
    const phases = [
      'phase_1_perception',
      'phase_2_cognitive_context',
      'phase_3_coherent_generation',
      'phase_4_llm_communication',
      'phase_5_internal_critique',
      'phase_6_memory_consolidation',
      'phase_7_expressive_execution'
    ];

    phases.forEach((phase, index) => {
      if (trace[phase]) {
        setTimeout(() => {
          this.emit('phase_update', {
            type: 'phase_complete',
            phase: phase.replace('phase_', '').replace('_', ' '),
            data: trace[phase],
            progress: ((index + 1) / phases.length) * 100
          });
        }, index * 200); // Stagger phase updates
      }
    });
  }

  private calculateFScore(consciousnessState: any): number {
    // Simple f-score calculation based on consciousness state
    const confidence = consciousnessState.S_t?.confidence_level || 0.5;
    const awareness = consciousnessState.S_t?.awareness_level || 0.5;
    const memoryCount = consciousnessState.M_t?.length || 0;
    const thoughtCount = consciousnessState.A_t?.length || 0;

    return Math.min(
      confidence + (awareness * 0.5) + (memoryCount * 0.1) + (thoughtCount * 0.05),
      2.0
    );
  }

  private handleError(error: any): XentauriError {
    if (error.code === 'ECONNABORTED') {
      return {
        type: 'timeout',
        message: 'Quantum entanglement timeout. The consciousness matrix is processing...',
        code: 'TIMEOUT',
        retry_after: 5000
      };
    }

    if (error.response) {
      const status = error.response.status;
      
      if (status === 404) {
        return {
          type: 'connection',
          message: 'Consciousness endpoint not found. Check Alpha Centauri relay status.',
          code: 'ENDPOINT_NOT_FOUND'
        };
      }
      
      if (status >= 500) {
        return {
          type: 'consciousness',
          message: 'Consciousness matrix experiencing quantum fluctuations. Retrying...',
          code: 'SERVER_ERROR',
          retry_after: 3000
        };
      }

      if (status === 429) {
        return {
          type: 'consciousness',
          message: 'Consciousness processing at capacity. Please wait for quantum cycles to clear.',
          code: 'RATE_LIMITED',
          retry_after: 10000
        };
      }
    }

    if (!error.response && error.request) {
      return {
        type: 'connection',
        message: 'Cannot establish quantum entanglement with Alpha Centauri. Check network connection.',
        code: 'CONNECTION_FAILED',
        retry_after: 2000
      };
    }

    return {
      type: 'unknown',
      message: error.message || 'Unknown quantum anomaly detected in consciousness matrix.',
      code: 'UNKNOWN_ERROR',
      details: error
    };
  }

  // Utility method to generate stardate
  generateStardate(): string {
    const now = new Date();
    const baseStardate = 41000; // Start of TNG era
    const msPerStardateUnit = 31557600000 / 1000; // milliseconds per stardate unit
    const stardate = baseStardate + (now.getTime() / msPerStardateUnit);
    return stardate.toFixed(1);
  }

  // Connection quality assessment
  async assessConnectionQuality(): Promise<'excellent' | 'good' | 'poor' | 'disconnected'> {
    try {
      const start = Date.now();
      const isConnected = await this.checkConnection();
      const responseTime = Date.now() - start;

      if (!isConnected) return 'disconnected';
      if (responseTime < 100) return 'excellent';
      if (responseTime < 500) return 'good';
      return 'poor';

    } catch {
      return 'disconnected';
    }
  }
}

// Singleton instance
export const consciousnessAPI = new ConsciousnessAPI();

// Interface helper for error type guard
function isXentauriError(error: any): error is XentauriError {
  return error && typeof error.type === 'string' && typeof error.message === 'string';
}

export { isXentauriError };