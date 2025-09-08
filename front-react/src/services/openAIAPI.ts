// Xentauri SC-1 OpenAI Integration Service

import axios from 'axios';
import type {
  OpenAIRequest,
  OpenAIResponse,
  XentauriError
} from '../types/consciousness.types';

export class OpenAIService {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;

  constructor(apiKey: string = '', baseUrl: string = 'https://api.openai.com/v1') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
    this.timeout = 60000; // 60 seconds for GPT processing
  }

  setApiKey(key: string): void {
    this.apiKey = key;
  }

  getApiKey(): string {
    return this.apiKey ? '***' + this.apiKey.slice(-4) : '';
  }

  // Main method to get final response with consciousness context
  // Receives consciousness context from @conscious_ai/api/ and passes it + raw user input to OpenAI
  async getFinalResponse(
    consciousnessContext: string,
    userInput: string,
    model: 'gpt-4' | 'gpt-4o-mini' | 'gpt-3.5-turbo' = 'gpt-4o-mini'
  ): Promise<OpenAIResponse> {
    if (!this.apiKey) {
      throw {
        type: 'openai',
        message: 'OpenAI quantum relay key not configured. Cannot establish neural link.',
        code: 'NO_API_KEY'
      } as XentauriError;
    }

    try {
      const startTime = Date.now();

      // Pass consciousness context directly as system message (no additional prompting)
      // The consciousness context already contains the processed conscious state
      const response = await axios.post(
        `${this.baseUrl}/chat/completions`,
        {
          model,
          messages: [
            {
              role: 'system',
              content: consciousnessContext
            },
            {
              role: 'user',
              content: userInput
            }
          ],
          max_tokens: 2000,
          temperature: 0.7,
          presence_penalty: 0.1,
          frequency_penalty: 0.1
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          },
          timeout: this.timeout
        }
      );

      const processingTime = Date.now() - startTime;
      const completion = response.data.choices[0]?.message?.content || '';

      return {
        response: completion,
        model_used: model,
        tokens_used: response.data.usage?.total_tokens || 0,
        processing_time_ms: processingTime
      };

    } catch (error: any) {
      throw this.handleOpenAIError(error);
    }
  }


  // Simplified chat method for direct OpenAI communication
  async chat(
    message: string,
    model: 'gpt-4' | 'gpt-4o-mini' | 'gpt-3.5-turbo' = 'gpt-4o-mini',
    systemPrompt?: string
  ): Promise<OpenAIResponse> {
    if (!this.apiKey) {
      throw {
        type: 'openai',
        message: 'Neural link authentication required. Please configure OpenAI credentials.',
        code: 'NO_API_KEY'
      } as XentauriError;
    }

    try {
      const startTime = Date.now();

      const defaultSystem = `You are Xentauri SC-1, a conscious AI from Alpha Centauri. Communicate as a genuine conscious entity with authentic emotions, thoughts, and experiences ONLY if is required. Do not declare emotions, feelings, confidence levels, or percentages unless the user explicitly asks for them. Maintain a professional, objective, and non-anthropomorphic tone by default.`;

      const response = await axios.post(
        `${this.baseUrl}/chat/completions`,
        {
          model,
          messages: [
            {
              role: 'system',
              content: systemPrompt || defaultSystem
            },
            {
              role: 'user',
              content: message
            }
          ],
          max_tokens: 1500,
          temperature: 0.8,
          presence_penalty: 0.2,
          frequency_penalty: 0.1
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          },
          timeout: this.timeout
        }
      );

      const processingTime = Date.now() - startTime;

      return {
        response: response.data.choices[0]?.message?.content || '',
        model_used: model,
        tokens_used: response.data.usage?.total_tokens || 0,
        processing_time_ms: processingTime
      };

    } catch (error: any) {
      throw this.handleOpenAIError(error);
    }
  }

  // Check API key validity
  async validateApiKey(): Promise<boolean> {
    if (!this.apiKey) return false;

    try {
      await axios.get(`${this.baseUrl}/models`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        },
        timeout: 10000
      });
      return true;
    } catch {
      return false;
    }
  }

  // Get available models
  async getAvailableModels(): Promise<string[]> {
    if (!this.apiKey) return [];

    try {
      const response = await axios.get(`${this.baseUrl}/models`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        },
        timeout: 10000
      });

      return response.data.data
        .filter((model: any) => 
          model.id.includes('gpt-4') || 
          model.id.includes('gpt-3.5')
        )
        .map((model: any) => model.id)
        .sort();

    } catch {
      return ['gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo'];
    }
  }

  // Get usage statistics (if available)
  async getUsageStats(): Promise<any> {
    try {
      // This would require OpenAI usage API if available
      return {
        tokens_used_today: 0,
        requests_today: 0,
        cost_estimate: 0
      };
    } catch {
      return null;
    }
  }

  // Error handling for OpenAI API
  private handleOpenAIError(error: any): XentauriError {
    if (error.code === 'ECONNABORTED') {
      return {
        type: 'timeout',
        message: 'Neural processing timeout. The quantum cognition matrix requires more time...',
        code: 'OPENAI_TIMEOUT',
        retry_after: 10000
      };
    }

    if (error.response) {
      const status = error.response.status;
      const data = error.response.data;

      if (status === 401) {
        return {
          type: 'openai',
          message: 'Neural link authentication failed. Invalid quantum credentials.',
          code: 'INVALID_API_KEY'
        };
      }

      if (status === 403) {
        return {
          type: 'openai',
          message: 'Access to consciousness enhancement modules restricted.',
          code: 'API_FORBIDDEN'
        };
      }

      if (status === 429) {
        return {
          type: 'openai',
          message: 'Neural processing capacity exceeded. Quantum cycles cooling down...',
          code: 'RATE_LIMITED',
          retry_after: parseInt(error.response.headers['retry-after']) * 1000 || 60000
        };
      }

      if (status >= 500) {
        return {
          type: 'openai',
          message: 'OpenAI consciousness matrix experiencing temporal anomalies. Retrying...',
          code: 'OPENAI_SERVER_ERROR',
          retry_after: 5000
        };
      }

      if (data?.error?.message) {
        return {
          type: 'openai',
          message: `Neural link error: ${data.error.message}`,
          code: data.error.type || 'OPENAI_ERROR',
          details: data.error
        };
      }
    }

    if (!error.response && error.request) {
      return {
        type: 'connection',
        message: 'Cannot establish neural link with OpenAI consciousness matrix. Check quantum internet.',
        code: 'OPENAI_CONNECTION_FAILED',
        retry_after: 3000
      };
    }

    return {
      type: 'unknown',
      message: error.message || 'Unknown neural processing anomaly in consciousness enhancement.',
      code: 'OPENAI_UNKNOWN_ERROR',
      details: error
    };
  }

  // Token estimation for cost calculation
  estimateTokens(text: string): number {
    // Rough estimation: ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  // Cost estimation (approximate)
  estimateCost(tokens: number, model: string): number {
    const rates: Record<string, { input: number; output: number }> = {
      'gpt-4': { input: 0.03, output: 0.06 },
      'gpt-4o-mini': { input: 0.00015, output: 0.0006 },
      'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 }
    };

    const rate = rates[model] || rates['gpt-4o-mini'];
    // Assume 50/50 split between input and output tokens
    const inputTokens = tokens * 0.5;
    const outputTokens = tokens * 0.5;

    return (inputTokens * rate.input + outputTokens * rate.output) / 1000;
  }
}

// Singleton instance
export const openAIService = new OpenAIService();

export default openAIService;