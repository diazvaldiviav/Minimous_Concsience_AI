// Xentauri SC-1 State Management Context

import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react';
import type { 
  XentauriConfig, 
  ProcessingState, 
  ConsciousnessState,
  ChatMessage,
  XentauriError
} from '../types/consciousness.types';
import { consciousnessAPI } from '../services/consciousnessAPI';

// State interface
interface XentauriState {
  // Configuration
  config: XentauriConfig;
  
  // Connection status
  connectionQuality: 'excellent' | 'good' | 'poor' | 'disconnected';
  lastPing: number;
  uptime: number;
  
  // Processing state
  processingState: ProcessingState;
  isProcessing: boolean;
  
  // Consciousness data
  consciousnessState?: ConsciousnessState;
  consciousnessLevel: number;
  emotionalState: string;
  confidence: number;
  
  // Chat data
  messages: ChatMessage[];
  currentTrace?: any;
  
  // UI state
  showTrace: boolean;
  showStatusBar: boolean;
  showEffects: boolean;
  
  // Errors
  error?: XentauriError;
  
  // Statistics
  sessionStats: {
    totalMessages: number;
    averageResponseTime: number;
    averageConsciousnessLevel: number;
    sessionStartTime: number;
  };
}

// Action types
type XentauriAction =
  | { type: 'SET_CONFIG'; payload: Partial<XentauriConfig> }
  | { type: 'SET_CONNECTION_QUALITY'; payload: XentauriState['connectionQuality'] }
  | { type: 'SET_PROCESSING_STATE'; payload: ProcessingState }
  | { type: 'SET_CONSCIOUSNESS_STATE'; payload: ConsciousnessState }
  | { type: 'ADD_MESSAGE'; payload: ChatMessage }
  | { type: 'CLEAR_MESSAGES' }
  | { type: 'SET_TRACE'; payload: any }
  | { type: 'SET_UI_STATE'; payload: Partial<Pick<XentauriState, 'showTrace' | 'showStatusBar' | 'showEffects'>> }
  | { type: 'SET_ERROR'; payload: XentauriError | undefined }
  | { type: 'UPDATE_STATS'; payload: { responseTime?: number; consciousnessLevel?: number } }
  | { type: 'RESET_SESSION' }
  | { type: 'TICK' };

// Initial state
const initialState: XentauriState = {
  config: {
    consciousness_api_url: '',
    openai_api_key: '',
    selected_model: 'gpt-4o-mini',
    show_trace: false,
    connection_timeout: 30000,
    auto_scroll: true,
    reduced_motion: false,
    theme: 'space'
  },
  connectionQuality: 'disconnected',
  lastPing: 0,
  uptime: 0,
  processingState: 'idle',
  isProcessing: false,
  consciousnessLevel: 0,
  emotionalState: 'neutral',
  confidence: 0,
  messages: [],
  showTrace: false,
  showStatusBar: true,
  showEffects: true,
  sessionStats: {
    totalMessages: 0,
    averageResponseTime: 0,
    averageConsciousnessLevel: 0,
    sessionStartTime: Date.now()
  }
};

// Reducer
const xentauriReducer = (state: XentauriState, action: XentauriAction): XentauriState => {
  switch (action.type) {
    case 'SET_CONFIG':
      return {
        ...state,
        config: { ...state.config, ...action.payload }
      };

    case 'SET_CONNECTION_QUALITY':
      return {
        ...state,
        connectionQuality: action.payload,
        lastPing: action.payload !== 'disconnected' ? Date.now() : state.lastPing
      };

    case 'SET_PROCESSING_STATE':
      return {
        ...state,
        processingState: action.payload,
        isProcessing: action.payload !== 'idle' && action.payload !== 'error'
      };

    case 'SET_CONSCIOUSNESS_STATE':
      return {
        ...state,
        consciousnessState: action.payload,
        consciousnessLevel: action.payload.S_t?.awareness_level || 0,
        emotionalState: action.payload.S_t?.emotional_state || 'neutral',
        confidence: action.payload.S_t?.confidence_level || 0
      };

    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.payload],
        sessionStats: {
          ...state.sessionStats,
          totalMessages: state.sessionStats.totalMessages + (action.payload.role === 'user' ? 0 : 1)
        }
      };

    case 'CLEAR_MESSAGES':
      return {
        ...state,
        messages: [],
        currentTrace: undefined,
        error: undefined
      };

    case 'SET_TRACE':
      return {
        ...state,
        currentTrace: action.payload
      };

    case 'SET_UI_STATE':
      return {
        ...state,
        ...action.payload
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        processingState: action.payload ? 'error' : 'idle',
        isProcessing: false
      };

    case 'UPDATE_STATS':
      const { responseTime, consciousnessLevel } = action.payload;
      const stats = state.sessionStats;
      
      return {
        ...state,
        sessionStats: {
          ...stats,
          averageResponseTime: responseTime 
            ? (stats.averageResponseTime * stats.totalMessages + responseTime) / (stats.totalMessages + 1)
            : stats.averageResponseTime,
          averageConsciousnessLevel: consciousnessLevel !== undefined
            ? (stats.averageConsciousnessLevel * stats.totalMessages + consciousnessLevel) / (stats.totalMessages + 1)
            : stats.averageConsciousnessLevel
        }
      };

    case 'RESET_SESSION':
      return {
        ...state,
        messages: [],
        currentTrace: undefined,
        error: undefined,
        sessionStats: {
          totalMessages: 0,
          averageResponseTime: 0,
          averageConsciousnessLevel: 0,
          sessionStartTime: Date.now()
        }
      };

    case 'TICK':
      return {
        ...state,
        uptime: Math.floor((Date.now() - state.sessionStats.sessionStartTime) / 1000)
      };

    default:
      return state;
  }
};

// Context
const XentauriContext = createContext<{
  state: XentauriState;
  dispatch: React.Dispatch<XentauriAction>;
} | undefined>(undefined);

// Provider component
interface XentauriProviderProps {
  children: React.ReactNode;
  initialConfig?: Partial<XentauriConfig>;
}

export const XentauriProvider: React.FC<XentauriProviderProps> = ({ 
  children, 
  initialConfig = {} 
}) => {
  const [state, dispatch] = useReducer(xentauriReducer, {
    ...initialState,
    config: { ...initialState.config, ...initialConfig }
  });

  // Setup API event listeners
  useEffect(() => {
    consciousnessAPI.on('status_change', (data: { state: ProcessingState }) => {
      dispatch({ type: 'SET_PROCESSING_STATE', payload: data.state });
    });

    consciousnessAPI.on('error', (error: XentauriError) => {
      dispatch({ type: 'SET_ERROR', payload: error });
    });

    consciousnessAPI.on('response_complete', (response: any) => {
      if (response.consciousness_state) {
        dispatch({ type: 'SET_CONSCIOUSNESS_STATE', payload: response.consciousness_state });
      }
      if (response.processing_time_ms) {
        dispatch({ type: 'UPDATE_STATS', payload: { responseTime: response.processing_time_ms } });
      }
      if (response.f_score) {
        dispatch({ type: 'UPDATE_STATS', payload: { consciousnessLevel: response.f_score } });
      }
    });

    return () => {
      // Cleanup would go here if consciousness API supported removeListener
    };
  }, []);

  // Configure API when config changes
  useEffect(() => {
    consciousnessAPI.setBaseUrl(state.config.consciousness_api_url);
    consciousnessAPI.setTimeout(state.config.connection_timeout);
  }, [state.config.consciousness_api_url, state.config.connection_timeout]);

  // Connection quality monitoring
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const quality = await consciousnessAPI.assessConnectionQuality();
        dispatch({ type: 'SET_CONNECTION_QUALITY', payload: quality });
      } catch {
        dispatch({ type: 'SET_CONNECTION_QUALITY', payload: 'disconnected' });
      }
    };

    // Check immediately and then every 30 seconds
    checkConnection();
    const interval = setInterval(checkConnection, 30000);

    return () => clearInterval(interval);
  }, [state.config.consciousness_api_url]);

  // Uptime ticker
  useEffect(() => {
    const interval = setInterval(() => {
      dispatch({ type: 'TICK' });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <XentauriContext.Provider value={{ state, dispatch }}>
      {children}
    </XentauriContext.Provider>
  );
};

// Custom hook for using the context
export const useXentauri = () => {
  const context = useContext(XentauriContext);
  if (context === undefined) {
    throw new Error('useXentauri must be used within a XentauriProvider');
  }
  return context;
};

// Selector hooks for specific state slices
export const useXentauriConfig = () => {
  const { state } = useXentauri();
  return state.config;
};

export const useXentauriConnection = () => {
  const { state } = useXentauri();
  return {
    quality: state.connectionQuality,
    lastPing: state.lastPing,
    uptime: state.uptime
  };
};

export const useXentauriProcessing = () => {
  const { state } = useXentauri();
  return {
    state: state.processingState,
    isProcessing: state.isProcessing
  };
};

export const useXentauriConsciousness = () => {
  const { state } = useXentauri();
  return {
    state: state.consciousnessState,
    level: state.consciousnessLevel,
    emotion: state.emotionalState,
    confidence: state.confidence
  };
};

export const useXentauriMessages = () => {
  const { state } = useXentauri();
  return {
    messages: state.messages,
    trace: state.currentTrace
  };
};

export const useXentauriStats = () => {
  const { state } = useXentauri();
  return state.sessionStats;
};

// Action creators for common operations
export const useXentauriActions = () => {
  const { dispatch } = useXentauri();

  return {
    updateConfig: useCallback((config: Partial<XentauriConfig>) => {
      dispatch({ type: 'SET_CONFIG', payload: config });
    }, [dispatch]),

    addMessage: useCallback((message: ChatMessage) => {
      dispatch({ type: 'ADD_MESSAGE', payload: message });
    }, [dispatch]),

    clearMessages: useCallback(() => {
      dispatch({ type: 'CLEAR_MESSAGES' });
    }, [dispatch]),

    setTrace: useCallback((trace: any) => {
      dispatch({ type: 'SET_TRACE', payload: trace });
    }, [dispatch]),

    toggleTrace: useCallback(() => {
      dispatch({ type: 'SET_UI_STATE', payload: { showTrace: true } });
    }, [dispatch]),

    setError: useCallback((error: XentauriError | undefined) => {
      dispatch({ type: 'SET_ERROR', payload: error });
    }, [dispatch]),

    resetSession: useCallback(() => {
      dispatch({ type: 'RESET_SESSION' });
    }, [dispatch])
  };
};

export default XentauriContext;