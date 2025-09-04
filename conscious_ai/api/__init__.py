"""
Consciousness API
================
REST API endpoints for external consumption of the consciousness pipeline.
"""

from .consciousness_endpoint import app, ConsciousnessRequest, ConsciousnessResponse, ModelStatusResponse, HealthResponse

__all__ = [
    'app',
    'ConsciousnessRequest',
    'ConsciousnessResponse', 
    'ModelStatusResponse',
    'HealthResponse'
]