"""
Módulo de utilidades para integración con Supabase.

Este módulo proporciona funciones y clases para interactuar con la base de datos
Supabase, incluyendo operaciones CRUD, consultas a vistas y ejecución de funciones.

Autor: Equipo de Desarrollo DermatologIA
Fecha: Octubre 2025
Versión: 1.0.0
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from django.conf import settings
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Cliente singleton para interactuar con Supabase.
    
    Proporciona métodos para realizar operaciones en la base de datos
    de manera consistente y con manejo de errores.
    """
    
    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        """Implementa patrón Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa el cliente de Supabase."""
        if self._client is None:
            try:
                self._client = create_client(
                    settings.SUPABASE_URL,
                    settings.SUPABASE_ANON_KEY
                )
                logger.info("✅ Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase client: {e}")
                raise
    
    @property
    def client(self) -> Client:
        """Retorna la instancia del cliente de Supabase."""
        if self._client is None:
            self.__init__()
        return self._client
    
    # ==================== SKIN IMAGE PREDICTIONS ====================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Crea una nueva predicción en la base de datos.
        
        Args:
            prediction_data: Diccionario con los datos de la predicción
                - image_path: str (requerido)
                - image_url: str (opcional)
                - predicted_class: str (opcional)
                - confidence_score: float (opcional)
                - probabilities: dict (opcional)
                - processing_time: float (opcional)
                - session_id: str (opcional)
                - image_size: str (opcional)
        
        Returns:
            Dict con los datos insertados o None si hay error
        """
        try:
            response = self.client.table('skin_image_prediction').insert(prediction_data).execute()
            logger.info(f"✅ Prediction created: ID {response.data[0]['id']}")
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"❌ Error creating prediction: {e}")
            return None
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """
        Obtiene una predicción por su ID.
        
        Args:
            prediction_id: ID de la predicción
        
        Returns:
            Dict con los datos de la predicción o None
        """
        try:
            response = self.client.table('skin_image_prediction').select('*').eq('id', prediction_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"❌ Error fetching prediction {prediction_id}: {e}")
            return None
    
    def update_prediction(self, prediction_id: int, update_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Actualiza una predicción existente.
        
        Args:
            prediction_id: ID de la predicción a actualizar
            update_data: Diccionario con los campos a actualizar
        
        Returns:
            Dict con los datos actualizados o None
        """
        try:
            response = self.client.table('skin_image_prediction').update(update_data).eq('id', prediction_id).execute()
            logger.info(f"✅ Prediction {prediction_id} updated")
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"❌ Error updating prediction {prediction_id}: {e}")
            return None
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Obtiene las predicciones más recientes usando la vista.
        
        Args:
            limit: Número máximo de predicciones a retornar
        
        Returns:
            Lista de diccionarios con las predicciones
        """
        try:
            response = self.client.table('v_recent_predictions').select('*').limit(limit).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching recent predictions: {e}")
            return []
    
    def get_high_risk_predictions(self, limit: int = 20) -> List[Dict]:
        """
        Obtiene predicciones de alto riesgo.
        
        Args:
            limit: Número máximo de predicciones a retornar
        
        Returns:
            Lista de predicciones de alto riesgo
        """
        try:
            response = self.client.table('v_high_risk_predictions').select('*').limit(limit).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching high risk predictions: {e}")
            return []
    
    def get_predictions_by_session(self, session_id: str, limit: int = 50) -> List[Dict]:
        """
        Obtiene todas las predicciones de una sesión.
        
        Args:
            session_id: UUID de la sesión
            limit: Número máximo de predicciones
        
        Returns:
            Lista de predicciones de la sesión
        """
        try:
            response = (self.client.table('skin_image_prediction')
                       .select('*')
                       .eq('session_id', session_id)
                       .order('uploaded_at', desc=True)
                       .limit(limit)
                       .execute())
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching predictions for session {session_id}: {e}")
            return []
    
    # ==================== DISEASE INFORMATION ====================
    
    def get_all_diseases(self) -> List[Dict]:
        """
        Obtiene información de todas las enfermedades.
        
        Returns:
            Lista de diccionarios con información de enfermedades
        """
        try:
            response = self.client.table('disease_information').select('*').execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching diseases: {e}")
            return []
    
    def get_disease_by_code(self, disease_code: str) -> Optional[Dict]:
        """
        Obtiene información de una enfermedad por su código.
        
        Args:
            disease_code: Código de la enfermedad (akiec, bcc, etc.)
        
        Returns:
            Dict con información de la enfermedad o None
        """
        try:
            response = self.client.table('disease_information').select('*').eq('disease_code', disease_code).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"❌ Error fetching disease {disease_code}: {e}")
            return None
    
    def get_prediction_stats_by_disease(self) -> List[Dict]:
        """
        Obtiene estadísticas de predicciones por enfermedad.
        
        Returns:
            Lista con estadísticas por enfermedad
        """
        try:
            response = self.client.table('v_prediction_stats_by_disease').select('*').execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching prediction stats: {e}")
            return []
    
    # ==================== USER SESSIONS ====================
    
    def create_or_update_session(
        self,
        session_token: str,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        device_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Crea o actualiza una sesión de usuario.
        
        Args:
            session_token: Token único de sesión
            user_agent: User agent del navegador
            ip_address: Dirección IP
            device_type: Tipo de dispositivo
        
        Returns:
            UUID de la sesión o None
        """
        try:
            # Intentar obtener sesión existente
            existing = self.client.table('user_sessions').select('id').eq('session_token', session_token).execute()
            
            if existing.data:
                # Actualizar sesión existente
                session_id = existing.data[0]['id']
                self.client.table('user_sessions').update({
                    'last_activity': datetime.utcnow().isoformat(),
                    'user_agent': user_agent,
                    'ip_address': ip_address,
                    'device_type': device_type
                }).eq('id', session_id).execute()
                logger.info(f"✅ Session {session_id} updated")
            else:
                # Crear nueva sesión
                response = self.client.table('user_sessions').insert({
                    'session_token': session_token,
                    'user_agent': user_agent,
                    'ip_address': ip_address,
                    'device_type': device_type
                }).execute()
                session_id = response.data[0]['id'] if response.data else None
                logger.info(f"✅ New session created: {session_id}")
            
            return session_id
        except Exception as e:
            logger.error(f"❌ Error managing session: {e}")
            return None
    
    def increment_session_predictions(self, session_id: str) -> bool:
        """
        Incrementa el contador de predicciones de una sesión.
        
        Args:
            session_id: UUID de la sesión
        
        Returns:
            True si fue exitoso, False en caso contrario
        """
        try:
            # Obtener contador actual
            response = self.client.table('user_sessions').select('total_predictions').eq('id', session_id).execute()
            
            if response.data:
                current_count = response.data[0].get('total_predictions', 0)
                self.client.table('user_sessions').update({
                    'total_predictions': current_count + 1,
                    'last_activity': datetime.utcnow().isoformat()
                }).eq('id', session_id).execute()
                logger.info(f"✅ Session {session_id} predictions incremented")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error incrementing session predictions: {e}")
            return False
    
    # ==================== PREDICTION FEEDBACK ====================
    
    def submit_feedback(self, feedback_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Envía feedback sobre una predicción.
        
        Args:
            feedback_data: Diccionario con el feedback
                - prediction_id: int (requerido)
                - session_id: str (opcional)
                - is_accurate: bool (opcional)
                - actual_diagnosis: str (opcional)
                - feedback_text: str (opcional)
                - rating: int (1-5, opcional)
        
        Returns:
            Dict con el feedback insertado o None
        """
        try:
            response = self.client.table('prediction_feedback').insert(feedback_data).execute()
            logger.info(f"✅ Feedback submitted for prediction {feedback_data.get('prediction_id')}")
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"❌ Error submitting feedback: {e}")
            return None
    
    # ==================== STATISTICS ====================
    
    def get_predictions_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Obtiene estadísticas de predicciones en un rango de fechas.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
        
        Returns:
            Lista con estadísticas por fecha
        """
        try:
            response = self.client.rpc('get_predictions_by_date_range', {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }).execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching predictions by date range: {e}")
            return []
    
    def calculate_model_metrics(self) -> Optional[Dict]:
        """
        Calcula métricas generales del modelo.
        
        Returns:
            Dict con métricas del modelo o None
        """
        try:
            response = self.client.rpc('calculate_model_metrics').execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"❌ Error calculating model metrics: {e}")
            return None
    
    def get_system_statistics(self, days: int = 7) -> List[Dict]:
        """
        Obtiene estadísticas del sistema de los últimos N días.
        
        Args:
            days: Número de días hacia atrás
        
        Returns:
            Lista con estadísticas diarias
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).date()
            response = (self.client.table('system_statistics')
                       .select('*')
                       .gte('stat_date', cutoff_date.isoformat())
                       .order('stat_date', desc=True)
                       .execute())
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"❌ Error fetching system statistics: {e}")
            return []


# Instancia global del cliente
supabase_client = SupabaseClient()


# Funciones de conveniencia para importar directamente
def create_prediction(prediction_data: Dict[str, Any]) -> Optional[Dict]:
    """Atajo para crear una predicción."""
    return supabase_client.create_prediction(prediction_data)


def get_recent_predictions(limit: int = 10) -> List[Dict]:
    """Atajo para obtener predicciones recientes."""
    return supabase_client.get_recent_predictions(limit)


def get_disease_info(disease_code: str) -> Optional[Dict]:
    """Atajo para obtener información de una enfermedad."""
    return supabase_client.get_disease_by_code(disease_code)


def submit_feedback(feedback_data: Dict[str, Any]) -> Optional[Dict]:
    """Atajo para enviar feedback."""
    return supabase_client.submit_feedback(feedback_data)
