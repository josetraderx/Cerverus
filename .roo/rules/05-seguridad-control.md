---
title: "Implementación Completa de Seguridad y Control - Proyecto Cerverus"
version: "1.0"
owner: "Equipo de Seguridad"
contact: "#team-security"
last_updated: "2025-09-09"
---

# =============================================================================
# IMPLEMENTACIÓN COMPLETA DE SEGURIDAD Y CONTROL - PROYECTO CERVERUS
# =============================================================================

## Checklist de Calidad para Seguridad y Control
- [ ] Arquitectura de seguridad por capas implementada
- [ ] Gestión de secretos y credenciales configurada
- [ ] MFA y RBAC habilitados y testeados
- [ ] Compliance GDPR verificado
- [ ] Auditoría y logging estructurado funcionando
- [ ] Rate limiting y protección DDoS activos

```python
import os
import secrets
import hashlib
import json
import re
import html
import time
import pyotp
import qrcode
import jwt
import redis
import boto3
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog
import bcrypt
from pydantic import BaseModel, validator, Field

# =============================================================================
# 1. ARQUITECTURA DE SEGURIDAD POR CAPAS
# =============================================================================

class SecurityLayer(Enum):
    """Capas de seguridad del sistema Cerverus"""
    NETWORK = "network"
    APPLICATION = "application" 
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"
    COMPLIANCE = "compliance"

class ThreatLevel(Enum):
    """Niveles de amenaza para el sistema"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Evento de seguridad estructurado"""
    event_type: str
    threat_level: ThreatLevel
    user_id: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    affected_layers: List[SecurityLayer]
    mitigation_actions: List[str] = field(default_factory=list)

# =============================================================================
# 2. GESTIÓN AVANZADA DE SECRETOS Y CREDENCIALES
# =============================================================================

class SecretManager:
    """Gestión enterprise de secretos y credenciales"""
    
    def __init__(self, use_aws_secrets: bool = True, use_vault: bool = False):
        self.use_aws = use_aws_secrets
        self.use_vault = use_vault
        self.logger = structlog.get_logger("secret_manager")
        
        if use_aws_secrets:
            self.secrets_client = boto3.client('secretsmanager')
        
        if use_vault:
            # Configuración para HashiCorp Vault
            self.vault_client = self._init_vault_client()
        
        # Encryption key para secretos locales
        self.encryption_key = self._get_or_create_master_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _init_vault_client(self):
        """Inicializar cliente HashiCorp Vault"""
        try:
            import hvac
            vault_url = os.environ.get('VAULT_URL', 'http://localhost:8200')
            vault_token = os.environ.get('VAULT_TOKEN')
            
            client = hvac.Client(url=vault_url, token=vault_token)
            if not client.is_authenticated():
                raise Exception("Vault authentication failed")
            return client
        except ImportError:
            self.logger.warning("hvac library not installed, Vault disabled")
            return None
    
    def _get_or_create_master_key(self) -> bytes:
        """Obtener o crear clave maestra para encriptación"""
        master_key_path = os.environ.get('MASTER_KEY_PATH', '/etc/cerverus/master.key')
        
        if os.path.exists(master_key_path):
            with open(master_key_path, 'rb') as f:
                return f.read()
        else:
            # Generar nueva clave maestra
            key = Fernet.generate_key()
            
            # En producción, esto debe manejarse de forma más segura
            os.makedirs(os.path.dirname(master_key_path), exist_ok=True)
            with open(master_key_path, 'wb') as f:
                f.write(key)
            
            # Establecer permisos restrictivos
            os.chmod(master_key_path, 0o600)
            
            self.logger.info("Generated new master key", path=master_key_path)
            return key
    
    def store_secret(self, secret_name: str, secret_value: str, 
                    metadata: Optional[Dict] = None) -> bool:
        """Almacenar secreto de forma segura"""
        try:
            audit_data = {
                'action': 'store_secret',
                'secret_name': secret_name,
                'timestamp': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat()
            }))
            mitigation_actions.append('24-hour temporary ban applied')
            
        elif threat_analysis['level'] == 'high':
            # Rate limit más estricto por 1 hora
            strict_key = f"rate_limit:strict:{identifier}"
            self.redis.setex(strict_key, 3600, json.dumps({
                'max_requests': 10,
                'window_seconds': 3600,
                'reason': 'High threat level detected'
            }))
            mitigation_actions.append('Strict rate limiting applied for 1 hour')
        
        # Log de mitigación
        mitigation_record = {
            'identifier': identifier,
            'threat_level': threat_analysis['level'],
            'actions': mitigation_actions,
            'indicators': threat_analysis['indicators'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.redis.lpush("threat_mitigation_log", json.dumps(mitigation_record))
        
        return {
            'actions': mitigation_actions,
            'threat_level': threat_analysis['level']
        }
    
    def _get_ip_request_count(self, ip_address: str) -> int:
        """Obtener conteo de requests por IP en la última hora"""
        now = time.time()
        hour_ago = now - 3600
        
        key = f"ip_requests:{ip_address}"
        return self.redis.zcount(key, hour_ago, now)

# =============================================================================
# 7. AUDITORÍA Y LOGGING DE SEGURIDAD AVANZADO
# =============================================================================

class SecurityEventType(Enum):
    """Tipos de eventos de seguridad para auditoría"""
    # Autenticación
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    MFA_SUCCESS = "mfa_success"
    MFA_FAILURE = "mfa_failure"
    PASSWORD_CHANGE = "password_change"
    
    # Autorización
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ROLE_CHANGE = "role_change"
    
    # Datos
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    
    # Sistema
    SYSTEM_LOGIN = "system_login"
    CONFIGURATION_CHANGE = "configuration_change"
    SECRET_ACCESS = "secret_access"
    SECRET_ROTATION = "secret_rotation"
    
    # Amenazas
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    DDOS_ATTEMPT = "ddos_attempt"
    INJECTION_ATTEMPT = "injection_attempt"
    
    # Compliance
    GDPR_REQUEST = "gdpr_request"
    AUDIT_LOG_ACCESS = "audit_log_access"
    COMPLIANCE_VIOLATION = "compliance_violation"

@dataclass
class SecurityAuditEvent:
    """Evento de auditoría de seguridad estructurado"""
    event_id: str
    event_type: SecurityEventType
    severity: str  # low, medium, high, critical
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent_hash: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    result: str  # success, failure, denied
    details: Dict[str, Any]
    correlation_id: Optional[str] = None

class AdvancedSecurityAuditor:
    """Auditoría avanzada de seguridad con correlación de eventos"""
    
    def __init__(self, redis_client: redis.Redis, secret_manager: SecretManager):
        self.redis = redis_client
        self.secret_manager = secret_manager
        self.logger = structlog.get_logger("security_auditor")
        
        # Configuración de retención de logs
        self.retention_periods = {
            'high': timedelta(days=2555),     # 7 años para eventos críticos
            'critical': timedelta(days=2555), # 7 años para eventos críticos
            'medium': timedelta(days=730),    # 2 años para eventos importantes
            'low': timedelta(days=90)         # 90 días para eventos normales
        }
        
        # Patrones de correlación para detectar ataques
        self.correlation_patterns = {
            'brute_force': {
                'events': [SecurityEventType.LOGIN_FAILURE],
                'threshold': 10,
                'window_minutes': 5,
                'action': 'alert_and_block'
            },
            'privilege_escalation': {
                'events': [SecurityEventType.ROLE_CHANGE, SecurityEventType.PRIVILEGE_ESCALATION],
                'threshold': 3,
                'window_minutes': 60,
                'action': 'alert_immediate'
            },
            'data_exfiltration': {
                'events': [SecurityEventType.DATA_ACCESS, SecurityEventType.DATA_EXPORT],
                'threshold': 100,
                'window_minutes': 30,
                'action': 'alert_and_investigate'
            }
        }
    
    def log_security_event(self, event_type: SecurityEventType, severity: str,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          resource_type: Optional[str] = None,
                          resource_id: Optional[str] = None,
                          action: str = "",
                          result: str = "success",
                          details: Optional[Dict[str, Any]] = None,
                          correlation_id: Optional[str] = None) -> str:
        """Registrar evento de seguridad con correlación automática"""
        
        event_id = f"sec_{secrets.token_hex(16)}"
        timestamp = datetime.utcnow()
        
        # Hash del user agent para privacidad
        user_agent_hash = None
        if user_agent:
            user_agent_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]
        
        # Hash de IP para privacidad (manteniendo capacidad de correlación)
        ip_hash = None
        if ip_address:
            ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
        
        audit_event = SecurityAuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=timestamp,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_hash,
            user_agent_hash=user_agent_hash,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            details=details or {},
            correlation_id=correlation_id
        )
        
        # Serializar evento
        event_data = {
            'event_id': event_id,
            'event_type': event_type.value,
            'severity': severity,
            'timestamp': timestamp.isoformat(),
            'user_id': user_id,
            'session_id': session_id,
            'ip_address_hash': ip_hash,
            'user_agent_hash': user_agent_hash,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'action': action,
            'result': result,
            'details': details or {},
            'correlation_id': correlation_id
        }
        
        # Almacenar en múltiples índices para búsquedas eficientes
        self._store_audit_event(event_data, severity)
        
        # Correlación de eventos para detección de ataques
        correlation_results = self._correlate_events(audit_event)
        if correlation_results:
            self._handle_correlation_alerts(correlation_results)
        
        # Log estructurado
        log_level = 'error' if severity in ['high', 'critical'] else 'info'
        getattr(self.logger, log_level)(
            "Security event recorded",
            event_id=event_id,
            event_type=event_type.value,
            severity=severity,
            user_id=user_id,
            result=result
        )
        
        return event_id
    
    def _store_audit_event(self, event_data: Dict[str, Any], severity: str):
        """Almacenar evento en múltiples índices"""
        retention_seconds = int(self.retention_periods[severity].total_seconds())
        event_json = json.dumps(event_data)
        
        # Índice principal por ID
        self.redis.setex(
            f"audit_event:{event_data['event_id']}",
            retention_seconds,
            event_json
        )
        
        # Índice por usuario (si existe)
        if event_data['user_id']:
            self.redis.zadd(
                f"user_audit:{event_data['user_id']}",
                {event_data['event_id']: time.time()}
            )
            self.redis.expire(f"user_audit:{event_data['user_id']}", retention_seconds)
        
        # Índice por tipo de evento
        self.redis.zadd(
            f"event_type_audit:{event_data['event_type']}",
            {event_data['event_id']: time.time()}
        )
        self.redis.expire(f"event_type_audit:{event_data['event_type']}", retention_seconds)
        
        # Índice por severidad
        self.redis.zadd(
            f"severity_audit:{severity}",
            {event_data['event_id']: time.time()}
        )
        self.redis.expire(f"severity_audit:{severity}", retention_seconds)
        
        # Índice temporal para correlación (últimas 24 horas)
        self.redis.zadd(
            "recent_security_events",
            {event_data['event_id']: time.time()}
        )
        self.redis.expire("recent_security_events", 86400)  # 24 horas
    
    def _correlate_events(self, event: SecurityAuditEvent) -> List[Dict[str, Any]]:
        """Correlacionar eventos para detectar patrones de ataque"""
        alerts = []
        
        for pattern_name, pattern_config in self.correlation_patterns.items():
            if event.event_type in pattern_config['events']:
                # Buscar eventos similares en la ventana de tiempo
                window_start = time.time() - (pattern_config['window_minutes'] * 60)
                
                # Construir clave de correlación
                if event.ip_address and event.user_id:
                    correlation_key = f"{event.ip_address}:{event.user_id}"
                elif event.ip_address:
                    correlation_key = event.ip_address
                elif event.user_id:
                    correlation_key = event.user_id
                else:
                    continue
                
                # Contar eventos similares
                pattern_key = f"correlation:{pattern_name}:{correlation_key}"
                event_count = self.redis.zcount(pattern_key, window_start, time.time())
                
                # Añadir evento actual
                self.redis.zadd(pattern_key, {event.event_id: time.time()})
                self.redis.expire(pattern_key, pattern_config['window_minutes'] * 60)
                
                # Verificar si se alcanzó el umbral
                if event_count >= pattern_config['threshold']:
                    alerts.append({
                        'pattern': pattern_name,
                        'correlation_key': correlation_key,
                        'event_count': event_count + 1,
                        'window_minutes': pattern_config['window_minutes'],
                        'action': pattern_config['action'],
                        'events': list(self.redis.zrange(pattern_key, 0, -1))
                    })
        
        return alerts
    
    def _handle_correlation_alerts(self, correlation_results: List[Dict[str, Any]]):
        """Manejar alertas de correlación"""
        for alert in correlation_results:
            alert_id = f"alert_{secrets.token_hex(16)}"
            
            alert_data = {
                'alert_id': alert_id,
                'pattern': alert['pattern'],
                'correlation_key': alert['correlation_key'],
                'event_count': alert['event_count'],
                'window_minutes': alert['window_minutes'],
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'active',
                'action_required': alert['action'],
                'related_events': alert['events']
            }
            
            # Almacenar alerta
            self.redis.setex(
                f"security_alert:{alert_id}",
                86400 * 7,  # 7 días
                json.dumps(alert_data)
            )
            
            # Añadir a lista de alertas activas
            self.redis.lpush("active_security_alerts", alert_id)
            
            # Log crítico
            self.logger.critical(
                "Security pattern detected",
                alert_id=alert_id,
                pattern=alert['pattern'],
                event_count=alert['event_count'],
                action_required=alert['action']
            )
            
            # Ejecutar acciones automáticas según configuración
            if alert['action'] == 'alert_and_block':
                self._auto_block_threat(alert['correlation_key'])
            elif alert['action'] == 'alert_immediate':
                self._send_immediate_alert(alert_data)
    
    def _auto_block_threat(self, correlation_key: str):
        """Bloquear automáticamente amenaza detectada"""
        # Aplicar bloqueo temporal
        block_key = f"auto_block:{correlation_key}"
        self.redis.setex(block_key, 3600, json.dumps({  # 1 hora
            'reason': 'Automatic block due to suspicious pattern',
            'timestamp': datetime.utcnow().isoformat(),
            'auto_generated': True
        }))
        
        self.logger.warning("Automatic threat block applied", 
                          correlation_key=correlation_key)
    
    def _send_immediate_alert(self, alert_data: Dict[str, Any]):
        """Enviar alerta inmediata al equipo de seguridad"""
        # Placeholder - integrar con sistema de notificaciones
        # (Slack, email, PagerDuty, etc.)
        self.logger.critical("Immediate security alert triggered", 
                           alert_id=alert_data['alert_id'])
    
    def search_audit_events(self, filters: Dict[str, Any], 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Buscar eventos de auditoría con filtros"""
        events = []
        
        # Construir consulta basada en filtros
        if 'user_id' in filters:
            event_ids = self.redis.zrevrange(
                f"user_audit:{filters['user_id']}", 
                0, limit - 1
            )
        elif 'event_type' in filters:
            event_ids = self.redis.zrevrange(
                f"event_type_audit:{filters['event_type']}", 
                0, limit - 1
            )
        elif 'severity' in filters:
            event_ids = self.redis.zrevrange(
                f"severity_audit:{filters['severity']}", 
                0, limit - 1
            )
        else:
            # Búsqueda general en eventos recientes
            event_ids = self.redis.zrevrange("recent_security_events", 0, limit - 1)
        
        # Recuperar eventos completos
        for event_id in event_ids:
            event_data = self.redis.get(f"audit_event:{event_id.decode()}")
            if event_data:
                event = json.loads(event_data)
                
                # Aplicar filtros adicionales
                if self._matches_filters(event, filters):
                    events.append(event)
        
        return events[:limit]
    
    def _matches_filters(self, event: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Verificar si el evento coincide con los filtros"""
        for key, value in filters.items():
            if key in event and event[key] != value:
                return False
        return True
    
    def generate_compliance_report(self, start_date: datetime, 
                                  end_date: datetime) -> Dict[str, Any]:
        """Generar reporte de compliance para auditorías"""
        report_id = f"compliance_report_{secrets.token_hex(16)}"
        
        # Recopilar estadísticas de eventos
        stats = {
            'report_id': report_id,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'event_counts': {},
            'user_activity': {},
            'access_patterns': {},
            'security_incidents': [],
            'compliance_events': []
        }
        
        # Buscar eventos en el período (simplificado)
        recent_events = self.redis.zrevrangebyscore(
            "recent_security_events",
            end_date.timestamp(),
            start_date.timestamp()
        )
        
        for event_id in recent_events:
            event_data = self.redis.get(f"audit_event:{event_id.decode()}")
            if event_data:
                event = json.loads(event_data)
                
                # Contar por tipo
                event_type = event['event_type']
                stats['event_counts'][event_type] = stats['event_counts'].get(event_type, 0) + 1
                
                # Actividad por usuario
                if event['user_id']:
                    user_id = event['user_id']
                    if user_id not in stats['user_activity']:
                        stats['user_activity'][user_id] = {
                            'total_events': 0,
                            'last_activity': event['timestamp']
                        }
                    stats['user_activity'][user_id]['total_events'] += 1
                
                # Eventos de compliance
                if 'gdpr' in event_type.lower() or 'compliance' in event_type.lower():
                    stats['compliance_events'].append(event)
        
        # Almacenar reporte
        self.redis.setex(
            f"compliance_report:{report_id}",
            86400 * 30,  # 30 días
            json.dumps(stats)
        )
        
        return stats

# =============================================================================
# 8. DECORADORES Y UTILIDADES DE SEGURIDAD
# =============================================================================

def require_permission(permission: Permission):
    """Decorador para verificar permisos"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener información de usuario del contexto (request, sesión, etc.)
            # Placeholder - implementar según su framework web
            user_id = kwargs.get('current_user_id')
            if not user_id:
                raise PermissionError("User not authenticated")
            
            # Verificar permiso (necesita instancia de RBACManager)
            # rbac_manager = get_rbac_manager()  # Implementar según DI
            # user_role = rbac_manager.get_user_role(user_id)
            # if not rbac_manager.check_permission(user_role, permission):
            #     raise PermissionError(f"Permission {permission.value} required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def audit_action(event_type: SecurityEventType, resource_type: str = None):
    """Decorador para auditar acciones automáticamente"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Log evento exitoso
                # auditor = get_security_auditor()  # Implementar según DI
                # auditor.log_security_event(
                #     event_type=event_type,
                #     severity='low',
                #     action=func.__name__,
                #     result='success',
                #     resource_type=resource_type,
                #     details={'execution_time': time.time() - start_time}
                # )
                
                return result
                
            except Exception as e:
                # Log evento fallido
                # auditor = get_security_auditor()  # Implementar según DI
                # auditor.log_security_event(
                #     event_type=event_type,
                #     severity='medium',
                #     action=func.__name__,
                #     result='failure',
                #     resource_type=resource_type,
                #     details={
                #         'error': str(e),
                #         'execution_time': time.time() - start_time
                #     }
                # )
                raise
        return wrapper
    return decorator

def rate_limit(rule_name: str):
    """Decorador para rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener identificador del cliente
            client_id = kwargs.get('client_id') or 'anonymous'
            
            # Verificar rate limit
            # rate_limiter = get_rate_limiter()  # Implementar según DI
            # result = rate_limiter.check_rate_limit(client_id, rule_name)
            # if not result['allowed']:
            #     raise RateLimitExceededError(
            #         f"Rate limit exceeded for {rule_name}. "
            #         f"Try again in {result.get('reset_time', 60)} seconds"
            #     )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def security_context(user_id: str, session_id: str, ip_address: str):
    """Context manager para operaciones con contexto de seguridad"""
    correlation_id = f"ctx_{secrets.token_hex(8)}"
    
    # Setup del contexto
    context_data = {
        'user_id': user_id,
        'session_id': session_id,
        'ip_address': ip_address,
        'correlation_id': correlation_id,
        'start_time': time.time()
    }
    
    try:
        yield context_data
        
        # Log successful context completion
        # auditor = get_security_auditor()
        # auditor.log_security_event(
        #     event_type=SecurityEventType.SYSTEM_LOGIN,
        #     severity='low',
        #     user_id=user_id,
        #     session_id=session_id,
        #     action='security_context_completed',
        #     result='success',
        #     correlation_id=correlation_id
        # )
        
    except Exception as e:
        # Log context failure
        # auditor = get_security_auditor()
        # auditor.log_security_event(
        #     event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
        #     severity='medium',
        #     user_id=user_id,
        #     session_id=session_id,
        #     action='security_context_failed',
        #     result='failure',
        #     details={'error': str(e)},
        #     correlation_id=correlation_id
        # )
        raise

# =============================================================================
# 9. EJEMPLO DE INTEGRACIÓN COMPLETA
# =============================================================================

class CerverusSecurityManager:
    """Gestión integrada de seguridad para el proyecto Cerverus"""
    
    def __init__(self, redis_client: redis.Redis, aws_region: str = 'us-east-1'):
        self.redis = redis_client
        
        # Inicializar componentes de seguridad
        self.secret_manager = SecretManager(use_aws_secrets=True)
        self.mfa_manager = EnhancedMFAManager(redis_client, self.secret_manager)
        self.rbac_manager = RBACManager(redis_client)
        self.gdpr_manager = GDPRComplianceManager(redis_client, self.secret_manager)
        self.rate_limiter = AdvancedRateLimiter(redis_client)
        self.auditor = AdvancedSecurityAuditor(redis_client, self.secret_manager)
        
        self.logger = structlog.get_logger("security_manager")
    
    async def authenticate_and_authorize(self, username: str, password: str,
                                       mfa_token: Optional[str],
                                       required_permission: Permission,
                                       ip_address: str, user_agent: str) -> Dict[str, Any]:
        """Flujo completo de autenticación y autorización"""
        try:
            # 1. Verificar rate limiting
            rate_check = self.rate_limiter.check_rate_limit(
                f"auth:{ip_address}", 
                'login',
                context={
                    'ip_address': ip_address,
                    'user_agent': user_agent
                }
            )
            
            if not rate_check['allowed']:
                self.auditor.log_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    'medium',
                    action='authentication_attempt',
                    result='denied',
                    ip_address=ip_address,
                    details={'reason': 'rate_limit_exceeded'}
                )
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            # 2. Autenticación inicial
            auth_result = self.mfa_manager.authenticate_user(
                username, password, ip_address, user_agent
            )
            
            if auth_result['result'] == AuthenticationResult.INVALID_CREDENTIALS:
                self.auditor.log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    'medium',
                    action='authentication_attempt',
                    result='failure',
                    ip_address=ip_address,
                    details={'username': username, 'reason': 'invalid_credentials'}
                )
                return {'success': False, 'error': 'Invalid credentials'}
            
            # 3. MFA si es requerido
            if auth_result['result'] == AuthenticationResult.MFA_REQUIRED:
                if not mfa_token:
                    return {
                        'success': False, 
                        'mfa_required': True,
                        'temp_session_id': auth_result['temp_session_id']
                    }
                
                mfa_result = self.mfa_manager.complete_mfa_authentication(
                    auth_result['temp_session_id'], mfa_token
                )
                
                if mfa_result['result'] != AuthenticationResult.SUCCESS:
                    self.auditor.log_security_event(
                        SecurityEventType.MFA_FAILURE,
                        'high',
                        action='mfa_verification',
                        result='failure',
                        ip_address=ip_address,
                        details={'username': username}
                    )
                    return {'success': False, 'error': 'Invalid MFA token'}
                
                auth_result = mfa_result
            
            # 4. Obtener información del usuario
            # user_info = self._get_user_info(username)  # Implementar
            user_id = "placeholder_user_id"  # Obtener del resultado de auth
            user_role = self.rbac_manager.get_user_role(user_id)
            
            if not user_role:
                return {'success': False, 'error': 'No role assigned'}
            
            # 5. Verificar autorización
            access_context = AccessContext(
                user_id=user_id,
                ip_address=ip_address,
                time_of_access=datetime.utcnow(),
                resource_type='api'
            )
            
            if not self.rbac_manager.check_permission(user_role, required_permission, access_context):
                self.auditor.log_security_event(
                    SecurityEventType.ACCESS_DENIED,
                    'medium',
                    user_id=user_id,
                    action='authorization_check',
                    result='denied',
                    ip_address=ip_address,
                    details={'required_permission': required_permission.value}
                )
                return {'success': False, 'error': 'Insufficient permissions'}
            
            # 6. Log autenticación exitosa
            self.auditor.log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                'low',
                user_id=user_id,
                session_id=auth_result['session_id'],
                action='full_authentication',
                result='success',
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # 7. Registrar procesamiento de datos para GDPR
            self.gdpr_manager.record_data_processing(
                user_id=user_id,
                data_categories=[DataCategory.PERSONAL_DATA, DataCategory.BEHAVIORAL_DATA],
                purpose="Authentication and session management",
                legal_basis=LegalBasis.LEGITIMATE_INTERESTS
            )
            
            return {
                'success': True,
                'session_id': auth_result['session_id'],
                'user_id': user_id,
                'user_role': user_role.value,
                'expires_at': auth_result['expires_at']
            }
            
        except Exception as e:
            self.logger.error("Authentication/authorization failed", 
                            username=username, error=str(e))
            
            self.auditor.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                'high',
                action='authentication_error',
                result='error',
                ip_address=ip_address,
                details={'error': str(e), 'username': username}
            )
            
            return {'success': False, 'error': 'Internal authentication error'}

# =============================================================================
# EJEMPLO DE USO EN APLICACIÓN FLASK/FASTAPI
# =============================================================================

"""
# Ejemplo de integración con FastAPI

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer
import redis

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)
security_manager = CerverusSecurityManager(redis_client)
security = HTTPBearer()

@app.post("/api/v1/authenticate")
async def authenticate(request: Request, credentials: dict):
    result = await security_manager.authenticate_and_authorize(
        username=credentials['username'],
        password=credentials['password'],
        mfa_token=credentials.get('mfa_token'),
         datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            if self.use_aws:
                response = self.secrets_client.create_secret(
                    Name=secret_name,
                    SecretString=secret_value,
                    Description=metadata.get('description', '') if metadata else ''
                )
                audit_data['storage_backend'] = 'aws_secrets_manager'
                audit_data['secret_arn'] = response.get('ARN')
                
            elif self.use_vault and self.vault_client:
                self.vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_name,
                    secret={'value': secret_value, 'metadata': metadata}
                )
                audit_data['storage_backend'] = 'hashicorp_vault'
                
            else:
                # Fallback a encriptación local
                encrypted_value = self.cipher.encrypt(secret_value.encode())
                with open(f'/etc/cerverus/secrets/{secret_name}.enc', 'wb') as f:
                    f.write(encrypted_value)
                audit_data['storage_backend'] = 'local_encrypted'
            
            self.logger.info("Secret stored successfully", **audit_data)
            return True
            
        except Exception as e:
            self.logger.error("Failed to store secret", 
                            secret_name=secret_name, error=str(e))
            return False
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Recuperar secreto de forma segura"""
        try:
            audit_data = {
                'action': 'retrieve_secret',
                'secret_name': secret_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if self.use_aws:
                response = self.secrets_client.get_secret_value(SecretId=secret_name)
                audit_data['storage_backend'] = 'aws_secrets_manager'
                self.logger.info("Secret retrieved", **audit_data)
                return response['SecretString']
                
            elif self.use_vault and self.vault_client:
                response = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=secret_name
                )
                audit_data['storage_backend'] = 'hashicorp_vault'
                self.logger.info("Secret retrieved", **audit_data)
                return response['data']['data']['value']
                
            else:
                # Fallback a encriptación local
                secret_path = f'/etc/cerverus/secrets/{secret_name}.enc'
                if os.path.exists(secret_path):
                    with open(secret_path, 'rb') as f:
                        encrypted_value = f.read()
                    audit_data['storage_backend'] = 'local_encrypted'
                    self.logger.info("Secret retrieved", **audit_data)
                    return self.cipher.decrypt(encrypted_value).decode()
                
            return None
            
        except Exception as e:
            self.logger.error("Failed to retrieve secret", 
                            secret_name=secret_name, error=str(e))
            return None
    
    def rotate_secret(self, secret_name: str, new_value: str) -> bool:
        """Rotar secreto manteniendo versión anterior"""
        try:
            # Backup del valor anterior
            old_value = self.get_secret(secret_name)
            if old_value:
                backup_name = f"{secret_name}_backup_{int(time.time())}"
                self.store_secret(backup_name, old_value, 
                                {'type': 'backup', 'original': secret_name})
            
            # Almacenar nuevo valor
            result = self.store_secret(secret_name, new_value, 
                                     {'rotated_at': datetime.utcnow().isoformat()})
            
            self.logger.info("Secret rotated successfully", 
                           secret_name=secret_name)
            return result
            
        except Exception as e:
            self.logger.error("Failed to rotate secret", 
                            secret_name=secret_name, error=str(e))
            return False

# =============================================================================
# 3. AUTENTICACIÓN MULTI-FACTOR (MFA) AVANZADA
# =============================================================================

class MFAMethod(Enum):
    """Métodos de autenticación multi-factor"""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"
    BIOMETRIC = "biometric"

class AuthenticationResult(Enum):
    """Resultados de autenticación"""
    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    MFA_REQUIRED = "mfa_required"
    MFA_INVALID = "mfa_invalid"
    ACCOUNT_LOCKED = "account_locked"
    EXPIRED_TOKEN = "expired_token"

@dataclass
class AuthSession:
    """Sesión de autenticación"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    mfa_completed: bool
    ip_address: str
    user_agent_hash: str
    permissions: Set[str]

class EnhancedMFAManager:
    """Gestión avanzada de autenticación multi-factor"""
    
    def __init__(self, redis_client: redis.Redis, secret_manager: SecretManager):
        self.redis = redis_client
        self.secret_manager = secret_manager
        self.logger = structlog.get_logger("mfa_manager")
        
        # Configuración de seguridad
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=8)
        self.mfa_timeout = timedelta(minutes=5)
    
    def setup_totp(self, user_id: str, user_email: str) -> Dict[str, Any]:
        """Configurar TOTP para usuario"""
        try:
            # Generar secreto único
            secret = pyotp.random_base32()
            
            # Crear URI de aprovisionamiento
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user_email,
                issuer_name="Cerverus Fraud Detection System"
            )
            
            # Generar QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=5
            )
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            # Generar códigos de respaldo
            backup_codes = [secrets.token_hex(4) for _ in range(10)]
            hashed_backup_codes = [
                bcrypt.hashpw(code.encode(), bcrypt.gensalt()).decode()
                for code in backup_codes
            ]
            
            # Almacenar configuración MFA
            mfa_config = {
                'secret': secret,
                'backup_codes': hashed_backup_codes,
                'setup_date': datetime.utcnow().isoformat(),
                'method': MFAMethod.TOTP.value
            }
            
            config_json = json.dumps(mfa_config)
            self.secret_manager.store_secret(
                f"mfa_config_{user_id}", 
                config_json,
                {'type': 'mfa_configuration', 'user_id': user_id}
            )
            
            self.logger.info("TOTP setup completed", user_id=user_id)
            
            return {
                'secret': secret,
                'qr_code_uri': totp_uri,
                'backup_codes': backup_codes,  # Solo retornamos una vez
                'setup_complete': True
            }
            
        except Exception as e:
            self.logger.error("TOTP setup failed", user_id=user_id, error=str(e))
            raise
    
    def verify_totp(self, user_id: str, token: str, 
                   allow_backup_code: bool = True) -> bool:
        """Verificar token TOTP o código de respaldo"""
        try:
            # Obtener configuración MFA
            config_json = self.secret_manager.get_secret(f"mfa_config_{user_id}")
            if not config_json:
                return False
            
            mfa_config = json.loads(config_json)
            
            # Verificar token TOTP
            if len(token) == 6 and token.isdigit():
                totp = pyotp.TOTP(mfa_config['secret'])
                if totp.verify(token, valid_window=2):  # ±2 ventanas de tiempo
                    self.logger.info("TOTP verification successful", user_id=user_id)
                    return True
            
            # Verificar código de respaldo si está permitido
            if allow_backup_code and len(token) == 8:
                for i, hashed_code in enumerate(mfa_config['backup_codes']):
                    if bcrypt.checkpw(token.encode(), hashed_code.encode()):
                        # Invalidar código usado
                        mfa_config['backup_codes'][i] = None
                        updated_config = json.dumps(mfa_config)
                        self.secret_manager.store_secret(
                            f"mfa_config_{user_id}", 
                            updated_config
                        )
                        self.logger.info("Backup code verification successful", 
                                       user_id=user_id)
                        return True
            
            self.logger.warning("MFA verification failed", user_id=user_id)
            return False
            
        except Exception as e:
            self.logger.error("MFA verification error", 
                            user_id=user_id, error=str(e))
            return False
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str, user_agent: str) -> Dict[str, Any]:
        """Autenticación completa con MFA"""
        user_agent_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]
        
        # Verificar intentos de login fallidos
        failed_attempts_key = f"failed_login:{username}:{ip_address}"
        failed_attempts = self.redis.get(failed_attempts_key)
        
        if failed_attempts and int(failed_attempts) >= self.max_login_attempts:
            self.logger.warning("Account locked due to failed attempts", 
                              username=username, ip_address=ip_address)
            return {
                'result': AuthenticationResult.ACCOUNT_LOCKED,
                'message': 'Account temporarily locked due to failed login attempts'
            }
        
        # Verificar credenciales (implementar según su sistema de usuarios)
        user_id = self._verify_password(username, password)
        if not user_id:
            # Incrementar contador de intentos fallidos
            self.redis.incr(failed_attempts_key)
            self.redis.expire(failed_attempts_key, int(self.lockout_duration.total_seconds()))
            
            self.logger.warning("Invalid credentials", 
                              username=username, ip_address=ip_address)
            return {
                'result': AuthenticationResult.INVALID_CREDENTIALS,
                'message': 'Invalid username or password'
            }
        
        # Limpiar intentos fallidos en login exitoso
        self.redis.delete(failed_attempts_key)
        
        # Verificar si MFA está habilitado
        mfa_config = self.secret_manager.get_secret(f"mfa_config_{user_id}")
        if mfa_config:
            # Crear sesión temporal para MFA
            temp_session_id = secrets.token_urlsafe(32)
            temp_session_data = {
                'user_id': user_id,
                'username': username,
                'ip_address': ip_address,
                'user_agent_hash': user_agent_hash,
                'created_at': datetime.utcnow().isoformat(),
                'mfa_required': True
            }
            
            self.redis.setex(
                f"temp_session:{temp_session_id}",
                int(self.mfa_timeout.total_seconds()),
                json.dumps(temp_session_data)
            )
            
            return {
                'result': AuthenticationResult.MFA_REQUIRED,
                'temp_session_id': temp_session_id,
                'message': 'MFA verification required'
            }
        
        # Crear sesión completa si no se requiere MFA
        return self._create_full_session(user_id, username, ip_address, user_agent_hash)
    
    def complete_mfa_authentication(self, temp_session_id: str, 
                                  mfa_token: str) -> Dict[str, Any]:
        """Completar autenticación MFA"""
        # Obtener sesión temporal
        temp_session_data = self.redis.get(f"temp_session:{temp_session_id}")
        if not temp_session_data:
            return {
                'result': AuthenticationResult.EXPIRED_TOKEN,
                'message': 'MFA session expired'
            }
        
        session_data = json.loads(temp_session_data)
        user_id = session_data['user_id']
        
        # Verificar token MFA
        if self.verify_totp(user_id, mfa_token):
            # Eliminar sesión temporal
            self.redis.delete(f"temp_session:{temp_session_id}")
            
            # Crear sesión completa
            return self._create_full_session(
                user_id, 
                session_data['username'],
                session_data['ip_address'],
                session_data['user_agent_hash']
            )
        else:
            return {
                'result': AuthenticationResult.MFA_INVALID,
                'message': 'Invalid MFA token'
            }
    
    def _verify_password(self, username: str, password: str) -> Optional[str]:
        """Verificar contraseña (implementar según su sistema)"""
        # Placeholder - implementar con su sistema de usuarios
        # Debería verificar hash de contraseña y retornar user_id si es válido
        pass
    
    def _create_full_session(self, user_id: str, username: str, 
                           ip_address: str, user_agent_hash: str) -> Dict[str, Any]:
        """Crear sesión completa autenticada"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + self.session_timeout
        
        session = AuthSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            mfa_completed=True,
            ip_address=ip_address,
            user_agent_hash=user_agent_hash,
            permissions=self._get_user_permissions(user_id)
        )
        
        # Almacenar sesión
        session_data = {
            'user_id': user_id,
            'username': username,
            'created_at': session.created_at.isoformat(),
            'expires_at': session.expires_at.isoformat(),
            'ip_address': ip_address,
            'user_agent_hash': user_agent_hash,
            'permissions': list(session.permissions)
        }
        
        self.redis.setex(
            f"session:{session_id}",
            int(self.session_timeout.total_seconds()),
            json.dumps(session_data)
        )
        
        self.logger.info("Authentication successful", 
                        user_id=user_id, session_id=session_id)
        
        return {
            'result': AuthenticationResult.SUCCESS,
            'session_id': session_id,
            'expires_at': expires_at.isoformat(),
            'message': 'Authentication successful'
        }
    
    def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Obtener permisos del usuario"""
        # Placeholder - implementar según su sistema RBAC
        return set()

# =============================================================================
# 4. CONTROL DE ACCESO BASADO EN ROLES (RBAC) AVANZADO
# =============================================================================

class Permission(Enum):
    """Permisos específicos del sistema Cerverus"""
    # Permisos de lectura
    READ_TRANSACTIONS = "read:transactions"
    READ_FRAUD_SIGNALS = "read:fraud_signals"
    READ_USER_DATA = "read:user_data"
    READ_AUDIT_LOGS = "read:audit_logs"
    READ_SYSTEM_METRICS = "read:system_metrics"
    
    # Permisos de escritura
    WRITE_TRANSACTIONS = "write:transactions"
    WRITE_FRAUD_RULES = "write:fraud_rules"
    WRITE_USER_DATA = "write:user_data"
    
    # Permisos administrativos
    MANAGE_USERS = "manage:users"
    MANAGE_MODELS = "manage:models"
    MANAGE_CONFIGURATION = "manage:configuration"
    MANAGE_SECRETS = "manage:secrets"
    
    # Permisos del sistema
    SYSTEM_DEPLOY = "system:deploy"
    SYSTEM_BACKUP = "system:backup"
    SYSTEM_MAINTENANCE = "system:maintenance"
    
    # Permisos de compliance
    COMPLIANCE_EXPORT = "compliance:export"
    COMPLIANCE_DELETE = "compliance:delete"
    COMPLIANCE_AUDIT = "compliance:audit"

class Role(Enum):
    """Roles del sistema Cerverus"""
    VIEWER = "viewer"
    ANALYST = "analyst"
    SENIOR_ANALYST = "senior_analyst"
    ADMIN = "admin"
    SYSTEM_ADMIN = "system_admin"
    COMPLIANCE_OFFICER = "compliance_officer"
    API_SERVICE = "api_service"

class AccessContext:
    """Contexto de acceso para decisiones de autorización"""
    def __init__(self, user_id: str, ip_address: str, time_of_access: datetime,
                 resource_type: str, resource_id: Optional[str] = None):
        self.user_id = user_id
        self.ip_address = ip_address
        self.time_of_access = time_of_access
        self.resource_type = resource_type
        self.resource_id = resource_id

class RBACManager:
    """Gestión avanzada de RBAC con contexto"""
    
    ROLE_PERMISSIONS = {
        Role.VIEWER: {
            Permission.READ_TRANSACTIONS,
            Permission.READ_FRAUD_SIGNALS,
            Permission.READ_SYSTEM_METRICS
        },
        Role.ANALYST: {
            Permission.READ_TRANSACTIONS,
            Permission.READ_FRAUD_SIGNALS,
            Permission.READ_SYSTEM_METRICS,
            Permission.READ_USER_DATA,
            Permission.WRITE_FRAUD_RULES
        },
        Role.SENIOR_ANALYST: {
            Permission.READ_TRANSACTIONS,
            Permission.READ_FRAUD_SIGNALS,
            Permission.READ_SYSTEM_METRICS,
            Permission.READ_USER_DATA,
            Permission.READ_AUDIT_LOGS,
            Permission.WRITE_FRAUD_RULES,
            Permission.WRITE_TRANSACTIONS,
            Permission.MANAGE_MODELS
        },
        Role.ADMIN: {
            Permission.READ_TRANSACTIONS,
            Permission.READ_FRAUD_SIGNALS,
            Permission.READ_SYSTEM_METRICS,
            Permission.READ_USER_DATA,
            Permission.READ_AUDIT_LOGS,
            Permission.WRITE_FRAUD_RULES,
            Permission.WRITE_TRANSACTIONS,
            Permission.WRITE_USER_DATA,
            Permission.MANAGE_USERS,
            Permission.MANAGE_MODELS,
            Permission.MANAGE_CONFIGURATION
        },
        Role.SYSTEM_ADMIN: {
            # Todos los permisos del admin plus sistema
            *{p for perms in [ROLE_PERMISSIONS[Role.ADMIN]] for p in perms},
            Permission.MANAGE_SECRETS,
            Permission.SYSTEM_DEPLOY,
            Permission.SYSTEM_BACKUP,
            Permission.SYSTEM_MAINTENANCE
        },
        Role.COMPLIANCE_OFFICER: {
            Permission.READ_TRANSACTIONS,
            Permission.READ_USER_DATA,
            Permission.READ_AUDIT_LOGS,
            Permission.COMPLIANCE_EXPORT,
            Permission.COMPLIANCE_DELETE,
            Permission.COMPLIANCE_AUDIT
        },
        Role.API_SERVICE: {
            Permission.READ_TRANSACTIONS,
            Permission.WRITE_TRANSACTIONS,
            Permission.READ_FRAUD_SIGNALS,
            Permission.WRITE_FRAUD_RULES
        }
    }
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = structlog.get_logger("rbac_manager")
        
        # Restricciones de tiempo por rol
        self.time_restrictions = {
            Role.ANALYST: {'start_hour': 6, 'end_hour': 22},  # 6 AM - 10 PM
            Role.VIEWER: {'start_hour': 8, 'end_hour': 18},   # 8 AM - 6 PM
        }
        
        # Restricciones de IP por rol (ejemplo)
        self.ip_restrictions = {
            Role.SYSTEM_ADMIN: ['10.0.0.0/8', '192.168.0.0/16'],  # Solo redes internas
        }
    
    def check_permission(self, user_role: Role, required_permission: Permission,
                        context: Optional[AccessContext] = None) -> bool:
        """Verificar permiso con contexto de acceso"""
        try:
            # Verificación básica de permiso
            if required_permission not in self.ROLE_PERMISSIONS.get(user_role, set()):
                self.logger.warning("Permission denied - insufficient role", 
                                  role=user_role.value, 
                                  permission=required_permission.value)
                return False
            
            # Verificaciones contextuales si se proporciona contexto
            if context:
                # Verificar restricciones de tiempo
                if not self._check_time_restrictions(user_role, context.time_of_access):
                    self.logger.warning("Permission denied - time restriction", 
                                      role=user_role.value, 
                                      time=context.time_of_access.isoformat())
                    return False
                
                # Verificar restricciones de IP
                if not self._check_ip_restrictions(user_role, context.ip_address):
                    self.logger.warning("Permission denied - IP restriction", 
                                      role=user_role.value, 
                                      ip=context.ip_address)
                    return False
                
                # Verificar acceso a recursos específicos
                if not self._check_resource_access(user_role, context):
                    self.logger.warning("Permission denied - resource restriction", 
                                      role=user_role.value, 
                                      resource=context.resource_type)
                    return False
            
            # Log acceso exitoso
            self.logger.info("Permission granted", 
                           role=user_role.value, 
                           permission=required_permission.value,
                           user_id=context.user_id if context else None)
            return True
            
        except Exception as e:
            self.logger.error("Permission check failed", 
                            role=user_role.value, 
                            permission=required_permission.value, 
                            error=str(e))
            return False
    
    def _check_time_restrictions(self, role: Role, access_time: datetime) -> bool:
        """Verificar restricciones de horario"""
        if role not in self.time_restrictions:
            return True
        
        restrictions = self.time_restrictions[role]
        hour = access_time.hour
        
        return restrictions['start_hour'] <= hour <= restrictions['end_hour']
    
    def _check_ip_restrictions(self, role: Role, ip_address: str) -> bool:
        """Verificar restricciones de IP"""
        if role not in self.ip_restrictions:
            return True
        
        # Implementar verificación de rangos CIDR
        import ipaddress
        
        allowed_networks = self.ip_restrictions[role]
        client_ip = ipaddress.ip_address(ip_address)
        
        for network_str in allowed_networks:
            network = ipaddress.ip_network(network_str)
            if client_ip in network:
                return True
        
        return False
    
    def _check_resource_access(self, role: Role, context: AccessContext) -> bool:
        """Verificar acceso a recursos específicos"""
        # Implementar lógica específica de recursos
        # Por ejemplo, analistas solo pueden ver sus propias investigaciones
        
        if role == Role.ANALYST and context.resource_type == "investigation":
            # Verificar si el analista es el propietario de la investigación
            owner_key = f"investigation_owner:{context.resource_id}"
            owner = self.redis.get(owner_key)
            return owner and owner.decode() == context.user_id
        
        return True
    
    def assign_role(self, user_id: str, role: Role, assigned_by: str) -> bool:
        """Asignar rol a usuario con auditoría"""
        try:
            # Verificar que quien asigna tiene permisos
            assignor_role = self.get_user_role(assigned_by)
            if assignor_role != Role.ADMIN and assignor_role != Role.SYSTEM_ADMIN:
                self.logger.warning("Unauthorized role assignment attempt", 
                                  assigned_by=assigned_by, target_user=user_id)
                return False
            
            # Asignar rol
            self.redis.set(f"user_role:{user_id}", role.value)
            
            # Auditar asignación
            audit_record = {
                'action': 'role_assignment',
                'target_user': user_id,
                'new_role': role.value,
                'assigned_by': assigned_by,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.redis.lpush("role_audit_log", json.dumps(audit_record))
            
            self.logger.info("Role assigned successfully", 
                           user_id=user_id, role=role.value, assigned_by=assigned_by)
            return True
            
        except Exception as e:
            self.logger.error("Role assignment failed", 
                            user_id=user_id, role=role.value, error=str(e))
            return False
    
    def get_user_role(self, user_id: str) -> Optional[Role]:
        """Obtener rol del usuario"""
        role_str = self.redis.get(f"user_role:{user_id}")
        if role_str:
            try:
                return Role(role_str.decode())
            except ValueError:
                self.logger.warning("Invalid role found for user", user_id=user_id)
        return None

# =============================================================================
# 5. IMPLEMENTACIÓN COMPLETA GDPR
# =============================================================================

class DataSubjectRights(Enum):
    """Derechos del titular de datos bajo GDPR"""
    ACCESS = "access"              # Art. 15
    RECTIFICATION = "rectification"  # Art. 16
    ERASURE = "erasure"            # Art. 17 (Right to be forgotten)
    PORTABILITY = "portability"    # Art. 20
    RESTRICTION = "restriction"    # Art. 18
    OBJECTION = "objection"        # Art. 21

class LegalBasis(Enum):
    """Bases legales para procesamiento de datos"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class DataCategory(Enum):
    """Categorías de datos personales"""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_DATA = "sensitive_data"
    FINANCIAL_DATA = "financial_data"
    BEHAVIORAL_DATA = "behavioral_data"
    TRANSACTION_DATA = "transaction_data"

@dataclass
class DataProcessingRecord:
    """Registro de actividad de procesamiento"""
    processing_id: str
    user_id: str
    data_categories: List[DataCategory]
    purpose: str
    legal_basis: LegalBasis
    timestamp: datetime
    retention_period: timedelta
    third_parties: List[str] = field(default_factory=list)
    automated_decision_making: bool = False

@dataclass
class ConsentRecord:
    """Registro de consentimiento"""
    consent_id: str
    user_id: str
    purposes: List[str]
    consent_given: bool
    timestamp: datetime
    ip_address: str
    user_agent_hash: str
    withdrawal_date: Optional[datetime] = None

class GDPRComplianceManager:
    """Gestión completa de cumplimiento GDPR"""
    
    def __init__(self, redis_client: redis.Redis, secret_manager: SecretManager):
        self.redis = redis_client
        self.secret_manager = secret_manager
        self.logger = structlog.get_logger("gdpr_compliance")
        
        # Configuración de retención por tipo de dato
        self.retention_policies = {
            DataCategory.PERSONAL_DATA: timedelta(days=2555),      # 7 años
            DataCategory.FINANCIAL_DATA: timedelta(days=2555),     # 7 años (requerimiento regulatorio)
            DataCategory.TRANSACTION_DATA: timedelta(days=2555),   # 7 años
            DataCategory.BEHAVIORAL_DATA: timedelta(days=730),     # 2 años
            DataCategory.SENSITIVE_DATA: timedelta(days=365)       # 1 año
        }
    
    def record_data_processing(self, user_id: str, data_categories: List[DataCategory],
                              purpose: str, legal_basis: LegalBasis,
                              automated_decision: bool = False,
                              third_parties: List[str] = None) -> str:
        """Registrar actividad de procesamiento de datos"""
        processing_id = f"proc_{secrets.token_hex(16)}"
        
        # Calcular período de retención (tomar el máximo para las categorías)
        retention_period = max(
            self.retention_policies.get(cat, timedelta(days=365)) 
            for cat in data_categories
        )
        
        record = DataProcessingRecord(
            processing_id=processing_id,
            user_id=user_id,
            data_categories=data_categories,
            purpose=purpose,
            legal_basis=legal_basis,
            timestamp=datetime.utcnow(),
            retention_period=retention_period,
            third_parties=third_parties or [],
            automated_decision_making=automated_decision
        )
        
        # Almacenar registro
        record_data = {
            'processing_id': processing_id,
            'user_id': user_id,
            'data_categories': [cat.value for cat in data_categories],
            'purpose': purpose,
            'legal_basis': legal_basis.value,
            'timestamp': record.timestamp.isoformat(),
            'retention_period_days': record.retention_period.days,
            'third_parties': record.third_parties,
            'automated_decision_making': automated_decision,
            'expires_at': (record.timestamp + retention_period).isoformat()
        }
        
        # Almacenar en Redis con TTL automático
        self.redis.setex(
            f"processing_record:{processing_id}",
            int(retention_period.total_seconds()),
            json.dumps(record_data)
        )
        
        # Indexar por usuario para búsquedas
        self.redis.sadd(f"user_processing:{user_id}", processing_id)
        
        self.logger.info("Data processing recorded", 
                        processing_id=processing_id, 
                        user_id=user_id, 
                        purpose=purpose)
        
        return processing_id
    
    def record_consent(self, user_id: str, purposes: List[str], 
                      consent_given: bool, ip_address: str, 
                      user_agent: str) -> str:
        """Registrar consentimiento del usuario"""
        consent_id = f"consent_{secrets.token_hex(16)}"
        user_agent_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]
        
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            purposes=purposes,
            consent_given=consent_given,
            timestamp=datetime.utcnow(),
            ip_address=hashlib.sha256(ip_address.encode()).hexdigest()[:16],  # IP hasheada
            user_agent_hash=user_agent_hash
        )
        
        consent_data = {
            'consent_id': consent_id,
            'user_id': user_id,
            'purposes': purposes,
            'consent_given': consent_given,
            'timestamp': consent.timestamp.isoformat(),
            'ip_address_hash': consent.ip_address,
            'user_agent_hash': user_agent_hash,
            'withdrawal_date': None
        }
        
        # Almacenar permanentemente (7 años para auditoría)
        self.redis.setex(
            f"consent_record:{consent_id}",
            int(timedelta(days=2555).total_seconds()),
            json.dumps(consent_data)
        )
        
        # Indexar por usuario
        self.redis.sadd(f"user_consents:{user_id}", consent_id)
        
        # Actualizar estado actual de consentimiento
        current_consent = {
            'purposes': {purpose: consent_given for purpose in purposes},
            'last_updated': consent.timestamp.isoformat(),
            'latest_consent_id': consent_id
        }
        
        existing_consent = self.redis.get(f"current_consent:{user_id}")
        if existing_consent:
            existing_data = json.loads(existing_consent)
            existing_data['purposes'].update(current_consent['purposes'])
            existing_data['last_updated'] = current_consent['last_updated']
            current_consent = existing_data
        
        self.redis.set(f"current_consent:{user_id}", json.dumps(current_consent))
        
        self.logger.info("Consent recorded", 
                        consent_id=consent_id, 
                        user_id=user_id, 
                        consent_given=consent_given)
        
        return consent_id
    
    def handle_access_request(self, user_id: str, requester_id: str) -> Dict[str, Any]:
        """Manejar solicitud de acceso a datos (Art. 15 GDPR)"""
        try:
            # Verificar autorización del solicitante
            if user_id != requester_id:
                # Solo compliance officers pueden acceder a datos de otros usuarios
                requester_role = self._get_user_role(requester_id)
                if requester_role != Role.COMPLIANCE_OFFICER:
                    raise PermissionError("Unauthorized access request")
            
            # Recopilar todos los datos del usuario
            access_data = {
                'request_id': f"access_{secrets.token_hex(16)}",
                'user_id': user_id,
                'requested_by': requester_id,
                'request_timestamp': datetime.utcnow().isoformat(),
                'data_categories': {},
                'processing_activities': [],
                'consents': [],
                'retention_info': {},
                'third_party_sharing': []
            }
            
            # Obtener registros de procesamiento
            processing_ids = self.redis.smembers(f"user_processing:{user_id}")
            for proc_id in processing_ids:
                proc_data = self.redis.get(f"processing_record:{proc_id.decode()}")
                if proc_data:
                    access_data['processing_activities'].append(json.loads(proc_data))
            
            # Obtener registros de consentimiento
            consent_ids = self.redis.smembers(f"user_consents:{user_id}")
            for consent_id in consent_ids:
                consent_data = self.redis.get(f"consent_record:{consent_id.decode()}")
                if consent_data:
                    access_data['consents'].append(json.loads(consent_data))
            
            # Obtener datos personales (implementar según su esquema de datos)
            personal_data = self._get_user_personal_data(user_id)
            access_data['personal_data'] = personal_data
            
            # Información de retención
            access_data['retention_info'] = {
                category.value: self.retention_policies[category].days
                for category in DataCategory
            }
            
            # Log de la solicitud
            self.logger.info("Access request processed", 
                           user_id=user_id, 
                           requester_id=requester_id,
                           request_id=access_data['request_id'])
            
            return access_data
            
        except Exception as e:
            self.logger.error("Access request failed", 
                            user_id=user_id, 
                            requester_id=requester_id, 
                            error=str(e))
            raise
    
    def handle_erasure_request(self, user_id: str, requester_id: str,
                              reason: str = "user_request") -> Dict[str, Any]:
        """Manejar solicitud de borrado (Art. 17 GDPR)"""
        try:
            # Verificar si hay base legal para retener datos
            legal_retention = self._check_legal_retention_requirements(user_id)
            if legal_retention['must_retain']:
                return {
                    'erasure_id': f"erasure_{secrets.token_hex(16)}",
                    'status': 'denied',
                    'reason': 'Legal retention requirements',
                    'details': legal_retention['reasons'],
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            erasure_id = f"erasure_{secrets.token_hex(16)}"
            
            # Anonimizar/eliminar datos
            erasure_results = {
                'erasure_id': erasure_id,
                'user_id': user_id,
                'requested_by': requester_id,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'completed',
                'actions_taken': []
            }
            
            # Eliminar datos personales identificables
            deleted_records = self._anonymize_user_data(user_id)
            erasure_results['actions_taken'].extend(deleted_records)
            
            # Mantener solo datos necesarios para compliance (anonimizados)
            compliance_data = self._create_compliance_record(user_id, erasure_id)
            erasure_results['compliance_record_id'] = compliance_data['record_id']
            
            # Invalidar sesiones activas
            self._invalidate_user_sessions(user_id)
            erasure_results['actions_taken'].append('Active sessions invalidated')
            
            # Registro de auditoría del borrado
            audit_record = {
                'action': 'data_erasure',
                'user_id': user_id,
                'erasure_id': erasure_id,
                'requested_by': requester_id,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
                'records_affected': len(deleted_records)
            }
            
            self.redis.lpush("erasure_audit_log", json.dumps(audit_record))
            
            self.logger.info("Erasure request completed", 
                           user_id=user_id, 
                           erasure_id=erasure_id,
                           records_affected=len(deleted_records))
            
            return erasure_results
            
        except Exception as e:
            self.logger.error("Erasure request failed", 
                            user_id=user_id, 
                            error=str(e))
            raise
    
    def handle_portability_request(self, user_id: str, requester_id: str,
                                  format_type: str = "json") -> bytes:
        """Manejar solicitud de portabilidad (Art. 20 GDPR)"""
        try:
            # Obtener datos portables (solo datos proporcionados por el usuario)
            portable_data = {
                'export_id': f"export_{secrets.token_hex(16)}",
                'user_id': user_id,
                'export_timestamp': datetime.utcnow().isoformat(),
                'format': format_type,
                'data': {}
            }
            
            # Datos de perfil del usuario
            user_profile = self._get_portable_user_profile(user_id)
            if user_profile:
                portable_data['data']['profile'] = user_profile
            
            # Preferencias y configuraciones
            user_preferences = self._get_user_preferences(user_id)
            if user_preferences:
                portable_data['data']['preferences'] = user_preferences
            
            # Datos de transacciones (solo los proporcionados directamente)
            user_transactions = self._get_portable_transactions(user_id)
            if user_transactions:
                portable_data['data']['transactions'] = user_transactions
            
            # Convertir a formato solicitado
            if format_type.lower() == "json":
                export_data = json.dumps(portable_data, indent=2, ensure_ascii=False)
                content_type = "application/json"
            elif format_type.lower() == "csv":
                export_data = self._convert_to_csv(portable_data)
                content_type = "text/csv"
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Log de la exportación
            export_record = {
                'export_id': portable_data['export_id'],
                'user_id': user_id,
                'requested_by': requester_id,
                'format': format_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data_size_bytes': len(export_data.encode())
            }
            
            self.redis.lpush("portability_audit_log", json.dumps(export_record))
            
            self.logger.info("Portability request completed", 
                           user_id=user_id,
                           export_id=portable_data['export_id'],
                           format=format_type)
            
            return export_data.encode()
            
        except Exception as e:
            self.logger.error("Portability request failed", 
                            user_id=user_id, 
                            error=str(e))
            raise
    
    def _get_user_role(self, user_id: str) -> Optional[Role]:
        """Obtener rol del usuario (helper method)"""
        role_str = self.redis.get(f"user_role:{user_id}")
        if role_str:
            try:
                return Role(role_str.decode())
            except ValueError:
                return None
        return None
    
    def _check_legal_retention_requirements(self, user_id: str) -> Dict[str, Any]:
        """Verificar requerimientos legales de retención"""
        # Verificar si hay investigaciones activas, requerimientos regulatorios, etc.
        # Placeholder - implementar lógica específica
        return {
            'must_retain': False,
            'reasons': []
        }
    
    def _get_user_personal_data(self, user_id: str) -> Dict[str, Any]:
        """Obtener datos personales del usuario"""
        # Placeholder - implementar según su esquema de datos
        return {}
    
    def _anonymize_user_data(self, user_id: str) -> List[str]:
        """Anonimizar datos del usuario"""
        # Placeholder - implementar lógica de anonimización
        actions = [
            f"Anonymized personal identifiers for user {user_id}",
            f"Removed PII from transaction records",
            f"Anonymized behavioral data"
        ]
        return actions
    
    def _create_compliance_record(self, user_id: str, erasure_id: str) -> Dict[str, Any]:
        """Crear registro de compliance para auditoría"""
        record_id = f"compliance_{secrets.token_hex(16)}"
        
        compliance_record = {
            'record_id': record_id,
            'original_user_id_hash': hashlib.sha256(user_id.encode()).hexdigest(),
            'erasure_id': erasure_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data_categories_processed': [cat.value for cat in DataCategory],
            'retention_period_compliance': True
        }
        
        # Almacenar registro de compliance (permanente para auditoría)
        self.redis.set(f"compliance_record:{record_id}", 
                      json.dumps(compliance_record))
        
        return compliance_record
    
    def _invalidate_user_sessions(self, user_id: str):
        """Invalidar todas las sesiones activas del usuario"""
        # Buscar y eliminar todas las sesiones del usuario
        # Placeholder - implementar según su sistema de sesiones
        pass
    
    def _get_portable_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Obtener datos de perfil portables"""
        # Placeholder - implementar según su esquema
        return {}
    
    def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Obtener preferencias del usuario"""
        # Placeholder - implementar según su esquema
        return {}
    
    def _get_portable_transactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener transacciones portables del usuario"""
        # Placeholder - implementar según su esquema
        return []
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convertir datos a formato CSV"""
        # Placeholder - implementar conversión a CSV
        import csv
        import io
        
        output = io.StringIO()
        # Implementar lógica de conversión específica
        return output.getvalue()

# =============================================================================
# 6. RATE LIMITING Y PROTECCIÓN DDOS AVANZADA
# =============================================================================

class RateLimitStrategy(Enum):
    """Estrategias de rate limiting"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

class ThreatMitigation(Enum):
    """Acciones de mitigación de amenazas"""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_BAN = "temporary_ban"
    PERMANENT_BAN = "permanent_ban"
    CAPTCHA_CHALLENGE = "captcha_challenge"

@dataclass
class RateLimitRule:
    """Regla de rate limiting"""
    name: str
    strategy: RateLimitStrategy
    requests_per_window: int
    window_size_seconds: int
    burst_capacity: Optional[int] = None
    mitigation: ThreatMitigation = ThreatMitigation.RATE_LIMIT

class AdvancedRateLimiter:
    """Rate limiter avanzado con protección DDoS"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = structlog.get_logger("rate_limiter")
        
        # Reglas de rate limiting por endpoint/acción
        self.rules = {
            'login': RateLimitRule(
                name='login_attempts',
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                requests_per_window=5,
                window_size_seconds=300,  # 5 minutos
                mitigation=ThreatMitigation.TEMPORARY_BAN
            ),
            'api_general': RateLimitRule(
                name='api_general',
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                requests_per_window=1000,
                window_size_seconds=3600,  # 1 hora
                burst_capacity=100,
                mitigation=ThreatMitigation.RATE_LIMIT
            ),
            'fraud_detection': RateLimitRule(
                name='fraud_detection',
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                requests_per_window=10000,
                window_size_seconds=60,  # 1 minuto
                mitigation=ThreatMitigation.RATE_LIMIT
            ),
            'data_export': RateLimitRule(
                name='data_export',
                strategy=RateLimitStrategy.FIXED_WINDOW,
                requests_per_window=5,
                window_size_seconds=86400,  # 1 día
                mitigation=ThreatMitigation.TEMPORARY_BAN
            )
        }
        
        # Configuración de detección de amenazas
        self.threat_detection = {
            'suspicious_patterns': {
                'requests_from_single_ip_threshold': 10000,  # requests/hour
                'failed_auth_threshold': 20,  # failed attempts/hour
                'geographic_anomaly_threshold': 5  # different countries/hour
            }
        }
    
    def check_rate_limit(self, identifier: str, rule_name: str,
                        context: Optional[Dict] = None) -> Dict[str, Any]:
        """Verificar rate limit con contexto de amenazas"""
        if rule_name not in self.rules:
            return {'allowed': True, 'message': 'No rule defined'}
        
        rule = self.rules[rule_name]
        
        # Verificar rate limit según estrategia
        if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            result = self._check_sliding_window(identifier, rule)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            result = self._check_token_bucket(identifier, rule)
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            result = self._check_fixed_window(identifier, rule)
        else:
            result = {'allowed': True, 'remaining': rule.requests_per_window}
        
        # Análisis de amenazas si está activado
        if context:
            threat_analysis = self._analyze_threats(identifier, context)
            result['threat_level'] = threat_analysis['level']
            result['threat_indicators'] = threat_analysis['indicators']
            
            # Aplicar mitigaciones adicionales si hay amenazas
            if threat_analysis['level'] in ['high', 'critical']:
                result['allowed'] = False
                result['mitigation'] = self._apply_threat_mitigation(
                    identifier, threat_analysis, rule
                )
        
        # Log del resultado
        self.logger.info("Rate limit check", 
                        identifier=identifier,
                        rule=rule_name,
                        allowed=result['allowed'],
                        remaining=result.get('remaining', 0))
        
        return result
    
    def _check_sliding_window(self, identifier: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Implementar sliding window rate limiting"""
        now = time.time()
        window_start = now - rule.window_size_seconds
        
        key = f"rate_limit:sliding:{rule.name}:{identifier}"
        
        # Limpiar entradas antiguas y contar requests en ventana
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, rule.window_size_seconds)
        
        results = pipe.execute()
        current_requests = results[1]
        
        allowed = current_requests < rule.requests_per_window
        remaining = max(0, rule.requests_per_window - current_requests - 1)
        
        return {
            'allowed': allowed,
            'remaining': remaining,
            'reset_time': now + rule.window_size_seconds,
            'strategy': rule.strategy.value
        }
    
    def _check_token_bucket(self, identifier: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Implementar token bucket rate limiting"""
        key = f"rate_limit:bucket:{rule.name}:{identifier}"
        now = time.time()
        
        # Obtener estado actual del bucket
        bucket_data = self.redis.hmget(key, 'tokens', 'last_refill')
        
        if bucket_data[0] is None:
            # Inicializar bucket
            tokens = rule.burst_capacity or rule.requests_per_window
            last_refill = now
        else:
            tokens = float(bucket_data[0])
            last_refill = float(bucket_data[1])
            
            # Calcular tokens a añadir
            time_passed = now - last_refill
            tokens_to_add = (time_passed / rule.window_size_seconds) * rule.requests_per_window
            tokens = min(rule.burst_capacity or rule.requests_per_window, 
                        tokens + tokens_to_add)
        
        # Verificar si hay tokens disponibles
        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False
        
        # Actualizar estado
        self.redis.hmset(key, {
            'tokens': tokens,
            'last_refill': now
        })
        self.redis.expire(key, rule.window_size_seconds * 2)
        
        return {
            'allowed': allowed,
            'remaining': int(tokens),
            'refill_rate': rule.requests_per_window / rule.window_size_seconds,
            'strategy': rule.strategy.value
        }
    
    def _check_fixed_window(self, identifier: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Implementar fixed window rate limiting"""
        now = time.time()
        window_start = int(now // rule.window_size_seconds) * rule.window_size_seconds
        
        key = f"rate_limit:fixed:{rule.name}:{identifier}:{window_start}"
        
        current_requests = self.redis.incr(key)
        if current_requests == 1:
            self.redis.expire(key, rule.window_size_seconds)
        
        allowed = current_requests <= rule.requests_per_window
        remaining = max(0, rule.requests_per_window - current_requests)
        
        return {
            'allowed': allowed,
            'remaining': remaining,
            'reset_time': window_start + rule.window_size_seconds,
            'strategy': rule.strategy.value
        }
    
    def _analyze_threats(self, identifier: str, context: Dict) -> Dict[str, Any]:
        """Analizar indicadores de amenazas"""
        indicators = []
        threat_score = 0
        
        # Analizar patrones sospechosos
        if 'ip_address' in context:
            ip_requests = self._get_ip_request_count(context['ip_address'])
            if ip_requests > self.threat_detection['suspicious_patterns']['requests_from_single_ip_threshold']:
                indicators.append(f"High request volume from IP: {ip_requests}/hour")
                threat_score += 30
        
        if 'failed_auth_count' in context:
            failed_auths = context['failed_auth_count']
            if failed_auths > self.threat_detection['suspicious_patterns']['failed_auth_threshold']:
                indicators.append(f"Multiple failed authentications: {failed_auths}")
                threat_score += 40
        
        if 'geographic_locations' in context:
            locations = len(context['geographic_locations'])
            if locations > self.threat_detection['suspicious_patterns']['geographic_anomaly_threshold']:
                indicators.append(f"Requests from multiple locations: {locations} countries")
                threat_score += 25
        
        # Determinar nivel de amenaza
        if threat_score >= 70:
            level = 'critical'
        elif threat_score >= 50:
            level = 'high'
        elif threat_score >= 25:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': threat_score,
            'indicators': indicators
        }
    
    def _apply_threat_mitigation(self, identifier: str, threat_analysis: Dict,
                               rule: RateLimitRule) -> Dict[str, Any]:
        """Aplicar medidas de mitigación de amenazas"""
        mitigation_actions = []
        
        if threat_analysis['level'] == 'critical':
            # Ban temporal de 24 horas
            ban_key = f"ban:critical:{identifier}"
            self.redis.setex(ban_key, 86400, json.dumps({
                'reason': 'Critical threat detected',
                'indicators': threat_analysis['indicators'],
                'timestamp':