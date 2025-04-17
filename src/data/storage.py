import logging
import os
import json
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DataStorage:
    """
    Storage system for bot detection session data.
    
    This class handles persistence of session data, analysis results,
    and provides utilities for querying and analyzing historical data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data storage system.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.storage_type = config.get("type", "sqlite")
        self.db_path = config.get("path", "data/sessions.db")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize storage
        self._initialize_storage()
        
        logger.info(f"Initialized DataStorage with type: {self.storage_type}, path: {self.db_path}")
    
    def _initialize_storage(self):
        """Initialize the storage system based on type"""
        if self.storage_type == "sqlite":
            self._initialize_sqlite()
        elif self.storage_type == "postgresql":
            self._initialize_postgresql()
        elif self.storage_type == "json":
            self._initialize_json()
        else:
            logger.warning(f"Unknown storage type: {self.storage_type}, falling back to sqlite")
            self.storage_type = "sqlite"
            self._initialize_sqlite()
    
    def _initialize_sqlite(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                ip_address TEXT,
                user_agent TEXT,
                is_bot INTEGER,
                confidence REAL,
                data TEXT
            )
            ''')
            
            # Create analysis_results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                session_id TEXT,
                expert_name TEXT,
                probability REAL,
                confidence REAL,
                explanation TEXT,
                features TEXT,
                PRIMARY KEY (session_id, expert_name),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            # Create model_metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                timestamp INTEGER,
                model_version TEXT,
                metric_name TEXT,
                metric_value REAL,
                PRIMARY KEY (timestamp, model_version, metric_name)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("SQLite database initialized")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {str(e)}", exc_info=True)
    
    def _initialize_postgresql(self):
        """Initialize PostgreSQL database (placeholder)"""
        logger.warning("PostgreSQL support not implemented, falling back to SQLite")
        self.storage_type = "sqlite"
        self._initialize_sqlite()
    
    def _initialize_json(self):
        """Initialize JSON storage"""
        try:
            # Check if JSON files exist, create if not
            sessions_path = os.path.join(os.path.dirname(self.db_path), "sessions.json")
            results_path = os.path.join(os.path.dirname(self.db_path), "analysis_results.json")
            metrics_path = os.path.join(os.path.dirname(self.db_path), "model_metrics.json")
            
            if not os.path.exists(sessions_path):
                with open(sessions_path, 'w') as f:
                    json.dump([], f)
            
            if not os.path.exists(results_path):
                with open(results_path, 'w') as f:
                    json.dump([], f)
            
            if not os.path.exists(metrics_path):
                with open(metrics_path, 'w') as f:
                    json.dump([], f)
            
            logger.info("JSON storage initialized")
        except Exception as e:
            logger.error(f"Error initializing JSON storage: {str(e)}", exc_info=True)
    
    def save_session(self, session_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> bool:
        """
        Save a session and its analysis result.
        
        Args:
            session_data: Dictionary containing session data
            analysis_result: Dictionary containing analysis result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_id = session_data.get("sessionId")
            if not session_id:
                logger.error("Session data missing sessionId")
                return False
            
            if self.storage_type == "sqlite":
                return self._save_session_sqlite(session_data, analysis_result)
            elif self.storage_type == "postgresql":
                return self._save_session_postgresql(session_data, analysis_result)
            elif self.storage_type == "json":
                return self._save_session_json(session_data, analysis_result)
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")
                return False
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}", exc_info=True)
            return False
    
    def _save_session_sqlite(self, session_data: Dict[str, Any], 
                           analysis_result: Dict[str, Any]) -> bool:
        """Save session to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract basic session info
            session_id = session_data.get("sessionId")
            timestamp = int(session_data.get("startTime", datetime.now().timestamp()))
            user_agent = session_data.get("browserFingerprint", {}).get("userAgent", "")
            ip_address = session_data.get("webrtcInfo", {}).get("ipAddress", "")
            
            # Extract analysis result
            is_bot = 1 if analysis_result.get("is_bot", False) else 0
            confidence = analysis_result.get("confidence", 0.0)
            
            # Save session data as JSON
            session_json = json.dumps(session_data)
            
            # Insert into sessions table
            cursor.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, timestamp, ip_address, user_agent, is_bot, confidence, session_json)
            )
            
            # Save expert analysis results
            expert_results = analysis_result.get("expert_results", {})
            for expert_name, result in expert_results.items():
                probability = result.get("probability", 0.0)
                expert_confidence = result.get("confidence", 0.0)
                explanation = json.dumps(result.get("explanation", []))
                features = json.dumps(result.get("features", {}))
                
                cursor.execute(
                    "INSERT OR REPLACE INTO analysis_results VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, expert_name, probability, expert_confidence, explanation, features)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved session {session_id} to SQLite database")
            return True
        except Exception as e:
            logger.error(f"Error saving session to SQLite: {str(e)}", exc_info=True)
            return False
    
    def _save_session_postgresql(self, session_data: Dict[str, Any], 
                               analysis_result: Dict[str, Any]) -> bool:
        """Save session to PostgreSQL database (placeholder)"""
        logger.warning("PostgreSQL support not implemented, falling back to SQLite")
        return self._save_session_sqlite(session_data, analysis_result)
    
    def _save_session_json(self, session_data: Dict[str, Any], 
                         analysis_result: Dict[str, Any]) -> bool:
        """Save session to JSON files"""
        try:
            # Load existing sessions
            sessions_path = os.path.join(os.path.dirname(self.db_path), "sessions.json")
            results_path = os.path.join(os.path.dirname(self.db_path), "analysis_results.json")
            
            with open(sessions_path, 'r') as f:
                sessions = json.load(f)
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Add session ID to analysis result
            analysis_result["session_id"] = session_data.get("sessionId")
            
            # Check if session already exists
            session_ids = [s.get("sessionId") for s in sessions]
            result_ids = [r.get("session_id") for r in results]
            
            if session_data.get("sessionId") in session_ids:
                # Update existing session
                idx = session_ids.index(session_data.get("sessionId"))
                sessions[idx] = session_data
            else:
                # Add new session
                sessions.append(session_data)
            
            if analysis_result.get("session_id") in result_ids:
                # Update existing result
                idx = result_ids.index(analysis_result.get("session_id"))
                results[idx] = analysis_result
            else:
                # Add new result
                results.append(analysis_result)
            
            # Save updated data
            with open(sessions_path, 'w') as f:
                json.dump(sessions, f)
            
            with open(results_path, 'w') as f:
                json.dump(results, f)
            
            logger.info(f"Saved session {session_data.get('sessionId')} to JSON storage")
            return True
        except Exception as e:
            logger.error(f"Error saving session to JSON: {str(e)}", exc_info=True)
            return False
    
    def load_session(self, session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Load a session and its analysis result.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            Tuple containing (session_data, analysis_result), both None if not found
        """
        try:
            if self.storage_type == "sqlite":
                return self._load_session_sqlite(session_id)
            elif self.storage_type == "postgresql":
                return self._load_session_postgresql(session_id)
            elif self.storage_type == "json":
                return self._load_session_json(session_id)
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")
                return None, None
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}", exc_info=True)
            return None, None
    
    def _load_session_sqlite(self, session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load session from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            # Load session data
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()
            
            if not session_row:
                logger.warning(f"Session {session_id} not found")
                return None, None
            
            # Parse session data
            session_data = json.loads(session_row["data"])
            
            # Load analysis results
            cursor.execute("SELECT * FROM analysis_results WHERE session_id = ?", (session_id,))
            result_rows = cursor.fetchall()
            
            # Combine results into analysis result
            expert_results = {}
            for row in result_rows:
                expert_name = row["expert_name"]
                expert_results[expert_name] = {
                    "probability": row["probability"],
                    "confidence": row["confidence"],
                    "explanation": json.loads(row["explanation"]),
                    "features": json.loads(row["features"])
                }
            
            analysis_result = {
                "session_id": session_id,
                "is_bot": bool(session_row["is_bot"]),
                "probability": max([r.get("probability", 0.0) for r in expert_results.values()], default=0.0),
                "confidence": session_row["confidence"],
                "expert_results": expert_results
            }
            
            conn.close()
            
            logger.info(f"Loaded session {session_id} from SQLite database")
            return session_data, analysis_result
        except Exception as e:
            logger.error(f"Error loading session from SQLite: {str(e)}", exc_info=True)
            return None, None
    
    def _load_session_postgresql(self, session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load session from PostgreSQL database (placeholder)"""
        logger.warning("PostgreSQL support not implemented, falling back to SQLite")
        return self._load_session_sqlite(session_id)
    
    def _load_session_json(self, session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load session from JSON files"""
        try:
            # Load existing sessions
            sessions_path = os.path.join(os.path.dirname(self.db_path), "sessions.json")
            results_path = os.path.join(os.path.dirname(self.db_path), "analysis_results.json")
            
            with open(sessions_path, 'r') as f:
                sessions = json.load(f)
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Find session and result
            session_data = next((s for s in sessions if s.get("sessionId") == session_id), None)
            analysis_result = next((r for r in results if r.get("session_id") == session_id), None)
            
            if not session_data:
                logger.warning(f"Session {session_id} not found")
                return None, None
            
            logger.info(f"Loaded session {session_id} from JSON storage")
            return session_data, analysis_result
        except Exception as e:
            logger.error(f"Error loading session from JSON: {str(e)}", exc_info=True)
            return None, None
    
    def save_model_metrics(self, model_version: str, metrics: Dict[str, float]) -> bool:
        """
        Save model performance metrics.
        
        Args:
            model_version: Version identifier for the model
            metrics: Dictionary mapping metric names to values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            timestamp = int(datetime.now().timestamp())
            
            if self.storage_type == "sqlite":
                return self._save_metrics_sqlite(timestamp, model_version, metrics)
            elif self.storage_type == "postgresql":
                return self._save_metrics_postgresql(timestamp, model_version, metrics)
            elif self.storage_type == "json":
                return self._save_metrics_json(timestamp, model_version, metrics)
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")
                return False
        except Exception as e:
            logger.error(f"Error saving model metrics: {str(e)}", exc_info=True)
            return False
    
    def _save_metrics_sqlite(self, timestamp: int, model_version: str, 
                           metrics: Dict[str, float]) -> bool:
        """Save metrics to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO model_metrics VALUES (?, ?, ?, ?)",
                    (timestamp, model_version, metric_name, metric_value)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved metrics for model {model_version} to SQLite database")
            return True
        except Exception as e:
            logger.error(f"Error saving metrics to SQLite: {str(e)}", exc_info=True)
            return False
    
    def _save_metrics_postgresql(self, timestamp: int, model_version: str, 
                               metrics: Dict[str, float]) -> bool:
        """Save metrics to PostgreSQL database (placeholder)"""
        logger.warning("PostgreSQL support not implemented, falling back to SQLite")
        return self._save_metrics_sqlite(timestamp, model_version, metrics)
    
    def _save_metrics_json(self, timestamp: int, model_version: str, 
                         metrics: Dict[str, float]) -> bool:
        """Save metrics to JSON files"""
        try:
            # Load existing metrics
            metrics_path = os.path.join(os.path.dirname(self.db_path), "model_metrics.json")
            
            with open(metrics_path, 'r') as f:
                all_metrics = json.load(f)
            
            # Create metric entries
            metric_entries = []
            for metric_name, metric_value in metrics.items():
                entry = {
                    "timestamp": timestamp,
                    "model_version": model_version,
                    "metric_name": metric_name,
                    "metric_value": metric_value
                }
                metric_entries.append(entry)
            
            # Add new metrics
            all_metrics.extend(metric_entries)
            
            # Save updated metrics
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f)
            
            logger.info(f"Saved metrics for model {model_version} to JSON storage")
            return True
        except Exception as e:
            logger.error(f"Error saving metrics to JSON: {str(e)}", exc_info=True)
            return False
    
    def get_recent_sessions(self, count: int = 100, is_bot: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get recent sessions, optionally filtered by bot status.
        
        Args:
            count: Maximum number of sessions to return
            is_bot: If provided, filter by bot status
            
        Returns:
            List of session data dictionaries
        """
        try:
            if self.storage_type == "sqlite":
                return self._get_recent_sessions_sqlite(count, is_bot)
            elif self.storage_type == "postgresql":
                return self._get_recent_sessions_postgresql(count, is_bot)
            elif self.storage_type == "json":
                return self._get_recent_sessions_json(count, is_bot)
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")
                return []
        except Exception as e:
            logger.error(f"Error getting recent sessions: {str(e)}", exc_info=True)
            return []
    
    def _get_recent_sessions_sqlite(self, count: int, is_bot: Optional[bool]) -> List[Dict[str, Any]]:
        """Get recent sessions from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM sessions"
            params = []
            
            if is_bot is not None:
                query += " WHERE is_bot = ?"
                params.append(1 if is_bot else 0)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(count)
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            sessions = []
            for row in rows:
                session_data = json.loads(row["data"])
                sessions.append(session_data)
            
            conn.close()
            
            logger.info(f"Retrieved {len(sessions)} recent sessions from SQLite database")
            return sessions
        except Exception as e:
            logger.error(f"Error getting recent sessions from SQLite: {str(e)}", exc_info=True)
            return []
    
    def _get_recent_sessions_postgresql(self, count: int, is_bot: Optional[bool]) -> List[Dict[str, Any]]:
        """Get recent sessions from PostgreSQL database (placeholder)"""
        logger.warning("PostgreSQL support not implemented, falling back to SQLite")
        return self._get_recent_sessions_sqlite(count, is_bot)
    
    def _get_recent_sessions_json(self, count: int, is_bot: Optional[bool]) -> List[Dict[str, Any]]:
        """Get recent sessions from JSON files"""
        try:
            # Load existing sessions and results
            sessions_path = os.path.join(os.path.dirname(self.db_path), "sessions.json")
            results_path = os.path.join(os.path.dirname(self.db_path), "analysis_results.json")
            
            with open(sessions_path, 'r') as f:
                sessions = json.load(f)
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Create mapping from session ID to is_bot status
            is_bot_map = {}
            for result in results:
                session_id = result.get("session_id")
                if session_id:
                    is_bot_map[session_id] = result.get("is_bot", False)
            
            # Filter and sort sessions
            filtered_sessions = []
            for session in sessions:
                session_id = session.get("sessionId")
                if not session_id:
                    continue
                
                # Filter by bot status if specified
                if is_bot is not None and session_id in is_bot_map:
                    if is_bot_map[session_id] != is_bot:
                        continue
                
                filtered_sessions.append(session)
            
            # Sort by timestamp (descending)
            filtered_sessions.sort(key=lambda s: s.get("startTime", 0), reverse=True)
            
            # Limit to specified count
            limited_sessions = filtered_sessions[:count]
            
            logger.info(f"Retrieved {len(limited_sessions)} recent sessions from JSON storage")
            return limited_sessions
        except Exception as e:
            logger.error(f"Error getting recent sessions from JSON: {str(e)}", exc_info=True)
            return []
    
    def get_model_metrics_history(self, model_version: str, metric_name: str) -> List[Tuple[int, float]]:
        """
        Get historical metrics for a specific model version and metric.
        
        Args:
            model_version: Version identifier for the model
            metric_name: Name of the metric
            
        Returns:
            List of (timestamp, value) tuples
        """
        try:
            if self.storage_type == "sqlite":
                return self._get_metrics_history_sqlite(model_version, metric_name)
            elif self.storage_type == "postgresql":
                return self._get_metrics_history_postgresql(model_version, metric_name)
            elif self.storage_type == "json":
                return self._get_metrics_history_json(model_version, metric_name)
            else:
                logger.error(f"Unknown storage type: {self.storage_type}")
                return []
        except Exception as e:
            logger.error(f"Error getting metrics history: {str(e)}", exc_info=True)
            return []
    
    def _get_metrics_history_sqlite(self, model_version: str, metric_name: str) -> List[Tuple[int, float]]:
        """Get metrics history from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(
                "SELECT timestamp, metric_value FROM model_metrics WHERE model_version = ? AND metric_name = ? ORDER BY timestamp",
                (model_version, metric_name)
            )
            rows = cursor.fetchall()
            
            conn.close()
            
            logger.info(f"Retrieved {len(rows)} metric history points from SQLite database")
            return rows
        except Exception as e:
            logger.error(f"Error getting metrics history from SQLite: {str(e)}", exc_info=True)
            return []
    
    def _get_metrics_history_postgresql(self, model_version: str, metric_name: str) -> List[Tuple[int, float]]:
        """Get metrics history from PostgreSQL database (placeholder)"""
        logger.warning("PostgreSQL support not implemented, falling back to SQLite")
        return self._get_metrics_history_sqlite(model_version, metric_name)
    
    def _get_metrics_history_json(self, model_version: str, metric_name: str) -> List[Tuple[int, float]]:
        """Get metrics history from JSON files"""
        try:
            # Load existing metrics
            metrics_path = os.path.join(os.path.dirname(self.db_path), "model_metrics.json")
            
            with open(metrics_path, 'r') as f:
                all_metrics = json.load(f)
            
            # Filter metrics
            filtered_metrics = [
                (m.get("timestamp", 0), m.get("metric_value", 0.0))
                for m in all_metrics
                if m.get("model_version") == model_version and m.get("metric_name") == metric_name
            ]
            
            # Sort by timestamp
            filtered_metrics.sort(key=lambda m: m[0])
            
            logger.info(f"Retrieved {len(filtered_metrics)} metric history points from JSON storage")
            return filtered_metrics
        except Exception as e:
            logger.error(f"Error getting metrics history from JSON: {str(e)}", exc_info=True)
            return []