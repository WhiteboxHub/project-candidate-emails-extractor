import mysql.connector
from mysql.connector import pooling
import os
import logging
from typing import Optional, List, Dict, Any, Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseClient:
    """
    Singleton Database Client handling MySQL connections and pooling.
    """
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseClient, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize the connection pool if not already initialized."""
        if self._pool is None:
            try:
                db_config = {
                    "host": os.getenv("DB_HOST", "localhost"),
                    "port": int(os.getenv("DB_PORT", 3306)),
                    "user": os.getenv("DB_USER", "root"),
                    "password": os.getenv("DB_PASSWORD", ""),
                    "database": os.getenv("DB_NAME", "automation_db"),
                }
                
                # Create a connection pool
                self._pool = mysql.connector.pooling.MySQLConnectionPool(
                    pool_name="automation_pool",
                    pool_size=5,
                    pool_reset_session=True,
                    **db_config
                )
                logger.info("Database connection pool initialized successfully.")
            except mysql.connector.Error as e:
                logger.error(f"Error initializing database pool: {e}")
                raise

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Context manager to get a connection from the pool.
        Yields a connection object.
        """
        if self._pool is None:
            self.initialize()
        
        connection = None
        try:
            connection = self._pool.get_connection()
            yield connection
        except mysql.connector.Error as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a read query and return dictionary results.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(query, params or ())
                result = cursor.fetchall()
                return result
            except mysql.connector.Error as e:
                logger.error(f"Error executing query: {query}. Error: {e}")
                raise
            finally:
                cursor.close()

    def execute_non_query(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute a write query (INSERT, UPDATE, DELETE).
        Returns the number of affected rows or last row id for inserts.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params or ())
                conn.commit()
                return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
            except mysql.connector.Error as e:
                logger.error(f"Error executing non-query: {query}. Error: {e}")
                conn.rollback()
                raise
            finally:
                cursor.close()

# Global instance accessor
def get_db_client() -> DatabaseClient:
    return DatabaseClient()
