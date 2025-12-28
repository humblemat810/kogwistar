    
#### log utils

import logging
import sqlite3
import threading
import os
import traceback
def safe_format_exception(exc: Exception, base_path: str = None):
    """Format exception with paths relative to project root."""
    if base_path is None:
        base_path = os.getcwd()  # default to current working directory

    lines = []
    for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
        if line.startswith('  File'):
            parts = line.split('"')
            if len(parts) >= 3:
                full_path = parts[1]
                try:
                    relative_path = os.path.relpath(full_path, base_path)
                    line = line.replace(full_path, relative_path)
                except ValueError:
                    # relpath failed (different drive?), keep original
                    pass
        lines.append(line)
    
    return ''.join(lines)

def trace_logger_hierarchy(logger):
    while logger:
        print(f"Logger Name: {logger.name}")
        print(f"  Level: {logging.getLevelName(logger.level)}")
        print(f"  Handlers: {logger.handlers}")
        print(f"  Propagate: {logger.propagate}")
        print("-" * 40)
        logger = logger.parent

class SQLiteHandler(logging.Handler):
    """
    Custom logging handler that writes log records to an SQLite database,
    including filename and line number information.
    """

    def __init__(self, db_path):
        """
        Initializes the handler with the database path.
        Ensures the log table exists.
        """
        super().__init__()
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
        # Set up a formatter to format the log records
        self.formatter = logging.Formatter('%(asctime)s', '%Y-%m-%d %H:%M:%S')

    def _initialize_database(self):
        """
        Creates the logs table if it doesn't already exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    module TEXT,
                    filename TEXT,
                    line_number INTEGER,
                    message TEXT
                )
            ''')
            conn.commit()

    def emit(self, record):
        """
        Inserts a new log record into the database.
        """
        try:
            # Ensure the record is formatted to populate all fields
            self.format(record)
            # Format the timestamp using the formatter
            timestamp = self.formatter.formatTime(record)
            #with self.lock:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO logs (timestamp, level, module, filename, line_number, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, record.levelname, record.module,
                        record.filename, record.lineno, record.getMessage()))
                conn.commit()
        except Exception:
            self.handleError(record)
    def __del__(self):
        """
        Destructor to perform a WAL checkpoint when the handler is destroyed.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('PRAGMA wal_checkpoint;')
                conn.commit()
        except Exception as e:
            print(f"Error during WAL checkpoint: {e}")
            
            
"""_summary_

usage:

import logging

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create the SQLite logging handler
sqlite_handler = SQLiteHandler('application_logs.db')
sqlite_handler.setLevel(logging.DEBUG)

# Add the handler to the logger
logger.addHandler(sqlite_handler)

# Register the handler's close method with the logging shutdown
logging.shutdown = sqlite_handler.close

# Log messages
logger.info('This is an info message.')
logger.error('This is an error message.')

# When the application is terminating
logging.shutdown()

"""