import json
import threading
import time
from collections import deque
from typing import Dict, Optional, List


class BatchJsonlLogger:
    _instances: Dict[str, 'BatchJsonlLogger'] = {}  # Singleton pool, keyed by file path
    _lock = threading.Lock()  # Lock for singleton creation

    def __new__(cls, file_path: str, max_workers: int = 1, batch_size: int = 100, flush_interval: int = 5):
        """
        Singleton logger: instances sharing the same file path are reused.
        :param file_path: Path to the log file
        :param max_workers: Number of writer threads (1 is usually sufficient)
        :param batch_size: Threshold for batch writes
        :param flush_interval: Periodic flush interval in seconds
        """
        with cls._lock:
            if file_path not in cls._instances:
                cls._instances[file_path] = super().__new__(cls)
                # Initialize instance attributes
                cls._instances[file_path]._init(
                    file_path=file_path,
                    max_workers=max_workers,
                    batch_size=batch_size,
                    flush_interval=flush_interval
                )
        return cls._instances[file_path]

    def _init(self, file_path: str, max_workers: int, batch_size: int, flush_interval: int):
        self.file_path = file_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue = deque()  # Thread-safe log queue
        self.queue_lock = threading.Lock()  # Lock for queue operations
        self.running = True
        self.workers = []
        
        # Start writer threads
        for _ in range(max_workers):
            worker = threading.Thread(target=self._write_worker, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _write_worker(self):
        """Background writer thread logic."""
        while self.running:
            try:
                # Extract up to batch_size log entries
                batch = self._get_batch()
                if batch:
                    self._write_batch(batch)
                else:
                    # Sleep when the queue is empty
                    time.sleep(self.flush_interval)
            except Exception as e:
                print(f"Log writer thread error: {e}")
                time.sleep(1)  # Sleep after error to avoid busy loop

    def _get_batch(self) -> List[Dict]:
        """Extract a batch of log entries from the queue."""
        batch = []
        with self.queue_lock:
            while len(batch) < self.batch_size and self.queue:
                batch.append(self.queue.popleft())
        return batch

    def _write_batch(self, batch: List[Dict]):
        """Write a batch of log entries to the file."""
        with open(self.file_path, "a", encoding="utf-8") as f:
            for data in batch:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def write(self, data: Dict):
        """Write a single log entry (thread-safe)."""
        with self.queue_lock:
            self.queue.append(data)

    def flush(self):
        """Force-flush all queued log entries to disk."""
        while True:
            batch = self._get_batch()
            if not batch:
                break
            self._write_batch(batch)

    def close(self):
        """Close the logger (call on program exit)."""
        self.running = True  # Stop the thread loop first
        self.flush()  # Force-flush remaining log entries
        # Wait for all worker threads to finish
        for worker in self.workers:
            worker.join(timeout=5)
        # Remove from the singleton pool
        with self._lock:
            if self.file_path in BatchJsonlLogger._instances:
                del BatchJsonlLogger._instances[self.file_path]

    def __del__(self):
        self.close()