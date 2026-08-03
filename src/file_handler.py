"""File handling operations for data collection.

This module provides the FileHandler class which encapsulates all file-related
operations including collection, filtering, chunking, and cleanup.
"""

import pathlib
import logging
from pathlib import Path

from src.constants import MAX_PAYLOAD_SIZE, MAX_DATA_DIR_SIZE

logger = logging.getLogger(__name__)


def _cleanup_empty_directories(dir_path: pathlib.Path, root_dir: pathlib.Path) -> None:
    """Recursively remove empty directories up to (but not including) root_dir.

    Args:
        dir_path: Directory path to check and potentially remove.
        root_dir: Root directory boundary. Directories will not be removed
            if they reach or would exceed this root.
    """
    # Stop if we've reached the root directory
    if dir_path == root_dir:
        return

    # Stop if dir_path is not a descendant of root_dir
    try:
        dir_path.relative_to(root_dir)
    except ValueError:
        # dir_path is not within root_dir, stop cleanup
        return

    try:
        # Only attempt to remove if directory exists and is empty
        if dir_path.exists() and dir_path.is_dir():
            # Check if directory is empty (no files or subdirectories)
            try:
                next(dir_path.iterdir())
                # Directory is not empty, stop here
                return
            except StopIteration:
                # Directory is empty, remove it
                logger.debug("Removing empty directory '%s'", dir_path)
                dir_path.rmdir()
                # Recursively check parent directory
                _cleanup_empty_directories(dir_path.parent, root_dir)
        elif dir_path.exists():
            # Path exists but is not a directory, stop here
            return
    except OSError as e:
        logger.debug(
            "Failed to remove empty directory '%s': %s (may not be empty)", dir_path, e
        )


def delete_files(
    file_paths: list[pathlib.Path], root_dir: pathlib.Path | None = None
) -> None:
    """Delete files from the provided paths.

    After deleting each file, if root_dir is provided, recursively removes empty
    parent directories up to (but not including) the root_dir.

    Args:
        file_paths: List of paths to the files to be deleted.
        root_dir: Optional root directory for cleanup. If provided, empty parent
            directories will be removed up to this root. If None, only files are deleted.
    """
    for file_path in file_paths:
        logger.debug("Removing '%s'", file_path)
        file_deleted = False
        try:
            file_path.unlink()
            file_deleted = True
        except FileNotFoundError:
            logger.debug("File '%s' already deleted or does not exist", file_path)
        except OSError as e:
            logger.error("Failed to remove '%s': %s", file_path, e)
        else:
            if file_path.exists():
                logger.error("Failed to remove '%s'", file_path)

        # Clean up empty parent directories if root_dir is provided and file was deleted
        # NOTE: Per-file cleanup is sufficient for expected volumes (hundreds of files).
        # If performance issues arise with higher volumes, consider batching: collect all
        # parent directories first, then do a single cleanup sweep after all files are deleted.
        if file_deleted and root_dir is not None:
            _cleanup_empty_directories(file_path.parent, root_dir)


def filter_symlinks(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Filter out symlinks from file list for security reasons.

    Args:
        files: List of file paths to filter.

    Returns:
        List of file paths with symlinks removed.
    """
    filtered_files = []
    for file_path in files:
        if file_path.is_symlink():
            logger.warning("Skipping symlink '%s' for security reasons", file_path)
        else:
            filtered_files.append(file_path)
    return filtered_files


def chunk_data(
    data: list[tuple[pathlib.Path, int]], chunk_max_size: int
) -> list[list[pathlib.Path]]:
    """Chunk the data into smaller parts.

    Args:
        data: List of tuples containing (file_path, file_size_bytes).
        chunk_max_size: Maximum size of a chunk.

    Returns:
        List of lists of paths to the chunked files.
    """
    # Create chunks that don't exceed the maximum chunk size
    chunk_size = 0
    chunks: list[list[pathlib.Path]] = []
    chunk: list[pathlib.Path] = []
    for file_path, file_size in data:
        # Start a new chunk if adding this file would exceed the size limit
        if chunk and chunk_size + file_size > chunk_max_size:
            chunks.append(chunk)
            chunk = []
            chunk_size = 0
        chunk.append(file_path)
        chunk_size += file_size
    if chunk:
        chunks.append(chunk)
    return chunks


class FileHandler:
    """Handles file collection, filtering, chunking, and cleanup operations."""

    def __init__(
        self,
        data_dir: Path,
        allowed_subdirs: list[str] | None = None,
        max_data_dir_size: int = MAX_DATA_DIR_SIZE,
        max_payload_size: int = MAX_PAYLOAD_SIZE,
    ):
        """Initialize the file handler.

        Args:
            data_dir: Directory to collect files from
            allowed_subdirs: List of allowed subdirectories to include. None or empty list means collect from all subdirectories.
            max_data_dir_size: Maximum total size for data directory
            max_payload_size: Maximum size for individual payloads/chunks
        """
        self.data_dir = data_dir
        self.allowed_subdirs = allowed_subdirs or []
        self.max_data_dir_size = max_data_dir_size
        self.max_payload_size = max_payload_size

    def filter_allowed_files(self, files: list[Path]) -> list[Path]:
        """Filter files to only include allowed subdirectories."""
        # If no allowed subdirs specified, collect all files
        if not self.allowed_subdirs:
            return files

        filtered_files: list[Path] = []
        for file in files:
            # Strip the data_dir prefix and get the first directory component
            relative_path = file.relative_to(self.data_dir)
            first_dir = relative_path.parts[0] if relative_path.parts else None
            if first_dir in self.allowed_subdirs:
                filtered_files.append(file)

        # Log warning if there are unknown files
        unknown_files = len(files) - len(filtered_files)
        if unknown_files > 0:
            logger.warning("Found %s unknown files", unknown_files)
        else:
            logger.debug("No unknown files found")

        return filtered_files

    def collect_files(self) -> list[tuple[Path, int]]:
        """Perform a single collection operation.

        Returns:
            List of tuples containing (file_path, file_size_bytes).
            Files larger than MAX_PAYLOAD_SIZE are filtered out with warnings.
        """
        if not self.data_dir.exists():
            logger.warning("Data directory %s does not exist", self.data_dir)
            return []

        # Collect all files to be packed into tarball
        all_files = list(self.data_dir.rglob("*.json"))

        # Filter out symlinks for security reasons
        all_files = filter_symlinks(all_files)

        # Filter by allowed subdirectories
        all_files = self.filter_allowed_files(all_files)

        logger.debug("Collected %d files from %s", len(all_files), self.data_dir)

        if not all_files:
            return []

        # Collect file sizes along with paths and remove oversized files
        files_with_sizes = []
        for file_path in all_files:
            file_size = file_path.stat().st_size
            if file_size > self.max_payload_size:
                logger.warning(
                    "File '%s' (size: %d bytes) is too big for export and was removed. "
                    "Maximum allowed size: %d bytes",
                    file_path,
                    file_size,
                    self.max_payload_size,
                )
                # Optionally delete the oversized file to prevent accumulation
                try:
                    file_path.unlink()
                    logger.info("Removed oversized file: %s", file_path)
                except OSError as e:
                    logger.error(
                        "Failed to remove oversized file '%s': %s", file_path, e
                    )
            else:
                files_with_sizes.append((file_path, file_size))

        return files_with_sizes

    def gather_data_chunks(
        self, collected_files: list[tuple[Path, int]]
    ) -> list[list[pathlib.Path]]:
        """Gather data chunks from the collected files."""
        data_chunks = chunk_data(collected_files, self.max_payload_size)
        if any(data_chunks):
            logger.info(
                "Collected %d files (split to %d chunks)",
                len(collected_files),
                len(data_chunks),
            )
        return data_chunks

    def delete_collected_files(self, file_paths: list[pathlib.Path]) -> None:
        """Delete files from the provided paths.

        After deletion, empty parent directories will be recursively removed
        up to the data directory root.

        Args:
            file_paths: List of paths to the files to be deleted.
        """
        delete_files(file_paths, root_dir=self.data_dir)

    def ensure_size_limit(self) -> None:
        """Safeguard to prevent data directory overflow when export/cleanup fails.

        This method acts as a safety mechanism to prevent unbounded data accumulation
        in the data directory. It's called after successful export when cleanup is
        enabled, but primarily serves to handle cases where:
        - Network issues prevent data export
        - Export failures leave data unprocessed
        - Cleanup operations fail
        - Service interruptions break the normal export-cleanup cycle

        When the total size exceeds MAX_DATA_DIR_SIZE, it removes files in collection
        order until the size limit is satisfied, preventing disk space exhaustion.

        The data are removed without any order or particular pattern
        """
        collected_files = self.collect_files()
        data_size = sum(file_size for _, file_size in collected_files)
        if data_size > self.max_data_dir_size:
            logger.error(
                "Data folder size is bigger than the maximum allowed size: %d > %d",
                data_size,
                self.max_data_dir_size,
            )
            logger.info("Removing files to fit the data into the limit...")
            extra_size = data_size - self.max_data_dir_size
            for file_path, file_size in collected_files:
                extra_size -= file_size
                self.delete_collected_files([file_path])
                if extra_size < 0:
                    break
