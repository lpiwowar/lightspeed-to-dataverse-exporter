"""Unit tests for file_handler module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, call
import logging

from src.file_handler import FileHandler, delete_files, chunk_data, filter_symlinks
from src.constants import MAX_PAYLOAD_SIZE, MAX_DATA_DIR_SIZE


class TestDeleteFiles:
    """Tests for the delete_files standalone function."""

    def test_delete_files_success(self, tmp_path):
        """Test successful deletion of files."""
        # Create test files
        file1 = tmp_path / "test1.json"
        file2 = tmp_path / "test2.json"
        file1.write_text('{"test": 1}')
        file2.write_text('{"test": 2}')

        assert file1.exists()
        assert file2.exists()

        # Delete files
        delete_files([file1, file2])

        assert not file1.exists()
        assert not file2.exists()

    def test_delete_files_with_missing_file(self, tmp_path, caplog):
        """Test deletion when some files don't exist."""
        file1 = tmp_path / "exists.json"
        file2 = tmp_path / "missing.json"
        file1.write_text('{"test": 1}')

        with caplog.at_level(logging.DEBUG):
            delete_files([file1, file2])

        assert not file1.exists()
        assert "Removing" in caplog.text
        assert "already deleted or does not exist" in caplog.text

    @patch("pathlib.Path.unlink")
    def test_delete_files_permission_error(self, mock_unlink, tmp_path, caplog):
        """Test deletion with permission errors."""
        file1 = tmp_path / "test.json"
        file1.write_text('{"test": 1}')

        # Mock unlink to raise PermissionError
        mock_unlink.side_effect = PermissionError("Permission denied")

        with caplog.at_level(logging.ERROR):
            delete_files([file1])

        # Should log the error
        assert "Failed to remove" in caplog.text
        assert "Permission denied" in caplog.text

    def test_delete_files_empty_list(self):
        """Test deletion with empty file list."""
        # Should not raise any errors
        delete_files([])

    @patch("src.file_handler.logger")
    def test_delete_files_still_exists_after_unlink(self, mock_logger, tmp_path):
        """Test delete_files when file still exists after unlink (edge case)."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        # Mock pathlib.Path.unlink to succeed but pathlib.Path.exists to return True
        with (
            patch("pathlib.Path.unlink") as mock_unlink,
            patch("pathlib.Path.exists", return_value=True) as mock_exists,
        ):
            delete_files([test_file])

            mock_unlink.assert_called_once()
            mock_exists.assert_called_once()
            mock_logger.error.assert_called_once_with(
                "Failed to remove '%s'", test_file
            )

    def test_delete_files_removes_empty_directories(self, tmp_path, caplog):
        """Test that empty directories are removed after file deletion."""
        # Create nested directory structure
        data_dir = tmp_path / "data"
        subdir1 = data_dir / "subdir1"
        subdir2 = subdir1 / "subdir2"
        subdir2.mkdir(parents=True)

        # Create file in nested directory
        file1 = subdir2 / "test.json"
        file1.write_text('{"test": 1}')

        assert file1.exists()
        assert subdir2.exists()
        assert subdir1.exists()
        assert data_dir.exists()

        # Delete file with root_dir specified
        with caplog.at_level(logging.DEBUG):
            delete_files([file1], root_dir=data_dir)

        # File should be deleted
        assert not file1.exists()
        # Empty parent directories should be removed
        assert not subdir2.exists()
        assert not subdir1.exists()
        # Root directory should remain
        assert data_dir.exists()
        assert "Removing empty directory" in caplog.text

    def test_delete_files_preserves_non_empty_directories(self, tmp_path):
        """Test that non-empty directories are not removed."""
        # Create nested directory structure
        data_dir = tmp_path / "data"
        subdir1 = data_dir / "subdir1"
        subdir2 = subdir1 / "subdir2"
        subdir2.mkdir(parents=True)

        # Create two files
        file1 = subdir2 / "test1.json"
        file2 = subdir1 / "test2.json"
        file1.write_text('{"test": 1}')
        file2.write_text('{"test": 2}')

        # Delete only one file
        delete_files([file1], root_dir=data_dir)

        # File should be deleted
        assert not file1.exists()
        # subdir2 should be removed (empty)
        assert not subdir2.exists()
        # subdir1 should remain (has file2)
        assert subdir1.exists()
        assert file2.exists()
        # Root directory should remain
        assert data_dir.exists()

    def test_delete_files_never_removes_root_dir(self, tmp_path):
        """Test that root_dir is never removed even if empty."""
        # Create file directly in root_dir
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file1 = data_dir / "test.json"
        file1.write_text('{"test": 1}')

        # Delete file
        delete_files([file1], root_dir=data_dir)

        # File should be deleted
        assert not file1.exists()
        # Root directory should remain even though it's empty
        assert data_dir.exists()

    def test_delete_files_stops_at_root_boundary(self, tmp_path):
        """Test that cleanup stops at root_dir boundary."""
        # Create structure with directories outside root
        outside_dir = tmp_path / "outside"
        data_dir = tmp_path / "data"
        subdir = data_dir / "subdir"
        subdir.mkdir(parents=True)
        outside_dir.mkdir()

        # Create file in subdir
        file1 = subdir / "test.json"
        file1.write_text('{"test": 1}')

        # Delete file
        delete_files([file1], root_dir=data_dir)

        # File and subdir should be deleted
        assert not file1.exists()
        assert not subdir.exists()
        # Outside directory should remain untouched
        assert outside_dir.exists()
        # Root directory should remain
        assert data_dir.exists()

    def test_delete_files_backward_compatible_no_root_dir(self, tmp_path):
        """Test backward compatibility when root_dir is not provided."""
        # Create nested directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file1 = subdir / "test.json"
        file1.write_text('{"test": 1}')

        # Delete file without root_dir
        delete_files([file1])

        # File should be deleted
        assert not file1.exists()
        # Directory should remain (backward compatible behavior)
        assert subdir.exists()

    def test_delete_files_handles_multiple_files_in_different_dirs(self, tmp_path):
        """Test cleanup when deleting multiple files in different directories."""
        data_dir = tmp_path / "data"
        subdir1 = data_dir / "subdir1"
        subdir2 = data_dir / "subdir2"
        subdir1.mkdir(parents=True)
        subdir2.mkdir(parents=True)

        # Create one file in each subdirectory
        file1 = subdir1 / "test1.json"
        file2 = subdir2 / "test2.json"
        file1.write_text('{"test": 1}')
        file2.write_text('{"test": 2}')

        # Delete both files
        delete_files([file1, file2], root_dir=data_dir)

        # Both files should be deleted
        assert not file1.exists()
        assert not file2.exists()
        # Both empty subdirectories should be removed
        assert not subdir1.exists()
        assert not subdir2.exists()
        # Root directory should remain
        assert data_dir.exists()


class TestFilterSymlinks:
    """Tests for the filter_symlinks standalone function."""

    def test_filter_symlinks_no_symlinks(self, tmp_path):
        """Test filtering when there are no symlinks."""
        # Create regular files
        file1 = tmp_path / "file1.json"
        file2 = tmp_path / "file2.json"
        file1.write_text("{}")
        file2.write_text("{}")

        files = [file1, file2]
        result = filter_symlinks(files)

        assert result == files
        assert len(result) == 2

    def test_filter_symlinks_with_symlinks(self, tmp_path, caplog):
        """Test filtering when symlinks are present."""
        # Create regular file and symlink
        regular_file = tmp_path / "regular.json"
        symlink_file = tmp_path / "symlink.json"

        regular_file.write_text("{}")

        try:
            symlink_file.symlink_to(regular_file)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        files = [regular_file, symlink_file]

        with caplog.at_level(logging.WARNING):
            result = filter_symlinks(files)

        # Should only return regular file
        assert result == [regular_file]
        assert len(result) == 1
        assert "Skipping symlink" in caplog.text
        assert str(symlink_file) in caplog.text

    def test_filter_symlinks_empty_list(self):
        """Test filtering with empty file list."""
        result = filter_symlinks([])
        assert result == []

    def test_filter_symlinks_all_symlinks(self, tmp_path, caplog):
        """Test filtering when all files are symlinks."""
        # Create target file and symlinks
        target_file = tmp_path / "target.json"
        symlink1 = tmp_path / "symlink1.json"
        symlink2 = tmp_path / "symlink2.json"

        target_file.write_text("{}")

        try:
            symlink1.symlink_to(target_file)
            symlink2.symlink_to(target_file)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        files = [symlink1, symlink2]

        with caplog.at_level(logging.WARNING):
            result = filter_symlinks(files)

        # Should return empty list
        assert result == []
        assert caplog.text.count("Skipping symlink") == 2


class TestChunkData:
    """Tests for the chunk_data standalone function."""

    def test_chunk_data_basic(self):
        """Test basic chunking functionality."""
        files = [
            (Path("file1.json"), 30),
            (Path("file2.json"), 40),
            (Path("file3.json"), 50),
        ]

        chunks = chunk_data(files, chunk_max_size=80)

        assert len(chunks) == 2
        assert len(chunks[0]) == 2  # file1 + file2 = 70 bytes
        assert len(chunks[1]) == 1  # file3 = 50 bytes
        assert chunks[0] == [Path("file1.json"), Path("file2.json")]
        assert chunks[1] == [Path("file3.json")]

    def test_chunk_data_single_chunk(self):
        """Test when all files fit in one chunk."""
        files = [
            (Path("file1.json"), 20),
            (Path("file2.json"), 30),
        ]

        chunks = chunk_data(files, chunk_max_size=100)

        assert len(chunks) == 1
        assert len(chunks[0]) == 2
        assert chunks[0] == [Path("file1.json"), Path("file2.json")]

    def test_chunk_data_one_file_per_chunk(self):
        """Test when each file requires its own chunk."""
        files = [
            (Path("file1.json"), 60),
            (Path("file2.json"), 70),
        ]

        chunks = chunk_data(files, chunk_max_size=80)

        assert len(chunks) == 2
        assert len(chunks[0]) == 1
        assert len(chunks[1]) == 1

    def test_chunk_data_empty_list(self):
        """Test chunking with empty file list."""
        chunks = chunk_data([], chunk_max_size=100)
        assert chunks == []

    def test_chunk_data_exact_size_match(self):
        """Test chunking when files exactly match chunk size."""
        files = [
            (Path("file1.json"), 50),
            (Path("file2.json"), 50),
        ]

        chunks = chunk_data(files, chunk_max_size=100)

        assert len(chunks) == 1
        assert len(chunks[0]) == 2


class TestFileHandler:
    """Tests for the FileHandler class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def handler(self, temp_data_dir):
        """Create a FileHandler instance for testing."""
        return FileHandler(temp_data_dir)

    def test_init_with_defaults(self, temp_data_dir):
        """Test FileHandler initialization with default values."""
        handler = FileHandler(temp_data_dir)

        assert handler.data_dir == temp_data_dir
        assert handler.allowed_subdirs == []
        assert handler.max_data_dir_size == MAX_DATA_DIR_SIZE
        assert handler.max_payload_size == MAX_PAYLOAD_SIZE

    def test_init_with_custom_values(self, temp_data_dir):
        """Test FileHandler initialization with custom values."""
        custom_subdirs = ["custom1", "custom2"]
        handler = FileHandler(
            temp_data_dir,
            allowed_subdirs=custom_subdirs,
            max_data_dir_size=1000,
            max_payload_size=500,
        )

        assert handler.data_dir == temp_data_dir
        assert handler.allowed_subdirs == custom_subdirs
        assert handler.max_data_dir_size == 1000
        assert handler.max_payload_size == 500

    def test_filter_allowed_files_success(self, temp_data_dir, caplog):
        """Test filtering files from allowed subdirectories."""
        # Create handler with specific allowed subdirs for filtering test
        handler = FileHandler(
            temp_data_dir, allowed_subdirs=["feedback", "transcripts"]
        )

        # Create test files in allowed and disallowed directories
        feedback_dir = temp_data_dir / "feedback"
        transcripts_dir = temp_data_dir / "transcripts"
        unknown_dir = temp_data_dir / "unknown"

        feedback_dir.mkdir()
        transcripts_dir.mkdir()
        unknown_dir.mkdir()

        feedback_file = feedback_dir / "test1.json"
        transcripts_file = transcripts_dir / "test2.json"
        unknown_file = unknown_dir / "test3.json"

        feedback_file.write_text("{}")
        transcripts_file.write_text("{}")
        unknown_file.write_text("{}")

        files = [feedback_file, transcripts_file, unknown_file]

        with caplog.at_level(logging.WARNING):
            filtered = handler.filter_allowed_files(files)

        assert len(filtered) == 2
        assert feedback_file in filtered
        assert transcripts_file in filtered
        assert unknown_file not in filtered
        assert "Found 1 unknown files" in caplog.text

    def test_filter_allowed_files_warns_on_unknown_files(self, temp_data_dir, caplog):
        """Test that warning is logged when there are unknown files."""
        # Create handler with specific allowed subdirs for filtering test
        handler = FileHandler(
            temp_data_dir, allowed_subdirs=["feedback", "transcripts"]
        )

        # Create test files in allowed and unknown directories
        feedback_dir = temp_data_dir / "feedback"
        unknown_dir1 = temp_data_dir / "unknown1"
        unknown_dir2 = temp_data_dir / "unknown2"

        feedback_dir.mkdir()
        unknown_dir1.mkdir()
        unknown_dir2.mkdir()

        feedback_file = feedback_dir / "test1.json"
        unknown_file1 = unknown_dir1 / "test2.json"
        unknown_file2 = unknown_dir2 / "test3.json"

        feedback_file.write_text("{}")
        unknown_file1.write_text("{}")
        unknown_file2.write_text("{}")

        files = [feedback_file, unknown_file1, unknown_file2]

        with caplog.at_level(logging.WARNING):
            filtered = handler.filter_allowed_files(files)

        assert len(filtered) == 1
        assert feedback_file in filtered
        assert unknown_file1 not in filtered
        assert unknown_file2 not in filtered
        # Should log warning with correct count of unknown files
        assert "Found 2 unknown files" in caplog.text
        # Verify it's logged at WARNING level
        warning_messages = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        assert any("Found 2 unknown files" in msg for msg in warning_messages)

    def test_filter_allowed_files_empty_list(self, temp_data_dir, caplog):
        """Test filtering with empty file list."""
        # Create handler with specific allowed subdirs for filtering test
        handler = FileHandler(
            temp_data_dir, allowed_subdirs=["feedback", "transcripts"]
        )
        with caplog.at_level(logging.DEBUG):
            filtered = handler.filter_allowed_files([])

        assert filtered == []
        # Should log debug message when there are no unknown files
        assert "No unknown files found" in caplog.text

    def test_filter_allowed_files_empty_allowed_subdirs(self, temp_data_dir):
        """Test filtering when allowed_subdirs is empty - should allow all files."""
        handler = FileHandler(temp_data_dir, allowed_subdirs=[])

        # Create test files in various locations
        feedback_dir = temp_data_dir / "feedback"
        unknown_dir = temp_data_dir / "unknown"
        feedback_dir.mkdir()
        unknown_dir.mkdir()

        feedback_file = feedback_dir / "test1.json"
        unknown_file = unknown_dir / "test2.json"
        root_file = temp_data_dir / "root.json"

        feedback_file.write_text("{}")
        unknown_file.write_text("{}")
        root_file.write_text("{}")

        files = [feedback_file, unknown_file, root_file]
        filtered = handler.filter_allowed_files(files)

        # Should return all files when allowed_subdirs is empty
        assert len(filtered) == 3
        assert feedback_file in filtered
        assert unknown_file in filtered
        assert root_file in filtered

    def test_collect_files_directory_not_exists(self, caplog):
        """Test collect_files when data directory doesn't exist."""
        non_existent_dir = Path("/non/existent/directory")
        handler = FileHandler(non_existent_dir)

        with caplog.at_level(logging.WARNING):
            result = handler.collect_files()

        assert result == []
        assert f"Data directory {non_existent_dir} does not exist" in caplog.text

    def test_collect_files_no_files(self, handler, temp_data_dir, caplog):
        """Test collect_files when no JSON files exist."""
        # Create allowed directory but no files
        feedback_dir = temp_data_dir / "feedback"
        feedback_dir.mkdir()

        with caplog.at_level(logging.DEBUG):
            result = handler.collect_files()

        assert result == []

    def test_collect_files_success(self, handler, temp_data_dir, caplog):
        """Test successful file collection."""
        # Create test files
        feedback_dir = temp_data_dir / "feedback"
        feedback_dir.mkdir()

        file1 = feedback_dir / "test1.json"
        file2 = feedback_dir / "test2.json"
        file1.write_text('{"size": "small"}')  # ~18 bytes
        file2.write_text('{"size": "medium content"}')  # ~25 bytes

        with caplog.at_level(logging.DEBUG):
            result = handler.collect_files()

        assert len(result) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(
            isinstance(path, Path) and isinstance(size, int) for path, size in result
        )
        assert f"Collected 2 files from {temp_data_dir}" in caplog.text

    def test_collect_files_removes_oversized(self, temp_data_dir, caplog):
        """Test that oversized files are removed."""
        # Create handler with small payload size
        handler = FileHandler(temp_data_dir, max_payload_size=10)

        feedback_dir = temp_data_dir / "feedback"
        feedback_dir.mkdir()

        small_file = feedback_dir / "small.json"
        large_file = feedback_dir / "large.json"

        small_file.write_text("{}")  # ~2 bytes
        large_file.write_text(
            '{"data": "this is a large file with lots of content"}'
        )  # >10 bytes

        assert large_file.exists()

        with caplog.at_level(logging.INFO):
            result = handler.collect_files()

        # Only small file should be in result
        assert len(result) == 1
        assert result[0][0] == small_file

        # Large file should be deleted
        assert not large_file.exists()
        assert "too big for export and was removed" in caplog.text
        assert "Removed oversized file" in caplog.text

    def test_collect_files_skips_symlinks(self, temp_data_dir, caplog):
        """Test that symlinks are skipped for security reasons."""
        handler = FileHandler(temp_data_dir)

        feedback_dir = temp_data_dir / "feedback"
        feedback_dir.mkdir()

        # Create a regular file and a symlink
        regular_file = feedback_dir / "regular.json"
        symlink_file = feedback_dir / "symlink.json"

        regular_file.write_text('{"type": "regular"}')

        # Create symlink pointing to the regular file
        try:
            symlink_file.symlink_to(regular_file)
        except OSError:
            # Skip test if symlinks aren't supported (e.g., Windows without privileges)
            pytest.skip("Symlinks not supported on this system")

        with caplog.at_level(logging.WARNING):
            result = handler.collect_files()

        # Should only collect the regular file, not the symlink
        assert len(result) == 1
        assert result[0][0] == regular_file
        assert "Skipping symlink" in caplog.text
        assert str(symlink_file) in caplog.text

    @patch("src.file_handler.logger")
    def test_collect_files_oversized_file_removal_fails(
        self, mock_logger, temp_data_dir
    ):
        """Test collect_files when removal of oversized file fails with OSError."""
        handler = FileHandler(temp_data_dir, max_payload_size=10)

        # Create an oversized file
        oversized_file = temp_data_dir / "large.json"
        large_content = '{"data": "' + "x" * 50 + '"}'  # >10 bytes
        oversized_file.write_text(large_content)

        # Create a mock that will raise OSError when unlink is called
        def mock_unlink_with_error(self):
            raise OSError("Permission denied")

        # Patch the unlink method on the specific file
        with patch.object(type(oversized_file), "unlink", mock_unlink_with_error):
            collected = handler.collect_files()

        # Should have no collected files
        assert len(collected) == 0

        # Should log the error
        mock_logger.error.assert_called_once()
        args = mock_logger.error.call_args[0]
        assert "Failed to remove oversized file" in args[0]
        assert oversized_file in args
        assert "Permission denied" in str(args[2])

    @patch("src.file_handler.chunk_data")
    def test_gather_data_chunks_with_data(self, mock_chunk_data, handler, caplog):
        """Test gather_data_chunks with data."""
        # Mock chunk_data to return test chunks
        mock_chunks = [[Path("file1.json")], [Path("file2.json"), Path("file3.json")]]
        mock_chunk_data.return_value = mock_chunks

        collected_files = [
            (Path("file1.json"), 100),
            (Path("file2.json"), 50),
            (Path("file3.json"), 30),
        ]

        with caplog.at_level(logging.INFO):
            result = handler.gather_data_chunks(collected_files)

        assert result == mock_chunks
        mock_chunk_data.assert_called_once_with(
            collected_files, handler.max_payload_size
        )
        assert "Collected 3 files (split to 2 chunks)" in caplog.text

    @patch("src.file_handler.chunk_data")
    def test_gather_data_chunks_empty(self, mock_chunk_data, handler, caplog):
        """Test gather_data_chunks with no data."""
        mock_chunk_data.return_value = []

        with caplog.at_level(logging.INFO):
            result = handler.gather_data_chunks([])

        assert result == []
        # Should not log when no chunks
        assert "Collected" not in caplog.text

    @patch("src.file_handler.delete_files")
    def test_delete_collected_files(self, mock_delete_files, handler):
        """Test delete_collected_files wrapper method passes root_dir."""
        test_files = [Path("file1.json"), Path("file2.json")]

        handler.delete_collected_files(test_files)

        # Should pass root_dir parameter
        mock_delete_files.assert_called_once_with(test_files, root_dir=handler.data_dir)

    def test_delete_collected_files_removes_empty_directories(
        self, handler, temp_data_dir
    ):
        """Test that delete_collected_files removes empty directories."""
        # Create nested directory structure
        subdir1 = temp_data_dir / "subdir1"
        subdir2 = subdir1 / "subdir2"
        subdir2.mkdir(parents=True)

        # Create file in nested directory
        file1 = subdir2 / "test.json"
        file1.write_text('{"test": 1}')

        assert file1.exists()
        assert subdir2.exists()
        assert subdir1.exists()

        # Delete file using delete_collected_files
        handler.delete_collected_files([file1])

        # File should be deleted
        assert not file1.exists()
        # Empty parent directories should be removed
        assert not subdir2.exists()
        assert not subdir1.exists()
        # Root directory should remain
        assert temp_data_dir.exists()

    @patch("src.file_handler.FileHandler.collect_files")
    @patch("src.file_handler.delete_files")
    def test_ensure_size_limit_under_limit(
        self, mock_delete_files, mock_collect_files, handler, caplog
    ):
        """Test ensure_size_limit when under the limit."""
        # Create handler with large limit
        handler.max_data_dir_size = 1000

        mock_collect_files.return_value = [
            (Path("file1.json"), 100),
            (Path("file2.json"), 200),
        ]

        with caplog.at_level(logging.ERROR):
            handler.ensure_size_limit()

        # Should not delete anything or log errors
        mock_delete_files.assert_not_called()
        assert "Data folder size is bigger" not in caplog.text

    @patch("src.file_handler.FileHandler.collect_files")
    @patch("src.file_handler.delete_files")
    def test_ensure_size_limit_over_limit(
        self, mock_delete_files, mock_collect_files, handler, caplog
    ):
        """Test ensure_size_limit when over the limit."""
        # Create handler with small limit
        handler.max_data_dir_size = 100

        mock_collect_files.return_value = [
            (Path("file1.json"), 80),
            (Path("file2.json"), 60),  # Total: 140 > 100
        ]

        with caplog.at_level(logging.INFO):
            handler.ensure_size_limit()

        # Should delete the first file (80 bytes removed, bringing total to 60 < 100)
        mock_delete_files.assert_called_once_with(
            [Path("file1.json")], root_dir=handler.data_dir
        )
        assert (
            "Data folder size is bigger than the maximum allowed size: 140 > 100"
            in caplog.text
        )
        assert "Removing files to fit the data into the limit" in caplog.text

    @patch("src.file_handler.FileHandler.collect_files")
    @patch("src.file_handler.delete_files")
    def test_ensure_size_limit_multiple_deletions(
        self, mock_delete_files, mock_collect_files, handler, caplog
    ):
        """Test ensure_size_limit when multiple files need deletion."""
        handler.max_data_dir_size = 50

        mock_collect_files.return_value = [
            (Path("file1.json"), 40),
            (Path("file2.json"), 30),
            (Path("file3.json"), 20),  # Total: 90 > 50
        ]

        with caplog.at_level(logging.INFO):
            handler.ensure_size_limit()

        # Should delete first two files (70 bytes removed, bringing total to 20 < 50)
        expected_calls = [
            call([Path("file1.json")], root_dir=handler.data_dir),
            call([Path("file2.json")], root_dir=handler.data_dir),
        ]
        mock_delete_files.assert_has_calls(expected_calls)


class TestIntegration:
    """Integration tests for FileHandler workflow."""

    @pytest.fixture
    def integration_setup(self):
        """Set up integration test environment."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)

        # Create directory structure
        feedback_dir = data_dir / "feedback"
        transcripts_dir = data_dir / "transcripts"
        unknown_dir = data_dir / "unknown"

        feedback_dir.mkdir()
        transcripts_dir.mkdir()
        unknown_dir.mkdir()

        yield {
            "data_dir": data_dir,
            "feedback_dir": feedback_dir,
            "transcripts_dir": transcripts_dir,
            "unknown_dir": unknown_dir,
        }

        shutil.rmtree(temp_dir)

    def test_full_workflow(self, integration_setup):
        """Test complete file handling workflow."""
        dirs = integration_setup
        # Create handler with specific allowed subdirs for filtering test
        handler = FileHandler(
            dirs["data_dir"],
            allowed_subdirs=["feedback", "transcripts"],
            max_payload_size=100,
        )

        # Create test files
        small_file = dirs["feedback_dir"] / "small.json"
        medium_file = dirs["transcripts_dir"] / "medium.json"
        large_file = dirs["feedback_dir"] / "large.json"
        unknown_file = dirs["unknown_dir"] / "unknown.json"

        small_file.write_text('{"type": "small"}')  # ~17 bytes
        medium_file.write_text(
            '{"type": "medium", "data": "some content"}'
        )  # ~40 bytes
        large_file.write_text(
            '{"type": "large", "data": "' + "x" * 200 + '"}'
        )  # >100 bytes
        unknown_file.write_text('{"type": "unknown"}')

        # Collect files - should filter out unknown and large files
        collected = handler.collect_files()

        # Should have 2 files (small and medium), large file deleted
        assert len(collected) == 2
        assert not large_file.exists()
        assert unknown_file.exists()  # Unknown files are not deleted, just filtered

        # Chunk the collected files
        chunks = handler.gather_data_chunks(collected)

        # All files should fit in one chunk since total < 100 bytes
        assert len(chunks) == 1
        assert len(chunks[0]) == 2
