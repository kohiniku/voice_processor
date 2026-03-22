"""Split an audio file into fixed-duration chunks using pydub."""

import os
import tempfile
from dataclasses import dataclass

from pydub import AudioSegment


@dataclass
class ChunkInfo:
    """Metadata for a single audio chunk."""

    index: int
    path: str
    total: int


class AudioChunker:
    """Split an audio file into time-based chunks."""

    def chunk(
        self, audio_path: str, chunk_duration_seconds: int
    ) -> list[ChunkInfo]:
        """Return a list of *ChunkInfo* for the given audio file.

        If the audio is shorter than *chunk_duration_seconds* a single
        chunk covering the entire file is returned.
        """
        audio = AudioSegment.from_file(audio_path)
        chunk_ms = chunk_duration_seconds * 1000
        total_ms = len(audio)

        if total_ms == 0:
            raise ValueError("Audio file is empty (0 ms duration)")

        # Calculate the total number of chunks
        total_chunks = (total_ms + chunk_ms - 1) // chunk_ms  # ceiling division

        chunks: list[ChunkInfo] = []
        for i in range(total_chunks):
            start = i * chunk_ms
            end = min(start + chunk_ms, total_ms)
            segment = audio[start:end]

            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav", prefix=f"chunk{i}_"
            )
            segment.export(tmp.name, format="wav")
            tmp.close()

            chunks.append(ChunkInfo(index=i, path=tmp.name, total=total_chunks))

        return chunks

    @staticmethod
    def cleanup(chunks: list[ChunkInfo]) -> None:
        """Remove temporary chunk files."""
        for c in chunks:
            try:
                os.remove(c.path)
            except OSError:
                pass
