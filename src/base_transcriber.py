from abc import ABC, abstractmethod


class TranscriptionSession(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> tuple[str, bool]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


class BaseTranscriber(ABC):
    @abstractmethod
    def create_session(self) -> TranscriptionSession:
        ...
