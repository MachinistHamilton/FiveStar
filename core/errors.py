class AssistantError(Exception):
    """An actionable error that can safely be displayed by the local UI."""


class PDFError(AssistantError):
    pass


class ModelError(AssistantError):
    pass


class IndexError(AssistantError):
    pass


class LibraryBusyError(AssistantError):
    pass


class GenerationError(AssistantError):
    pass
