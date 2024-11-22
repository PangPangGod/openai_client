class MissingAPIKeyError(Exception):
    """Raised when API key is missing."""
    def __init__(self, message="OpenAI API 키가 제공되지 않았습니다. .env 파일 또는 환경 변수를 확인해주세요."):
        super().__init__(message)