"""Initialize the Docker database and register its backend service token."""

from datetime import timedelta
import os
from pathlib import Path
from secrets import token_urlsafe
from tempfile import NamedTemporaryFile

from config import Settings
from db.database import Database
from db.repositories import AuthRepository
from scripts.init_mysql import main as init_mysql
from service.auth import AuthService
from utils.time import utc_now


def main():
    init_mysql([])
    settings = Settings.load()
    service = AuthService(AuthRepository(Database(settings)), settings.setup_secret)
    path = Path(os.environ["DOCKER_LLM_TOKEN_FILE"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        token = path.read_text(encoding="ascii").strip()
        if service.is_valid_token(token):
            return

    token = token_urlsafe(48)
    service.create_token(settings.setup_secret, token, utc_now() + timedelta(days=365),
                         label="docker-business-backend")
    with NamedTemporaryFile(mode="w", encoding="ascii", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(token + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print("Docker backend model token registered.")


if __name__ == "__main__":
    main()
