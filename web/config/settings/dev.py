import os

from config.settings.base import *

DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"

# 0.0.0.0 included so the container is reachable from the host by any of the
# usual spellings -- the previous build allowed only "localhost".
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", "django", "[::1]"]
