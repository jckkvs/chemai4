"""WSGI config for ChemAI ML Studio."""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chemai_django.settings")
application = get_wsgi_application()
