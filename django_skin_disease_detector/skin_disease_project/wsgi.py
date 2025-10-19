"""
WSGI config for skin_disease_project project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_project.settings')

application = get_wsgi_application()
