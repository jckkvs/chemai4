"""core/urls.py - ページURLルーティング"""
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("session/new/", views.new_session, name="new_session"),
    path("session/<uuid:session_id>/", views.session_detail, name="session_detail"),
    path("doe/", views.doe_page, name="doe"),
    path("help/", views.help_page, name="help"),
]

