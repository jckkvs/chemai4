"""core/api_urls.py - API URLルーティング"""
from django.urls import path
from . import views

urlpatterns = [
    path("session/<uuid:session_id>/upload/", views.upload_data, name="api_upload"),
    path("session/<uuid:session_id>/columns/", views.set_columns, name="api_set_columns"),
    path("session/<uuid:session_id>/sample/", views.load_sample, name="api_load_sample"),
    path("session/<uuid:session_id>/descriptors/", views.calculate_descriptors, name="api_calculate_descriptors"),
    path("session/<uuid:session_id>/run/", views.run_analysis, name="api_run_analysis"),
    path("session/<uuid:session_id>/run_multi/", views.run_multi_analysis, name="api_run_multi_analysis"),
    path("session/<uuid:session_id>/results/", views.get_results, name="api_get_results"),
    path("session/<uuid:session_id>/status/", views.check_status, name="api_check_status"),
    # パラメータ自動UI用API
    path("params/model/<str:model_key>/", views.get_model_params_schema, name="api_model_params"),
    path("params/adapter/<str:adapter_name>/", views.get_adapter_params_schema, name="api_adapter_params"),
    # DoE API
    path("doe/upload_existing/", views.doe_upload_existing, name="api_doe_upload"),
    path("doe/run/", views.doe_run, name="api_doe_run"),
    # 逆解析・ベイズ最適化・リーケージ・セッション内DoE
    path("session/<uuid:session_id>/inverse/", views.run_inverse, name="api_run_inverse"),
    path("session/<uuid:session_id>/bayesian_suggest/", views.run_bayesian_suggest, name="api_bayesian_suggest"),
    path("session/<uuid:session_id>/leakage/", views.check_leakage, name="api_check_leakage"),
    path("session/<uuid:session_id>/doe_factors/", views.session_doe_factors, name="api_session_doe_factors"),
    path("session/<uuid:session_id>/doe/", views.session_doe_run, name="api_session_doe_run"),
]


