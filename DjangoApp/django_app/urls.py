from django.urls import path
from . import views

urlpatterns = [
    path("stream/", views.stream_status, name="stream_status"),
    path("stream/toggle/", views.toggle_stream, name="toggle_stream"),
    path("api/messages/latest/", views.latest_messages_json, name="latest_messages_json"),
    path("stream-metrics/", views.stream_metrics_json, name="stream_metrics_json"),


    path("update-config/", views.update_config, name="update_config"),


    path('search-messages/', views.search_classified_messages, name='search_classified_messages'),
    path('search/', views.message_search_page, name='message_search_page'),



    path('performance-logs/', views.performance_logs_json, name='performance_logs_json'),
    path('performance/', views.performance_logs_page, name='performance_logs_page'),


    path('chart-data/throughput/', views.chart_throughput_data, name='chart_throughput_data'),
    path('chart-data/categories/', views.chart_category_distribution, name='chart_category_distribution'),



    path('category-settings/', views.category_settings_page, name='category_settings'),
    path('categories/', views.get_categories, name='get_categories'),
    path('categories/update/', views.update_category_metadata, name='update_category_metadata'),
    path('categories/delete/<str:cat_id>/', views.delete_category_metadata, name='delete_category_metadata'),
]