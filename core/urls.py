from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='Home'),
    path('search_products', views.search, name='search')
]
