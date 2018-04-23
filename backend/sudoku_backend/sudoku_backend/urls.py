from django.conf.urls import url
 
from . import view
 
urlpatterns = [
    url(r'^index$', view.index),
    url(r'^solve$', view.solve),
    url(r'^process$', view.process),
    url(r'^solve_origin$', view.solve_origin)
]