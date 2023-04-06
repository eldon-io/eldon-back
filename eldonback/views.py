from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from eldonback.serializers import UserSerializer, GroupSerializer

from django.http import JsonResponse
from eldonback.ia_model import algorithm_gpt3

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = User.objects.all().order_by("-date_joined")
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """

    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

def index(request):
    return JsonResponse({"text": "Je fais unn changement pour le CI-CD"})

def response_model(request, comment):
    return JsonResponse({"result": str(algorithm_gpt3(comment))})