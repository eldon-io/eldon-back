from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from eldonback.serializers import UserSerializer, GroupSerializer
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from eldonback.ia_model import algorithm_gpt3
import json

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

@csrf_exempt
def response_model(request):
    response_ = []
    
    if request.method == 'POST':
        data = json.loads(request.body) # On récupère le post
        n = len(data['tweets'])
    
        for i in range(n):
            text_ = data['tweets'][i]["text"]
            id_ = data['tweets'][i]["id"]
            result_ = algorithm_gpt3(text_)
            
            response_.append(
                {
                    "text" : text_,
                    "id" : id_,
                    "result" : result_ })
        
    return JsonResponse({"answer": response_})