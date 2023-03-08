from MonApp.models import AuthToken
from django.utils import timezone
import secrets

def create_auth_token(user_id):
    token = secrets.token_urlsafe(32)
    auth_token = AuthToken(token=token, user_id=user_id, created_at=timezone.now())
    auth_token.save()
    return token

def get_auth_token(token):
    try:
        auth_token = AuthToken.objects.get(token=token)
    except AuthToken.DoesNotExist:
        return None
    return auth_token
