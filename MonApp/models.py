from django.db import models

class AuthToken(models.Model):
    token = models.CharField(max_length=255)
    user_id = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
