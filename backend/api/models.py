from django.contrib.auth.models import AbstractUser
from django.db import models
from enum import Enum


class Stage(str, Enum):
    BEGINNING = "Beginning"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]


class UserGroup(str, Enum):
    CONTROL = "control"  # simple bot (RCT control group)
    INTERVENTION = "intervention"  # main chatbot (RCT intervention group)
    PLACEBO = "placebo"  # placebo bot (minimal prompt chatbot)

    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]


class User(AbstractUser):
    stage = models.CharField(
        max_length=20,
        choices=Stage.choices(),
        default=Stage.BEGINNING
    )
    group = models.CharField(
        max_length=20,
        choices=UserGroup.choices(),
        default=UserGroup.CONTROL,
        help_text="User's assigned group (control or intervention). Should be set once and not changed."
    )


class Message(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="messages")
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    session_id = models.IntegerField()
    is_user = models.BooleanField(default=True)  # True if message is from user, False if from LLM
    state = models.CharField(max_length=50, null=True, blank=True, help_text="FSM state when message was created")

    def __str__(self):
        return f"{self.user.username}: {self.text[:30]} ({self.session_id})"


class UserMemoryState(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="memory_state")
    current_memory = models.TextField(default="")  # Stores the LLM-generated memory
    last_processed_message = models.ForeignKey(
        Message,
        on_delete=models.SET_NULL,
        null=True,
        related_name="last_processed"
    )

    def __str__(self):
        return f"Memory state for {self.user.username}"


class UserDayProgress(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="day_progress")
    start_date = models.DateField(auto_now_add=True)  # First interaction date
    current_day = models.IntegerField(default=1)  # Current day (1-8)
    last_updated = models.DateField(auto_now=True)  # Last day calculation

    def __str__(self):
        return f"Day progress for {self.user.username}: Day {self.current_day}"

    def calculate_current_day(self):
        """Calculate current day based on calendar days since start"""
        from django.utils import timezone
        today = timezone.now().date()
        days_passed = (today - self.start_date).days + 1  # +1 because day 1 is the start date

        # Cap at day 8
        self.current_day = min(days_passed, 8)
        self.save()
        return self.current_day
