from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.urls import path
from django.shortcuts import render
from django.db.models import Count, Q
from .models import User, Message, UserMemoryState

class UserAdmin(BaseUserAdmin):
    fieldsets = BaseUserAdmin.fieldsets + (
        ('RCT Group', {'fields': ('group',)}),
    )
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ('RCT Group', {'fields': ('group',)}),
    )
    list_display = BaseUserAdmin.list_display + ('group',)

# REMOVE or COMMENT OUT this line:
# admin.site.unregister(User)

admin.site.register(User, UserAdmin)

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'text_preview', 'session_id', 'state_display', 'timestamp', 'is_user')
    list_filter = ('user', 'session_id', 'state', 'timestamp', 'is_user')
    search_fields = ('text', 'user__username', 'state')
    readonly_fields = ('timestamp',)
    ordering = ('-timestamp',)
    actions = ['export_conversation_flow']
    fieldsets = (
        ('Message Information', {
            'fields': ('user', 'text', 'is_user', 'session_id')
        }),
        ('State Information', {
            'fields': ('state', 'timestamp'),
            'classes': ('collapse',)
        }),
    )
    
    def text_preview(self, obj):
        return obj.text[:50] + "..." if len(obj.text) > 50 else obj.text
    text_preview.short_description = 'Message Text'
    
    def state_display(self, obj):
        if obj.state:
            return obj.state.replace('_', ' ').title()
        return 'No State'
    state_display.short_description = 'FSM State'
    
    def export_conversation_flow(self, request, queryset):
        """Export conversation flow with state transitions for selected messages"""
        from django.http import HttpResponse
        import csv
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="conversation_flow.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['User', 'Session ID', 'Timestamp', 'State', 'Is User', 'Message'])
        
        for message in queryset.order_by('user', 'session_id', 'timestamp'):
            writer.writerow([
                message.user.username,
                message.session_id,
                message.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                message.state or 'No State',
                'User' if message.is_user else 'Assistant',
                message.text[:200] + '...' if len(message.text) > 200 else message.text
            ])
        
        return response
    export_conversation_flow.short_description = "Export conversation flow to CSV"
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('conversation-stats/', self.admin_site.admin_view(self.conversation_stats_view), name='message_conversation_stats'),
        ]
        return custom_urls + urls
    
    def changelist_view(self, request, extra_context=None):
        """Override changelist view to add custom links"""
        extra_context = extra_context or {}
        extra_context['show_conversation_stats_link'] = True
        return super().changelist_view(request, extra_context)
    
    def conversation_stats_view(self, request):
        """Custom admin view to show conversation statistics"""
        # Get statistics
        total_messages = Message.objects.count()
        total_users = User.objects.count()
        messages_with_state = Message.objects.filter(state__isnull=False).count()
        
        # State distribution
        state_distribution = Message.objects.filter(state__isnull=False).values('state').annotate(
            count=Count('state')
        ).order_by('-count')
        
        # User activity
        user_activity = Message.objects.values('user__username').annotate(
            message_count=Count('id'),
            session_count=Count('session_id', distinct=True)
        ).order_by('-message_count')[:10]
        
        # Recent conversations
        recent_conversations = Message.objects.select_related('user').order_by('-timestamp')[:20]
        
        context = {
            'title': 'Conversation Statistics',
            'total_messages': total_messages,
            'total_users': total_users,
            'messages_with_state': messages_with_state,
            'state_distribution': state_distribution,
            'user_activity': user_activity,
            'recent_conversations': recent_conversations,
        }
        
        return render(request, 'admin/api/message/conversation_stats.html', context)

@admin.register(UserMemoryState)
class UserMemoryStateAdmin(admin.ModelAdmin):
    list_display = ('user', 'memory_state_preview', 'last_processed_message')
    list_filter = ('user',)
    search_fields = ('user',)
    
    def memory_state_preview(self, obj):
        return obj.current_memory[:50] + "..." if len(obj.current_memory) > 50 else obj.current_memory
    memory_state_preview.short_description = 'Current Memory'

    def last_processed_message(self, obj):
        if obj.last_processed_message:
            return f"{obj.last_processed_message.text[:30]}... ({obj.last_processed_message.timestamp})"
        return "No messages processed"
    last_processed_message.short_description = 'Last Processed Message'