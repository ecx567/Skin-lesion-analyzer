from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('skin_detector', '0003_add_user_field'),
    ]

    operations = [
        migrations.CreateModel(
            name='SocialAccount',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('provider', models.CharField(max_length=32)),
                ('provider_user_id', models.CharField(max_length=255)),
                ('email', models.EmailField(max_length=254)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='social_accounts', to='auth.user')),
            ],
            options={
                'unique_together': {('provider', 'provider_user_id')},
            },
        ),
        migrations.AddIndex(
            model_name='socialaccount',
            index=models.Index(fields=['provider', 'provider_user_id'], name='social_provider_idx'),
        ),
    ]
