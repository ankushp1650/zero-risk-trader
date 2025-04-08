from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('register-alpha-vantage/', views.alpha_vantage_register, name='alpha_vantage_register'),
    # path('api_key_form/', api_key_form, name='api_key_form'),
    path('save-api-key/', views.save_api_key, name='save_api_key'),
    path('select_symbol/', views.symbol_selection_stock, name='select_symbol'),
    path('stock_analysis/', views.stock_dataframe, name='stock_analysis'),
    path('transaction_history/', views.transaction_history_view, name='transaction_history'),
    # path('graph/', views.display_graph, name='display_graph'),
    path('investment_view/', views.investment_platform, name='investment_view'),
    path('buy_stock/', views.buy_stock, name='buy_stock'),
    path('sell_stock/', views.sell_stock, name='sell_stock'),
    path('platform/', views.investment_view, name='platform'),
    path('activate/<uidb64>/<token>/', views.activate_view, name='activate'),
    path('verify-email/<uidb64>/<token>/', views.verify_email, name='verify_email'),
    path('email-verification-sent/', views.email_verification_sent_view, name='email_verification_sent'),
    path('password_reset/', views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),

]
