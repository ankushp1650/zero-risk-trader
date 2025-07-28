from django.contrib.auth.models import User
from django.utils import timezone
from django.db import models
from django.db import models
from django.contrib.auth.models import User


class Final_holding(models.Model):  # Class name should follow Python naming conventions (CamelCase)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='holdings')
    stock_symbol = models.CharField(max_length=20)  # For example, 'RELIANCE.BSE'
    quantity = models.PositiveIntegerField(default=0)  # Quantity of the stock held
    investment_value = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    current_value = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    pnl = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)

    def __str__(self):
        return f"{self.user.username} - {self.stock_symbol}: {self.quantity}"

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'stock_symbol'], name='unique_user_stock')
        ]  # Ensure no duplicate holdings for the same user and stock


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    api_key = models.CharField(max_length=255, blank=True, null=True)
    last_data_fetch_date = models.DateField(blank=True, null=True)

    def __str__(self):
        return f"{self.user.username}'s API Key: {self.api_key}"


class Portfolio(models.Model):
    """
    Model to represent a user's portfolio, including their cash balance.
    Each user has one portfolio.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # The user who made the transaction

    symbol = models.CharField(max_length=255, unique=True, null=True, blank=True)  # Stock symbol like 'BSE'
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)

    cash_balance = models.DecimalField(max_digits=12, decimal_places=2,
                                       default=1000000.00)  # Default balance is 100,000
    available_margin = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    invested_margin = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    current_value = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    pnl = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)

    def __str__(self):
        return f"{self.user.username}'s Portfolio - Cash Balance: {self.cash_balance:.2f}"


class Transaction(models.Model):
    """
    Model to represent transactions (buying or selling) of stocks.
    Each transaction is linked to a user and a stock.
    """
    TRANSACTION_TYPES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)  # The user who made the transaction
    stock = models.CharField(max_length=255, null=True, blank=True)  # The stock that was traded
    transaction_type = models.CharField(max_length=4, choices=TRANSACTION_TYPES)  # Whether it's a buy or sell
    quantity = models.IntegerField()  # Quantity of stock bought or sold
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    price_at_transaction = models.DecimalField(max_digits=10,
                                               decimal_places=2)  # Price of stock at the time of transaction
    # date = models.DateTimeField(auto_now_add=True)  # The date and time when the transaction was made
    date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username} {self.transaction_type} {self.quantity} {self.stock} {self.cost}   {self.price_at_transaction} on {self.date}"


class StockJason(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    last_refresh_data = models.DateField(default='2000-01-01')  # Default date for existing rows
    stock_name = models.CharField(max_length=100)  # Field for stock name
    meta_data = models.JSONField()  # Field for storing JSON data

    def __str__(self):
        return f'{self.user.username} - {self.stock_name}: {self.last_refresh_data}'

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'stock_name', 'last_refresh_data'], name='unique_user_stock_mydata')
        ]
        indexes = [
            models.Index(fields=['last_refresh_data'], name='last_refresh_idx')  # Adding the index on last_refresh_data
        ]


class CurrentPrice(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Foreign key to User
    date = models.DateField(default='2000-01-01')  # Default date for existing rows
    stock_name = models.CharField(max_length=100, null=True)  # Field for stock name
    open = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Field for 'Open' price
    high = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Field for 'High' price
    low = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Field for 'Low' price
    close = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Field for 'Close' price
    volume = models.DecimalField(max_digits=10, decimal_places=2, null=True)  # Field for stock 'Volume'

    def __str__(self):
        return f'{self.user.username} - {self.stock_name}: {self.date}'

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'stock_name', 'date', 'close'], name='unique_stock_currentprice')
        ]
        indexes = [
            models.Index(fields=['date'], name='date_index')
        ]


class UserTrade(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock_symbol = models.CharField(max_length=50)
    buy_price = models.DecimalField(max_digits=10, decimal_places=2)
    sell_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    quantity = models.IntegerField()
    buy_date = models.DateTimeField()
    sell_date = models.DateTimeField(null=True, blank=True)
    cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)  # <<== New field

    def __str__(self):
        return f"{self.user.username} - {self.stock_symbol} ({self.quantity})"


class LSTMModelStorage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock_name = models.CharField(max_length=100)
    model_blob = models.BinaryField()
    scaler_blob = models.BinaryField()
    sequence_length = models.IntegerField(default=60)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'stock_name')

    def __str__(self):
        return f"LSTM Model for {self.stock_name} (User: {self.user.username})"



class StockPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)

    stock_name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=50)
    last_refreshed = models.DateField()
    date = models.DateField()
    low = models.FloatField()
    open = models.FloatField()
    high = models.FloatField()
    close = models.FloatField()

    linear_model = models.FloatField()
    lstm_model = models.FloatField()
    decision_tree_model = models.FloatField()
    random_forest_model = models.FloatField()
    svm_model = models.FloatField()


class StockPerformance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)

    stock_name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=50)

    linear_model_success_rate = models.FloatField()
    linear_model_directional_success_rate = models.FloatField()
    linear_model_avg_error = models.FloatField()

    decision_tree_model_success_rate = models.FloatField()
    decision_tree_model_directional_success_rate = models.FloatField()
    decision_tree_model_avg_error = models.FloatField()

    random_forest_model_success_rate = models.FloatField()
    random_forest_model_directional_success_rate = models.FloatField()
    random_forest_model_avg_error = models.FloatField()

    svm_model_success_rate = models.FloatField()
    svm_model_directional_success_rate = models.FloatField()
    svm_model_avg_error = models.FloatField()

    lstm_model_success_rate = models.FloatField()
    lstm_model_directional_success_rate = models.FloatField()
    lstm_model_avg_error = models.FloatField()


class BestModelRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    stock_name = models.CharField(max_length=100)
    model_name = models.CharField(max_length=100)
    success_rate = models.FloatField()
    directional_success_rate = models.FloatField()
    average_error = models.FloatField()
    normalized_models_score = models.FloatField()
    best_model = models.CharField(max_length=3)  # "Yes" or "No"

    class Meta:
        unique_together = ('user', 'stock_name', 'model_name', 'best_model')