from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm


class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class ApiKeyForm(forms.Form):
    api_key = forms.CharField(max_length=255, widget=forms.TextInput(attrs={'placeholder': 'Enter your API key'}))


class AlphaVantageRegistrationForm(forms.Form):
    email = forms.EmailField(label='Your Email', required=True)


def int_choices(start, end, step=1):
    return [(i, str(i)) for i in range(start, end + 1, step)]


class HyperparameterForm(forms.Form):
    # Static Stock Symbols
    stock_symbol = forms.ChoiceField(
        choices=[('TCS.BSE', 'TCS.BSE'),
                 ('RELIANCE.BSE', 'RELIANCE.BSE'),
                 ('HDFCBANK.BSE', 'HDFCBANK.BSE'),
                 ('BHARTIARTL.BSE', 'BHARTIARTL.BSE'),
                 ('ICICIBANK.BSE', 'ICICIBANK.BSE')],
        widget=forms.Select(),
        initial='TCS.BSE'
    )

    # Linear Regression
    fit_intercept = forms.ChoiceField(
        choices=[('True', 'True'), ('False', 'False')],
        widget=forms.Select(),
        initial='True'
    )
    regularization_type = forms.ChoiceField(
        choices=[('none', 'None'), ('ridge', 'Ridge'), ('lasso', 'Lasso')],
        widget=forms.Select(),
        initial='none'
    )
    alpha = forms.FloatField(
        initial=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01'}),
        required=False,
        label='Alpha (for Ridge/Lasso)'
    )

    # Decision Tree
    max_depth = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in range(1, 31)] ,
        widget=forms.Select(),
        initial=5
    )
    min_samples_split = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in range(1, 31)] ,
        widget=forms.Select(),
        initial=2
    )

    criterion = forms.ChoiceField(
        choices=[('squared_error', 'squared_error'),
                 ('friedman_mse', 'friedman_mse'),
                 ('absolute_error', 'absolute_error')],
        widget=forms.Select(),
        initial='squared_error'
    )

    # Random Forest
    n_estimators = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in [50, 100, 150, 200]],
        widget=forms.Select(),
        initial='100',
        label='Number of Estimators'
    )

    rf_max_depth = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in range(1, 21)] + [('None', 'None')],
        widget=forms.Select(),
        initial='None',
        label='Max Depth'
    )

    min_samples_split_rf = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in range(2, 21)],
        widget=forms.Select(),
        initial='2',
        label='Min Samples Split'
    )

    criterion_rf = forms.ChoiceField(
        choices=[
            ('squared_error', 'squared_error'),
            ('absolute_error', 'absolute_error'),
            ('friedman_mse', 'friedman_mse')
        ],
        widget=forms.Select(),
        initial='squared_error',
        label='Criterion (Split Quality)'
    )

    # SVM
    kernel = forms.ChoiceField(
        choices=[('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid')],
        widget=forms.Select(),
        initial='rbf'
    )

    C = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in [0.01, 0.1, 1, 10, 100, 1000]],  # Dropdown with common values
        widget=forms.Select(),
        initial='1.0'
    )

    epsilon = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]],  # Dropdown with common values
        widget=forms.Select(),
        initial='0.1'
    )

    gamma = forms.ChoiceField(
        choices=[('scale', 'scale'), ('auto', 'auto')],
        widget=forms.Select(),
        initial='scale'
    )

    degree = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in range(1, 11)],  # Dropdown for degree (1-10)
        widget=forms.Select(),
        initial='3'
    )

    coef0 = forms.ChoiceField(
        choices=[(str(i), str(i)) for i in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]],  # Dropdown options for coef0
        widget=forms.Select(),
        initial='0.0'
    )

    # LSTM
    lstm_units = forms.ChoiceField(
        choices=int_choices(16, 512, 16),  # e.g., 16, 32, ..., 512
        initial=32,
        widget=forms.Select()
    )

    epochs = forms.ChoiceField(
        choices=int_choices(10, 1000, 10),  # e.g., 10, 20, ..., 1000
        initial=20,
        widget=forms.Select()
    )

    batch_size = forms.ChoiceField(
        choices=int_choices(8, 512, 8),  # e.g., 8, 16, ..., 512
        initial=32,
        widget=forms.Select()
    )

    num_layers = forms.ChoiceField(
        choices=int_choices(1, 5),
        initial=1,
        widget=forms.Select()
    )

    learning_rate = forms.ChoiceField(
        choices=[(str(v), str(v)) for v in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]],
        initial='0.001',
        widget=forms.Select()
    )

    dropout = forms.ChoiceField(
        choices=[(str(v), str(v)) for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]],
        initial='0.2',
        widget=forms.Select()
    )

    optimizer = forms.ChoiceField(
        choices=[('adam', 'Adam'), ('sgd', 'SGD'), ('rmsprop', 'RMSProp')],
        initial='adam',
        widget=forms.Select()
    )

    loss_function = forms.ChoiceField(
        choices=[('mse', 'Mean Squared Error'), ('mae', 'Mean Absolute Error')],
        initial='mse',
        widget=forms.Select()
    )

    activation_function = forms.ChoiceField(
        choices=[('linear', 'Linear'), ('relu', 'ReLU'), ('sigmoid', 'Sigmoid')],
        initial='linear',
        widget=forms.Select()
    )
